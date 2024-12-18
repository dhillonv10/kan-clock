"""
This script implements a feature selection and training pipeline for a stacked, age-specific
Kolmogorov Arnold Network (KAN) regression model, designed to estimate the biological age 
from epigenetic (DNA methylation) data—also known as an epigenetic clock. The process includes 
data loading, preprocessing (variance filtering, MICE imputation, scaling), and feature 
selection through stability tests and multiple scoring metrics (correlation, mutual 
information, and ElasticNet). It then trains a custom StackedAgeKAN model, which partitions 
samples by age bins and learns stacked residual models to produce more age-aware predictions.

Results are evaluated on both training/test splits and external validation datasets, then 
summarized and saved for future reference and deployment.

Authors: Vikram Dhillon, Suresh Balasubramanian
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kan import MultKAN, create_dataset
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from biolearn.data_library import DataLibrary

TRAIN_PATH = '/home/hmhvxd/scratch/clock/EWA/HMB/consolidated_train.csv'
TEST_PATH = '/home/hmhvxd/scratch/clock/EWA/HMB/consolidated_test.csv'
N_FEATURES = 200

class StackedAgeKAN:
    def __init__(self, n_stacks=5, n_age_bins=10, input_dim=100, device='cuda'):
        """
        Initialize the StackedAgeKAN model which fits multiple stacked KAN models 
        across different age bins. Each age bin gets its own stack of models, and 
        the predictions are combined to handle age-dependent variations.

        Parameters
        ----------
        n_stacks : int
            Number of stacked models per age bin.
        n_age_bins : int
            Number of age bins to split the data into.
        input_dim : int
            Dimensionality of the input features.
        device : str
            Device for PyTorch computations, e.g., 'cuda' or 'cpu'.
        """
        self.device = device
        self.n_stacks = n_stacks
        self.n_age_bins = n_age_bins
        self.age_bins = None
        self.age_models = []
        # Base network architecture definition: input -> hidden layers -> output
        self.base_width = [input_dim, 256, 128, 1]
        
    def _create_age_bins(self, ages):
        """
        Create age bins using percentiles for a balanced distribution of samples 
        across bins.

        Parameters
        ----------
        ages : torch.Tensor
            Tensor of age values.

        Returns
        -------
        torch.Tensor
            Tensor of age bin boundaries.
        """
        # Compute percentiles and create bin edges
        age_percentiles = torch.linspace(0, 100, self.n_age_bins + 1)
        self.age_bins = torch.tensor(
            np.percentile(ages.cpu().numpy(), age_percentiles),
            device=self.device
        )
        return self.age_bins
    
    def _get_age_bin_idx(self, ages):
        """
        Determine which bin each age belongs to.

        Parameters
        ----------
        ages : torch.Tensor
            Tensor of age values.

        Returns
        -------
        torch.Tensor
            Tensor of integer bin indices for each age.
        """
        # Create age bins if not already done
        if self.age_bins is None:
            self._create_age_bins(ages)
        
        # Bucketize ages into bin indices, then clamp to valid range
        bin_indices = torch.bucketize(ages.squeeze(), self.age_bins) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.n_age_bins - 1)
        return bin_indices
    
    def _create_age_specific_model(self, age_bin):
        """
        Create a stack of models (MultKAN) for a particular age bin.
        This allows for bin-specific model complexities and noise scales.

        Parameters
        ----------
        age_bin : int
            The current age bin index.

        Returns
        -------
        list
            A list of trained MultKAN models for the given age bin.
        """
        models = []
        # For each stack, we slightly modify the network width and noise_scale
        # to introduce diversity and specialization at different stack layers.
        for i in range(self.n_stacks):
            width = [self.base_width[0]] + \
                   [max(n//(i+1), 64) for n in self.base_width[1:-1]] + [1]
            
            model = MultKAN(
                width=width,
                grid=5,
                k=3,
                noise_scale=0.1/(i+1),
                base_fun='silu',
                device=self.device
            )
            models.append(model)
        return models
        
    def fit(self, dataset, steps=500):
        """
        Fit the stacked KAN models to the training data across age bins.

        Parameters
        ----------
        dataset : dict
            Dictionary containing training and test sets:
            {
                'train_input': torch.Tensor,
                'train_label': torch.Tensor,
                'test_input': torch.Tensor,
                'test_label': torch.Tensor
            }
        steps : int
            Number of optimization steps (iterations) per model.
        """
        X_train = dataset['train_input']
        y_train = dataset['train_label']
        
        # Create age bins based on training labels if not already created
        if self.age_bins is None:
            self._create_age_bins(y_train)
        
        # For each age bin, create a stack of models and train them sequentially
        for age_bin in range(self.n_age_bins):
            print(f"\nTraining models for age bin {age_bin}")
            print(f"Age range: {self.age_bins[age_bin]:.1f} - {self.age_bins[age_bin+1]:.1f}")
            
            # Select samples belonging to the current age bin
            bin_indices = self._get_age_bin_idx(y_train) == age_bin
            if not torch.any(bin_indices):
                # If no samples fall into this bin, skip training
                print(f"No samples in age bin {age_bin}, skipping...")
                continue
                
            X_bin = X_train[bin_indices]
            y_bin = y_train[bin_indices]
            
            # Create and train the stack of models for this bin
            current_targets = y_bin.clone()
            bin_models = self._create_age_specific_model(age_bin)
            
            for i, model in enumerate(bin_models):
                print(f"Training stack {i+1} for age bin {age_bin}")
                
                # Create a temporary dataset for this stack's training
                bin_dataset = {
                    'train_input': X_bin,
                    'train_label': current_targets,
                    'test_input': dataset['test_input'],
                    'test_label': dataset['test_label']
                }
                
                # Train the model using LBFGS with a decreasing learning rate and
                # increasing lambda (regularization) for each subsequent stack.
                model.fit(
                    bin_dataset,
                    opt="LBFGS",
                    steps=steps,
                    lamb=1*(i+1),
                    lr=0.0005/(i+1)
                )
                
                # After training, update the residuals (current_targets)
                # This will allow each subsequent model in the stack to 
                # learn to predict the remaining unexplained variance.
                with torch.no_grad():
                    predictions = model(X_bin)
                    current_targets = current_targets - predictions
            
            self.age_models.append(bin_models)

    def predict(self, X, ages):
        """
        Predict ages using the stacked KAN models.

        Parameters
        ----------
        X : torch.Tensor
            Input features for prediction.
        ages : torch.Tensor
            Age values for the samples (used to determine which bin-specific model to use).

        Returns
        -------
        torch.Tensor
            Predicted age values.
        """
        predictions = torch.zeros((X.shape[0], 1), device=self.device)
        
        # Determine the bin each age belongs to
        bin_indices = self._get_age_bin_idx(ages)
        
        # For each bin, collect predictions from the respective stack of models
        for age_bin in range(self.n_age_bins):
            bin_mask = bin_indices == age_bin
            if not torch.any(bin_mask):
                # If no samples fall in this bin, skip
                continue
                
            X_bin = X[bin_mask]
            bin_predictions = torch.zeros((X_bin.shape[0], 1), device=self.device)
            
            # Sum predictions from all stacked models in this bin
            for model in self.age_models[age_bin]:
                with torch.no_grad():
                    bin_predictions += model(X_bin)
                    
            # Assign these predictions to the corresponding samples
            predictions[bin_mask] = bin_predictions
            
        return predictions

    def state_dict(self):
        """
        Get a state dictionary for saving the trained model.
        
        Returns
        -------
        dict
            A dictionary containing model state, age bins, and model configurations.
        """
        state = {
            'age_bins': self.age_bins,
            'n_stacks': self.n_stacks,
            'n_age_bins': self.n_age_bins,
            'base_width': self.base_width,
        }
        
        # Save state for each model in each age bin
        state['age_models'] = []
        for bin_models in self.age_models:
            bin_state = []
            for model in bin_models:
                bin_state.append(model.state_dict())
            state['age_models'].append(bin_state)
        
        return state

    def load_state_dict(self, state_dict):
        """
        Load a previously saved state dictionary into the model.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing previously saved model state, including
            'age_bins', 'n_stacks', 'n_age_bins', 'base_width', and 'age_models'.
        """
        self.age_bins = state_dict['age_bins']
        self.n_stacks = state_dict['n_stacks']
        self.n_age_bins = state_dict['n_age_bins']
        self.base_width = state_dict['base_width']
        
        # Rebuild and load each model for each age bin
        self.age_models = []
        for bin_state in state_dict['age_models']:
            bin_models = self._create_age_specific_model(len(self.age_models))
            for model, model_state in zip(bin_models, bin_state):
                model.load_state_dict(model_state)
            self.age_models.append(bin_models)
            
def calculate_metrics(y_true, y_pred, prefix=""):
    """
    Calculate standard regression evaluation metrics: MAE, MSE, RMSE, R^2, and 
    Explained Variance.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    prefix : str
        Optional prefix for metric names (useful for labeling different datasets).

    Returns
    -------
    dict
        Dictionary of computed metrics with prefixed names.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    ev = float(explained_variance_score(y_true, y_pred))
    
    metrics = {
        f'{prefix}MAE': mae,
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': rmse,
        f'{prefix}R2': r2,
        f'{prefix}Explained_Variance': ev
    }
    return metrics
    
def load_and_prepare_data(train_path, test_size=0.3):
    """
    Load methylation data and corresponding ages from a CSV file, then 
    split into training and test sets.

    Parameters
    ----------
    train_path : str
        Path to the training data CSV file.
    test_size : float
        Fraction of the data to reserve as the test set.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Training and test feature sets.
    y_train, y_test : np.ndarray
        Training and test labels (ages).
    feature_names : np.ndarray
        Array of feature names.
    ids_train, ids_test : np.ndarray
        Sample IDs for training and test sets.
    """
    # Read training data from CSV
    train_data = pd.read_csv(train_path)
    
    # Extract necessary columns
    sample_ids = train_data['SampleID'].values
    ages = train_data['Age'].values
    feature_names = train_data.columns[2:].values
    methylation = train_data.iloc[:, 2:].values
    
    # Convert to DataFrame for convenience
    X = pd.DataFrame(methylation, columns=feature_names)
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, ages, sample_ids, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, feature_names, ids_train, ids_test

def efficient_methylation_feature_selection(X, y, feature_names, n_features=N_FEATURES, n_cv=5):
    """
    Perform an efficient feature selection pipeline for methylation data, 
    including low-variance filtering, stability selection via bootstrapped 
    correlation checks, and a combination of correlation, mutual information, 
    and ElasticNet scores.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of methylation data.
    y : np.ndarray
        Array of corresponding age values.
    feature_names : np.ndarray
        Names of features in X.
    n_features : int
        Desired number of final selected features.
    n_cv : int
        Number of cross-validation folds for ElasticNetCV.

    Returns
    -------
    list
        List of selected feature names.
    """
    # Filter out low-variance features
    variance_threshold = 0.001
    variances = X.var()
    X = X.loc[:, variances > variance_threshold]
    feature_names = feature_names[variances > variance_threshold]
    print(f"Features after variance thresholding: {X.shape[1]}")
    
    # Perform stability selection via bootstrapped correlation
    n_bootstraps = 100
    correlation_stability = np.zeros(X.shape[1])
    for i in range(n_bootstraps):
        # Sample with replacement
        boot_idx = np.random.choice(len(y), len(y), replace=True)
        # Compute Spearman correlation for each bootstrap sample
        correlations = X.iloc[boot_idx].apply(
            lambda col: stats.spearmanr(col, y[boot_idx])[0]
        )
        # Count how many times correlation exceeds threshold
        correlation_stability += np.abs(correlations) > 0.2
    
    # Keep features stable in at least 60% of bootstraps
    stability_mask = correlation_stability >= (n_bootstraps * 0.6)
    stable_features = X.columns[stability_mask]
    X_stable = X[stable_features]
    print(f"Features after stability selection: {len(stable_features)}")
    
    # Compute correlation scores with the full dataset
    correlations = X_stable.apply(lambda col: abs(stats.spearmanr(col, y)[0]))
    
    # Compute Mutual Information scores
    mi_scores = mutual_info_regression(X_stable.values, y, n_jobs=-1)
    mi_scores = pd.Series(mi_scores, index=X_stable.columns)
    
    # Fit an ElasticNetCV for additional feature importance
    scaler_elastic = StandardScaler()
    X_scaled = scaler_elastic.fit_transform(X_stable)
    elasticnet = ElasticNetCV(
        cv=n_cv, 
        random_state=42, 
        max_iter=70000, 
        n_jobs=-1, 
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
    )
    elasticnet.fit(X_scaled, y)
    elastic_scores = pd.Series(np.abs(elasticnet.coef_), index=X_stable.columns)
    
    # Normalize and combine scores
    correlations_norm = (correlations - correlations.min()) / (correlations.max() - correlations.min())
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    elastic_scores_norm = (elastic_scores - elastic_scores.min()) / (elastic_scores.max() - elastic_scores.min())
    
    # Combine all three normalized scores into a final score
    final_scores = (
        0.3 * correlations_norm +
        0.3 * mi_scores_norm +
        0.4 * elastic_scores_norm
    )
    
    selected_features = final_scores.nlargest(n_features).index    
    print(f"Final number of selected features: {len(selected_features)}")
    
    # Save feature importance scores for reference
    feature_importance = pd.DataFrame({
        'correlation': correlations[selected_features],
        'mutual_info': mi_scores[selected_features],
        'elastic_net': elastic_scores[selected_features],
        'final_score': final_scores[selected_features]
    })
    feature_importance.to_csv('feature_importance_scores.csv')
    
    return selected_features.tolist()

# Main execution flow
print("Loading and splitting training data...")
X_train, X_test, y_train, y_test, feature_names, train_ids, test_ids = load_and_prepare_data(TRAIN_PATH)

# Perform feature selection
print("Performing feature selection...")
top_cpg_names = efficient_methylation_feature_selection(X_train, y_train, feature_names, n_features=N_FEATURES)
print(f"Selected {len(top_cpg_names)} CpG sites")

# Prepare training data with only selected CpG sites
X_train_top = X_train[top_cpg_names]
X_test_top = X_test[top_cpg_names]

# Initialize and fit MICE imputer
print("Performing MICE imputation...")
mice_imputer = IterativeImputer(max_iter=100, random_state=42)
X_train_imputed = pd.DataFrame(
    mice_imputer.fit_transform(X_train_top),
    columns=X_train_top.columns
)
X_test_imputed = pd.DataFrame(
    mice_imputer.transform(X_test_top),
    columns=X_test_top.columns
)

# Scale the data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Prepare dictionary for training
training_dataset = {
    'train_input': X_train_tensor,
    'train_label': y_train_tensor,
    'test_input': X_test_tensor,
    'test_label': y_test_tensor
}

# Initialize and train the stacked age-specific KAN model
print("\nTraining Stacked Age-Specific KAN Models...")
stacked_model = StackedAgeKAN(
    n_stacks=5,
    n_age_bins=10,
    input_dim=N_FEATURES,
    device=device
)

stacked_model.fit(training_dataset, steps=300)

# Evaluate on training and test sets
print("\nEvaluating on training/test splits...")
with torch.no_grad():
    train_predictions = stacked_model.predict(X_train_tensor, y_train_tensor)
    test_predictions = stacked_model.predict(X_test_tensor, y_test_tensor)

train_metrics = calculate_metrics(
    y_train_tensor.cpu().numpy(),
    train_predictions.cpu().numpy(),
    prefix="Train_"
)

test_metrics = calculate_metrics(
    y_test_tensor.cpu().numpy(),
    test_predictions.cpu().numpy(),
    prefix="Test_"
)

print("\nTraining Data Performance:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTest Split Performance:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Validate on external test set
print("\nValidating on external test set...")
validation_data = pd.read_csv(TEST_PATH)
val_sample_ids = validation_data['SampleID'].values
val_ages = validation_data['Age'].values
X_val_full = validation_data.iloc[:, 2:]
X_val = X_val_full[top_cpg_names]

# Apply MICE imputation on validation set
X_val_imputed = pd.DataFrame(
    mice_imputer.transform(X_val),
    columns=X_val.columns
)

# Scale validation data
X_val_scaled = scaler.transform(X_val_imputed)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
val_ages_tensor = torch.tensor(val_ages, dtype=torch.float32).to(device)

# Generate validation predictions
with torch.no_grad():
    val_predictions = stacked_model.predict(X_val_tensor, val_ages_tensor)

validation_metrics = calculate_metrics(
    val_ages,
    val_predictions.cpu().numpy(),
    prefix="Validation_"
)

print("\nExternal Validation Set Performance:")
for metric, value in validation_metrics.items():
    print(f"{metric}: {value:.4f}")
    
from datetime import datetime
import os

# Create a results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)

# Inference on multiple biolearn datasets
print("\nPerforming inference on multiple biolearn datasets...")
datasets = [
    "GSE40279", "GSE42861", "GSE52588", "GSE51032", "GSE64495", 
    "GSE41169", "GSE69270", "GSE157131", "GSE51057", "GSE73103", 
    "GSE132203"
]

inference_results = {}

for dataset_name in datasets:
    print(f"\nProcessing dataset: {dataset_name}")
    try:
        # Load external dataset from biolearn
        data = DataLibrary().get(dataset_name).load()
        
        # Extract methylation data and corresponding ages
        methylation_data = data.dnam.transpose().values
        ages = data.metadata['age'].values
        
        # Filter out samples without age information
        valid_age_mask = ~np.isnan(ages)
        if not np.all(valid_age_mask):
            print(f"Warning: Removing {np.sum(~valid_age_mask)} samples with missing ages")
            methylation_data = methylation_data[valid_age_mask]
            ages = ages[valid_age_mask]
        
        if len(ages) == 0:
            raise ValueError("No valid samples with age information")
        
        # Identify which CpG sites are common to both training and this dataset
        available_cpgs = data.dnam.index.values
        common_cpgs = list(set(top_cpg_names) & set(available_cpgs))
        
        if len(common_cpgs) < len(top_cpg_names):
            print(f"Warning: Only {len(common_cpgs)}/{len(top_cpg_names)} CpG sites available")
        
        # Create a DataFrame to hold data for all selected features, 
        # filling missing features with NaN
        methylation_df = pd.DataFrame(
            methylation_data,
            columns=data.dnam.index.values
        )
        X_inference = pd.DataFrame(
            index=methylation_df.index,
            columns=top_cpg_names
        )
        
        for cpg in top_cpg_names:
            if cpg in methylation_df.columns:
                X_inference[cpg] = methylation_df[cpg]
            else:
                X_inference[cpg] = np.nan
        
        # Impute missing values in the inference dataset
        print(f"Applying MICE imputation to {dataset_name}...")
        X_inference_imputed = pd.DataFrame(
            mice_imputer.transform(X_inference),
            columns=X_inference.columns
        )
        
        # Scale the inference data
        X_inference_scaled = scaler.transform(X_inference_imputed)
        X_inference_tensor = torch.tensor(X_inference_scaled, dtype=torch.float32).to(device)
        ages_tensor = torch.tensor(ages, dtype=torch.float32).to(device)
        
        # Generate predictions
        with torch.no_grad():
            predictions = stacked_model.predict(X_inference_tensor, ages_tensor)
        
        # Calculate and store metrics
        dataset_metrics = calculate_metrics(
            ages,
            predictions.cpu().numpy(),
            prefix=f"{dataset_name}_"
        )
        
        inference_results[dataset_name] = {
            'metrics': dataset_metrics,
            'predictions': predictions.cpu().numpy().flatten(),
            'true_ages': ages,
            'n_samples': len(ages),
            'n_common_cpgs': len(common_cpgs),
            'coverage': len(common_cpgs) / len(top_cpg_names)
        }
        
    except Exception as e:
        # In case of errors, store the error message
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        inference_results[dataset_name] = {'error': str(e)}
        
# Save model and preprocessing states
print("\nSaving all results...")

print("Saving models and preprocessing components...")
torch.save({
    'stacked_model_state': stacked_model.state_dict(),
    'age_bins': stacked_model.age_bins,
    'model_config': {
        'n_stacks': stacked_model.n_stacks,
        'n_age_bins': stacked_model.n_age_bins,
        'input_dim': N_FEATURES
    }
}, os.path.join(results_dir, 'stacked_models.pt'))

joblib.dump({
    'scaler': scaler,
    'mice_imputer': mice_imputer,
    'top_cpg_names': top_cpg_names
}, os.path.join(results_dir, 'preprocessing.joblib'))

# Consolidate all metrics into a single JSON
all_metrics = {
    "Training_Metrics": train_metrics,
    "Test_Split_Metrics": test_metrics,
    "Validation_Metrics": validation_metrics,
    "Inference_Metrics": {
        dataset: results['metrics'] 
        for dataset, results in inference_results.items() 
        if 'error' not in results
    }
}

with open(os.path.join(results_dir, 'all_metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=4)

# Save detailed inference results, converting arrays to lists for JSON
serializable_results = {}
for dataset_name, results in inference_results.items():
    if 'error' in results:
        serializable_results[dataset_name] = {'error': str(results['error'])}
    else:
        serializable_results[dataset_name] = {
            'metrics': results['metrics'],
            'predictions': results['predictions'].tolist(),
            'true_ages': results['true_ages'].tolist(),
            'n_samples': results['n_samples'],
            'n_common_cpgs': results.get('n_common_cpgs', 0),
            'coverage': results.get('coverage', 0)
        }

with open(os.path.join(results_dir, 'inference_details.json'), 'w') as f:
    json.dump(serializable_results, f, indent=4)

# Create and save a summary DataFrame for inference results
summary_data = []
for dataset_name, results in inference_results.items():
    if 'error' not in results:
        summary_data.append({
            'Dataset': dataset_name,
            'N_Samples': results['n_samples'],
            'N_Common_CpGs': results.get('n_common_cpgs', 0),
            'CpG_Coverage_%': results.get('coverage', 0) * 100,
            'MAE': results['metrics'][f'{dataset_name}_MAE'],
            'R2': results['metrics'][f'{dataset_name}_R2'],
            'RMSE': results['metrics'][f'{dataset_name}_RMSE']
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(results_dir, 'inference_summary.csv'), index=False)

print(f"\nAll results saved to {results_dir}/")
print("\nOverall Performance Summary:")
print(summary_df.to_string())

# Final completion message
print("\nExecution completed successfully!")
print(f"Results are saved in: {results_dir}")
print("\nKey files saved:")
print(f"- Models and age bins: {os.path.join(results_dir, 'stacked_models.pt')}")
print(f"- Preprocessing components: {os.path.join(results_dir, 'preprocessing.joblib')}")
print(f"- All metrics: {os.path.join(results_dir, 'all_metrics.json')}")
print(f"- Detailed inference results: {os.path.join(results_dir, 'inference_details.json')}")
print(f"- Summary report: {os.path.join(results_dir, 'inference_summary.csv')}")
