"""
Epigenetic Clock Implementation using Kolmogorov-Arnold-Networks (KAN)
-------------------------------------------------------------------

This implementation provides a robust epigenetic clock that predicts biological age
from DNA methylation data using a combination of feature selection and neural network
models based on Kolmogorov Arnold Networks (KAN).

Authors: Vikram Dhillon, Suresh Balasubramanian
License: MIT License
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kan import MultKAN, create_dataset
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from biolearn.data_library import DataLibrary
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import joblib
import traceback
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Number of CpG sites to select for the final model
N_FEATURES = 200

def efficient_methylation_feature_selection(X, y, feature_names, n_features=N_FEATURES, n_cv=5):
    """    
    This function implements a multi-step feature selection process:
    1. Removes low-variance features
    2. Performs stability selection through bootstrapped correlation analysis
    3. Calculates multiple feature importance metrics
    4. Combines different metrics for final feature selection
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Methylation data matrix (samples × CpG sites)
    y : numpy.array
        Target variable (chronological age)
    feature_names : numpy.array
        Names of the CpG sites
    n_features : int
        Number of features to select (default: 200)
    n_cv : int
        Number of cross-validation folds for ElasticNet (default: 5)
        
    Returns:
    --------
    tuple
        (selected_features, scaler) where:
        - selected_features: list of selected CpG site names
        - scaler: fitted StandardScaler for the selected features
    """
    # Remove low-variance features
    variance_threshold = 0.001
    variances = X.var()
    X = X.loc[:, variances > variance_threshold]
    feature_names = feature_names[variances > variance_threshold]
    print(f"Features after variance thresholding: {X.shape[1]}")
    
    # Stability selection through bootstrapped correlation
    n_bootstraps = 100
    correlation_stability = np.zeros(X.shape[1])
    for i in range(n_bootstraps):
        boot_idx = np.random.choice(len(y), len(y), replace=True)
        correlations = X.iloc[boot_idx].apply(
            lambda col: stats.spearmanr(col, y[boot_idx])[0]
        )
        correlation_stability += np.abs(correlations) > 0.3
    
    # Select stable features
    stability_mask = correlation_stability >= (n_bootstraps * 0.6)
    stable_features = X.columns[stability_mask]
    X_stable = X[stable_features]
    print(f"Features after stability selection: {len(stable_features)}")
    
    # Calculate multiple feature importance metrics
    correlations = X_stable.apply(lambda col: abs(stats.spearmanr(col, y)[0]))
    mi_scores = mutual_info_regression(X_stable.values, y, n_jobs=-1)
    mi_scores = pd.Series(mi_scores, index=X_stable.columns)
    
    # ElasticNet feature importance
    scaler_elastic = StandardScaler()
    X_scaled = scaler_elastic.fit_transform(X_stable)
    elasticnet = ElasticNetCV(
        cv=n_cv,
        random_state=42,
        max_iter=50000,
        n_jobs=-1,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
    )
    elasticnet.fit(X_scaled, y)
    elastic_scores = pd.Series(np.abs(elasticnet.coef_), index=X_stable.columns)
    
    # Normalize and combine scores
    correlations_norm = (correlations - correlations.min()) / (correlations.max() - correlations.min())
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    elastic_scores_norm = (elastic_scores - elastic_scores.min()) / (elastic_scores.max() - elastic_scores.min())
    
    combined_scores = pd.DataFrame({
        'correlation': correlations_norm,
        'mutual_info': mi_scores_norm,
        'elastic_net': elastic_scores_norm
    })
    
    final_scores = (
        0.3 * combined_scores['correlation'] +
        0.3 * combined_scores['mutual_info'] +
        0.4 * combined_scores['elastic_net']
    )
    
    # Select final features and create scaler
    selected_features = final_scores.nlargest(n_features).index
    final_scaler = StandardScaler()
    final_scaler.fit(X[selected_features])
    
    # Save feature importance scores
    feature_importance = pd.DataFrame({
        'correlation': correlations[selected_features],
        'mutual_info': mi_scores[selected_features],
        'elastic_net': elastic_scores[selected_features],
        'final_score': final_scores[selected_features]
    })
    feature_importance.to_csv('feature_importance_scores.csv')
    
    return selected_features.tolist(), final_scaler

def perform_inference(dataset_name, initial_model, residual_model, top_cpg_names, scaler, device):
    """
    Performs inference on a new dataset using the trained epigenetic clock model.
    
    This function:
    1. Loads and pre-processes a new dataset
    2. Handles missing CpG sites through imputation
    3. Makes predictions using both initial and residual models
    4. Calculates epigenetic age acceleration
    5. Saves and returns performance metrics
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset in the biolearn library
    initial_model : MultKAN
        Trained initial KAN model
    residual_model : MultKAN
        Trained residual KAN model
    top_cpg_names : list
        Names of the selected CpG sites
    scaler : StandardScaler
        Fitted scaler for the features
    device : torch.device
        Device to run the models on
        
    Returns:
    --------
    tuple
        (mae, mse, r2) - Mean Absolute Error, Mean Squared Error, and R-squared scores
    """
    print(f"\nPerforming inference on {dataset_name}:")

    try:
        # Load and prepare inference data
        inference_data = DataLibrary().get(dataset_name).load()
        X_inference = inference_data.dnam.transpose()
        y_inference = inference_data.metadata['age'].values

        # Print diagnostic information
        print(f"Number of selected CpG sites: {len(top_cpg_names)}")
        print(f"Number of CpG sites in inference data: {len(X_inference.columns)}")
        print(f"Number of matching CpG sites: {len(set(top_cpg_names).intersection(set(X_inference.columns)))}")

        # Prepare data for inference
        pruned_data = X_inference.reindex(columns=top_cpg_names)
        missing_columns = pruned_data.columns[pruned_data.isna().all()].tolist()
        print(f"Warning: {len(missing_columns)} CpG sites are completely missing in the inference data.")
        pruned_data[missing_columns] = pruned_data[missing_columns].fillna(0)

        # Perform imputation
        mice_imputer = IterativeImputer(max_iter=20, random_state=0)
        imputed_values = mice_imputer.fit_transform(pruned_data)
        pruned_data_imputed = pd.DataFrame(
            imputed_values, 
            columns=top_cpg_names,
            index=pruned_data.index
        )

        # Scale the data
        pruned_data_scaled = scaler.transform(pruned_data_imputed)
        X_inference_tensor = torch.tensor(pruned_data_scaled, dtype=torch.float32).to(device)

        # Make predictions
        initial_model.eval()
        residual_model.eval()
        with torch.no_grad():
            initial_pred = initial_model(X_inference_tensor)
            residual_pred = residual_model(X_inference_tensor)
            final_pred = initial_pred + residual_pred

        final_pred_np = final_pred.cpu().numpy().flatten()

        # Calculate metrics
        inference_mae = mean_absolute_error(y_inference, final_pred_np)
        inference_mse = mean_squared_error(y_inference, final_pred_np)
        inference_r2 = r2_score(y_inference, final_pred_np)

        # Calculate epigenetic age acceleration
        epigenetic_age_acceleration = final_pred_np - y_inference

        # Save results
        inference_results = pd.DataFrame({
            'True Age': y_inference,
            'Predicted Age': final_pred_np,
            'Epigenetic Age Acceleration': epigenetic_age_acceleration
        })
        inference_results.to_csv(f'inference_results_{dataset_name}.csv', index=False)

        return inference_mae, inference_mse, inference_r2

    except Exception as e:
        print(f"Error occurred during inference on {dataset_name}: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None, None, None

"""
Main execution script
----------------------------------------------------------

This script implements a two-stage model for epigenetic age prediction:
1. Initial KAN model for base predictions
2. Residual KAN model for error correction

The implementation includes:
- Feature selection
- Model training
- Model evaluation
- Cross-dataset validation
- Result storage and analysis
"""

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------ Data Loading and Preprocessing ------------------------#
# Load the training data
file_path = 'processed_data.csv'
data = pd.read_csv(file_path, header=None, low_memory=False)

# Extract features and target variable
feature_names = data.iloc[1, 4:].values
y = data.iloc[2:, 2].astype(float).values  # Age values
X = data.iloc[2:, 4:].astype(float).values  # Methylation values

# Convert to DataFrame for easier manipulation
X = pd.DataFrame(X, columns=feature_names)

#------------------------ Feature Selection ------------------------#
print("Performing efficient feature selection...")
top_cpg_names, scaler = efficient_methylation_feature_selection(
    X, y, feature_names, n_features=N_FEATURES
)
print(f"Selected {len(top_cpg_names)} CpG sites")

# Prepare scaled dataset with selected features
X_top = X[top_cpg_names]
X_top_scaled = scaler.transform(X_top)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_top_scaled, y, test_size=0.2, random_state=42
)

#------------------------ Data Preparation ------------------------#
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Create dataset dictionary for KAN models
dataset = {
    'train_input': X_train_tensor,
    'train_label': y_train_tensor,
    'test_input': X_test_tensor,
    'test_label': y_test_tensor
}

#------------------------ Initial Model Training ------------------------#
# Define and train the initial KAN model
initial_model = MultKAN(
    width=[N_FEATURES, 1024, 512, 1],
    grid=5,
    k=3,
    noise_scale=0.1,
    base_fun='silu',
    device=device
)

# Train initial model
initial_results = initial_model.fit(
    dataset, 
    opt="LBFGS", 
    steps=200, 
    lamb=0.1, 
    lr=0.01, 
    batch=256, 
    log=10
)

#------------------------ Residual Model Training ------------------------#
# Calculate residuals for residual model training
initial_model.eval()
with torch.no_grad():
    y_train_pred = initial_model(X_train_tensor)
    y_test_pred = initial_model(X_test_tensor)

residuals_train = y_train_tensor - y_train_pred
residuals_test = y_test_tensor - y_test_pred

# Create residual dataset
residual_dataset = {
    'train_input': X_train_tensor,
    'train_label': residuals_train,
    'test_input': X_test_tensor,
    'test_label': residuals_test
}

# Define and train residual model
residual_model = MultKAN(
    width=[N_FEATURES, 2048, 1024, 1],
    grid=5,
    k=3,
    noise_scale=0.05,
    base_fun='silu',
    device=device
)

# Train residual model
residual_results = residual_model.fit(
    residual_dataset, 
    opt="LBFGS", 
    steps=200, 
    lamb=50, 
    lr=0.5, 
    batch=256, 
    log=10
)

#------------------------ Model Evaluation ------------------------#
# Evaluate combined model performance
initial_model.eval()
residual_model.eval()
with torch.no_grad():
    initial_pred = initial_model(X_test_tensor)
    residual_pred = residual_model(X_test_tensor)
    final_pred = initial_pred + residual_pred

# Calculate performance metrics
final_pred_np = final_pred.cpu().numpy()
y_test_np = y_test_tensor.cpu().numpy()

r2 = r2_score(y_test_np, final_pred_np)
explained_var = explained_variance_score(y_test_np, final_pred_np)
mae = mean_absolute_error(y_test_np, final_pred_np)
mse = mean_squared_error(y_test_np, final_pred_np)

print("Combined Model Performance:")
print(f"R^2: {r2}")
print(f"Explained Variance: {explained_var}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

#------------------------ Save Models and Parameters ------------------------#
# Save model states
torch.save(initial_model.state_dict(), 'initial_kan_model.pth')
torch.save(residual_model.state_dict(), 'residual_kan_model.pth')

# Save model architectures
model_params = {
    'initial_model': {
        'width': initial_model.width,
        'grid': initial_model.grid,
        'k': initial_model.k,
        'noise_scale': 0.1,
        'base_fun': initial_model.base_fun_name
    },
    'residual_model': {
        'width': residual_model.width,
        'grid': residual_model.grid,
        'k': residual_model.k,
        'noise_scale': 0.05,
        'base_fun': residual_model.base_fun_name
    }
}
with open('model_params.json', 'w') as f:
    json.dump(model_params, f)

# Save feature selection results
np.save(f'top_{N_FEATURES}_cpg_sites.npy', top_cpg_names)
joblib.dump(scaler, 'scaler.joblib')

#------------------------ Validation ------------------------#
# Test on two external datasets
datasets = ["GSE40279", "GSE52588"]
inference_results = {}

for dataset in datasets:
    mae, mse, r2 = perform_inference(
        dataset, initial_model, residual_model, 
        top_cpg_names, scaler, device
    )
    if mae is not None:
        inference_results[dataset] = {"MAE": mae, "MSE": mse, "R-squared": r2}
    else:
        print(f"Inference failed for dataset {dataset}")

#------------------------ Results Summary ------------------------#
# Print and save validation results
print("\nSummary of inference results for all datasets:")
for dataset, metrics in inference_results.items():
    print(f"\n{dataset}:")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R-squared: {metrics['R-squared']:.4f}")

if inference_results:
    all_results = pd.DataFrame(inference_results).T
    all_results.to_csv('all_inference_results.csv')
    print("\nAll inference results saved to 'all_inference_results.csv'")
else:
    print("\nNo valid inference results to save.")
