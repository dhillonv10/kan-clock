# Kolmogorov Arnold Network (KAN) Based Stacked Age-Specific Epigenetic Clock

**Authors:** Vikram Dhillon, Suresh Balasubramanian

Epigenetic clocks are biomarkers of aging that leverage DNA methylation patterns to estimate a biological age, which may differ from the chronological age, providing deeper insights into the molecular mechanisms of aging. In this project, we extend the [KAN 2.0](https://github.com/KindXiaoming/pykan) framework to develop a more sophisticated epigenetic clock model. Instead of a simple two-stage approach, we implement a *stacked, age-specific KAN model*, where samples are partitioned into age bins, and separate KAN-based residual models are trained for each bin to refine predictions. This approach enables the model to capture age-dependent nonlinearities and variations, improving accuracy across a wide range of ages.

**Key Features**
- **Robust Feature Selection:**  
  - Variance thresholding to remove low-informative features  
  - Stability selection via bootstrapped correlation  
  - Mutual information (MI) scoring  
  - ElasticNet-based feature importance  
- **Stacked Age-Specific Architecture:**  
  - Partition training samples into distinct age bins  
  - Train multiple stacked KAN models within each bin, each refining the residuals of the previous  
  - Produce more age-aware and stable predictions across the entire age spectrum  
- **Comprehensive Preprocessing and Validation:**  
  - MICE imputation for missing CpG values  
  - Scaling and normalization for consistent feature representation  
  - Extensive validation across different datasets to ensure generalizability

### Data Source and Processing
The primary data source for this project is studies deposited on NCBI Gene Expression Omnibus (GEO), and the [ComputAge Bench Dataset](https://huggingface.co/datasets/computage/computage_bench). Post-processing steps taken include:

- **Data Transformation:**  
  Original data was transposed so that each CpG site corresponds to a column and each sample (patient ID) corresponds to a row, aligning with standard machine learning data formats.
  
- **Missing Value Filtering:**  
  CpG sites with more than 30% missing values were removed to ensure data quality, leaving approximately 9,000 samples for model training.

- **Imputation and Normalization:**  
  Post-filtering, MICE imputation was used to estimate missing CpG values, followed by scaling to standardize feature distributions.

The prepared training and validation datasets can be accessed from [Hugging Face](https://huggingface.co/datasets/dhillonv10/KAN-clock) for replication and further analysis.


## Model training log

```console
Loading and splitting training data...
Performing feature selection...
Features after variance thresholding: 7905
Features after stability selection: 909
Final number of selected features: 200
Selected 200 CpG sites
Performing MICE imputation...
Scaling data...

Training Stacked Age-Specific KAN Models...

Training models for age bin 0
Age range: 0.0 - 31.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 0
saving model version 0.1
Training stack 2 for age bin 0
saving model version 0.1
Training stack 3 for age bin 0
saving model version 0.1
Training stack 4 for age bin 0
saving model version 0.1
Training stack 5 for age bin 0
saving model version 0.1

Training models for age bin 1
Age range: 31.0 - 39.7
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 1
saving model version 0.1
Training stack 2 for age bin 1
saving model version 0.1
Training stack 3 for age bin 1
saving model version 0.1
Training stack 4 for age bin 1
saving model version 0.1
Training stack 5 for age bin 1
saving model version 0.1

Training models for age bin 2
Age range: 39.7 - 45.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 2
saving model version 0.1
Training stack 2 for age bin 2
saving model version 0.1
Training stack 3 for age bin 2
saving model version 0.1
Training stack 4 for age bin 2
saving model version 0.1
Training stack 5 for age bin 2
saving model version 0.1

Training models for age bin 3
Age range: 45.0 - 49.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 3
saving model version 0.1
Training stack 2 for age bin 3
saving model version 0.1
Training stack 3 for age bin 3
saving model version 0.1
Training stack 4 for age bin 3
saving model version 0.1
Training stack 5 for age bin 3
saving model version 0.1

Training models for age bin 4
Age range: 49.0 - 53.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 4
saving model version 0.1
Training stack 2 for age bin 4
saving model version 0.1
Training stack 3 for age bin 4
saving model version 0.1
Training stack 4 for age bin 4
saving model version 0.1
Training stack 5 for age bin 4
saving model version 0.1

Training models for age bin 5
Age range: 53.0 - 57.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 5
saving model version 0.1
Training stack 2 for age bin 5
saving model version 0.1
Training stack 3 for age bin 5
saving model version 0.1
Training stack 4 for age bin 5
saving model version 0.1
Training stack 5 for age bin 5
saving model version 0.1

Training models for age bin 6
Age range: 57.0 - 62.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 6
saving model version 0.1
Training stack 2 for age bin 6
saving model version 0.1
Training stack 3 for age bin 6
saving model version 0.1
Training stack 4 for age bin 6
saving model version 0.1
Training stack 5 for age bin 6
saving model version 0.1

Training models for age bin 7
Age range: 62.0 - 68.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 7
saving model version 0.1
Training stack 2 for age bin 7
saving model version 0.1
Training stack 3 for age bin 7
saving model version 0.1
Training stack 4 for age bin 7
saving model version 0.1
Training stack 5 for age bin 7
saving model version 0.1

Training models for age bin 8
Age range: 68.0 - 75.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 8
saving model version 0.1
Training stack 2 for age bin 8
saving model version 0.1
Training stack 3 for age bin 8
saving model version 0.1
Training stack 4 for age bin 8
saving model version 0.1
Training stack 5 for age bin 8
saving model version 0.1

Training models for age bin 9
Age range: 75.0 - 96.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
checkpoint directory created: ./model
saving model version 0.0
Training stack 1 for age bin 9
saving model version 0.1
Training stack 2 for age bin 9
saving model version 0.1
Training stack 3 for age bin 9
saving model version 0.1
Training stack 4 for age bin 9
saving model version 0.1
Training stack 5 for age bin 9
saving model version 0.1

Evaluating on training/test splits...

Training Data Performance:
Train_MAE: 1.7821
Train_MSE: 6.5586
Train_RMSE: 2.5610
Train_R2: 0.9747
Train_Explained_Variance: 0.9747

Test Split Performance:
Test_MAE: 1.9035
Test_MSE: 7.8132
Test_RMSE: 2.7952
Test_R2: 0.9694
Test_Explained_Variance: 0.9694

Validating on external test set...

External Validation Set Performance:
Validation_MAE: 1.9215
Validation_MSE: 8.2296
Validation_RMSE: 2.8687
Validation_R2: 0.9673
Validation_Explained_Variance: 0.9674

Performing inference on multiple biolearn datasets...

Processing dataset: GSE40279
Applying MICE imputation to GSE40279...

Processing dataset: GSE42861
Applying MICE imputation to GSE42861...

Processing dataset: GSE52588
Applying MICE imputation to GSE52588...

Processing dataset: GSE51032
Applying MICE imputation to GSE51032...

Processing dataset: GSE64495
Applying MICE imputation to GSE64495...

Processing dataset: GSE41169
Applying MICE imputation to GSE41169...

Processing dataset: GSE69270
Warning: Only 199/200 CpG sites available
Applying MICE imputation to GSE69270...

Processing dataset: GSE157131
Applying MICE imputation to GSE157131...

Processing dataset: GSE51057
Applying MICE imputation to GSE51057...

Processing dataset: GSE73103
Warning: Only 199/200 CpG sites available
Applying MICE imputation to GSE73103...

Processing dataset: GSE132203
Applying MICE imputation to GSE132203...

Saving all results...
Saving models and preprocessing components...

All results saved to results_20241215_040541/

Overall Performance Summary:
      Dataset  N_Samples  N_Common_CpGs  CpG_Coverage_%       MAE        R2      RMSE
0    GSE40279        656            200           100.0  2.299605  0.951189  3.253330
1    GSE42861        689            200           100.0  1.531849  0.973413  1.921934
2    GSE52588         87            200           100.0  3.351902  0.925994  4.871161
3    GSE51032        845            200           100.0  1.347121  0.913410  2.123756
4    GSE64495        113            200           100.0  4.936124  0.773398  8.851532
5    GSE41169         95            200           100.0  2.516828  0.908641  3.091654
6    GSE69270        184            199            99.5  1.529045  0.713646  1.733030
7   GSE157131        946            200           100.0  1.749867  0.940061  2.371022
8    GSE51057        329            200           100.0  1.204809  0.959221  1.432153
9    GSE73103        355            199            99.5  4.962290 -0.642951  6.124137
10  GSE132203        795            200           100.0  3.608828  0.733986  6.345746

Execution completed successfully!
```
