# Kolmogorov Arnold Network (KAN) Based Epigenetic Clock

Authors: Vikram Dhillon, Suresh Balasubramanian

Epigenetic clocks are biomarkers of aging based on DNA methylation levels. These clocks can predict biological age, which may differ from chronological age, providing insights into the biology of aging. DNA methylation is measured at specific CpG sites, where values range from 0 (unmethylated) to 1 (fully methylated). In this project, I extended the [KAN 2.0](https://github.com/KindXiaoming/pykan) implementation to develop a biological application: constructing an epigenetic clock that predicts biological age from DNA methylation data. The model uses a novel two-stage approach combining an initial KAN model with a residual model for improved accuracy. 

**Features**
  - Advanced feature selection using multiple methods:
    - Stability selection through bootstrapped correlation analysis
    - Mutual information scoring
    - ElasticNet-based feature importance
    - Variance thresholding
  - Two-stage model architecture:
    - Initial KAN model for base predictions
    - Residual KAN model for error correction
  - Comprehensive cross-dataset validation
  - Built-in handling of missing CpG sites through MICE imputation
  - Calculation of epigenetic age acceleration

### Data Source and Processing
The data source for this project was the EWAS Data Hub, specifically the dataset on [DNA methylation changes with age](https://download.cncb.ac.cn/ewas/datahub/download/age_methylation_v1.zip). After downloading the raw dataset, initial preprocessing steps were performed to structure the data for model training.

Data Transformation: The raw data was transposed so that CpG sites (methylation features) became columns, and samples (patient IDs) became rows, following the typical format for machine learning tasks where features are represented as columns for each observation.

Handling Missing Values: Columns (CpG sites) with more than 30% missing values were removed to ensure data quality. This filtering process resulted in a cleaned dataset with 8,000 samples available for modeling.

Data Availability: The original training dataset can be accessed directly from the EWAS Data Hub. A fully processed version used in this analysis is also publicly available on [Hugging Face](https://huggingface.co/datasets/dhillonv10/EWAS/tree/main)

### Results

Initial model training:
```
Performing efficient feature selection...
Features after variance thresholding: 69803
Features after stability selection: 2296
Final number of selected features: 200
Selected 200 CpG sites

Combined Model Performance:
R^2: 0.9536537528038025
Explained Variance: 0.953670084476471
Mean Absolute Error (MAE): 3.6282405853271484
Mean Squared Error (MSE): 26.685400009155273
```

Inference:

```
Performing inference on GSE40279:
Number of selected CpG sites: 200
Number of CpG sites in inference data: 473034
Number of matching CpG sites: 200
Warning: 0 CpG sites are completely missing in the inference data.
Completely missing CpG sites: []
Total missing values before imputation: 0
Shape of imputed values: (656, 200)
Shape expected based on top CpG sites: (656, 200)
Missing values after imputation: 0
Inference tensor shape: torch.Size([656, 200])
Expected shape based on top CpG sites: (656, 200)
Inference MAE: 4.502884608943288
Inference MSE: 33.16915284143427
Inference R-squared: 0.8470330649019633

First few rows of inference results for GSE40279:
   True Age  Predicted Age  Epigenetic Age Acceleration
0      67.0      66.039261                    -0.960739
1      89.0      83.227829                    -5.772171
2      66.0      67.181793                     1.181793
3      64.0      72.526657                     8.526657
4      62.0      58.510544                    -3.489456
```

```
Performing inference on GSE52588:
Number of selected CpG sites: 200
Number of CpG sites in inference data: 485577
Number of matching CpG sites: 200
Warning: 0 CpG sites are completely missing in the inference data.
Completely missing CpG sites: []
Total missing values before imputation: 6
Shape of imputed values: (87, 200)
Shape expected based on top CpG sites: (87, 200)
Missing values after imputation: 0
Inference tensor shape: torch.Size([87, 200])
Expected shape based on top CpG sites: (87, 200)
Inference MAE: 4.76802161644245
Inference MSE: 33.89300837538403
Inference R-squared: 0.8942906977423506

First few rows of inference results for GSE52588:
   True Age  Predicted Age  Epigenetic Age Acceleration
0      18.0      21.857203                     3.857203
1      12.0      19.268435                     7.268435
2      13.0      24.474918                    11.474918
3      24.0      28.435495                     4.435495
4      33.0      38.924927                     5.924927
```

Summary of inference results:
```
GSE40279:
  MAE: 4.5029
  MSE: 33.1692
  R-squared: 0.8470

GSE52588:
  MAE: 4.7680
  MSE: 33.8930
  R-squared: 0.8943
```
