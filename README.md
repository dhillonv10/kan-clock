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
