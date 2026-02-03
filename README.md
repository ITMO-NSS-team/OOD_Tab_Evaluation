# OOD Tabular Evaluation: Meta-Feature Based Distribution Shift Protocol

## 🔍 Overview

This repository implements a universal, reproducible Out-of-Distribution (OOD) evaluation protocol for tabular data. It uses evolutionary optimization to create train-test splits that maximize meta-feature differences, enabling controlled investigation of model behavior under distributional shifts.


## ✨ Key Features

- **📊 Meta-Feature Based Splitting**: Optimizes data splits using dataset characteristics like mutual information and class concentration
- **🧬 Synthetic Data Generation**: Creates synthetic datasets matching target meta-feature distributions
- **🛡️ Robust Model Evaluation**: Includes IRM and DRO model implementations for OOD testing
- **📈 Comprehensive Benchmarking**: Tests on real-world tabular datasets with known shifts

## 🎯 Problem Statement

In empirical machine learning settings, the core assumption that training and test distributions are identical is often violated. This is particularly challenging in high-stakes domains (medical diagnostics, finance, climate monitoring) where model performance degradation under distributional shifts can have significant real-world implications.

Traditional tabular datasets lack mechanisms for constructing well-defined distributional shifts, making systematic OOD evaluation difficult. This work addresses this gap by introducing a principled protocol that enables controlled manipulation of dataset characteristics.

## 🧠 Methodology

### Proposed Approach

Our approach enhances OOD evaluation through meta-feature based splitting, enabling controlled distributional shifts without architectural changes. Unlike random splits that may not capture meaningful distributional differences, our evolutionary algorithm systematically constructs train-test partitions that maximize meta-feature disparities. The method applies constraints through the fitness function rather than modifying the data generation process, maintaining dataset integrity while enforcing interpretable geometric relationships. Additionally, our framework supports synthetic data generation that preserves specific meta-feature distributions, allowing researchers to create controlled datasets with desired statistical properties for comprehensive robustness testing. 

## 1️⃣ Meta-Feature Based Splitting

The core idea is formulating train-test partitioning as an optimization problem:

```
maximize: [mean(meta_feature₁(train))/mean(meta_feature₁(test)), ..., mean(meta_featureₙ(train))/mean(meta_featureₙ(test))]
subject to: |test_set| = α × |dataset|
```

Where meta-features include, for example:

 ℹ️ **Info-theory:**
 
- **Attribute Entropy** (`attr_ent`): Measures feature distribution complexity
- **Class Concentration** (`class_conc`): Quantifies class imbalance
- **Mutual Information** (`mut_inf`): Captures feature-target relationships
- **Interquartile Range** (`iq_range`): Describes distribution spread
  
 🔢 **Statistical:**
- **Joint Entropy** (`joint_ent`): Measures overall dataset complexity
- **Kurtosis** (`kurtosis`): Measures distribution tail heaviness
- **Eigenvalues** (`eigenvalues`): Captures data structure variance

### Evolutionary Algorithm

The optimization uses a genetic algorithm with:
- **Population**: Lists of indices representing test set assignments
- **Fitness**: Ratio-based distance between meta-feature values (train/test ratios)
- **Selection**: NSGA-II multi-objective selection
- **Crossover**: Index exchange between individuals while maintaining test set size
- **Mutation**: Index replacement with available indices from train set

### Approach Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '10px' }}}%%
graph TD
    A[Input Data<br/>X, y] --> C[Evolutionary<br/>Optimization]
    
    subgraph Meta-Features["<div style='text-align: center; margin-top: 10px; color:#287786;'>Meta-Features</div>"]
        D["Statistical,<br/>Info-theory"]
    end
    
    subgraph Optimization["<div style='text-align: right; margin-top: 5px; margin-right: 10px; color:#287786; '>Optimization</div>"]
        E["Population<br/>Train/Test Splits"] --> F["Meta-Feature<br/>Extraction"]
        F --> G["Fitness Function<br/>ratio = MF_train / MF_test"]
        G --> H["Selection"]
        H --> I["Crossover"]
        I --> J["Mutation"]
        J --> E
    end
    
    C --> E
    C --> K["Final Split"]
    K --> L["Train Set"] & M["Test Set"]
    
    style A fill:#761A29,stroke:#761A29,stroke-width:1px,color:#fff
    style C fill:#8A8F35,stroke:#8A8F35,stroke-width:1px,color:#fff
    style K fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style L fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style M fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style D fill:#66B8C8,stroke:#66B8C8,stroke-width:1px,color:#fff
    style E fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style F fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style G fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style H fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style I fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style J fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    
    linkStyle default stroke:#287786,stroke-width:1.5px
```



### 📊 Experimental Results: Train/Test Split Analysis
📌 **Bold** indicates best result in category

| Split type | Dataset | LR | XGB | IRM | DRO |
|--------|---------|----|-----|-----|-----|
| **Random Split**| taxi | 0.752 ± 0.01 | 0.778 ± 0.01 | 0.790 ± 0.02 | 0.712 ± 0.02 |
| **Best MF** | taxi | **0.526** ± 0.10 | **0.592** ± 0.07 | **0.773** ± 0.10 | **0.505** ± 0.10 |
| **Random Split** | electricity | 0.798 ± 0.00 | 0.832 ± 0.00 | 0.813 ± 0.01 | 0.814 ± 0.02 |
| **Best MF** | electricity | **0.735** ± 0.02 | **0.749** ± 0.01 | **0.795** ± 0.02 | **0.766** ± 0.01 |

**Best MF** reports worst-case performance across all tested meta-features, demonstrating robustness under adversarial distributional shifts. 

## 2️⃣ Synthetic Data Generation

The synthetic generation approach formulates data creation as an optimization problem:
```
minimize: ||meta_features(synthetic) - meta_features(target)||₂
subject to: synthetic ∈ feasible_space(source)
```
### Evolutionary Algorithm

The optimization uses a genetic algorithm with:
- **Population**: Samples generated by Forest diffusion from source data
- **Fitness**: Euclidean distance between synthetic and target meta-feature vectors
- **Selection**: NSGA-III multi-objective selection with reference points
- **Crossover**: Row-wise or column-wise exchange between data matrices
- **Mutation**: Multiple strategies: noise addition, distribution sampling, or covariance-based generation
  
### Approach Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '10px' }}}%%
graph TD
    A[Source Data] --> C[Forest Diffusion<br/>Model Training]
    B[Target Data] --> D[Target Meta-Features<br/>Extraction]
    
    subgraph Meta-Features["<div style='text-align: center; margin-top: 90px; color:#287786;'>Meta-Features</div>"]
        E["Statistical,<br/>Info-theory"]
    end
    
    subgraph Evolutionary Process["<div style='text-align: left; margin-top: 5px; margin-left: 30px; color:#287786;'>Optimization</div>"]
        F["Population<br/>Synthetic Data"] --> G["Fitness Function<br/>||MF_synthetic - MF_target||"]
        G --> H["Selection"]
        H --> I["Crossover"]
        I --> J["Mutation<br/>Noise/Distribution/Covariance"]
        J --> F
    end
    
    C --> F
    D --> E
    G --> K["Best Synthetic Data"]
    
    style A fill:#761A29,stroke:#761A29,stroke-width:1px,color:#fff
    style B fill:#761A29,stroke:#761A29,stroke-width:1px,color:#fff
    style C fill:#8A8F35,stroke:#8A8F35,stroke-width:1px,color:#fff
    style D fill:#8A8F35,stroke:#8A8F35,stroke-width:1px,color:#fff
    style K fill:#287786,stroke:#287786,stroke-width:1px,color:#fff
    style E fill:#66B8C8,stroke:#66B8C8,stroke-width:1px,color:#fff
    style F fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style G fill:#DBA494,stroke:#DBA494,stroke-width:1px,color:#fff
    style H fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style I fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    style J fill:#9FB88E,stroke:#9FB88E,stroke-width:1px,color:#fff
    
    linkStyle default stroke:#287786,stroke-width:1.5px
```

### 📊 Experimental Results: Synthetic Data Generation Analysis

Synthetic OOD data generated by matching target meta-feature distributions (shown in parentheses).

| Dataset | LR | XGB | DRO | IRM |
|---------|----|-----|-----|-----|
| **electricity** (mut-inf, class-conc, iq-range) | 0.613 ± 0.08 | 0.641 ± 0.09 | 0.587 ± 0.08 | 0.613 ± 0.08 |
| **electricity** (mut-inf, class-conc) | 0.611 ± 0.01 | 0.625 ± 0.01 | 0.589 ± 0.01 | 0.632 ± 0.02 |

## 3️⃣ Source-Target Split Reconstruction

Validates whether meta-features capture meaningful distribution shifts by recovering the original source/target boundary from pooled data.

**Method**: Pool source and target datasets → split using meta-feature optimization → compute ARI against original boundary.

**Best MF** shows highest ARI across all tested meta-features. Baseline is MMD-based clustering [[1]](#references) using Maximum Mean Discrepancy.

### 📊 Experimental Results: Split Reconstruction

| Dataset | Best MF ARI | MMD ARI |
|---------|-------------|---------|
| **electricity** | **0.384** | 0.040 |
| **taxi** | **0.003** | 0.000 |

Meta-feature-based splits achieve higher ARI than MMD baseline, demonstrating better recovery of original source/target partition.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ITMO-NSS-team/OOD_Tab_Evaluation.git
cd OOD_Tab_Evaluation

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Reproducing Experiments

### 1. Download Datasets

The repository includes several tabular datasets with known distributional shifts:

```
data/
├── electricity_source.csv      # Source domain data
├── electricity_target.csv      # Target domain data
...
├── taxi_source.csv
└── taxi_target.csv
```


### 2. Run Meta-Feature Splitting
```python
from mfs_based_algs.mfs_split import run_split
import pandas as pd

# Load your data (can be source+target concatenated)
data = pd.read_csv('your_data.csv')

# Run split optimization
run_split(
    file=data,
    target_column_name='target',  # Your target column name
    file_prefix_name='split_by_class_conc',  # Output file prefix
    meta_features=['class_conc'],  # Meta-feature to optimize
    population_size=50,
    generations=300
)
```

This will create a directory `split_by_class_conc_pareto_solutions/` containing:
- `split_by_class_conc_solution_XX_train.csv` - Training split
- `split_by_class_conc_solution_XX_test.csv` - Test split
- `split_by_class_conc_solution_XX_info.txt` - Split metadata
- `pareto_solutions_summary.csv` - Summary of all solutions
### 3. Run Synthetic Data Generation
```python
from mfs_based_algs.mfs_synthetic import run_shift_convergence_experiment

# Generate synthetic data
results = run_shift_convergence_experiment(
    meta_features=['class_conc', 'mut_inf'],  # Meta-features to match
    mutation_type='all',  # Mutation strategy
    n_samples=dataset_length, # Number of samples to generate
    generations=200,
    source_file='data/source.csv',
    target_file='data/target.csv'
)
```

### 4. Run Source-Target Split Reconstruction

Computes ARI between reconstructed splits and original source/target boundaries.

**Prerequisites:** Splits in `split_by_*_pareto_solutions/` and datasets in `data/` as `*_source.csv`/`*_target.csv`.

```bash
python evaluate_reconstruction.py
```

**Output:**
```
Processing: split_by_class_conc_pareto_solutions
  dataset_name - Solution 1: ARI=0.0234, Source in train: 6/13, Target in test: 1/3
```

## 📁 Repository Structure

```
OOD_Tab_Evaluation/
├── mfs_based_algs/              # Meta-feature based algorithms
│   ├── mfs_split.py             # Meta-feature based splitting
│   └── mfs_synthetic.py         # Synthetic data generation
├── robust_models/               # Robust model implementations
│   ├── IRM_model/               # Invariant Risk Minimization
│   └── DRO_model/               # Distributionally Robust Optimization
├── data/                        # Dataset files
│   ├── *_source.csv            # Source domain datasets
│   └── *_target.csv            # Target domain datasets
├── evaluate_reconstruction.py  # Source-target split evaluation experiment
└── README.md                    # This file
```

## 🎯 Key Findings

1. **Controlled OOD Evaluation**: Meta-feature splits enable systematic investigation under distributional shifts
2. **Interpretable Shifts**: Meta-features provide semantic interpretation of distributional differences
3. **Reconstruction**: Meta-features recover source/target boundaries better than MMD baseline

## 📚 Dependencies

- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `torch>=1.9.0`
- `xgboost>=1.5.0`
- `deap>=1.3.0`
- `pymfe>=0.4.0`
- `matplotlib>=3.5.0`


## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{ood_tabular_evaluation,
  title={Meta-Feature Based Distribution Shift Protocol for Tabular Data},
}
```

## 📚 References

1. Napoli, A., & White, C. (2025). Clustering-based validation splits for model evaluation. [[Paper]](https://arxiv.org/abs/2405.19461)

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.


