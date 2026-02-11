# Replica Trick Estimators for V* (Offline Estimation)

Implementation of replica trick-based estimators for offline estimation of V* in the A*PO algorithm framework.

## Overview

This repository contains the implementation and experiments for my senior thesis on **replica trick estimators** for offline value function estimation under KL-regularized control. The code implements and compares four estimators:

1. **LME (Log-Mean-Exp)** - Baseline biased estimator
2. **Single-Replica** - Fixed replica order n > 1
3. **Multi-n Slope** - Linear extrapolation across multiple replica orders
4. **Multi-n Slope + Jackknife** - Slope estimator with delete-one jackknife bias correction

## Repository Structure

```
.
├── src/
│   ├── estimators.py      # Core V* estimators
│   ├── ground_truth.py    # Analytical V* computation
│   └── metrics.py         # Bias, variance, RMSE metrics
├── run_experiment1.py     # Main experiment runner (scalar setting)
├── results/               # Experiment outputs (plots, CSV)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd replica-value-estimation
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computing
- `scipy` - Statistical functions
- `pandas` - Data handling
- `matplotlib` - Plotting

## Running Experiments

### Experiment 1: Scalar Estimation

Pure scalar setting with Gaussian rewards to validate estimator properties.

#### Quick Test (for debugging)
```bash
python run_experiment1.py --quick
```
- 3 N_tot values: [32, 128, 512]
- 50 Monte Carlo trials
- 2 single-replica variants
- 1 multi-n set
- **Completes in ~0.3 seconds**

#### Full Experiment (for thesis results)
```bash
python run_experiment1.py
```
- 8 N_tot values: [16, 32, 64, 128, 256, 512, 1024, 2048]
- 1000 Monte Carlo trials
- 5 single-replica variants (n ∈ {2, 3, 4, 5, 8})
- 5 multi-n order sets (each with/without jackknife)
- **Completes in ~5-10 minutes**

### Output Files

All results are saved to the `results/` directory:

**Plots** (9 files):
- `bias_beta{X}.png` - Bias vs N_tot for β=X
- `variance_beta{X}.png` - Variance vs N_tot for β=X
- `rmse_beta{X}.png` - RMSE vs N_tot for β=X

Where X ∈ {0.5, 1.0, 2.0}

**Data**:
- `experiment1_results.csv` - Full numerical results

## Configuration

You can modify experiment parameters in `run_experiment1.py`:

### Key Parameters

```python
# Reward distribution
MU_R = 0.0         # Reward mean
SIGMA_R = 1.0      # Reward standard deviation

# KL regularization parameters
BETAS = [0.5, 1.0, 2.0]

# Sample budgets
N_TOT_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

# Monte Carlo trials
T_TRIALS = 1000

# Single-replica configurations to test
SINGLE_REPLICA_N_VALUES = [2, 3, 4, 5, 8]

# Multi-n order sets to test
MULTI_N_ORDER_SETS = [
    [2, 3],           # Minimal set
    [2, 3, 4],        # Small set
    [2, 3, 4, 5],     # Original set
    [2, 3, 4, 5, 6],  # Extended set
    [2, 4, 6, 8],     # Even orders only
]

# Random seed for reproducibility
SEED = 42
```

## Understanding the Estimators

### 1. LME (Log-Mean-Exp)
```python
V_hat = beta * log(mean(exp(r_i / beta)))
```
- Simple and fast
- **Negatively biased** (Jensen's inequality)
- Baseline for comparison

### 2. Single-Replica (n=N)
```python
# Partition N samples into blocks of size n
# Compute product within each block, then average
V_hat = beta * log(mean(product_of_n_samples))
```
- Reduces bias compared to LME
- Performance depends on choice of n

### 3. Multi-n Slope
```python
# Fit linear model: phi_hat(n) ~ a + b*n
# where phi_hat(n) = log(mean(block_products))
V_hat = beta * b  # Extract slope
```
- Uses multiple replica orders to extrapolate
- Theoretically less biased
- Current implementation: each n uses **full N_tot budget**

### 4. Multi-n Slope + Jackknife
- Applies delete-one jackknife to each phi_hat(n)
- Reduces bias further at cost of increased variance

## Interpreting Results

### Plots

Each plot shows one metric for one β value. All estimators are overlaid for comparison.

**Bias plots**: How far estimates are from true V* (should be near 0)
**Variance plots**: Spread of estimates across trials (lower is better)
**RMSE plots**: Overall error combining bias and variance (lower is better)

### CSV Data

The results CSV contains:
- `beta`: KL regularization parameter
- `n_tot`: Sample budget
- `method`: Estimator name (e.g., "lme", "single_n2", "multi_[2,3,4]")
- `bias`: Bias of the estimator
- `variance`: Variance of the estimator
- `rmse`: Root mean squared error
- `v_star`: Ground truth V*
- `mean_estimate`: Average estimate across trials
- `n_valid`: Number of valid (non-NaN) estimates

## Scientific Questions

This codebase helps answer:

1. **How does replica order affect single-replica performance?**
   - Compare single_n2, single_n3, single_n4, single_n5, single_n8

2. **How does the multi-n set choice affect slope estimator?**
   - Compare different sets: [2,3], [2,3,4], [2,3,4,5], etc.

3. **Does jackknife help or hurt?**
   - Compare methods with/without `_jk` suffix

4. **How does performance scale with β?**
   - Compare across β ∈ {0.5, 1.0, 2.0}

5. **What is the sample efficiency?**
   - Look at RMSE vs N_tot curves

## Troubleshooting

### Import Errors
Make sure you've activated your virtual environment and installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Slow Execution
Use `--quick` mode for testing:
```bash
python run_experiment1.py --quick
```

### Memory Issues
Reduce `T_TRIALS` or the number of `N_TOT_VALUES` in the configuration.

### Plot Display Issues
Plots are saved to files automatically. If matplotlib warnings appear, they can usually be ignored.

## Future Work

- **Experiment 2**: Multi-context (contextual bandit) setting
- **Ablation studies**: Effect of reward distribution parameters
- **Variance reduction**: Other bias correction techniques
- **Online estimation**: Extend to online A*PO implementation

## Citation

If you use this code in your research, please cite:

```
[Your thesis citation here]
```

## License

[Your license choice]

## Contact

[Your contact information]
