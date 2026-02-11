# Codebase Explanation

## Overview

This codebase implements and evaluates **four estimators** for V* = β log Z in the KL-regularized RL setting, where Z = E[exp(r/β)] is the partition function. The goal is to compare their **bias**, **variance**, and **RMSE** under different sample budgets.

---

## File Structure

```
replica-value-estimation/
├── src/                      # Core library modules
│   ├── __init__.py          # Empty (makes src a package)
│   ├── ground_truth.py      # Analytical V* for Gaussian rewards
│   ├── metrics.py           # Bias, variance, RMSE computation
│   └── estimators.py        # Four V* estimators
├── run_experiment1.py       # Main experiment runner
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
├── README.md               # User documentation
└── results/                # Experiment outputs (plots, CSV)
```

---

## Module 1: `src/ground_truth.py`

### Purpose
Provides **analytical ground truth** for V* when rewards are Gaussian.

### Mathematical Background
For r ~ N(μᵣ, σᵣ²):
- Z = E[exp(r/β)] = exp(μᵣ/β + σᵣ²/(2β²))  [moment generating function]
- V* = β log Z = μᵣ + σᵣ²/(2β)

### Functions

#### `compute_v_star_gaussian(mu_r, sigma_r, beta)`
**What it does:** Returns the exact V* = μᵣ + σᵣ²/(2β)

**Why it's correct:** This is the closed-form solution for the Gaussian case, derived from the MGF of the normal distribution.

**Used for:** Computing the ground truth for bias/RMSE calculations.

---

## Module 2: `src/metrics.py`

### Purpose
Computes evaluation metrics from T Monte Carlo trial estimates.

### Functions

#### `compute_bias(estimates, v_star)`
**Formula:** Bias = mean(V̂) - V*

**What it does:** Measures systematic error (how far the average estimate is from truth)

**Correct implementation:** Uses `np.nanmean` to handle potential NaN values from failed trials.

#### `compute_variance(estimates)`
**Formula:** Variance = (1/(T-1)) Σ(V̂ᵢ - mean(V̂))²

**What it does:** Measures estimate spread (using unbiased sample variance with ddof=1)

**Correct implementation:** 
- Filters out NaN values
- Uses ddof=1 for unbiased variance
- Returns NaN if fewer than 2 valid estimates

#### `compute_rmse(estimates, v_star)`
**Formula:** RMSE = sqrt(mean((V̂ᵢ - V*)²))

**What it does:** Measures overall error combining bias and variance

**Relationship:** RMSE² ≈ Bias² + Variance (exact for large T)

**Correct implementation:** Uses `np.nanmean` to handle NaN values.

---

## Module 3: `src/estimators.py` (Core Logic)

This is the heart of the codebase. Let me explain each estimator in detail.

### Estimator 1: LME (Log-Mean-Exp)

#### `estimate_lme(rewards, beta)`

**Formula:** V̂_LME = β log((1/N) Σᵢ exp(rᵢ/β))

**How it works:**
1. Scale rewards by 1/β: scaled = r/β
2. Compute log-mean-exp using logsumexp for stability:
   - log(mean(exp(x))) = logsumexp(x) - log(N)
3. Multiply by β to get V̂

**Why it's biased:**
- Jensen's inequality: E[log(mean)] ≤ log(E[mean])
- The estimator is **negatively biased** (underestimates V*)

**Numerical stability:** Uses `scipy.special.logsumexp` instead of direct exp/log to avoid overflow.

**Correctness check:** ✅
- Matches standard A*PO Stage 1 formula
- Logsumexp usage is correct
- Should have n=1 case, which this is

---

### Helper Functions for Replica Methods

#### `_partition_and_compute_log_block_products(rewards, beta, n)`

**Purpose:** Partition N samples into blocks of size n, compute log(Wⱼ) for each block.

**How it works:**
1. Calculate M = floor(N/n) blocks
2. Take exactly M×n samples (discard remainder if N not divisible by n)
3. Reshape into (M, n) array
4. For each block j: log(Wⱼ) = Σₖ rⱼₖ/β (sum within block, then divide by β)

**Why this is correct:**
- Wⱼ = ∏ₖ exp(rⱼₖ/β) = exp(Σₖ rⱼₖ/β)
- So log(Wⱼ) = Σₖ rⱼₖ/β  ✅

**Numerical stability:** Working in log-space throughout.

#### `_compute_log_psi_hat(log_W)`

**Purpose:** Compute log(ψ̂(n)) where ψ̂(n) = (1/M) Σⱼ Wⱼ

**How it works:**
1. log(mean(W)) = logsumexp(log_W) - log(M)

**Why this is correct:**
- mean(W) = (1/M) Σ exp(log_Wⱼ)
- log(mean(W)) = log(Σ exp(log_Wⱼ)) - log(M) = logsumexp(log_W) - log(M)  ✅

#### `_compute_jackknife_phi(log_W)`

**Purpose:** Apply delete-one jackknife bias correction to φ̂(n) = log(ψ̂(n))

**Jackknife formula:**
- Full estimate: φ̂ = log(ψ̂)
- Leave-one-out: φ̂₍₋ⱼ₎ = log(ψ̂ without Wⱼ)
- Jackknife: φ_JK = M·φ̂ - (M-1)·mean(φ̂₍₋ⱼ₎)

**Why jackknife helps:**
- Reduces O(1/M) bias to O(1/M²)
- Standard technique for bias reduction

**How the leave-one-out is computed:**
1. For each j, need: log(Σᵢ≠ⱼ Wᵢ) = log(Σ_all - Wⱼ)
2. Numerically stable formula:
   - ratio = Wⱼ / Σ_all = exp(log_Wⱼ - log_Σ_all)
   - log(Σ_all - Wⱼ) = log_Σ_all + log(1 - ratio) = log_Σ_all + log1p(-ratio)
3. Edge case: if ratio ≈ 1 (one block dominates), fall back to direct computation

**Correctness check:** ✅
- Jackknife formula is standard
- Numerical stability carefully handled
- Edge case protection prevents log(0)

---

### Estimator 2: Single-Replica

#### `estimate_single_replica(rewards, beta, n)`

**Formula:** V̂ = (β/n) log(ψ̂(n))

**How it works:**
1. Partition N_tot samples into M = floor(N_tot/n) blocks of size n
2. Compute log(Wⱼ) for each block
3. Compute log(ψ̂(n)) = log(mean(W))
4. Return V̂ = (β/n) log(ψ̂(n))

**Theory:**
- E[Wⱼ] = Z^n (because Wⱼ is the product of n independent exp(r/β) terms)
- So ψ̂(n) ≈ Z^n
- Therefore log(ψ̂(n)) ≈ n log(Z)
- And V̂ = (β/n) log(ψ̂(n)) ≈ (β/n) · n log(Z) = β log(Z) = V*  ✅

**Why n=1 is just LME:**
- If n=1: M blocks, each containing 1 sample
- Wⱼ = exp(rⱼ/β)
- ψ̂(1) = mean(exp(r/β))
- V̂ = β log(ψ̂(1)) = LME  ✅

**Correctness check:** ✅ Matches replica trick theory.

---

### Estimator 3 & 4: Multi-n Slope

#### `estimate_multi_n_slope(rewards, beta, replica_orders, use_jackknife)`

**Formula:** Fit φ̂(n) ~ a + b·n, return V̂ = β·b

**How it works:**
1. For each n in replica_orders:
   - Re-partition **all N_tot samples** into blocks of size n
   - Compute φ̂(n) = log(ψ̂(n)) [with or without jackknife]
2. Fit linear regression: φ̂(n) = a + b·n using `np.polyfit`
3. Extract slope b
4. Return V̂ = β·b

**Theory:**
- True: φ(n) = log(Z^n) = n log(Z) is **exactly linear** in n
- Slope = log(Z)
- Therefore V* = β log(Z) = β · slope  ✅
- Intercept 'a' absorbs constant bias across n values

**Critical design choice: Budget allocation**

**Current implementation:** Each n uses **all N_tot samples**, creating different block structures from the same data.

**Pros:**
- Maximum data usage
- Each φ̂(n) is estimated from all available samples
- Consistent with "using the replica trick to improve estimates"

**Cons:**
- φ̂(n) values are **correlated** (share the same raw samples)
- This correlation doesn't affect consistency but might affect variance

**Alternative approach (not implemented):** 
- Split N_tot into K chunks, give each n a separate N_tot/K budget
- Makes φ̂(n) independent but wastes samples

**Which is correct?** This depends on your thesis specification. The current implementation maximizes sample efficiency. If your thesis specifies budget splitting, this needs to be changed.

**Linear regression:**
- `np.polyfit(n_arr, phi_arr, 1)` returns [slope, intercept]
- Standard least-squares fitting  ✅

**Correctness check:** ✅ (assuming budget allocation matches thesis intent)

---

## Module 4: `run_experiment1.py`

### Configuration Constants (lines 60-85)

```python
MU_R = 0.0                    # Reward mean
SIGMA_R = 1.0                 # Reward std dev
BETAS = [0.5, 1.0, 2.0]       # KL regularization parameters
N_TOT_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]  # Sample budgets
T_TRIALS = 1000               # Monte Carlo trials
SINGLE_REPLICA_N_VALUES = [2, 3, 4, 5, 8]  # Replica orders to test
MULTI_N_ORDER_SETS = [        # Sets of orders for multi-n slope
    [2, 3],
    [2, 3, 4],
    [2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [2, 4, 6, 8],
]
SEED = 42                     # For reproducibility
```

### Core Experiment Functions

#### `run_single_trial(rng, beta, n_tot, single_n_values, multi_n_sets)`

**What it does:** Run ONE Monte Carlo trial for a given (β, N_tot) configuration.

**How it works:**
1. Draw N_tot i.i.d. samples: `rewards = rng.normal(MU_R, SIGMA_R, size=n_tot)`
2. Run LME estimator
3. Run single-replica for each n in single_n_values
4. Run multi-n slope (no JK) for each set in multi_n_sets
5. Run multi-n slope (with JK) for each set in multi_n_sets
6. Return dict of all estimates

**Key design: Fair comparison**
- ALL estimators receive the **same** N_tot samples
- This ensures fair comparison under the same data conditions

**Correctness:** ✅ Each method gets the same input data.

#### `run_experiment(betas, n_tot_values, t_trials, single_n_values, multi_n_sets, seed)`

**What it does:** Run full experiment sweep over all (β, N_tot) configurations.

**How it works:**
1. For each β:
   - Compute true V* using `compute_v_star_gaussian`
2. For each N_tot:
   - Run T_trials independent trials
   - Collect all estimates for each method
   - Compute bias, variance, RMSE
3. Store results in DataFrame

**Output:** DataFrame with columns:
- beta, n_tot, method, bias, variance, rmse, v_star, mean_estimate, n_valid

**Correctness:** ✅ Standard Monte Carlo evaluation protocol.

### Plotting Functions

#### `_classify_methods(df)`
**Purpose:** Group methods into 3 families for visualization.

**Output:** Dict with keys {"single_replica", "multi_n", "multi_n_jk"}, each containing list of (method_key, label, style) tuples.

#### `_plot_one(ax, df_beta, metric, family_items, show_ylabel)`
**Purpose:** Plot one panel (one family, one metric, one β).

**Features:**
- Log₂ scale on x-axis (N_tot)
- Log scale on y-axis for variance/RMSE
- Horizontal line at y=0 for bias
- Shared y-axis when used in triptych

#### `plot_results(df, output_dir)`
**Purpose:** Generate triptych plots (3 families side-by-side).

**Output:** 9 plots = 3 metrics × 3 β values

**Features:**
- **Shared y-axis** across 3 panels for direct comparison
- LME baseline (black) in every panel
- Family-specific color schemes
- 1790×515 pixels (under 2000px limit)

**Correctness:** ✅ Matplotlib sharey=True ensures aligned scales.

### Main Entry Point

#### `main()`

**Handles:**
- `--quick`: Reduced config for fast testing
- `--plot-only`: Regenerate plots from existing CSV without re-running trials

**Flow:**
1. Parse arguments
2. Load or select configuration
3. Run experiment (or skip if plot-only)
4. Save CSV
5. Generate plots

---

## Key Design Decisions & Their Correctness

### ✅ 1. Numerical Stability
**Decision:** Use `logsumexp` throughout instead of raw exp/log operations.
**Why correct:** Prevents overflow when exp(r/β) is large. Standard practice in ML.

### ✅ 2. Fair Comparison
**Decision:** All estimators receive identical N_tot samples in each trial.
**Why correct:** Isolates estimator differences from data variability.

### ✅ 3. Shared Random Samples in Multi-n
**Decision:** Each replica order n uses all N_tot samples (re-partitioned).
**Why correct (probably):** Maximizes sample efficiency. Standard in replica trick literature.
**Caveat:** Verify this matches your thesis Section 3.2 specification.

### ✅ 4. Handling NaN Values
**Decision:** Use `np.nanmean`, `np.nanvar` for metrics computation.
**Why correct:** Some trials may fail (e.g., insufficient blocks), so we handle NaN gracefully.

### ✅ 5. Jackknife Numerical Stability
**Decision:** Special handling when one block dominates (ratio ≈ 1).
**Why correct:** Prevents log(0) errors in leave-one-out computation.

### ✅ 6. Unbiased Variance
**Decision:** Use ddof=1 in variance computation.
**Why correct:** Sample variance estimator is unbiased with Bessel's correction.

---

## Potential Issues to Check

### ⚠️ 1. Budget Allocation in Multi-n Slope
**Current:** Each n uses full N_tot → φ̂(n) values are correlated
**Alternative:** Split N_tot across n values → φ̂(n) independent

**Action needed:** Check your thesis Section 3.2 to confirm which approach is intended.

### ⚠️ 2. Why LME Wins in Results?
The experiments show LME beating replica methods. Possible explanations:
1. **Gaussian setting is "too easy"** → bias is small even for LME
2. **Large N regime** → LME bias vanishes, low variance wins
3. **Budget allocation** → If thesis intended split budget, current implementation over-allocates to multi-n

**Action needed:** Verify results match theoretical expectations from your thesis.

---

## Summary: Is the Code Correct?

**Yes, with one caveat:**

✅ **Numerics:** All estimators are correctly implemented with proper numerical stability.

✅ **Theory:** Formulas match the replica trick literature and A*PO framework.

✅ **Evaluation:** Monte Carlo protocol, metrics, and plotting are all correct.

⚠️ **Budget allocation:** The multi-n slope estimator uses **full N_tot for each n**. This is a valid choice but you should **verify** it matches your thesis specification. This is the most critical design decision to check against your advisor's expectations.

**Recommended action:** Re-read your thesis Section 3.2 (or consult with your advisor) to confirm the intended budget allocation strategy. If it requires splitting, I can quickly fix that.

