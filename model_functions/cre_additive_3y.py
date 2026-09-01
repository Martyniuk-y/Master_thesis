# ============================================================
# Correlated Random Effects (CRE / Mundlak) Model ONLY
# ADJUSTED: ADDITIVE MODEL (NO INTERACTION) & NO FSI / NO Prod
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 – PREPARE VARIABLES
# ============================================================

def prepare_panel(df):
    df = df.copy()
    df['entity'] = df['geo'].astype(str) + '_' + df['nace_r2'].astype(str)
    return df

# ============================================================
# STEP 2 – MUNDLAK MEANS (Dual-Level)
# ============================================================

def add_mundlak_means(df, time_varying_ij, time_varying_j):
    df = df.copy()
    
    entity_means = (
        df.groupby('entity')[time_varying_ij]
        .transform('mean')
        .rename(columns={v: v + '_mean' for v in time_varying_ij})
    )
    
    country_means = (
        df.groupby('geo')[time_varying_j]
        .transform('mean')
        .rename(columns={v: v + '_mean' for v in time_varying_j})
    )
    
    return pd.concat([df, entity_means, country_means], axis=1)

# ============================================================
# STEP 3 – CRE MODEL (WITH DATA LOSS DIAGNOSTICS)
# ============================================================

def run_cre_additive(df, ict_var, edu_control, label):
    used_vars = [
        'log_wage', 'ai_adoption', ict_var,
        edu_control, 'log_gdp', 'infl_r', 'unempl_r',
        'ai_adoption_mean', f'{ict_var}_mean',
        f'{edu_control}_mean', 'log_gdp_mean', 'infl_r_mean', 'unempl_r_mean',
        'year', 'nace_r2', 'entity'
    ]
    
    n_before = len(df)
    df_clean = df.dropna(subset=used_vars).copy()
    n_after = len(df_clean)
    
    print(f"\n  [Data Diagnostics] {label}:")
    print(f"  - Initial observations: {n_before}")
    print(f"  - Clean observations:   {n_after}")
    print(f"  - Dropped (Missing):    {n_before - n_after} rows")
    
    micro_within = f'ai_adoption + {ict_var}'
    sector_within = f'{edu_control}'
    macro_within = 'log_gdp + infl_r + unempl_r'
    
    micro_between = f'ai_adoption_mean + {ict_var}_mean'
    sector_between = f'{edu_control}_mean'
    macro_between = 'log_gdp_mean + infl_r_mean + unempl_r_mean'
    
    fixed_effects = 'C(year) + C(nace_r2)'

    formula = (
        f'log_wage ~ {micro_within} + {sector_within} + {macro_within} + '
        f'{micro_between} + {sector_between} + {macro_between} + '
        f'{fixed_effects}'
    )

    result = smf.ols(formula, data=df_clean).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_clean['entity']}
    )
    
    return result, edu_control, df_clean

# ============================================================
# STEP 4 – MUNDLAK-HAUSMAN TEST
# ============================================================

def mundlak_hausman_test(result, mean_vars):
    restriction_list = [f'({v} = 0)' for v in mean_vars]
    wald = result.wald_test(' , '.join(restriction_list), use_f=False)
    return float(wald.statistic), len(mean_vars), float(wald.pvalue)

# ============================================================
# STEP 5 – VIF 
# ============================================================

def compute_vif_additive(df, ict_var, edu_control):
    vif_vars  = ['ai_adoption', ict_var, edu_control, 'log_gdp', 'infl_r', 'unempl_r']

    X_raw = df[vif_vars].copy()

    entity_means = X_raw.copy()
    entity_means[:] = df.groupby('entity')[vif_vars].transform('mean').values

    time_means = X_raw.copy()
    time_means[:] = df.groupby('year')[vif_vars].transform('mean').values

    grand_mean  = X_raw.mean()
    X_demeaned  = X_raw - entity_means - time_means + grand_mean

    vif_df = pd.DataFrame({
        'Variable': vif_vars,
        'VIF': [variance_inflation_factor(X_demeaned.values, i)
                for i in range(len(vif_vars))]
    }).sort_values('VIF', ascending=False)

    return vif_df

# ============================================================
# STEP 6 – DISPLAY RESULTS
# ============================================================

def display_results_additive(result, ict_var, edu_control, label, df):
    print(f"\n{'='*68}")
    print(f"  ADDITIVE CRE / MUNDLAK MODEL — {label}")
    print(f"{'='*68}")

    core_vars = [
        'ai_adoption', ict_var,
        edu_control, 'log_gdp', 'infl_r', 'unempl_r',
        'ai_adoption_mean', f'{ict_var}_mean',
        f'{edu_control}_mean', 'log_gdp_mean', 'infl_r_mean', 'unempl_r_mean'
    ]

    print(f"\n  {'Variable':<33} {'Coef':>10}  {'SE':>9}  {'p-val':>8}  Sig")
    print(f"  {'-'*66}")
    for v in core_vars:
        if v in result.params.index:
            c, se, p = result.params[v], result.bse[v], result.pvalues[v]
            s = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            if v == 'ai_adoption_mean':
                print(f"  {'--- Mundlak means (BETWEEN effects) ---'}")
            print(f"  {v:<33} {c:>10.5f}  {se:>9.5f}  {p:>8.4f}  {s}")

    print(f"\n  R² = {result.rsquared:.4f}   Adj.R² = {result.rsquared_adj:.4f}   N = {int(result.nobs)}")

    mean_vars = [
        'ai_adoption_mean', f'{ict_var}_mean',
        f'{edu_control}_mean', 'log_gdp_mean', 'infl_r_mean', 'unempl_r_mean'
    ]
    chi2, df_t, pv = mundlak_hausman_test(result, mean_vars)
    print(f"\n  Mundlak-Hausman Test  H₀: entity means jointly = 0")
    print(f"  χ²({df_t}) = {chi2:.3f},  p = {pv:.4f}")
    verdict = ("→ REJECT H₀ — FE needed" if pv < 0.05 else
               "→ FAIL TO REJECT H₀ — RE sufficient")
    print(f"  {verdict}")

    vif_df = compute_vif_additive(df, ict_var, edu_control)
    print(f"\n  === VIF (within-demeaned, no dummies) ===")
    print(vif_df.to_string(index=False))

    return vif_df

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_additive_analysis(df_master):
    df = prepare_panel(df_master)

    time_varying_ij = ['ai_adoption', 'spec_ict', 'training_ict', 'tert_edu']
    time_varying_j = ['log_gdp', 'infl_r', 'unempl_r']

    df = add_mundlak_means(df, time_varying_ij, time_varying_j)
    results = {}

    print("\n" + "#"*68)
    print("  ADDITIVE SPECIFICATION A — ICT SPECIALISTS + tert_edu (No FSI/Prod)")
    print("#"*68)
    res_a, edu_a, df_clean_a = run_cre_additive(df, 'spec_ict', 'tert_edu', 'ICT Specialists')
    vif_a = display_results_additive(res_a, 'spec_ict', edu_a, 'ICT SPECIALISTS', df_clean_a)
    results.update({'spec_spec_ict': res_a, 'vif_spec_ict': vif_a})

    print("\n" + "#"*68)
    print("  ADDITIVE SPECIFICATION B — ICT TRAINING + tert_edu (No FSI/Prod)")
    print("#"*68)
    res_b, edu_b, df_clean_b = run_cre_additive(df, 'training_ict', 'tert_edu', 'ICT Training')
    vif_b = display_results_additive(res_b, 'training_ict', edu_b, 'ICT TRAINING', df_clean_b)
    results.update({'spec_training_ict': res_b, 'vif_training_ict': vif_b})

    return results

print("CRE/Mundlak ADDITIVE script loaded. Run: results_add = run_additive_analysis(df_master)")