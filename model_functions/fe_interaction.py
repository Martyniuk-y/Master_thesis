# ============================================================
# Two-Way Fixed Effects (TWFE) Robustness Check
# ============================================================

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1 – PREPARE VARIABLES
# ============================================================

def prepare_panel(df):
    df = df.copy()
    # Create country-sector identifier (ij)
    df['entity'] = df['geo'].astype(str) + '_' + df['nace_r2'].astype(str)
    
    # Interaction terms
    df['ai_x_spec_ict']     = df['ai_adoption'] * df['spec_ict']
    df['ai_x_training_ict'] = df['ai_adoption'] * df['training_ict']
    return df


# ============================================================
# STEP 2 – RUN FE MODEL
# ============================================================

def run_fe_model(df, ict_var, edu_control, label):
    inter_var = f'ai_x_{ict_var}'
    
    # Track variables used in the FE model to drop NaNs properly
    used_vars = [
        'log_wage', 'ai_adoption', ict_var, inter_var,
        'log_prod', 'FSI', edu_control,
        'log_gdp', 'infl_r', 'unempl_r',
        'year', 'entity'
    ]
    
    n_before = len(df)
    df_clean = df.dropna(subset=used_vars).copy()
    n_after = len(df_clean)
    
    print(f"\n{'='*68}")
    print(f"  TWO-WAY FIXED EFFECTS MODEL — {label}")
    print(f"{'='*68}")
    print(f"  [Data Diagnostics]:")
    print(f"  - Initial observations: {n_before}")
    print(f"  - Clean observations:   {n_after}")
    print(f"  - Dropped (Missing):    {n_before - n_after} rows\n")
    
    # linearmodels requires a MultiIndex (Entity, Time)
    panel_df = df_clean.set_index(['entity', 'year'])
    
    # TWFE Formula
    formula_fe = (
        f'log_wage ~ ai_adoption + {ict_var} + {inter_var} + '
        f'log_prod + FSI + {edu_control} + '
        f'log_gdp + infl_r + unempl_r + '
        f'EntityEffects + TimeEffects'
    )
    
    mod_fe = PanelOLS.from_formula(formula_fe, data=panel_df)
    
    # Fit with clustered standard errors
    result = mod_fe.fit(cov_type='clustered', cluster_entity=True)
    
    # Output the standard summary
    print(result.summary)
    
    return result


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_full_fe_analysis(df_master):
    df = prepare_panel(df_master)
    results = {}

    # --- Specification A: ICT Specialists + tert_edu ---
    res_a = run_fe_model(df, 'spec_ict', 'tert_edu', 'ICT Specialists')
    results['fe_spec_ict'] = res_a

    # --- Specification B: ICT Training + tert_edu ---
    res_b = run_fe_model(df, 'training_ict', 'tert_edu', 'ICT Training')
    results['fe_training_ict'] = res_b

    return results

print("Two-Way Fixed Effects script loaded. Run: fe_results = run_full_fe_analysis(df_master)")