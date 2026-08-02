import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 – PREPARE VARIABLES
# (no interaction terms needed for additive model)
# ============================================================

def prepare_panel(df):
    df = df.copy()
    df['entity'] = df['geo'].astype(str) + '_' + df['nace_r2'].astype(str)
    return df

# ============================================================
# STEP 2 – MUNDLAK MEANS
# ============================================================

def add_mundlak_means(df, time_varying_vars):
    df = df.copy()
    entity_means = (
        df.groupby('entity')[time_varying_vars]
        .transform('mean')
        .rename(columns={v: v + '_mean' for v in time_varying_vars})
    )
    return pd.concat([df, entity_means], axis=1)

# ============================================================
# STEP 3 – CRE MODEL (ADDITIVE, NO INTERACTION)
# Spec A (spec_ict)     : + tert_edu
# Spec B (training_ict) : + share_high_skill
# ============================================================

def run_cre_model_additive(df, ict_var, label):
    ict_mean = f'{ict_var}_mean'
    ai_mean  = 'ai_adoption_mean'

    edu_control = 'tert_edu' if ict_var == 'spec_ict' else 'share_high_skill'

    formula = (
        'log_wage ~ '
        'ai_adoption + ' + ict_var + ' + '
        f'{ai_mean} + {ict_mean} + '
        'log_prod + FSI + ' + edu_control + ' + '
        'C(year) + C(nace_r2)'
    )

    result = smf.ols(formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['entity']}
    )
    return result, edu_control

# ============================================================
# STEP 4 – MUNDLAK-HAUSMAN TEST
# ============================================================

def mundlak_hausman_test(result, mean_vars):
    restriction_list = [f'({v} = 0)' for v in mean_vars]
    wald = result.wald_test(' , '.join(restriction_list), use_f=False)
    return float(wald.statistic), len(mean_vars), float(wald.pvalue)

# ============================================================
# STEP 5 – MARGINAL EFFECTS: MEM and AME (ADDITIVE)
# In additive model: ME(ai) = β_ai (constant) [no ICT dependence]
# ============================================================

def compute_marginal_effects_additive(result):
    b_ai = result.params.get('ai_adoption', np.nan)
    se_ai = result.bse.get('ai_adoption', np.nan)
    dof = result.df_resid

    t = b_ai / se_ai if se_ai > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df=dof)) if not np.isnan(t) else np.nan
    ci_lo = b_ai - 1.96 * se_ai
    ci_hi = b_ai + 1.96 * se_ai
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))

    # MEM = AME = β_ai in linear additive model
    mem = {
        'ME': b_ai,
        'SE': se_ai,
        't': t,
        'p': p,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'Sig': stars
    }
    ame = mem.copy()

    return mem, ame

# ============================================================
# STEP 6 – VIF (within-demeaned, no dummies, no interaction)
# ============================================================

def compute_vif_additive(df, ict_var, edu_control):
    vif_vars = ['ai_adoption', ict_var, 'log_prod', 'FSI', edu_control]

    X_raw = df[vif_vars].copy()
    entity_means = df.groupby('entity')[vif_vars].transform('mean')
    time_means   = df.groupby('year')[vif_vars].transform('mean')
    grand_mean   = X_raw.mean()
    X_demeaned   = X_raw - entity_means - time_means + grand_mean

    vif_df = pd.DataFrame({
        'Variable': vif_vars,
        'VIF': [variance_inflation_factor(X_demeaned.values, i)
                for i in range(len(vif_vars))]
    }).sort_values('VIF', ascending=False)

    return vif_df

# ============================================================
# STEP 7 – DISPLAY RESULTS (ADDITIVE)
# ============================================================

def display_results_additive(result, ict_var, edu_control, label, df):
    print(f"\n{'='*68}")
    print(f"  CRE / MUNDLAK MODEL (ADDITIVE) — {label}")
    print(f"{'='*68}")

    core_vars = [
        'ai_adoption', ict_var,
        'ai_adoption_mean', f'{ict_var}_mean',
        'log_prod', 'FSI', edu_control
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

    # Mundlak-Hausman test
    mean_vars = ['ai_adoption_mean', f'{ict_var}_mean']
    chi2, df_t, pv = mundlak_hausman_test(result, mean_vars)
    print(f"\n  Mundlak-Hausman Test  H₀: entity means jointly = 0")
    print(f"  χ²({df_t}) = {chi2:.3f},  p = {pv:.4f}")
    verdict = ("→ REJECT H₀ — FE needed"
               if pv < 0.05 else
               "→ FAIL TO REJECT H₀ — RE sufficient")
    print(f"  {verdict}")

    # Marginal effects
    mem, ame = compute_marginal_effects_additive(result)
    print(f"\n  === Marginal Effect of ai_adoption (additive) ===")
    print(f"  ME (MEM = AME) : {mem['ME']:.5f} {mem['Sig']}")
    print(f"  SE             : {mem['SE']:.5f}  t = {mem['t']:.3f}  p = {mem['p']:.4f}")
    print(f"  95% CI         : [{mem['ci_lo']:.5f}, {mem['ci_hi']:.5f}]")

    # VIF
    vif_df = compute_vif_additive(df, ict_var, edu_control)
    print(f"\n  === VIF (within-demeaned, no dummies, additive spec) ===")
    print(vif_df.to_string(index=False))
    print(f"\n  Significance: *** p<0.01  ** p<0.05  * p<0.1")

    return mem, ame, vif_df

# ============================================================
# STEP 8 – CRE vs FE CONSISTENCY CHECK (ADDITIVE)
# ============================================================

def compare_with_fe_additive(df, ict_var, edu_control):
    from linearmodels.panel import PanelOLS
    panel_df  = df.set_index(['entity', 'year'])

    formula_fe = (
        f'log_wage ~ ai_adoption + {ict_var} + '
        f'log_prod + FSI + {edu_control} + EntityEffects + TimeEffects'
    )
    mod_fe = PanelOLS.from_formula(formula_fe, data=panel_df)
    return mod_fe.fit(cov_type='clustered', cluster_entity=True)

# ============================================================
# MAIN EXECUTION (ADDITIVE SPEC)
# ============================================================

def run_full_analysis_additive(df_master):
    df = prepare_panel(df_master)

    time_varying = [
        'ai_adoption', 'spec_ict', 'training_ict',
        'log_prod', 'FSI'
    ]
    df = add_mundlak_means(df, time_varying)

    results = {}

    # --- Spec A: ICT Specialists + tert_edu (additive) ---
    print("\n" + "#"*68)
    print("  SPECIFICATION A (ADDITIVE) — ICT SPECIALISTS + tert_edu")
    print("#"*68)
    res_a, edu_a = run_cre_model_additive(df, 'spec_ict', 'ICT Specialists (additive)')
    mem_a, ame_a, vif_a = display_results_additive(res_a, 'spec_ict', edu_a, 'ICT SPECIALISTS (additive)', df)
    results.update({'add_spec_spec_ict': res_a,
                    'add_mem_spec_ict': mem_a,
                    'add_ame_spec_ict': ame_a,
                    'add_vif_spec_ict': vif_a})

    # --- Spec B: ICT Training + share_high_skill (additive) ---
    print("\n" + "#"*68)
    print("  SPECIFICATION B (ADDITIVE) — ICT TRAINING + share_high_skill")
    print("#"*68)
    res_b, edu_b = run_cre_model_additive(df, 'training_ict', 'ICT Training (additive)')
    mem_b, ame_b, vif_b = display_results_additive(res_b, 'training_ict', edu_b, 'ICT TRAINING (additive)', df)
    results.update({'add_spec_training_ict': res_b,
                    'add_mem_training_ict': mem_b,
                    'add_ame_training_ict': ame_b,
                    'add_vif_training_ict': vif_b})

    # --- Consistency check with FE (additive) ---
    print("\n" + "="*68)
    print("  CONSISTENCY CHECK: CRE (additive) within β  vs  Two-Way FE β")
    print("="*68)
    try:
        fe_a = compare_with_fe_additive(df, 'spec_ict', edu_a)
        fe_b = compare_with_fe_additive(df, 'training_ict', edu_b)
        for v in ['ai_adoption']:
            print(f"  {v}:")
            print(f"    Spec A — CRE β = {res_a.params.get(v, np.nan):.5f}  |  FE β = {fe_a.params.get(v, np.nan):.5f}")
            print(f"    Spec B — CRE β = {res_b.params.get(v, np.nan):.5f}  |  FE β = {fe_b.params.get(v, np.nan):.5f}")
        results.update({'add_fe_spec_ict': fe_a, 'add_fe_training_ict': fe_b})
    except Exception as e:
        print(f"  FE comparison (additive) skipped: {e}")

    return results

print("CRE/Mundlak ADDITIVE script loaded. Run: add_results = run_full_analysis_additive(df_master)")