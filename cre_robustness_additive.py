import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ROBUSTNESS CHECK 1 — CRE with macro controls, ADDITIVE
# ============================================================

def run_cre_macro_additive(df, ict_var, edu_control):
    ict_mean = f'{ict_var}_mean'
    ai_mean  = 'ai_adoption_mean'

    formula = (
        'log_wage ~ '
        f'ai_adoption + {ict_var} + '
        f'{ai_mean} + {ict_mean} + '
        f'log_prod + FSI + {edu_control} + '
        'unempl_r + infl_r + '
        'C(nace_r2)'
    )

    result = smf.ols(formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['entity']}
    )
    return result

# ============================================================
# ROBUSTNESS CHECK 2 — Two-Way FE (additive)
# ============================================================

def run_twoway_fe_additive(df, ict_var, edu_control):
    from linearmodels.panel import PanelOLS

    panel_df  = df.set_index(['entity', 'year']).copy()

    cols = ['log_wage', 'ai_adoption', ict_var,
            'log_prod', 'FSI', edu_control]
    panel_df = panel_df[cols].dropna()
    panel_df['const'] = 1.0

    endog = panel_df['log_wage']
    exog  = panel_df[['const', 'ai_adoption', ict_var,
                      'log_prod', 'FSI', edu_control]]

    mod = PanelOLS(
        dependent=endog,
        exog=exog,
        entity_effects=True,
        time_effects=True
    )
    return mod.fit(cov_type='clustered', cluster_entity=True)

# ============================================================
# MEM / AME helper (additive)
# ============================================================

def compute_mem_ame_additive(result):
    b_ai = result.params.get('ai_adoption', np.nan)
    se_ai = result.bse.get('ai_adoption', np.nan)
    dof = result.df_resid

    t = b_ai / se_ai if se_ai > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df=dof)) if not np.isnan(t) else np.nan
    ci_lo = b_ai - 1.96 * se_ai
    ci_hi = b_ai + 1.96 * se_ai
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))

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

def print_mem_ame_additive(result):
    mem, ame = compute_mem_ame_additive(result)
    print(f"  ME (MEM = AME) : {mem['ME']:.5f} {mem['Sig']}")
    print(f"  SE             : {mem['SE']:.5f}  t={mem['t']:.3f}  p={mem['p']:.4f}")
    print(f"  95% CI         : [{mem['ci_lo']:.5f},{mem['ci_hi']:.5f}]")

# ============================================================
# VIF (additive, within-demeaned)
# ============================================================

def compute_vif_additive(df, ict_var, edu_control):
    vif_vars = ['ai_adoption', ict_var, 'log_prod', 'FSI', edu_control]

    X_raw = df[vif_vars].copy()
    entity_means = df.groupby('entity')[vif_vars].transform('mean')
    time_means   = df.groupby('year')[vif_vars].transform('mean')
    grand_mean   = X_raw.mean()
    X_dem        = X_raw - entity_means - time_means + grand_mean

    vif_df = pd.DataFrame({
        'Variable': vif_vars,
        'VIF': [variance_inflation_factor(X_dem.values, i)
                for i in range(len(vif_vars))]
    }).sort_values('VIF', ascending=False)
    return vif_df

# ============================================================
# COMPARISON TABLE (ADDITIVE)
# ============================================================

def compare_all_specs_additive(df, ict_var, edu_control,
                               res_cre_year, res_cre_macro, res_fe):
    vars_to_show = [
        'ai_adoption', ict_var, edu_control,
        'log_prod', 'FSI',
        'ai_adoption_mean', f'{ict_var}_mean',
        'unempl_r', 'infl_r'
    ]

    print(f"\n{'='*82}")
    print(f"  COEFFICIENT COMPARISON (ADDITIVE) — {ict_var.upper()}  |  edu: {edu_control}")
    print(f"  Col 1: CRE + year dummies | Col 2: CRE + macro controls | Col 3: Two-Way FE")
    print(f"{'='*82}")
    print(f"  {'Variable':<32} {'CRE+Year':>14} {'CRE+Macro':>14} {'Two-Way FE':>14}")
    print(f"  {'-'*74}")

    def fmt(res, var):
        try:
            c = res.params[var]
            p = res.pvalues[var]
            s = '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))
            return f"{c:>10.5f}{s:<3}"
        except (KeyError, AttributeError):
            return f"{'—':>13}"

    section_headers = {
        'ai_adoption_mean': '--- Mundlak means (BETWEEN) ---',
        'unempl_r':         '--- Macro time controls ---'
    }

    for v in vars_to_show:
        in_cre = any(v in r.params for r in [res_cre_year, res_cre_macro] if r is not None)
        in_fe  = (res_fe is not None) and (v in res_fe.params)
        if not (in_cre or in_fe):
            continue
        if v in section_headers:
            print(f"  {section_headers[v]}")
        col1 = fmt(res_cre_year,  v) if res_cre_year  is not None else f"{'—':>13}"
        col2 = fmt(res_cre_macro, v) if res_cre_macro is not None else f"{'—':>13}"
        col3 = fmt(res_fe,        v) if res_fe        is not None else f"{'—':>13}"
        print(f"  {v:<32} {col1:>14} {col2:>14} {col3:>14}")

    print(f"  {'-'*74}")
    for label, res in [('CRE+Year', res_cre_year), ('CRE+Macro', res_cre_macro)]:
        if res is not None:
            print(f"  {label:<12} R²={res.rsquared:.4f}  Adj.R²={res.rsquared_adj:.4f}  N={int(res.nobs)}")
    if res_fe is not None:
        try:
            print(f"  {'Two-Way FE':<12} R²(within)={res_fe.rsquared:.4f}  N={int(res_fe.nobs)}")
        except:
            pass
    print(f"  Significance: * p<0.1  ** p<0.05  *** p<0.01")
    print(f"  Clustered SE at entity (geo×nace_r2) level throughout")

    print(f"\n  --- MEM / AME for CRE+Macro robustness (additive, {ict_var}) ---")
    if res_cre_macro is not None:
        print_mem_ame_additive(res_cre_macro)

    print(f"\n  --- VIF for CRE+Macro robustness (additive, {ict_var}) ---")
    vif_df = compute_vif_additive(df, ict_var, edu_control)
    print(vif_df.to_string(index=False))

# ============================================================
# MAIN (ADDITIVE ROBUSTNESS)
# ============================================================

def run_robustness_additive(df, res_cre_specict, res_cre_training):
    edu_a = 'tert_edu'
    edu_b = 'share_high_skill'

    print("\n" + "#"*68)
    print("  ROBUSTNESS (ADDITIVE) — CRE WITH MACRO TIME CONTROLS")
    print("#"*68)
    rc1_spec  = run_cre_macro_additive(df, 'spec_ict',     edu_a)
    rc1_train = run_cre_macro_additive(df, 'training_ict', edu_b)

    print("\n" + "#"*68)
    print("  ROBUSTNESS (ADDITIVE) — TWO-WAY FE (linearmodels)")
    print("#"*68)
    try:
        fe_spec  = run_twoway_fe_additive(df, 'spec_ict',     edu_a)
        fe_train = run_twoway_fe_additive(df, 'training_ict', edu_b)
        print("  Two-Way FE (additive) estimated successfully.")
    except Exception as e:
        print(f"  Two-Way FE (additive) failed: {e}")
        fe_spec = fe_train = None

    print("\n" + "#"*68)
    print("  COMPARISON TABLES + MEM/AME + VIF (ADDITIVE)")
    print("#"*68)
    compare_all_specs_additive(df, 'spec_ict',     edu_a,
                               res_cre_specict,  rc1_spec,  fe_spec)
    compare_all_specs_additive(df, 'training_ict', edu_b,
                               res_cre_training, rc1_train, fe_train)

    return {
        'add_cre_macro_spec_ict':     rc1_spec,
        'add_cre_macro_training_ict': rc1_train,
        'add_fe_spec_ict':            fe_spec,
        'add_fe_training_ict':        fe_train,
    }

print("Additive robustness script loaded.")
print("Call: add_rc = run_robustness_additive(df, add_results['add_spec_spec_ict'], add_results['add_spec_training_ict'])")