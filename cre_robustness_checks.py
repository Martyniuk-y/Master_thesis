# # ============================================================
# # ROBUSTNESS CHECKS FOR CRE MODEL
# # 1. CRE with macro time controls (unempl_r, infl_r) instead of year dummies
# # 2. Fixed CRE vs Two-Way FE consistency check (linearmodels API fix)
# # ============================================================

# import numpy as np
# import pandas as pd
# import statsmodels.formula.api as smf
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')


# # ============================================================
# # ROBUSTNESS CHECK 1
# # CRE with explicit macro controls instead of year dummies
# # ------------------------------------------------------------
# # Motivation (supervisor Point 4): year dummies absorb the
# # trend in AI adoption, making the AI coefficient hard to
# # identify. Replacing year dummies with unempl_r + infl_r
# # captures macro time variation more parsimoniously, letting
# # the AI coefficient reflect within-sector dynamics net of
# # business cycle fluctuations rather than a common time trend.
# # ============================================================

# def run_cre_macro(df, ict_var, label):
#     """
#     CRE model with unempl_r + infl_r as macro time controls
#     instead of year dummies. Entity means (Mundlak device)
#     replace country dummies.
#     """
#     inter_var  = f'ai_x_{ict_var}'
#     ict_mean   = f'{ict_var}_mean'
#     ai_mean    = 'ai_adoption_mean'
#     inter_mean = f'{inter_var}_mean'

#     formula = (
#         'log_wage ~ '
#         # Within (time-varying) components
#         f'ai_adoption + {ict_var} + {inter_var} + '
#         # Mundlak means (between / structural components)
#         f'{ai_mean} + {ict_mean} + {inter_mean} + '
#         # Standard controls
#         'log_prod + FSI + '
#         # Macro time controls (replacing year dummies)
#         'unempl_r + infl_r + '
#         # Sector fixed effects only
#         'C(nace_r2)'
#     )

#     result = smf.ols(formula, data=df).fit(
#         cov_type='cluster',
#         cov_kwds={'groups': df['entity']}
#     )
#     return result


# # ============================================================
# # ROBUSTNESS CHECK 2
# # Two-Way FE via linearmodels (API-corrected)
# # ------------------------------------------------------------
# # Fixed: use EntityEffects + TimeEffects absorb=True in
# # PanelOLS constructor, not keyword arguments.
# # ============================================================

# def run_twoway_fe(df, ict_var):
#     """
#     Two-Way FE using linearmodels PanelOLS (corrected API).
#     Entity effects + time effects, clustered SE at entity level.
#     """
#     from linearmodels.panel import PanelOLS
#     import statsmodels.api as sm

#     inter_var = f'ai_x_{ict_var}'

#     panel_df = df.set_index(['entity', 'year']).copy()

#     # Drop rows with any NaN in required columns
#     cols = ['log_wage', 'ai_adoption', ict_var, inter_var, 'log_prod', 'FSI']
#     panel_df = panel_df[cols].dropna()

#     # Add constant (linearmodels does not add it automatically)
#     panel_df['const'] = 1.0

#     endog = panel_df['log_wage']
#     exog  = panel_df[['const', 'ai_adoption', ict_var, inter_var, 'log_prod', 'FSI']]

#     mod = PanelOLS(
#         dependent=endog,
#         exog=exog,
#         entity_effects=True,
#         time_effects=True
#     )
#     result = mod.fit(cov_type='clustered', cluster_entity=True)
#     return result


# # ============================================================
# # DISPLAY COMPARISON TABLE
# # CRE (year dummies) vs CRE (macro controls) vs Two-Way FE
# # ============================================================

# def compare_all_specs(df, ict_var, res_cre_year, res_cre_macro, res_fe):
#     """
#     Print a side-by-side coefficient table for the three
#     specifications on the key technology variables.
#     """
#     inter_var = f'ai_x_{ict_var}'

#     vars_to_show = ['ai_adoption', ict_var, inter_var,
#                     'ai_adoption_mean', f'{ict_var}_mean', f'{inter_var}_mean',
#                     'log_prod', 'FSI', 'unempl_r', 'infl_r']

#     print(f"\n{'='*80}")
#     print(f"  COEFFICIENT COMPARISON — {ict_var.upper()}")
#     print(f"  Col 1: CRE + year dummies | Col 2: CRE + macro controls | Col 3: Two-Way FE")
#     print(f"{'='*80}")
#     print(f"  {'Variable':<30} {'CRE+Year':>14} {'CRE+Macro':>14} {'Two-Way FE':>14}")
#     print(f"  {'-'*72}")

#     def fmt(res, var):
#         try:
#             c = res.params[var]
#             p = res.pvalues[var]
#             s = '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))
#             return f"{c:>10.5f}{s:<3}"
#         except (KeyError, AttributeError):
#             return f"{'—':>13}"

#     for v in vars_to_show:
#         in_any = any(v in r.params for r in [res_cre_year, res_cre_macro]
#                      if r is not None)
#         try:
#             in_fe = v in res_fe.params
#         except:
#             in_fe = False
#         if not (in_any or in_fe):
#             continue

#         col1 = fmt(res_cre_year,  v) if res_cre_year  is not None else f"{'—':>13}"
#         col2 = fmt(res_cre_macro, v) if res_cre_macro is not None else f"{'—':>13}"
#         col3 = fmt(res_fe,        v) if res_fe        is not None else f"{'—':>13}"

#         separator = "--- Mundlak means ---" if v == 'ai_adoption_mean' else (
#                     "--- Macro controls ---" if v == 'unempl_r' else '')
#         if separator:
#             print(f"  {separator}")
#         print(f"  {v:<30} {col1:>14} {col2:>14} {col3:>14}")

#     # Model fit
#     print(f"  {'-'*72}")
#     for label, res in [('CRE+Year', res_cre_year), ('CRE+Macro', res_cre_macro)]:
#         if res is not None:
#             print(f"  {label} R² = {res.rsquared:.4f}  Adj.R² = {res.rsquared_adj:.4f}  N = {int(res.nobs)}")
#     if res_fe is not None:
#         try:
#             print(f"  Two-Way FE  R² (within) = {res_fe.rsquared:.4f}  N = {int(res_fe.nobs)}")
#         except:
#             pass

#     print(f"  Significance: * p<0.1  ** p<0.05  *** p<0.01")
#     print(f"  Clustered SE at entity (geo×nace_r2) level throughout")


# # ============================================================
# # MAIN: run robustness checks
# # ============================================================

# def run_robustness(df, res_cre_specict, res_cre_training):
#     """
#     Run all robustness checks and print comparison tables.

#     Parameters
#     ----------
#     df               : prepared DataFrame (with Mundlak means already added
#                        — i.e. output of prepare_panel + add_mundlak_means)
#     res_cre_specict  : CRE result from run_cre_model(df, 'spec_ict', ...)
#     res_cre_training : CRE result from run_cre_model(df, 'training_ict', ...)
#     """

#     print("\n" + "#"*68)
#     print("  ROBUSTNESS CHECK 1 — CRE WITH MACRO TIME CONTROLS")
#     print("  (unempl_r + infl_r replace year dummies)")
#     print("#"*68)

#     rc1_spec  = run_cre_macro(df, 'spec_ict',     'ICT Specialists — macro controls')
#     rc1_train = run_cre_macro(df, 'training_ict', 'ICT Training — macro controls')

#     print("\n" + "#"*68)
#     print("  ROBUSTNESS CHECK 2 — TWO-WAY FE (linearmodels, API fixed)")
#     print("#"*68)

#     try:
#         fe_spec  = run_twoway_fe(df, 'spec_ict')
#         fe_train = run_twoway_fe(df, 'training_ict')
#         print("  Two-Way FE estimated successfully.")
#     except Exception as e:
#         print(f"  Two-Way FE failed: {e}")
#         fe_spec  = None
#         fe_train = None

#     print("\n" + "#"*68)
#     print("  SIDE-BY-SIDE COMPARISON TABLES")
#     print("#"*68)

#     compare_all_specs(df, 'spec_ict',     res_cre_specict,  rc1_spec,  fe_spec)
#     compare_all_specs(df, 'training_ict', res_cre_training, rc1_train, fe_train)

#     return {
#         'cre_macro_spec_ict':     rc1_spec,
#         'cre_macro_training_ict': rc1_train,
#         'fe_spec_ict':            fe_spec,
#         'fe_training_ict':        fe_train,
#     }


# # ============================================================
# # HOW TO USE IN YOUR NOTEBOOK
# # ============================================================
# #
# # Step 1 — run the main CRE analysis first (if not done yet):
# #   results = run_full_analysis(df_master)
# #
# # Step 2 — run robustness checks:
# #   from cre_mundlak_model_v2 import prepare_panel, add_mundlak_means
# #   df = prepare_panel(df_master)
# #   df = add_mundlak_means(df, ['ai_adoption','spec_ict','training_ict',
# #                                'ai_x_spec_ict','ai_x_training_ict',
# #                                'log_prod','FSI'])
# #   rc = run_robustness(df,
# #                       results['spec_spec_ict'],
# #                       results['spec_training_ict'])
# #
# # Step 3 — access individual results:
# #   rc['cre_macro_training_ict'].summary()
# #   rc['fe_spec_ict'].summary
# # ============================================================

# print("Robustness check script loaded.")
# print("Call: rc = run_robustness(df, results['spec_spec_ict'], results['spec_training_ict'])")

# ============================================================
# ROBUSTNESS CHECKS FOR CRE MODEL — ADJUSTED
# Spec A (spec_ict):     + tert_edu
# Spec B (training_ict): + share_high_skill
# + MEM/AME marginal effects
# + VIF (within-demeaned)
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ROBUSTNESS CHECK 1
# CRE with macro time controls instead of year dummies
# + spec-specific education control
# ============================================================

def run_cre_macro(df, ict_var, edu_control):
    inter_var  = f'ai_x_{ict_var}'
    ict_mean   = f'{ict_var}_mean'
    ai_mean    = 'ai_adoption_mean'
    inter_mean = f'{inter_var}_mean'

    formula = (
        'log_wage ~ '
        f'ai_adoption + {ict_var} + {inter_var} + '
        f'{ai_mean} + {ict_mean} + {inter_mean} + '
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
# ROBUSTNESS CHECK 2
# Two-Way FE via linearmodels + spec-specific education control
# ============================================================

def run_twoway_fe(df, ict_var, edu_control):
    from linearmodels.panel import PanelOLS

    inter_var = f'ai_x_{ict_var}'
    panel_df  = df.set_index(['entity', 'year']).copy()

    cols = ['log_wage', 'ai_adoption', ict_var, inter_var,
            'log_prod', 'FSI', edu_control]
    panel_df = panel_df[cols].dropna()
    panel_df['const'] = 1.0

    endog = panel_df['log_wage']
    exog  = panel_df[['const', 'ai_adoption', ict_var, inter_var,
                       'log_prod', 'FSI', edu_control]]

    mod = PanelOLS(
        dependent=endog,
        exog=exog,
        entity_effects=True,
        time_effects=True
    )
    return mod.fit(cov_type='clustered', cluster_entity=True)


# ============================================================
# SHARED: MEM + AME helper
# (mirrors cre_mundlak_model.py compute_marginal_effects)
# ============================================================

def me_stats(z_val, b_ai, b_inter, var_b, var_g, cov_bg, dof):
    me    = b_ai + b_inter * z_val
    var   = var_b + z_val**2 * var_g + 2 * z_val * cov_bg
    se    = np.sqrt(max(var, 0))
    t     = me / se if se > 0 else np.nan
    p     = 2 * (1 - stats.t.cdf(abs(t), df=dof)) if not np.isnan(t) else np.nan
    ci_lo = me - 1.96 * se
    ci_hi = me + 1.96 * se
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    return me, se, t, p, ci_lo, ci_hi, stars


def print_mem_ame(result, ict_var, df):
    b_ai    = result.params.get('ai_adoption', np.nan)
    b_inter = result.params.get(f'ai_x_{ict_var}', np.nan)
    cov_p   = result.cov_params()
    var_b   = cov_p.loc['ai_adoption', 'ai_adoption']
    var_g   = cov_p.loc[f'ai_x_{ict_var}', f'ai_x_{ict_var}']
    cov_bg  = cov_p.loc['ai_adoption', f'ai_x_{ict_var}']
    dof     = result.df_resid
    z       = df[ict_var]
    z_mean  = z.mean()

    me_m, se_m, t_m, p_m, ci_lo_m, ci_hi_m, st_m = me_stats(
        z_mean, b_ai, b_inter, var_b, var_g, cov_bg, dof)
    me_all = b_ai + b_inter * z
    me_a, se_a, t_a, p_a, ci_lo_a, ci_hi_a, st_a = me_stats(
        z_mean, b_ai, b_inter, var_b, var_g, cov_bg, dof)

    print(f"  === MEM  ({ict_var} mean = {z_mean:.4f}%) ===")
    print(f"  ME={me_m:.5f}{st_m}  SE={se_m:.5f}  t={t_m:.3f}  p={p_m:.4f}"
          f"  95%CI=[{ci_lo_m:.5f},{ci_hi_m:.5f}]")
    print(f"  === AME ===")
    print(f"  Mean of individual MEs={me_all.mean():.5f}{st_a}  SE={se_a:.5f}"
          f"  t={t_a:.3f}  p={p_a:.4f}  95%CI=[{ci_lo_a:.5f},{ci_hi_a:.5f}]")


# ============================================================
# SHARED: VIF (within-demeaned, no dummies)
# ============================================================

def compute_vif(df, ict_var, edu_control):
    inter_var = f'ai_x_{ict_var}'
    vif_vars  = ['ai_adoption', ict_var, inter_var,
                 'log_prod', 'FSI', edu_control]

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
# COMPARISON TABLE (extended: now includes edu_control row)
# ============================================================

def compare_all_specs(df, ict_var, edu_control,
                      res_cre_year, res_cre_macro, res_fe):
    inter_var = f'ai_x_{ict_var}'

    vars_to_show = [
        'ai_adoption', ict_var, inter_var,
        edu_control,
        'log_prod', 'FSI',
        'ai_adoption_mean', f'{ict_var}_mean', f'{inter_var}_mean',
        'unempl_r', 'infl_r'
    ]

    print(f"\n{'='*82}")
    print(f"  COEFFICIENT COMPARISON — {ict_var.upper()}  |  edu control: {edu_control}")
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
        in_cre  = any(v in r.params for r in [res_cre_year, res_cre_macro] if r is not None)
        in_fe   = (res_fe is not None) and (v in res_fe.params)
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

    # MEM/AME for macro CRE spec
    print(f"\n  --- MEM / AME for CRE+Macro robustness ({ict_var}) ---")
    if res_cre_macro is not None:
        print_mem_ame(res_cre_macro, ict_var, df)

    # VIF for macro CRE spec
    print(f"\n  --- VIF for CRE+Macro robustness ({ict_var}, within-demeaned) ---")
    vif_df = compute_vif(df, ict_var, edu_control)
    print(vif_df.to_string(index=False))
    print(f"  Note: high VIF on {inter_var} is expected (interaction term).")


# ============================================================
# MAIN
# ============================================================

def run_robustness(df, res_cre_specict, res_cre_training):
    """
    Parameters
    ----------
    df                : prepared DataFrame with Mundlak means added
    res_cre_specict   : CRE result from run_cre_model(df, 'spec_ict', ...)
    res_cre_training  : CRE result from run_cre_model(df, 'training_ict', ...)
    """
    edu_a = 'tert_edu'
    edu_b = 'share_high_skill'

    print("\n" + "#"*68)
    print("  ROBUSTNESS CHECK 1 — CRE WITH MACRO TIME CONTROLS")
    print("#"*68)
    rc1_spec  = run_cre_macro(df, 'spec_ict',     edu_a)
    rc1_train = run_cre_macro(df, 'training_ict', edu_b)

    print("\n" + "#"*68)
    print("  ROBUSTNESS CHECK 2 — TWO-WAY FE (linearmodels)")
    print("#"*68)
    try:
        fe_spec  = run_twoway_fe(df, 'spec_ict',     edu_a)
        fe_train = run_twoway_fe(df, 'training_ict', edu_b)
        print("  Two-Way FE estimated successfully.")
    except Exception as e:
        print(f"  Two-Way FE failed: {e}")
        fe_spec = fe_train = None

    print("\n" + "#"*68)
    print("  COMPARISON TABLES + MEM/AME + VIF")
    print("#"*68)
    compare_all_specs(df, 'spec_ict',     edu_a,
                      res_cre_specict,  rc1_spec,  fe_spec)
    compare_all_specs(df, 'training_ict', edu_b,
                      res_cre_training, rc1_train, fe_train)

    return {
        'cre_macro_spec_ict':     rc1_spec,
        'cre_macro_training_ict': rc1_train,
        'fe_spec_ict':            fe_spec,
        'fe_training_ict':        fe_train,
    }


print("Robustness script loaded.")
print("Call: rc = run_robustness(df, results['spec_spec_ict'], results['spec_training_ict'])")