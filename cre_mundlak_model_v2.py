# ============================================================
# Correlated Random Effects (CRE / Mundlak) Model
# ADJUSTED: tert_edu added to spec_ict spec, share_high_skill
#           added to training_ict spec, + VIF + MEM/AME
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1 – PREPARE VARIABLES
# ============================================================

def prepare_panel(df):
    df = df.copy()
    df['entity'] = df['geo'].astype(str) + '_' + df['nace_r2'].astype(str)
    df['ai_x_spec_ict']     = df['ai_adoption'] * df['spec_ict']
    df['ai_x_training_ict'] = df['ai_adoption'] * df['training_ict']
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
# STEP 3 – CRE MODEL
# Spec A (spec_ict)     : + tert_edu
# Spec B (training_ict) : + share_high_skill
# ============================================================

def run_cre_model(df, ict_var, label):
    inter_var  = f'ai_x_{ict_var}'
    ict_mean   = f'{ict_var}_mean'
    ai_mean    = 'ai_adoption_mean'
    inter_mean = f'{inter_var}_mean'

    # Specification-specific education control
    edu_control = 'tert_edu' 

    formula = (
        f'log_wage ~ '
        f'ai_adoption + {ict_var} + {inter_var} + '
        f'{ai_mean} + {ict_mean} + {inter_mean} + '
        f'log_prod  + '
        # f'log_prod + FSI + {edu_control} + '
        f'{edu_control} + '
        f'C(year) + C(nace_r2)'
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
# STEP 5 – MARGINAL EFFECTS: MEM and AME
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


def compute_marginal_effects(result, ict_var, df):
    b_ai    = result.params.get('ai_adoption', np.nan)
    b_inter = result.params.get(f'ai_x_{ict_var}', np.nan)
    cov_p   = result.cov_params()
    var_b   = cov_p.loc['ai_adoption', 'ai_adoption']
    var_g   = cov_p.loc[f'ai_x_{ict_var}', f'ai_x_{ict_var}']
    cov_bg  = cov_p.loc['ai_adoption', f'ai_x_{ict_var}']
    dof     = result.df_resid
    z       = df[ict_var]

    # --- Percentile table (P10–P90) ---
    rows = []
    for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
        z_val = z.quantile(p)
        me, se, t, pv, ci_lo, ci_hi, stars = me_stats(
            z_val, b_ai, b_inter, var_b, var_g, cov_bg, dof)
        rows.append({
            'Percentile': f'P{int(p*100)}',
            f'{ict_var} (%)': round(z_val, 2),
            'ME': round(me, 6), 'SE': round(se, 6),
            't': round(t, 3), 'p-value': round(pv, 4),
            'Sig': stars,
            '95% CI': f'[{ci_lo:.5f}, {ci_hi:.5f}]'
        })
    me_table = pd.DataFrame(rows)

    # --- MEM ---
    z_mean = z.mean()
    me_mem, se_mem, t_mem, p_mem, ci_lo_mem, ci_hi_mem, stars_mem = me_stats(
        z_mean, b_ai, b_inter, var_b, var_g, cov_bg, dof)

    # --- AME (= MEM for linear model, individual MEs for display) ---
    me_all  = b_ai + b_inter * z
    me_ame, se_ame, t_ame, p_ame, ci_lo_ame, ci_hi_ame, stars_ame = me_stats(
        z_mean, b_ai, b_inter, var_b, var_g, cov_bg, dof)

    return me_table, {
        'MEM': (z_mean, me_mem, se_mem, t_mem, p_mem, ci_lo_mem, ci_hi_mem, stars_mem),
        'AME': (me_all.mean(), me_ame, se_ame, t_ame, p_ame, ci_lo_ame, ci_hi_ame, stars_ame)
    }


# ============================================================
# STEP 6 – VIF (within-demeaned, no dummies)
# ============================================================

def compute_vif(df, ict_var, edu_control):
    """
    Two-way within-demeaning (entity + time) mirrors PanelOLS / CRE
    transformation. VIF on demeaned numeric regressors only.
    """
    inter_var = f'ai_x_{ict_var}'
    vif_vars  = ['ai_adoption', ict_var, inter_var,
                   'log_prod', edu_control]
                #   'FSI', edu_control]

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
# STEP 7 – DISPLAY RESULTS
# ============================================================

def display_results(result, ict_var, edu_control, label, df):
    print(f"\n{'='*68}")
    print(f"  CRE / MUNDLAK MODEL — {label}")
    print(f"{'='*68}")

    core_vars = [
        'ai_adoption', ict_var, f'ai_x_{ict_var}',
        'ai_adoption_mean', f'{ict_var}_mean', f'ai_x_{ict_var}_mean',
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
    mean_vars = ['ai_adoption_mean', f'{ict_var}_mean', f'ai_x_{ict_var}_mean']
    chi2, df_t, pv = mundlak_hausman_test(result, mean_vars)
    print(f"\n  Mundlak-Hausman Test  H₀: entity means jointly = 0")
    print(f"  χ²({df_t}) = {chi2:.3f},  p = {pv:.4f}")
    verdict = ("→ REJECT H₀ — FE needed" if pv < 0.05 else
               "→ FAIL TO REJECT H₀ — RE sufficient")
    print(f"  {verdict}")

    # Marginal effects
    me_table, summary = compute_marginal_effects(result, ict_var, df)
    z_mean, me_m, se_m, t_m, p_m, ci_lo_m, ci_hi_m, st_m = summary['MEM']
    me_a_val, me_a, se_a, t_a, p_a, ci_lo_a, ci_hi_a, st_a = summary['AME']

    print(f"\n  === Marginal Effect at the Mean (MEM) ===")
    print(f"  {ict_var} mean : {z_mean:.4f}%")
    print(f"  ME           : {me_m:.5f} {st_m}")
    print(f"  SE           : {se_m:.5f}  t = {t_m:.3f}  p = {p_m:.4f}")
    print(f"  95% CI       : [{ci_lo_m:.5f}, {ci_hi_m:.5f}]")

    print(f"\n  === Average Marginal Effect (AME) ===")
    print(f"  Mean of individual MEs : {me_a_val:.5f} {st_a}")
    print(f"  SE (delta at z̄)        : {se_a:.5f}  t = {t_a:.3f}  p = {p_a:.4f}")
    print(f"  95% CI                 : [{ci_lo_a:.5f}, {ci_hi_a:.5f}]")

    print(f"\n  === Marginal Effects across ICT distribution ===")
    print(me_table.to_string(index=False))

    # VIF
    vif_df = compute_vif(df, ict_var, edu_control)
    print(f"\n  === VIF (within-demeaned, no dummies) ===")
    print(vif_df.to_string(index=False))
    print(f"  Note: VIF > 10 flags concern; high VIF on {f'ai_x_{ict_var}'}")
    print(f"  is expected due to interaction term construction.")

    print("\n  Significance: *** p<0.01  ** p<0.05  * p<0.1")

    return me_table, vif_df


# ============================================================
# STEP 8 – CRE vs FE CONSISTENCY CHECK
# ============================================================

def compare_with_fe(df, ict_var, edu_control):
    from linearmodels.panel import PanelOLS
    inter_var = f'ai_x_{ict_var}'
    panel_df  = df.set_index(['entity', 'year'])
    formula_fe = (
        f'log_wage ~ ai_adoption + {ict_var} + {inter_var} + '
        f'log_prod + {edu_control} + EntityEffects + TimeEffects'
        # f'log_prod + FSI + {edu_control} + EntityEffects + TimeEffects'
    )
    mod_fe = PanelOLS.from_formula(formula_fe, data=panel_df)
    return mod_fe.fit(cov_type='clustered', cluster_entity=True)


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_full_analysis(df_master):
    df = prepare_panel(df_master)

    time_varying = [
        'ai_adoption', 'spec_ict', 'training_ict',
        'ai_x_spec_ict', 'ai_x_training_ict', 'log_prod'
        # 'log_prod', 'FSI'
    ]
    df = add_mundlak_means(df, time_varying)

    results = {}

    # --- Specification A: ICT Specialists + tert_edu ---
    print("\n" + "#"*68)
    print("  SPECIFICATION A — ICT SPECIALISTS + tert_edu")
    print("#"*68)
    res_a, edu_a = run_cre_model(df, 'spec_ict', 'ICT Specialists')
    me_a, vif_a  = display_results(res_a, 'spec_ict', edu_a, 'ICT SPECIALISTS', df)
    results.update({'spec_spec_ict': res_a, 'me_spec_ict': me_a, 'vif_spec_ict': vif_a})

    # --- Specification B: ICT Training + share_high_skill ---
    print("\n" + "#"*68)
    print("  SPECIFICATION B — ICT TRAINING + share_high_skill")
    print("#"*68)
    res_b, edu_b = run_cre_model(df, 'training_ict', 'ICT Training')
    me_b, vif_b  = display_results(res_b, 'training_ict', edu_b, 'ICT TRAINING', df)
    results.update({'spec_training_ict': res_b, 'me_training_ict': me_b, 'vif_training_ict': vif_b})

    # --- Consistency check ---
    print("\n" + "="*68)
    print("  CONSISTENCY CHECK: CRE within β  vs  Two-Way FE β")
    print("="*68)
    try:
        fe_a = compare_with_fe(df, 'spec_ict', edu_a)
        fe_b = compare_with_fe(df, 'training_ict', edu_b)
        for v in ['ai_adoption']:
            print(f"  {v}:")
            print(f"    Spec A — CRE β = {res_a.params.get(v, np.nan):.5f}  |  FE β = {fe_a.params.get(v, np.nan):.5f}")
            print(f"    Spec B — CRE β = {res_b.params.get(v, np.nan):.5f}  |  FE β = {fe_b.params.get(v, np.nan):.5f}")
        results.update({'fe_spec_ict': fe_a, 'fe_training_ict': fe_b})
    except Exception as e:
        print(f"  FE comparison skipped: {e}")

    return results

print("CRE/Mundlak script loaded. Run: results = run_full_analysis(df_master)")