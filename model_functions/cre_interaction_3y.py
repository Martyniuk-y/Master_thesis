# ============================================================
# Correlated Random Effects (CRE / Mundlak) Model ONLY
# ADJUSTED: WITH INTERACTION & NO FSI / NO Productivity
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

def run_cre_model(df, ict_var, edu_control, label):
    inter_var  = f'ai_x_{ict_var}'
    
    used_vars = [
        'log_wage', 'ai_adoption', ict_var, inter_var,
        edu_control, 'log_gdp', 'infl_r', 'unempl_r',
        'ai_adoption_mean', f'{ict_var}_mean', f'{inter_var}_mean',
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
    
    micro_within = f'ai_adoption + {ict_var} + {inter_var}'
    sector_within = f'{edu_control}'
    macro_within = 'log_gdp + infl_r + unempl_r'
    
    micro_between = f'ai_adoption_mean + {ict_var}_mean + {inter_var}_mean'
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

def linear_combo_variance(cov_p, vars_a, vars_b=None):
    if vars_b is None:
        vars_b = vars_a

    value = 0.0
    for var_a in vars_a:
        for var_b in vars_b:
            value += cov_p.loc[var_a, var_b]
    return value


def calculate_me_summary(z, b_ai, b_inter, var_b, var_g, cov_bg, dof, ict_label):
    z = pd.Series(z).dropna()
    rows = []

    for pctl in [0.10, 0.25, 0.50, 0.75, 0.90]:
        z_val = z.quantile(pctl)
        me, se, t, pv, ci_lo, ci_hi, stars = me_stats(
            z_val, b_ai, b_inter, var_b, var_g, cov_bg, dof
        )
        rows.append({
            'Percentile': f'P{int(pctl * 100)}',
            f'{ict_label} (%)': round(z_val, 2),
            'ME': round(me, 6),
            'SE': round(se, 6),
            't': round(t, 3),
            'p-value': round(pv, 4),
            'Sig': stars,
            '95% CI': f'[{ci_lo:.5f}, {ci_hi:.5f}]'
        })

    me_table = pd.DataFrame(rows)
    z_mean = z.mean()

    mem = me_stats(z_mean, b_ai, b_inter, var_b, var_g, cov_bg, dof)
    ame = me_stats(z_mean, b_ai, b_inter, var_b, var_g, cov_bg, dof)

    summary = {
        'mean_ict': z_mean,
        'MEM': mem,
        'AME': ame
    }

    return me_table, summary


def compute_marginal_effects(result, ict_var, df_clean):
    inter_var = f'ai_x_{ict_var}'
    ict_mean_var = f'{ict_var}_mean'
    inter_mean_var = f'{inter_var}_mean'
    cov_p = result.cov_params()
    dof = result.df_resid

    within_b_ai = result.params['ai_adoption']
    within_b_inter = result.params[inter_var]
    within_var_ai = cov_p.loc['ai_adoption', 'ai_adoption']
    within_var_inter = cov_p.loc[inter_var, inter_var]
    within_cov = cov_p.loc['ai_adoption', inter_var]

    within_table, within_summary = calculate_me_summary(
        df_clean[ict_var],
        within_b_ai,
        within_b_inter,
        within_var_ai,
        within_var_inter,
        within_cov,
        dof,
        ict_var
    )

    ai_vars = ['ai_adoption', 'ai_adoption_mean']
    inter_vars = [inter_var, inter_mean_var]

    between_b_ai = result.params['ai_adoption'] + result.params['ai_adoption_mean']
    between_b_inter = result.params[inter_var] + result.params[inter_mean_var]
    between_var_ai = linear_combo_variance(cov_p, ai_vars)
    between_var_inter = linear_combo_variance(cov_p, inter_vars)
    between_cov = linear_combo_variance(cov_p, ai_vars, inter_vars)

    entity_ict_means = (
        df_clean[['entity', ict_mean_var]]
        .drop_duplicates(subset='entity')[ict_mean_var]
    )

    between_table, between_summary = calculate_me_summary(
        entity_ict_means,
        between_b_ai,
        between_b_inter,
        between_var_ai,
        between_var_inter,
        between_cov,
        dof,
        ict_mean_var
    )

    return {
        'within_table': within_table,
        'within_summary': within_summary,
        'between_table': between_table,
        'between_summary': between_summary,
        'within_coefficients': {
            'ai': within_b_ai,
            'interaction': within_b_inter
        },
        'between_coefficients': {
            'ai': between_b_ai,
            'interaction': between_b_inter
        }
    }


def print_me_block(title, ict_label, coefficients, summary, me_table, between=False):
    z_mean = summary['mean_ict']
    me_mem, se_mem, t_mem, p_mem, ci_lo_mem, ci_hi_mem, stars_mem = summary['MEM']
    me_ame, se_ame, t_ame, p_ame, ci_lo_ame, ci_hi_ame, stars_ame = summary['AME']

    print(f"\n  === {title} ===")
    if between:
        print(f"  Formula: (b_AI + b_AI_mean) + "
              f"(b_AI×ICT + b_AI×ICT_mean) × {ict_label}_mean")
        print(f"  Implied between AI coefficient: {coefficients['ai']:.5f}")
        print(f"  Implied between interaction coefficient: {coefficients['interaction']:.5f}")
    else:
        print(f"  Formula: b_AI + b_AI×ICT × {ict_label}")
        print(f"  Within AI coefficient: {coefficients['ai']:.5f}")
        print(f"  Within interaction coefficient: {coefficients['interaction']:.5f}")

    print(f"\n  --- Marginal Effect at the Mean ---")
    print(f"  {ict_label} mean : {z_mean:.4f}%")
    print(f"  ME               : {me_mem:.5f} {stars_mem}")
    print(f"  SE               : {se_mem:.5f}  t = {t_mem:.3f}  p = {p_mem:.4f}")
    print(f"  95% CI           : [{ci_lo_mem:.5f}, {ci_hi_mem:.5f}]")

    print(f"\n  --- Average Marginal Effect ---")
    print(f"  Mean of individual MEs : {me_ame:.5f} {stars_ame}")
    print(f"  SE                     : {se_ame:.5f}  t = {t_ame:.3f}  p = {p_ame:.4f}")
    print(f"  95% CI                 : [{ci_lo_ame:.5f}, {ci_hi_ame:.5f}]")

    print(f"\n  --- Marginal Effects Across Distribution ---")
    print(me_table.to_string(index=False))

# ============================================================
# STEP 6 – VIF 
# ============================================================

def compute_vif(df, ict_var, edu_control):
    inter_var = f'ai_x_{ict_var}'
    vif_vars  = ['ai_adoption', ict_var, inter_var, edu_control, 'log_gdp', 'infl_r', 'unempl_r']

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
        edu_control, 'log_gdp', 'infl_r', 'unempl_r',
        'ai_adoption_mean', f'{ict_var}_mean', f'ai_x_{ict_var}_mean',
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
        'ai_adoption_mean', f'{ict_var}_mean', f'ai_x_{ict_var}_mean',
        f'{edu_control}_mean', 'log_gdp_mean', 'infl_r_mean', 'unempl_r_mean'
    ]
    chi2, df_t, pv = mundlak_hausman_test(result, mean_vars)
    print(f"\n  Mundlak-Hausman Test  H₀: entity means jointly = 0")
    print(f"  χ²({df_t}) = {chi2:.3f},  p = {pv:.4f}")
    verdict = ("→ REJECT H₀ — FE needed" if pv < 0.05 else
               "→ FAIL TO REJECT H₀ — RE sufficient")
    print(f"  {verdict}")

    me_results = compute_marginal_effects(result, ict_var, df)

    print_me_block(
        'WITHIN-ENTITY MARGINAL EFFECTS',
        ict_var,
        me_results['within_coefficients'],
        me_results['within_summary'],
        me_results['within_table'],
        between=False
    )

    print_me_block(
        'IMPLIED BETWEEN-ENTITY MARGINAL EFFECTS',
        ict_var,
        me_results['between_coefficients'],
        me_results['between_summary'],
        me_results['between_table'],
        between=True
    )

    vif_df = compute_vif(df, ict_var, edu_control)
    print(f"\n  === VIF (within-demeaned, no dummies) ===")
    print(vif_df.to_string(index=False))

    return me_results, vif_df

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_full_analysis(df_master):
    df = prepare_panel(df_master)

    time_varying_ij = [
        'ai_adoption', 'spec_ict', 'training_ict',
        'ai_x_spec_ict', 'ai_x_training_ict', 'tert_edu'
    ]
    time_varying_j = ['log_gdp', 'infl_r', 'unempl_r']

    df = add_mundlak_means(df, time_varying_ij, time_varying_j)
    results = {}

    print("\n" + "#"*68)
    print("  SPECIFICATION A — ICT SPECIALISTS + tert_edu (No FSI/Prod)")
    print("#"*68)
    res_a, edu_a, df_clean_a = run_cre_model(df, 'spec_ict', 'tert_edu', 'ICT Specialists')
    me_a, vif_a  = display_results(res_a, 'spec_ict', edu_a, 'ICT SPECIALISTS', df_clean_a)
    results.update({'spec_spec_ict': res_a, 'me_spec_ict': me_a, 'vif_spec_ict': vif_a})

    print("\n" + "#"*68)
    print("  SPECIFICATION B — ICT TRAINING + tert_edu (No FSI/Prod)")
    print("#"*68)
    res_b, edu_b, df_clean_b = run_cre_model(df, 'training_ict', 'tert_edu', 'ICT Training')
    me_b, vif_b  = display_results(res_b, 'training_ict', edu_b, 'ICT TRAINING', df_clean_b)
    results.update({'spec_training_ict': res_b, 'me_training_ict': me_b, 'vif_training_ict': vif_b})

    return results

print("CRE/Mundlak INTERACTION script loaded. Run: results = run_full_analysis(df_master)")