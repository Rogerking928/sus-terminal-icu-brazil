"""Tables (from out/*.csv) and the figures book / supplement, via the shared docbuild.
Every number printed in a table comes from a CSV written by analysis.py or validate.py."""
import json, os, sys
import numpy as np, pandas as pd

import docbuild as DB                                  # noqa: E402
from rules import apply_three_rules                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"
MS = f"{ROOT}/manuscript"
AGE_LAB = ['18-44', '45-64', '65-74', '75-84', '85+']


def n(x): return f"{int(round(x)):,}"
def p(x): return f"{x:.1f}"


def _or_lookup():
    o = pd.read_csv(f"{OUT}/t2_model_or_cluster.csv", index_col=0)
    return {t.split('[T.')[1].rstrip(']'): x for t, x in o.iterrows() if '[T.' in t}


def table1_csp():
    """Cadernos de Saude Publica: one value per cell, portrait 17 cm. Decedents, ICU care and the adjusted odds ratio
    (logistic model, clustered by hospital) for every characteristic, including year of death (the trend)."""
    t = pd.read_csv(f"{OUT}/t1_characteristics.csv").set_index('row')
    ty = pd.read_csv(f"{OUT}/t_trend_year.csv")
    ors = _or_lookup()
    rows = [["Characteristic", "Decedents, n", "Decedents, %", "With ICU care, n", "With ICU care, % of row",
             "Adjusted OR", "95% CI"]]
    def r(label, key, lev, ref):
        x = t.loc[key]
        o = ors.get(lev)
        rows.append([label, n(x.n), p(x.pct_all), n(x.n_icu), p(x.pct_with_icu),
                     "1" if lev == ref else f"{o.OR:.2f}", "Reference" if lev == ref else f"{o.lo:.2f}-{o.hi:.2f}"])
    x = t.loc["All"]
    rows.append(["All adult decedents", n(x.n), p(x.pct_all), n(x.n_icu), p(x.pct_with_icu), "", ""])
    rows.append(["Age group, years", "", "", "", "", "", ""])
    for a in AGE_LAB: r(f"  {a}", f"age_grp:{a}", a, '65-74')
    rows.append(["Sex", "", "", "", "", "", ""])
    for s in ["Female", "Male"]: r(f"  {s}", f"sex:{s}", s, 'Male')
    rows.append(["Admission", "", "", "", "", "", ""])
    r("  Urgent", "admission:urgent", "Urgent", "Elective"); r("  Elective", "admission:elective", "Elective", "Elective")
    rows.append(["Principal diagnosis (ICD-10 chapter)", "", "", "", "", "", ""])
    ks = [k for k in t.index if k.startswith('cause:')]
    for k in sorted(ks, key=lambda k: -t.loc[k].n): r(f"  {k.split(':')[1]}", k, k.split(':')[1], 'Circulatory')
    rows.append(["Region of the hospital", "", "", "", "", "", ""])
    for g in ['North', 'Northeast', 'Centre-West', 'Southeast', 'South']: r(f"  {g}", f"region:{g}", g, 'Southeast')
    rows.append(["Year of death", "", "", "", "", "", ""])
    tot = ty.n.sum()
    for y in ty.itertuples():
        lev = str(int(y.year)); o = ors.get(lev)
        rows.append([f"  {lev}", n(y.n), p(100 * y.n / tot), n(y.n_icu), p(y.pct),
                     "1" if lev == '2015' else f"{o.OR:.2f}", "Reference" if lev == '2015' else f"{o.lo:.2f}-{o.hi:.2f}"])
    return rows


def table_year():
    """deaths in hospitals without ICU billing and the share of adult ICU care used by stays ending in death, by year"""
    r = pd.read_csv(f"{OUT}/t4_resource.csv")
    a = pd.read_csv(f"{OUT}/t_access_year.csv").set_index('year')
    rows = [["Year", "Hospitals billing adult ICU care, n", "Decedents in a hospital without ICU billing, %",
             "Adult ICU stays, n", "ICU stays ending in death, %", "ICU days in stays ending in death, %",
             "ICU payments in stays ending in death, %"]]
    for x in r.itertuples():
        rows.append([str(int(x.year)), n(x.hospitals_icu), p(a.loc[x.year, 'pct_no_icu_hosp']), n(x.stays),
                     p(x.pct_stays_dying), p(x.pct_days_dying), p(x.pct_val_dying)])
    t = r[['stays', 'stays_dying', 'days', 'days_dying', 'val', 'val_dying']].sum()
    rows.append(["2015-2024", "", p(100 * a.no_icu_hosp.sum() / a.n.sum()), n(t.stays), p(100 * t.stays_dying / t.stays),
                 p(100 * t.days_dying / t.days), p(100 * t.val_dying / t.val)])
    return rows


def table_distance():
    """care received by distance, one block per outcome (portrait)"""
    c = pd.read_csv(f"{OUT}/t_iv_crude.csv")
    a = pd.read_csv(f"{OUT}/t_iv_adjusted.csv")
    rows = [["Outcome and distance", "Urgent admissions, n", "With the outcome, %", "Adjusted OR", "95% CI"]]
    for o, lab in (('noicu', 'Admitted to a hospital without ICU billing'), ('icu', 'ICU care during the admission'),
                   ('transfer', 'Transferred to another hospital')):
        rows.append([lab, "", "", "", ""])
        for x in c.itertuples():
            y = a[(a.outcome == o) & (a.band == x.band)]
            rows.append([f"  {x.band}", n(x.n), p(getattr(x, o)),
                         "1" if not len(y) else f"{y.iloc[0].OR:.2f}",
                         "Reference" if not len(y) else f"{y.iloc[0].lo:.2f}-{y.iloc[0].hi:.2f}"])
    return rows


def flow_note():
    f = json.load(open(f"{OUT}/numbers.json"))['flow']
    return (f"Of {n(f['in_years'])} stays of adults ending in death in 2015-2024, {n(f['in_years'] - f['non_covid'])} "
            f"with a COVID-19 code in any diagnosis field were excluded.")


def table3():
    r = pd.read_csv(f"{OUT}/t4_resource.csv")
    rows = [["Year", "Hospitals billing adult ICU care", "Adult ICU stays", "Ending in death, %", "ICU days",
             "In stays ending in death, %", "ICU payments, BRL million, nominal", "In stays ending in death, % (nominal)",
             "ICU payments, BRL million, December 2024"]]
    for x in r.itertuples():
        rows.append([str(int(x.year)), n(x.hospitals_icu), n(x.stays), p(x.pct_stays_dying), n(x.days),
                     p(x.pct_days_dying), f"{x.val / 1e6:,.1f}", p(x.pct_val_dying), f"{x.val_real / 1e6:,.1f}"])
    t = r[['stays', 'stays_dying', 'days', 'days_dying', 'val', 'val_dying', 'val_real']].sum()
    rows.append(["2015-2024", "-", n(t.stays), p(100 * t.stays_dying / t.stays), n(t.days),
                 p(100 * t.days_dying / t.days), f"{t.val / 1e6:,.1f}", p(100 * t.val_dying / t.val),
                 f"{t.val_real / 1e6:,.1f}"])
    return rows


def tableS_distance_cond():
    b = pd.read_csv(f"{OUT}/t_iv_by_condition.csv")
    bands = list(pd.read_csv(f"{OUT}/t_iv_crude.csv").band)
    rows = [["Group"] + [f"{x}, % (n)" for x in bands]]
    rows.append(["ICU care during the admission"] + [""] * len(bands))
    for cond in ['Sepsis', 'Pneumonia', 'Stroke', 'Myocardial infarction', 'Died in hospital (all four)']:
        y = b[b.cond == cond].set_index('band')
        rows.append([cond] + [f"{p(y.loc[k, 'icu'])} ({n(y.loc[k, 'n'])})" for k in bands])
    return rows


def tableS_hosp_state():
    h = pd.read_csv(f"{OUT}/t_hosp_by_state.csv").sort_values('uf')
    rows = [["State", "Hospitals", "Observed/expected ratio, 10th percentile", "Observed/expected ratio, 90th percentile",
             "Ratio of 90th to 10th percentile"]]
    for x in h.itertuples():
        rows.append([x.uf, str(int(x.hosp)), f"{x.p10:.2f}", f"{x.p90:.2f}", f"{x.p90 / x.p10:.1f}"])
    return rows


def tableS_validation():
    v = pd.read_csv(f"{OUT}/t_s_validation.csv")
    g = v.groupby('year').agg(months=('month', 'size'), tabnet_aih=('tabnet_aih', 'sum'), mirror_aih=('mirror_aih', 'sum'),
                              tabnet_adm=('tabnet_adm', 'sum'), mirror_adm=('mirror_adm', 'sum'),
                              tabnet_deaths=('tabnet_deaths', 'sum'), mirror_deaths=('mirror_deaths', 'sum'),
                              maxdiff=('diff_deaths_pct', lambda s: 0)).reset_index()
    for k in ('aih', 'adm', 'deaths'):
        v[f'd_{k}'] = v[f'mirror_{k}'] - v[f'tabnet_{k}']
        v[f'p_{k}'] = 100 * v[f'd_{k}'] / v[f'tabnet_{k}']
    v['ex'] = (v[['d_aih', 'd_adm', 'd_deaths']] == 0).all(axis=1)          # identical counts, not rounded percentages
    v['mxn'] = v[['d_aih', 'd_adm', 'd_deaths']].abs().max(axis=1)
    v['mx'] = v[['p_aih', 'p_adm', 'p_deaths']].abs().max(axis=1)
    g['exact'] = v.groupby('year').ex.sum().values
    g['maxn'] = v.groupby('year').mxn.max().values
    g['maxdiff'] = v.groupby('year').mx.max().values
    rows = [["Processing year", "Months", "Approved AIHs, TabNet", "Approved AIHs, analysed files",
             "Admissions, TabNet", "Admissions, analysed files", "Deaths, TabNet", "Deaths, analysed files",
             "Months identical on all three counts", "Largest monthly difference, records",
             "Largest monthly difference, %"]]
    for x in g.itertuples():
        rows.append([str(x.year), str(x.months), n(x.tabnet_aih), n(x.mirror_aih), n(x.tabnet_adm), n(x.mirror_adm),
                     n(x.tabnet_deaths), n(x.mirror_deaths), str(int(x.exact)), n(x.maxn), f"{x.maxdiff:.4f}"])
    rows.append(["2015-2025", str(int(g.months.sum())), n(g.tabnet_aih.sum()), n(g.mirror_aih.sum()), n(g.tabnet_adm.sum()),
                 n(g.mirror_adm.sum()), n(g.tabnet_deaths.sum()), n(g.mirror_deaths.sum()), str(int(g.exact.sum())),
                 n(g.maxn.max()), f"{g.maxdiff.max():.4f}"])
    return rows


def tableS_state():
    s = pd.read_csv(f"{OUT}/t3_state.csv").sort_values(['region', 'uf'])
    rows = [["State", "Region", "Adult decedents", "With ICU care, crude %", "Observed/expected (95% CI)",
             "Standardised %", "Died in a hospital without ICU billing, %"]]
    for x in s.itertuples():
        rows.append([x.uf, x.region, n(x.n), p(x.crude_pct), f"{x.oe:.2f} ({x.oe_lo:.2f}-{x.oe_hi:.2f})",
                     p(x.std_pct), p(x.pct_no_icu_hosp)])
    return rows


def tableS_access():
    a = pd.read_csv(f"{OUT}/t_age_access.csv")
    rows = [["Age, years", "Adult decedents", "With ICU care, %", "Died in a hospital without ICU billing, %",
             "With ICU care among deaths in hospitals with ICU billing, %"]]
    num = json.load(open(f"{OUT}/numbers.json"))
    for x in a.itertuples():
        rows.append([x.age_grp, n(x.n), p(x.pct_icu), p(x.pct_no_icu_hosp), p(x.pct_icu_within_icu_hosp)])
    rows.append(["All adults", n(a.n.sum()), p(num['pct_icu']), p(num['pct_die_no_icu_hosp']),
                 p(num['pct_icu_within_icu_hosp'])])
    return rows


def tableS_trend():
    a = pd.read_csv(f"{OUT}/t_trend_year.csv")
    t = pd.read_csv(f"{OUT}/t_trend_year_age.csv")
    rows = [["Year of death", "Adult decedents", "All ages, %"] + [f"{g} y, %" for g in AGE_LAB]]
    for x in a.itertuples():
        rows.append([str(int(x.year)), n(x.n), p(x.pct)] +
                    [p(t[(t.year == x.year) & (t.age_grp == g)].pct.iloc[0]) for g in AGE_LAB])
    return rows


def tableS_sens():
    s = pd.read_csv(f"{OUT}/t5_sensitivity.csv")
    rows = [["Analysis", "Adult decedents", "With ICU care, %"]]
    for x in s.itertuples():
        rows.append([x.analysis, n(x.n), p(x.pct_icu)])
    return rows


def trend_note():
    t = pd.read_csv(f"{OUT}/t_trend_year.csv").set_index('year')
    a1 = (t.pct[2021.0] - t.pct[2015.0]) / 6
    a2 = (t.pct[2024.0] - t.pct[2021.0]) / 3
    return (f"In (A), the share for all adults changed by {a1:+.1f} percentage points a year from 2015 to 2021 and by "
            f"{a2:+.1f} a year from 2021 to 2024.")


def hosp_note():
    q = json.load(open(f"{OUT}/numbers.json"))['hospital_variation']
    return (f"Hospitals with at least 200 adult deaths in years in which they billed adult intensive care "
            f"({q['hospitals']:,} hospitals, {q['deaths_in_them']:,} deaths). Expected numbers come from a model of age, "
            f"sex, principal diagnosis, admission type and year; a ratio of 1 means intensive care as often as expected for "
            f"the hospital's decedents. Each point is a hospital; the solid line is a ratio of 1. 10th percentile "
            f"{q['p10']:.2f}, median {q['p50']:.2f}, 90th percentile {q['p90']:.2f} (ratio of 90th to 10th, "
            f"{q['p90_over_p10']:.1f}). The variance of (O-E)/sqrt(E) was {q['overdispersion']:.1f} times that expected by "
            f"chance. From the variance of the log ratio beyond sampling error (1/O), the median ratio between two hospitals "
            f"chosen at random was {q['mor_hospital']:.2f} ({q['mor_hospital_within_state']:.2f} within the same state), and "
            f"{q['pct_var_within_state']:.1f}% of that variance lay within states (the variance of the log ratio around "
            f"each state's mean, minus the mean of 1/O, over the same quantity around the national mean); these log-scale estimates exclude the "
            f"{q['hospitals_zero_icu']} hospitals with no decedent with ICU care, which are plotted at 0.")




def items():
    """2026-10-28, Cadernos de Saude Publica (Artigo: at most 5 illustrations, each panel of a composite figure counting
    as one; one value per table cell; supplementary material only as a complement). Main text: Table 1 (decedents,
    adjusted odds ratios, trend), Figure 1 (states), Figure 2 (hospitals), Table 2 (deaths in hospitals without ICU
    billing and ICU resources, by year), Table 3 (distance). Population mortality, differential distance, E-values and
    in-hospital death by distance are not part of the article."""
    T = lambda rows, cap, note, sz=9.0: {"kind": "table", "caption": cap, "rows": rows, "body_size": sz, "note": note}
    F = lambda f, cap, note: {"kind": "figure", "caption": cap, "path": f"{OUT}/{f}.png", "note": note}
    num = json.load(open(f"{OUT}/numbers.json"))
    iv = json.load(open(f"{OUT}/iv_feasibility.json"))
    a = num['age_median'], *num['age_iqr']
    dd = num['icu_days_median_icu'], *num['icu_days_iqr_icu']
    main = [
        T(table1_csp(), "Table 1. Adults whose hospital stay funded by the Sistema Único de Saúde ended in death, "
          "2015-2024: intensive care during the terminal admission by characteristic.",
          "ICU - intensive care unit; OR - odds ratio; CI - confidence interval. Decedents, % is the column percentage; "
          "with ICU care, % of row is the share of each row with at least one ICU day billed. Odds ratios come from one "
          "logistic regression including every characteristic shown, with confidence intervals allowing for clustering "
          f"of decedents within hospitals. Median age {a[0]:.0f} years (interquartile range {a[1]:.0f}-{a[2]:.0f}); "
          f"median ICU days among decedents with ICU care {dd[0]:.0f} (interquartile range {dd[1]:.0f}-{dd[2]:.0f}). "
          + flow_note() + " Principal diagnosis is that of the terminal admission, not the underlying cause of death.",
          8.5),
        F("figure1", "Figure 1. Intensive care during the terminal admission by state, standardised for age, sex, "
          "principal diagnosis, admission type and year.",
          "Filled markers are standardised percentages with 95% confidence intervals (observed/expected ratio multiplied "
          "by the national percentage), open markers crude percentages, and the dashed line the national percentage. "
          "States are identified by their two-letter abbreviations and coloured by region; values are in Table S3 and "
          "a map in Figure S4."),
        F("figure2", "Figure 2. Intensive care during the terminal admission by hospital: observed/expected ratio "
          "against the expected number of decedents with intensive care.",
          "Dashed and dotted curves are 95% and 99.8% Poisson control limits. " + hosp_note()),
        T(table_year(), "Table 2. Deaths in hospitals without intensive care billing and adult intensive care used by "
          "stays ending in death, by year.",
          "ICU - intensive care unit. A hospital without ICU billing is a facility that billed no adult ICU day to the "
          "Sistema Único de Saúde in that year. ICU stays, days and payments (the ICU component of the hospital bill) "
          "cover every adult stay with at least one ICU day billed, excluding stays with a COVID-19 code; absolute days "
          "and payments, nominal and in December 2024 reais, are in Table S6.", 8.5),
        T(table_distance(), "Table 3. Care received in urgent admissions for four conditions that often need intensive "
          "care, by distance from the patient's municipality to the nearest hospital with ICU billing.",
          "ICU - intensive care unit; OR - odds ratio against residents of a municipality with a hospital with ICU "
          "billing (0 km), adjusted for age group, sex, condition, year and state of residence; CI - confidence "
          "interval clustered by municipality of residence. Adults aged 18 years or older, urgent admissions with a "
          "principal diagnosis of sepsis (A40-A41), pneumonia (J12-J18), stroke (I60-I64) or acute myocardial "
          "infarction (I21-I22), 2015-2024, excluding admissions with a COVID-19 code "
          f"({iv['analysed']:,} admissions). Distance is the great-circle distance between municipal seats in that year. "
          "Care after transfer is not captured, because a transferred patient's next admission cannot be linked.", 8.5),
    ]
    los = num['pct_los_lt2']
    hv = num['hospital_variation']
    supp = [
        F("figureS1", "Figure S1. Selection of adult decedents.",
          "AIH - authorisation for hospital admission. Records are from the reduced files of the Sistema de Informações "
          "Hospitalares; a stay is all AIHs sharing a number. Duplicate death records are AIHs of the same stay "
          "submitted again. The year of a stay is the year of discharge (death). COVID-19 codes are U07.1, U07.2 and "
          "B34.2 in any diagnosis field of any AIH of the stay."),
        T(tableS_validation(), "Table S1. The analysed files against the official DATASUS TabNet tabulation.",
          "National totals by processing month, summed to the year. Admissions are AIHs of type 1; approved AIHs "
          "include continuation AIHs of long stays. A month is identical when all three counts are equal. The largest "
          "difference is the largest of the three monthly differences, analysed files minus TabNet, in records and as a "
          "percentage of TabNet; the two need not come from the same month. For December 2020 the tabulation by place of "
          "admission is incomplete and the tabulation of the same files by place of residence was used.", 7.5),
        F("figureS2", "Figure S2. Intensive care during the terminal admission by year and age.",
          "Share of adult decedents with ICU care during the terminal admission (values in Table S2); the shaded band "
          "marks 2020-2021. " + trend_note()),
        T(tableS_trend(), "Table S2. Intensive care during the terminal admission by year and age.",
          "Values plotted in Figure S2. Percentages are of adult decedents in each year and age group.", 9.0),
        F("figureS3", "Figure S3. Intensive care during the terminal admission by principal diagnosis and age.",
          "Share of adult decedents with ICU care, by ICD-10 chapter of the principal diagnosis of the terminal "
          "admission (ordered by number of deaths) and age at death, 2015-2024."),
        T(tableS_state(), "Table S3. Intensive care during the terminal admission by state.",
          "Expected counts come from the logistic model without geography. A hospital without ICU billing is a "
          "facility that billed no adult ICU day to the Sistema Único de Saúde in that year. States are identified "
          "by their two-letter abbreviations.", 8.0),
        F("figureS4", "Figure S4. Standardised share of decedents with intensive care during the terminal admission, "
          "by state.",
          "Standardised for age, sex, principal diagnosis, admission type and year (values in Table S3). State "
          "boundaries from IBGE."),
        T(tableS_hosp_state(), "Table S4. Spread of intensive care during the terminal admission between hospitals "
          "within each state.",
          f"The {hv['states_with_10plus_hosp']} states with at least 10 hospitals meeting the criteria of Figure 2. "
          "Percentiles are of the hospitals' observed/expected ratios. Median ratio of 90th to 10th percentile across "
          f"these states, {hv['median_within_state_p90_over_p10']:.1f}.", 9.0),
        T(tableS_access(), "Table S5. Deaths in hospitals without intensive care billing, by age.",
          "A hospital without ICU billing is a facility that billed no adult ICU day to the Sistema Único de "
          "Saúde in the year of death.", 9.0),
        T(table3(), "Table S6. Adult intensive care funded by the Sistema Único de Saúde spent in stays that ended "
          "in death, 2015-2024: stays, days and payments.",
          "ICU - intensive care unit; BRL - Brazilian real. ICU payments are the ICU component of the hospital bill "
          "(VAL_UTI), as billed (nominal) and converted to December 2024 reais with the mean IPCA consumer price index "
          "of each year (IBGE). Hospitals billing adult ICU care are facilities with at least one adult ICU day billed "
          "in the year. Stays with a COVID-19 code are excluded. Within each year the share of payments in stays ending in "
          "death is the same in nominal and deflated values; over 2015-2024 it was "
          f"{num['add_ipca']['pct_val_dying_real']:.1f}% of deflated payments. "
          "Stays are counted in each year in which they have an ICU billing record, so a stay whose ICU billing spans "
          f"two years is counted in both: {num['cross_year_stays']['all']} stays in "
          f"the 2015-2024 total, {num['cross_year_stays']['dying']} of them ending in "
          f"death (hence {num['cross_year_stays']['dying']} more than the decedents with ICU care in Table 1).", 8.0),
        F("figureS5", "Figure S5. Adult ICU stays billed and hospitals billing adult ICU care, 2015-2024.",
          "Each series indexed to 2015 = 100 (values in Table S6); the shaded band marks 2020-2021. Stays with a "
          "COVID-19 code are excluded."),
        F("figureS6", "Figure S6. Care received in urgent admissions for the four conditions by distance to the nearest "
          "hospital with ICU billing.",
          "Distance from the seat of the municipality of residence to the nearest municipal seat with a hospital with "
          "ICU billing (0 km: such a hospital in the patient's own municipality). (A) Admissions with at least one ICU "
          "day billed, by condition (values in Table S7). (B) Admissions to a hospital without ICU billing and discharges "
          "by transfer to another hospital (values in Table 3)."),
        T(tableS_distance_cond(), "Table S7. Intensive care during urgent admissions by condition and distance to the "
          "nearest hospital with ICU billing.",
          "Percentages with the number of admissions. ICU care is at least one ICU day billed; the row 'Died in "
          "hospital (all four)' is restricted to admissions for any of the four conditions that ended in death in "
          "hospital.", 8.0),
        T(tableS_sens(), "Table S8. Sensitivity analyses of the primary estimate.",
          "The primary analysis counts a stay as receiving intensive care if any of its AIHs billed at least one "
          f"ICU day. Stays shorter than 2 days were {los:.1f}% of adult decedents.", 9.0),
        F("figureS7", "Figure S7. Sensitivity analyses of the primary estimate.",
          "Share of adult decedents with ICU care during the terminal admission under each definition in Table S8; "
          "the dashed line is the primary analysis."),
    ]
    return main, supp


def build_portrait(items_, path):
    """Cadernos de Saude Publica: tables in portrait, at most 17 cm wide. The shared builder lays pages out in
    landscape Letter; its geometry constants are read at call time, so they are set to portrait Letter with a 6.7 in
    (17.0 cm) text block for this file only, and the section orientation is corrected afterwards."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    keep = {k: getattr(DB, k) for k in ("PAGE_W", "PAGE_H", "MARGIN_LR", "MARGIN_TB", "TEXT_W", "TEXT_H", "IMG_MAX_W")}
    DB.PAGE_W, DB.PAGE_H, DB.MARGIN_LR, DB.MARGIN_TB = 8.5, 11.0, 0.9, 0.9
    DB.TEXT_W, DB.TEXT_H, DB.IMG_MAX_W = 6.7, 9.2, 6.5
    try:
        DB.build(items_, path)
    finally:
        for k, v in keep.items(): setattr(DB, k, v)
    d = Document(path)
    for sec in d.sections:
        sec.orientation = WD_ORIENT.PORTRAIT
    d.save(path)


def main():
    os.makedirs(MS, exist_ok=True)
    m, s = items()
    build_portrait(m, f"{MS}/figures_tables.docx")
    DB.build(s, f"{MS}/supplement.docx")
    for f in ("figures_tables.docx", "supplement.docx"):
        print(f, apply_three_rules(f"{MS}/{f}"), "tables ruled")
    return m, s


if __name__ == '__main__':
    main()
