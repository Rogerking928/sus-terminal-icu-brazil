"""ICU use in the terminal admission of adults who die in SUS hospitals, Brazil 2015-2024.

Inputs are the per-partition files written by extract.py. Unit of analysis is the
hospital stay, identified by N_AIH: a long stay is billed as a first AIH (IDENT 1)
plus continuation AIHs (IDENT 5) under the same number, so ICU days billed on any
segment of the stay count towards that stay. Year is the year of discharge (death).

Writes every table behind the manuscript to out/*.csv and a summary to out/numbers.json.
"""
import glob, json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"
Y0, Y1 = 2015, 2024
COVID_YEARS = (2020, 2021)

UF = {'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO','21':'MA','22':'PI','23':'CE',
      '24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP',
      '41':'PR','42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'}
REGION = {'1':'North','2':'Northeast','3':'Southeast','4':'South','5':'Centre-West'}
AGE_BINS = [18, 45, 65, 75, 85, 200]
AGE_LAB = ['18-44', '45-64', '65-74', '75-84', '85+']


def cause(icd):
    c = str(icd).strip().upper()
    if not c or c[0] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': return 'Other'
    l, n = c[0], int(c[1:3]) if c[1:3].isdigit() else -1
    if l in 'AB': return 'Infectious'
    if l == 'C' or (l == 'D' and n <= 48): return 'Neoplasm'
    if l == 'I': return 'Circulatory'
    if l == 'J': return 'Respiratory'
    if l == 'K': return 'Digestive'
    if l == 'N': return 'Genitourinary'
    if l == 'E': return 'Endocrine'
    if l == 'G' or (l == 'F'): return 'Nervous/mental'
    if l in 'STVWXY': return 'Injury'
    return 'Other'


def tags():
    """partitions whose extraction finished (the json is written last)"""
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{ROOT}/data/agg/*.json"))


DCOLS = ['N_AIH', 'IDENT', 'ANO_CMPT', 'MES_CMPT', 'CNES', 'MUNIC_MOV', 'SEXO', 'IDADE', 'DT_SAIDA', 'DIAS_PERM',
         'CAR_INT', 'DIAG_PRINC', 'MARCA_UTI', 'icu_days', 'covid']
ICOLS = ['N_AIH', 'DT_SAIDA', 'icu_days', 'val_uti', 'adult', 'covid']


CAUSES = ['Infectious', 'Neoplasm', 'Circulatory', 'Respiratory', 'Digestive', 'Genitourinary', 'Endocrine',
          'Nervous/mental', 'Injury', 'Other']


def _num(s, dtype):
    return pd.to_numeric(s.astype(str).str.strip(), errors='coerce').astype(dtype)


def _compact(x):
    """per-file reduction to numeric and fixed-category columns, so ~5 million deaths and ~12 million
    ICU segments concatenate without falling back to Python strings"""
    y = pd.DataFrame({'N_AIH': _num(x.N_AIH, 'int64'), 'DT_SAIDA': _num(x.DT_SAIDA.astype(str).str[:4], 'float32'),
                      'icu_days': x.icu_days.astype('int32'), 'covid': x.covid.astype(bool)})
    if 'MUNIC_MOV' in x:
        mm = x.MUNIC_MOV.astype(str)
        y['IDENT'] = _num(x.IDENT, 'float32')
        y['cmpt'] = _num(x.ANO_CMPT, 'float32') * 100 + _num(x.MES_CMPT, 'float32')
        y['CNES'] = _num(x.CNES, 'float64')
        y['SEXO'] = _num(x.SEXO, 'float32')
        y['age'] = _num(x.IDADE, 'float32')
        y['los'] = _num(x.DIAS_PERM, 'float32')
        y['urgent'] = x.CAR_INT.astype(str).str.strip() != '01'
        y['marca'] = _num(x.MARCA_UTI, 'float32')
        y['uf'] = pd.Categorical(mm.str[:2].map(UF), categories=sorted(UF.values()))
        y['region'] = pd.Categorical(mm.str[:1].map(REGION), categories=sorted(REGION.values()))
        y['cause'] = pd.Categorical(x.DIAG_PRINC.map(cause), categories=CAUSES)
    else:
        y['val_uti'] = x.val_uti.astype('float32')
        y['adult'] = x.adult.astype(bool)
    return y


def load():
    t = tags()
    d = pd.concat([_compact(pd.read_parquet(f"{ROOT}/data/raw_deaths/{x}.parquet", columns=DCOLS)) for x in t],
                  ignore_index=True)
    icu = pd.concat([_compact(pd.read_parquet(f"{ROOT}/data/raw_icu/{x}.parquet", columns=ICOLS)) for x in t],
                    ignore_index=True)
    return d, icu


def build(d, icu):
    n_rows = len(d)
    # one record per stay: the last death record by competence (resubmissions duplicate rows)
    d = d.sort_values('cmpt').drop_duplicates('N_AIH', keep='last')
    n_stays = len(d)
    # ICU exposure from every billed segment of the stay
    seg = icu.groupby('N_AIH').agg(icu_days_stay=('icu_days', 'sum'), icu_segments=('icu_days', 'size'),
                                  icu_covid=('covid', 'max')).astype({'icu_covid': 'float32'})
    d = d.join(seg, on='N_AIH')
    d['icu'] = d.icu_days_stay.fillna(0) > 0
    d['icu_days_stay'] = d.icu_days_stay.fillna(0)
    d['year'] = d.DT_SAIDA
    d['age_grp'] = pd.cut(d.age, AGE_BINS, right=False, labels=AGE_LAB)
    d['sex'] = d.SEXO.map({1: 'Male', 2: 'Female', 3: 'Female'})
    d['covid_any'] = d.covid | (d.icu_covid.fillna(0) > 0)
    flow = dict(death_records=n_rows, stays=n_stays)
    d = d[d.year.between(Y0, Y1)]
    flow['in_years'] = len(d)
    d = d[~d.covid_any]
    flow['non_covid'] = len(d)
    d = d[d.sex.notna() & d.uf.notna() & d.age_grp.notna()]
    flow['analysed'] = len(d)
    return d, flow


def hospital_icu(years):
    """CNES-year with any adult ICU day billed = hospital has an ICU that year."""
    h = pd.concat([pd.read_parquet(f"{ROOT}/data/agg/{x}_hosp.parquet") for x in tags()], ignore_index=True)
    h['year'] = h.tag.str[:4].astype(int)
    h['CNES'] = pd.to_numeric(h.CNES, errors='coerce')
    h = h.groupby(['CNES', 'year']).agg(n=('n', 'sum'), icu_days=('icu_days', 'sum')).reset_index()
    h['has_icu'] = h.icu_days > 0
    return h[h.year.between(*years)]


def pct(x): return round(100 * float(x), 1)


def main():
    d, icu = load()
    d, flow = build(d, icu)
    num = {'flow': flow}

    # ---- Table 1: decedent characteristics by ICU use
    rows = []
    def add(label, s):
        tot, yes, no = s.sum(), s[d.icu].sum(), s[~d.icu].sum()
        rows.append([label, int(tot), pct(tot / len(d)), int(yes), pct(yes / d.icu.sum()),
                     int(no), pct(no / (~d.icu).sum()), pct(yes / tot) if tot else np.nan])
    add('All', pd.Series(True, index=d.index))
    for v in ['age_grp', 'sex', 'cause', 'region']:
        for k in (AGE_LAB if v == 'age_grp' else sorted(d[v].dropna().unique())):
            add(f"{v}:{k}", d[v] == k)
    add('admission:urgent', d.urgent); add('admission:elective', ~d.urgent)
    t1 = pd.DataFrame(rows, columns=['row', 'n', 'pct_all', 'n_icu', 'pct_icu_col', 'n_noicu', 'pct_noicu_col',
                                     'pct_with_icu'])
    t1.to_csv(f"{OUT}/t1_characteristics.csv", index=False)
    num['n_deaths'] = len(d); num['n_icu'] = int(d.icu.sum()); num['pct_icu'] = pct(d.icu.mean())
    num['age_median'] = float(d.age.median()); num['age_iqr'] = [float(d.age.quantile(.25)), float(d.age.quantile(.75))]
    num['icu_days_median_icu'] = float(d.loc[d.icu, 'icu_days_stay'].median())
    num['icu_days_iqr_icu'] = [float(d.loc[d.icu, 'icu_days_stay'].quantile(q)) for q in (.25, .75)]

    # ---- trend by year and age group
    tr = d.groupby(['year']).agg(n=('icu', 'size'), n_icu=('icu', 'sum')).reset_index()
    tr['pct'] = (100 * tr.n_icu / tr.n).round(1)
    tr.to_csv(f"{OUT}/t_trend_year.csv", index=False)
    tra = d.groupby(['year', 'age_grp'], observed=True).agg(n=('icu', 'size'), n_icu=('icu', 'sum')).reset_index()
    tra['pct'] = (100 * tra.n_icu / tra.n).round(1)
    tra.to_csv(f"{OUT}/t_trend_year_age.csv", index=False)

    # ---- cause x age
    ca = d.groupby(['cause', 'age_grp'], observed=True).agg(n=('icu', 'size'), n_icu=('icu', 'sum')).reset_index()
    ca['pct'] = (100 * ca.n_icu / ca.n).round(1)
    ca.to_csv(f"{OUT}/t_cause_age.csv", index=False)

    # ---- multivariable model on aggregated cells (binomial GLM, identical to row-level logistic)
    d['yr'] = d.year.astype(int).astype(str)
    keys = ['age_grp', 'sex', 'cause', 'urgent', 'region', 'yr', 'uf']
    g = d.groupby(keys, observed=True).agg(n=('icu', 'size'), k=('icu', 'sum')).reset_index()
    g['urgent'] = g.urgent.map({True: 'Urgent', False: 'Elective'})
    g['fail'] = g.n - g.k
    ref = ("C(age_grp, Treatment('65-74')) + C(sex, Treatment('Male')) + C(cause, Treatment('Circulatory'))"
           " + C(urgent, Treatment('Elective')) + C(yr, Treatment('2015'))")
    m1 = smf.glm(f"k + fail ~ {ref} + C(region, Treatment('Southeast'))", data=g,
                 family=sm.families.Binomial()).fit()
    ors = pd.DataFrame({'OR': np.exp(m1.params), 'lo': np.exp(m1.conf_int()[0]), 'hi': np.exp(m1.conf_int()[1]),
                        'p': m1.pvalues}).round(4)
    ors.to_csv(f"{OUT}/t2_model_or.csv")
    # state indirect standardisation: expected from the model without geography
    m0 = smf.glm(f"k + fail ~ {ref}", data=g, family=sm.families.Binomial()).fit()
    g['exp'] = m0.predict(g) * g.n
    st = g.groupby(['uf', 'region'], observed=True).agg(n=('n', 'sum'), k=('k', 'sum'), exp=('exp', 'sum')).reset_index()
    st['crude_pct'] = (100 * st.k / st.n).round(1)
    st['oe'] = (st.k / st.exp).round(2)
    nat = d.icu.mean()
    st['std_pct'] = (100 * st.oe * nat).round(1)
    # Byar CI for O/E
    o = st.k
    st['oe_lo'] = (o * (1 - 1 / (9 * o) - 1.96 / (3 * np.sqrt(o))) ** 3 / st.exp).round(2)
    o1 = o + 1
    st['oe_hi'] = (o1 * (1 - 1 / (9 * o1) + 1.96 / (3 * np.sqrt(o1))) ** 3 / st.exp).round(2)
    st = st.sort_values('std_pct')
    st.to_csv(f"{OUT}/t3_state.csv", index=False)
    num['state_std_range'] = [float(st.std_pct.min()), float(st.std_pct.max())]
    num['state_crude_range'] = [float(st.crude_pct.min()), float(st.crude_pct.max())]

    # ---- hospital ICU availability: decedents dying in hospitals without any ICU billing that year
    h = hospital_icu((Y0, Y1))
    d = d.merge(h[['CNES', 'year', 'has_icu']], on=['CNES', 'year'], how='left')
    d['has_icu'] = d.has_icu.astype('boolean').fillna(False).astype(bool)
    hv = d.groupby('uf', observed=True).agg(n=('has_icu', 'size'), in_icu_hosp=('has_icu', 'sum'),
                            icu=('icu', 'sum')).reset_index()
    hv['pct_no_icu_hosp'] = (100 * (1 - hv.in_icu_hosp / hv.n)).round(1)
    st2 = st.merge(hv[['uf', 'pct_no_icu_hosp']], on='uf')
    st2.to_csv(f"{OUT}/t3_state.csv", index=False)
    num['pct_die_no_icu_hosp'] = pct(1 - d.has_icu.mean())
    within = d[d.has_icu]
    num['pct_icu_within_icu_hosp'] = pct(within.icu.mean())
    ag = d.groupby('age_grp', observed=True).agg(n=('icu', 'size'), no_icu_hosp=('has_icu', lambda s: (~s).sum()),
                                                icu=('icu', 'sum')).reset_index()
    ag['pct_no_icu_hosp'] = (100 * ag.no_icu_hosp / ag.n).round(1)
    ag['pct_icu'] = (100 * ag.icu / ag.n).round(1)
    wi = within.groupby('age_grp', observed=True).icu.mean().mul(100).round(1).rename('pct_icu_within_icu_hosp')
    ag = ag.join(wi, on='age_grp')
    ag.to_csv(f"{OUT}/t_age_access.csv", index=False)
    ya = d.groupby('year').agg(n=('has_icu', 'size'), no_icu_hosp=('has_icu', lambda s: (~s).sum())).reset_index()
    ya['pct_no_icu_hosp'] = (100 * ya.no_icu_hosp / ya.n).round(1)
    ya.to_csv(f"{OUT}/t_access_year.csv", index=False)
    num['corr_state_std_vs_noicuhosp'] = float(np.corrcoef(st2.std_pct, st2.pct_no_icu_hosp)[0, 1].round(2))

    # ---- ICU resource share: adult non-COVID ICU days and spending in stays that end in death
    covid_stays = np.unique(icu.N_AIH.values[icu.covid.values])
    m = icu.adult.values & ~icu.covid.values & (icu.DT_SAIDA.values >= Y0) & (icu.DT_SAIDA.values <= Y1)
    ia = pd.DataFrame({'year': icu.DT_SAIDA.values[m], 'N_AIH': icu.N_AIH.values[m],
                       'days': icu.icu_days.values[m].astype('int64'), 'val': icu.val_uti.values[m].astype('float64')})
    ia = ia[~np.isin(ia.N_AIH.values, covid_stays)]
    ia['dies'] = np.isin(ia.N_AIH.values, d.N_AIH.values)
    del icu
    a1 = ia.groupby('year').agg(stays=('N_AIH', 'nunique'), days=('days', 'sum'), val=('val', 'sum'))
    a2 = ia[ia.dies].groupby('year').agg(stays_dying=('N_AIH', 'nunique'), days_dying=('days', 'sum'),
                                         val_dying=('val', 'sum'))
    rs = a1.join(a2).reset_index()
    rs['pct_stays_dying'] = (100 * rs.stays_dying / rs.stays).round(1)
    rs['pct_days_dying'] = (100 * rs.days_dying / rs.days).round(1)
    rs['pct_val_dying'] = (100 * rs.val_dying / rs.val).round(1)
    rs.to_csv(f"{OUT}/t4_resource.csv", index=False)
    # stays whose ICU billing spans two years are counted in both years (note to the resource table)
    xy = ia.groupby('N_AIH').agg(ny=('year', 'nunique'), dies=('dies', 'max'))
    num['cross_year_stays'] = dict(all=int((xy.ny > 1).sum()), dying=int(((xy.ny > 1) & xy.dies).sum()))
    tot = rs[['stays', 'stays_dying', 'days', 'days_dying', 'val', 'val_dying']].sum()
    num['resource_all_years'] = dict(pct_stays_dying=pct(tot.stays_dying / tot.stays),
                                     pct_days_dying=pct(tot.days_dying / tot.days),
                                     pct_val_dying=pct(tot.val_dying / tot.val),
                                     days=int(tot.days), days_dying=int(tot.days_dying))

    # ---- sensitivity analyses
    sens = []
    sens.append(['Primary (2015-2024, COVID admissions excluded)', len(d), pct(d.icu.mean())])
    nc = d[~d.year.isin(COVID_YEARS)]
    sens.append(['Excluding 2020-2021 entirely', len(nc), pct(nc.icu.mean())])
    one = d[d.IDENT == 1]
    sens.append(['Stays billed on a single AIH (IDENT 1 death record)', len(one), pct(one.icu.mean())])
    alt = d.icu | (d.marca.fillna(0).between(1, 98))
    sens.append(['ICU defined by ICU days or any ICU type code', len(d), pct(alt.mean())])
    sens.append(['Hospitals with ICU billing that year only', len(within), pct(within.icu.mean())])
    ls = d[d.los >= 2]
    sens.append(['Stays of 2 days or longer', len(ls), pct(ls.icu.mean())])
    num['pct_los_lt2'] = pct((d.los < 2).mean())
    pd.DataFrame(sens, columns=['analysis', 'n', 'pct_icu']).to_csv(f"{OUT}/t5_sensitivity.csv", index=False)

    json.dump(num, open(f"{OUT}/numbers.json", 'w'), indent=1, default=str)
    print(json.dumps(num, indent=1, default=str))
    print(tr.to_string()); print(st2.to_string())
    print(ors.to_string()); print(rs.to_string()); print(ag.to_string())
    print(pd.DataFrame(sens).to_string())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
