"""Distance to intensive care and the care received in ICU-sensitive emergencies (2026-09-25).

Urgent adult admissions for sepsis, pneumonia, stroke or acute myocardial infarction (extract2.py, cohort2).
For each municipality of residence and year: great-circle distance from its seat to the nearest municipal seat
that contains a hospital billing adult ICU days that year (0 if the patient's own municipality has one).
Outcomes: admission to a hospital without ICU billing that year, ICU days billed during the admission, and discharge
by transfer to another hospital; logistic models adjusted for age group, sex, condition, year and state of residence, with standard errors
clustered by municipality of residence.
In-hospital death is not analysed: admission itself depends on distance (remote residents are admitted with a milder
case mix), so the admitted cohort is not comparable across distance bands for mortality (checked 2026-09-25).
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import analysis as A

ROOT, OUT = A.ROOT, A.OUT
MUN = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/municipios_ibge2022.csv"
BANDS = [-1, 0.5, 25, 50, 100, 1e6]
BLAB = ['0 (own municipality)', '<25 km', '25-49 km', '50-99 km', '100+ km']


def load():
    ufs = sorted(A.UF)
    conds = ['Myocardial infarction', 'Pneumonia', 'Sepsis', 'Stroke']
    def rd(f):
        x = pd.read_parquet(f, columns=['N_AIH', 'IDENT', 'CNES', 'uf2', 'mov6', 'res6', 'sex', 'age', 'year', 'death',
                                        'transfer', 'icu_days', 'cond', 'covid'])
        x['uf2'] = pd.Categorical(x.uf2, categories=ufs)
        x['cond'] = pd.Categorical(x.cond, categories=conds)
        return x
    c = pd.concat([rd(f) for f in sorted(glob.glob(f"{ROOT}/data/cohort2/*.parquet"))], ignore_index=True)
    cv = np.unique(c.loc[c.covid, 'N_AIH'].values)
    c = c.sort_values(['N_AIH', 'IDENT'])
    s = c.groupby('N_AIH', sort=False).agg(CNES=('CNES', 'first'), mov6=('mov6', 'first'), res6=('res6', 'first'),
                                           sex=('sex', 'first'), age=('age', 'first'), cond=('cond', 'first'),
                                           year=('year', 'max'), death=('death', 'max'),
                                           transfer=('transfer', 'max'), icu_days=('icu_days', 'sum')).reset_index()
    del c
    s = s[s.year.between(A.Y0, A.Y1) & ~np.isin(s.N_AIH.values, cv)]
    return s


def distances(s):
    m = pd.read_csv(MUN)
    m['mun6'] = (m.codigo_ibge // 10).astype(int)
    lat, lon = np.radians(m.latitude.values), np.radians(m.longitude.values)
    P = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    idx = pd.Series(np.arange(len(m)), index=m.mun6)
    h = A.hospital_icu((A.Y0, A.Y1))
    loc = s.groupby('CNES').mov6.agg(lambda x: x.mode().iloc[0])
    h['mun6'] = h.CNES.map(loc)
    out = []
    for y in range(A.Y0, A.Y1 + 1):
        icu_m = np.unique(h.loc[(h.year == y) & h.has_icu & h.mun6.notna(), 'mun6'].astype(int))
        icu_m = icu_m[np.isin(icu_m, idx.index)]
        Q = P[idx[icu_m].values]
        if len(Q) == 0: continue
        # chord -> great-circle km, nearest ICU municipality for every municipality
        d = np.empty(len(m))
        for i in range(0, len(m), 500):
            dot = P[i:i + 500] @ Q.T
            d[i:i + 500] = 6371 * np.arccos(np.clip(dot.max(axis=1), -1, 1))
        out.append(pd.DataFrame({'res6': m.mun6.values, 'year': y, 'dist': d}))
    return pd.concat(out, ignore_index=True), h


def cluster_se(m, s, keys, outcome):
    """sandwich standard errors clustered by municipality of residence, accumulated in chunks of municipalities:
    V = c * B (sum_g u_g u_g') B, B = (X'WX)^-1 from the fit, u_g = sum of cell scores X (k - n p) in cluster g,
    c = G/(G-1). Same as statsmodels cov_type='cluster' (without the (N-1)/(N-K) term, negligible at this N)."""
    from patsy import build_design_matrices
    di = m.model.data.design_info
    B = np.asarray(m.normalized_cov_params)
    muns = np.unique(s.res6.values)
    U = np.zeros((len(muns), B.shape[0]))
    for i in range(0, len(muns), 400):
        part = s[s.res6.isin(muns[i:i + 400])]
        g = part.groupby(['res6'] + keys, observed=True).agg(n=(outcome, 'size'), k=(outcome, 'sum')).reset_index()
        X = np.asarray(build_design_matrices([di], g)[0])
        p = 1 / (1 + np.exp(-(X @ m.params.values)))
        np.add.at(U, np.searchsorted(muns, g.res6.values), X * (g.k.values - g.n.values * p)[:, None])
    G = len(muns)
    V = G / (G - 1) * B @ (U.T @ U) @ B
    return pd.Series(np.sqrt(np.diag(V)), index=m.params.index)


def main():
    s = load()
    n0 = len(s)
    dist, h = distances(s)
    s['res6'] = s.res6.astype('float64')
    s = s.merge(dist.astype({'res6': 'float64', 'year': 'float32'}), on=['res6', 'year'], how='left')
    miss = s.dist.isna().mean()
    s = s[s.dist.notna()].copy()
    s = s.merge(h[['CNES', 'year', 'has_icu']].astype({'year': 'float32'}), on=['CNES', 'year'], how='left')
    s['noicu'] = ~s.has_icu.astype('boolean').fillna(False).astype(bool)
    s['band'] = pd.cut(s.dist, BANDS, labels=BLAB)
    s['uf'] = (s.res6 // 10000).astype(int).astype(str).map(A.UF)
    s['sexs'] = s.sex.map({1: 'Male', 2: 'Female', 3: 'Female'})
    s['age_grp'] = pd.cut(s.age, A.AGE_BINS, right=False, labels=A.AGE_LAB)
    s = s[s.uf.notna() & s.sexs.notna() & s.age_grp.notna()]
    s['icu'] = s.icu_days > 0
    s['yr'] = s.year.astype(int).astype(str)
    res = dict(stays=int(n0), no_distance_pct=round(100 * miss, 2), analysed=int(len(s)))
    crude = s.groupby('band', observed=True).agg(n=('noicu', 'size'), noicu=('noicu', 'mean'), icu=('icu', 'mean'),
                                                 transfer=('transfer', 'mean')).reset_index()
    res['pct_by_band'] = (100 * crude.n / crude.n.sum()).round(1).tolist()
    for c in ['noicu', 'icu', 'transfer']:
        crude[c] = (100 * crude[c]).round(1)
    print(crude.to_string())
    crude.to_csv(f"{OUT}/t_iv_crude.csv", index=False)
    # ICU care by band within each condition, and among admissions that ended in death
    bc = s.groupby(['cond', 'band'], observed=True).agg(n=('icu', 'size'), icu=('icu', 'mean')).reset_index()
    bd = s[s.death].groupby('band', observed=True).agg(n=('icu', 'size'), icu=('icu', 'mean')).reset_index()
    bd['cond'] = 'Died in hospital (all four)'
    bc = pd.concat([bc, bd], ignore_index=True)
    bc['icu'] = (100 * bc.icu).round(1)
    print(bc.pivot(index='cond', columns='band', values='icu').to_string())
    bc.to_csv(f"{OUT}/t_iv_by_condition.csv", index=False)
    # covariate patterns within each municipality of residence; standard errors clustered by municipality,
    # the level at which distance is assigned
    keys = ['band', 'age_grp', 'sexs', 'cond', 'yr', 'uf']
    rows = []
    for outcome in ['noicu', 'icu', 'transfer']:
        g = s.groupby(keys, observed=True).agg(n=(outcome, 'size'), k=(outcome, 'sum')).reset_index()
        g['fail'] = g.n - g.k
        m = smf.glm("k + fail ~ C(band) + C(age_grp) + C(sexs) + C(cond) + C(yr) + C(uf)", data=g,
                    family=sm.families.Binomial()).fit()
        se = cluster_se(m, s, keys, outcome)
        for t in [x for x in m.params.index if x.startswith('C(band)')]:
            b_, z = m.params[t], 1.959964
            ci = pd.Series([b_ - z * se[t], b_ + z * se[t]])
            rows.append(dict(outcome=outcome, band=t.split('T.')[1].rstrip(']'), OR=round(float(np.exp(m.params[t])), 2),
                             lo=round(float(np.exp(ci[0])), 2), hi=round(float(np.exp(ci[1])), 2)))
    adj = pd.DataFrame(rows)
    print(adj.to_string())
    adj.to_csv(f"{OUT}/t_iv_adjusted.csv", index=False)
    print(res)
    json.dump(res, open(f"{OUT}/iv_feasibility.json", 'w'), indent=1)


if __name__ == '__main__':
    main()
