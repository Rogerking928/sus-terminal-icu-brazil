"""Variation between hospitals (added after the first draft).

A. Hospital variation in ICU use at the end of life: for hospitals with ICU billing and at least 200 adult deaths
   in 2015-2024, the observed/expected ratio of decedents with intensive care (expected from the same patient-level
   model used for states, without geography), turned into a standardised percentage; distribution across hospitals
   and the share of its variance that lies within states.
"""
import glob, json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import analysis as A

ROOT, OUT = A.ROOT, A.OUT


def deaths_cached():
    """analysis.build() on the deaths, keeping only the ICU segments of stays that end in death; cached to disk"""
    f = f"{ROOT}/data/deaths_built.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    t = A.tags()
    d = pd.concat([A._compact(pd.read_parquet(f"{ROOT}/data/raw_deaths/{x}.parquet", columns=A.DCOLS)) for x in t],
                  ignore_index=True)
    ds = np.unique(d.N_AIH.values)
    parts = []
    for x in t:
        i = A._compact(pd.read_parquet(f"{ROOT}/data/raw_icu/{x}.parquet", columns=A.ICOLS))
        k = np.searchsorted(ds, i.N_AIH.values).clip(max=len(ds) - 1)
        parts.append(i[ds[k] == i.N_AIH.values])
    icu = pd.concat(parts, ignore_index=True)
    d, _ = A.build(d, icu)
    keep = ['N_AIH', 'CNES', 'year', 'age_grp', 'sex', 'cause', 'urgent', 'uf', 'region', 'icu']
    d = d[keep].reset_index(drop=True)
    d.to_parquet(f, index=False)
    return d


def part_a():
    d = deaths_cached()
    h = A.hospital_icu((A.Y0, A.Y1))
    d = d.merge(h[['CNES', 'year', 'has_icu']], on=['CNES', 'year'], how='left')
    n_all = len(d)
    d = d[d.has_icu.astype('boolean').fillna(False).astype(bool)].copy()   # deaths in hospital-years with ICU billing
    d['yr'] = d.year.astype(int).astype(str)
    keys = ['age_grp', 'sex', 'cause', 'urgent', 'yr']
    g = d.groupby(keys, observed=True).agg(n=('icu', 'size'), k=('icu', 'sum')).reset_index()
    g['fail'] = g.n - g.k
    g['urgent'] = g.urgent.map({True: 'Urgent', False: 'Elective'})
    m0 = smf.glm("k + fail ~ C(age_grp) + C(sex) + C(cause) + C(urgent) + C(yr)", data=g,
                 family=sm.families.Binomial()).fit()
    g['p'] = m0.predict(g)
    d['urgent_s'] = d.urgent.map({True: 'Urgent', False: 'Elective'})
    d = d.merge(g[keys[:3] + ['yr', 'p']].assign(urgent_s=g.urgent), on=['age_grp', 'sex', 'cause', 'urgent_s', 'yr'],
                how='left')
    hh = d.groupby('CNES').agg(n=('icu', 'size'), k=('icu', 'sum'), e=('p', 'sum'),
                               uf=('uf', lambda s: s.mode().iloc[0])).reset_index()
    hh = hh[hh.n >= 200].copy()
    nat = d.icu.mean()
    hh['oe'] = hh.k / hh.e
    hh['loe'] = np.log(hh.oe.clip(lower=0.01))
    q = hh.oe.quantile([.1, .25, .5, .75, .9])
    w = hh.n / hh.n.sum()
    mu = (w * hh.loe).sum()
    tot = (w * (hh.loe - mu) ** 2).sum()
    hh['state_mean'] = hh.groupby('uf', observed=True).loe.transform(lambda x: np.average(x, weights=hh.loc[x.index, 'n']))
    within = (w * (hh.loe - hh.state_mean) ** 2).sum()
    res = dict(hospitals=int(len(hh)), deaths_in_them=int(hh.n.sum()),
               pct_of_all_deaths=round(100 * hh.n.sum() / n_all, 1), pct_icu_in_icu_hosp_years=round(100 * nat, 1),
               p10=round(float(q[.1]), 2), p25=round(float(q[.25]), 2), p50=round(float(q[.5]), 2),
               p75=round(float(q[.75]), 2), p90=round(float(q[.9]), 2),
               p90_over_p10=round(float(q[.9] / q[.1]), 1),
               pct_var_within_state=round(100 * within / tot, 1))
    # between-hospital heterogeneity beyond chance: overdispersion of (O - E)/sqrt(E), and a moment estimate of the
    # variance of the true log ratio (observed variance minus mean Poisson sampling variance 1/O), turned into a
    # median odds ratio-like quantity, MOR = exp(0.6745 * sqrt(2 * tau2)) (Merlo et al. 2006)
    zz = (hh.k - hh.e) / np.sqrt(hh.e)
    res['overdispersion'] = round(float((zz ** 2).sum() / (len(hh) - 1)), 1)
    pos = hh[hh.k > 0]
    lo_ = np.log(pos.k / pos.e)
    tau2 = max(float(lo_.var() - (1 / pos.k).mean()), 0)
    within = lo_ - lo_.groupby(pos.uf).transform('mean')
    tau2_w = max(float(within.var() - (1 / pos.k).mean()), 0)
    res['mor_hospital'] = round(float(np.exp(0.6745 * np.sqrt(2 * tau2))), 2)
    res['mor_hospital_within_state'] = round(float(np.exp(0.6745 * np.sqrt(2 * tau2_w))), 2)
    # same scale as the MOR: variance of the true log ratio within states over the total (hospitals with k > 0)
    res['pct_var_within_state'] = round(100 * tau2_w / tau2, 1)
    res['hospitals_zero_icu'] = int((hh.k == 0).sum())
    ps = hh.groupby('uf', observed=True).agg(hosp=('oe', 'size'), p10=('oe', lambda x: x.quantile(.1)),
                                             p90=('oe', lambda x: x.quantile(.9))).reset_index()
    ps = ps[ps.hosp >= 10]
    res['states_with_10plus_hosp'] = int(len(ps))
    res['median_within_state_p90_over_p10'] = round(float((ps.p90 / ps.p10).median()), 1)
    ps.round(2).to_csv(f"{OUT}/t_hosp_by_state.csv", index=False)
    hh.round(3).to_csv(f"{OUT}/t_hosp_variation.csv", index=False)
    return res


if __name__ == '__main__':
    import sys
    num = json.load(open(f"{OUT}/numbers.json"))
    if 'a' in sys.argv[1:]:
        num['hospital_variation'] = part_a(); print(num['hospital_variation'])
    json.dump(num, open(f"{OUT}/numbers.json", 'w'), indent=1, default=str)
