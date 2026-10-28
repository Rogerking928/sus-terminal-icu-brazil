"""Additions after internal review (2026-09-25).

cluster  - the Table 2 model refitted on covariate patterns within each hospital, with standard errors clustered by
           hospital (decedents are nested in hospitals); point estimates are unchanged.
ipca     - ICU payments deflated to December 2024 reais with the IPCA (IBGE SIDRA table 1737, monthly index number),
           using the mean index of each year of discharge.
hosp     - hospitals billing adult ICU days to SUS, per year.
"""
import json, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import analysis as A

ROOT, OUT = A.ROOT, A.OUT
REF = ("C(age_grp, Treatment('65-74')) + C(sex, Treatment('Male')) + C(cause, Treatment('Circulatory'))"
       " + C(urgent, Treatment('Elective')) + C(yr, Treatment('2015')) + C(region, Treatment('Southeast'))")


def cluster():
    d = pd.read_parquet(f"{ROOT}/data/deaths_built.parquet")
    d['yr'] = d.year.astype(int).astype(str)
    d['urgent'] = d.urgent.map({True: 'Urgent', False: 'Elective'})
    d['CNES'] = d.CNES.fillna(-1)
    keys = ['CNES', 'age_grp', 'sex', 'cause', 'urgent', 'region', 'yr']
    g = d.groupby(keys, observed=True).agg(n=('icu', 'size'), k=('icu', 'sum')).reset_index()
    del d
    g['fail'] = g.n - g.k
    print('cells', len(g), 'hospitals', g.CNES.nunique(), flush=True)
    m = smf.glm(f"k + fail ~ {REF}", data=g, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(g.CNES)[0]})
    ci = m.conf_int()
    ors = pd.DataFrame({'OR': np.exp(m.params), 'lo': np.exp(ci[0]), 'hi': np.exp(ci[1]), 'p': m.pvalues}).round(4)
    ors.to_csv(f"{OUT}/t2_model_or_cluster.csv")
    old = pd.read_csv(f"{OUT}/t2_model_or.csv", index_col=0)
    print('max |OR change| vs unclustered fit:', float((ors.OR - old.OR.reindex(ors.index)).abs().max()))
    return dict(cells=int(len(g)), hospitals=int(g.CNES.nunique()))


def ipca():
    j = json.load(open(f"{ROOT}/data/ibge/ipca_1737.json"))
    x = pd.DataFrame(j[1:]).rename(columns=j[0])
    x['ym'] = x['Mês (Código)'].astype(int)
    x['v'] = pd.to_numeric(x['Valor'], errors='coerce')
    x['year'] = x.ym // 100
    dec24 = float(x.loc[x.ym == 202412, 'v'].iloc[0])
    f = (dec24 / x[x.year.between(A.Y0, A.Y1)].groupby('year').v.mean()).round(4)
    f.rename('factor_to_dec2024').to_csv(f"{OUT}/t_ipca_factor.csv")
    rs = pd.read_csv(f"{OUT}/t4_resource.csv")
    rs['val_real'] = rs.val * rs.year.astype(int).map(f)
    rs['val_dying_real'] = rs.val_dying * rs.year.astype(int).map(f)
    rs.to_csv(f"{OUT}/t4_resource.csv", index=False)
    return dict(val_real_total=float(rs.val_real.sum()), val_dying_real_total=float(rs.val_dying_real.sum()),
                pct_val_dying_real=round(100 * rs.val_dying_real.sum() / rs.val_real.sum(), 1))


def state():
    """recompute observed/expected ratios, standardised percentages and Byar limits from unrounded values"""
    st = pd.read_csv(f"{OUT}/t3_state.csv")
    ty = pd.read_csv(f"{OUT}/t_trend_year.csv")
    nat = ty.n_icu.sum() / ty.n.sum()
    o, e = st.k, st.exp
    st['oe'] = o / e
    st['oe_lo'] = o * (1 - 1 / (9 * o) - 1.96 / (3 * np.sqrt(o))) ** 3 / e
    o1 = o + 1
    st['oe_hi'] = o1 * (1 - 1 / (9 * o1) + 1.96 / (3 * np.sqrt(o1))) ** 3 / e
    st['std_pct'] = 100 * st.oe * nat
    st = st.sort_values('std_pct')          # full precision; rounding happens once, when printed
    st.to_csv(f"{OUT}/t3_state.csv", index=False)
    num = json.load(open(f"{OUT}/numbers.json"))
    num['state_std_range'] = [round(float(st.std_pct.min()), 1), round(float(st.std_pct.max()), 1)]
    json.dump(num, open(f"{OUT}/numbers.json", 'w'), indent=1, default=str)
    return dict(nat=round(100 * nat, 3), range=num['state_std_range'])


def hosp():
    h = A.hospital_icu((A.Y0, A.Y1))
    c = h[h.has_icu].groupby('year').CNES.nunique().rename('hospitals_icu')
    rs = pd.read_csv(f"{OUT}/t4_resource.csv")
    rs['hospitals_icu'] = rs.year.astype(int).map(c)
    rs.to_csv(f"{OUT}/t4_resource.csv", index=False)
    return c.to_dict()


if __name__ == '__main__':
    for part in sys.argv[1:]:
        r = {'cluster': cluster, 'ipca': ipca, 'hosp': hosp, 'state': state}[part]()
        print(part, r, flush=True)
        num = json.load(open(f"{OUT}/numbers.json"))   # re-read: other parts may run concurrently
        num[f'add_{part}'] = r
        json.dump(num, open(f"{OUT}/numbers.json", 'w'), indent=1, default=str)
