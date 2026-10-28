"""Extract adult in-hospital deaths, ICU-billed AIH segments and per-file aggregates
from SIH/SUS RD microdata (healthbr-data Parquet mirror of DATASUS), competence years 2015-2025.
Resumable: one output per (year, month, state) partition."""
import os, sys, json, time
import pyarrow.fs as fs, pyarrow.parquet as pq, pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# public read-only credentials of the healthbr-data mirror, as published by its maintainer in
# the healthbR R package (R/utils-r2.R); see https://github.com/SidneyBissoli/healthbr-data
S3 = fs.S3FileSystem(endpoint_override="https://5c499208eebced4e34bd98ffa204f2fb.r2.cloudflarestorage.com",
    access_key="28c72d4b3e1140fa468e367ae472b522",
    secret_key="2937b2106736e2ba64e24e92f2be4e6c312bba3355586e41ce634b14c1482951", region="auto")
COLS = ['N_AIH','IDENT','ANO_CMPT','MES_CMPT','CNES','MUNIC_MOV','MUNIC_RES','NASC','SEXO','COD_IDADE','IDADE',
        'DT_INTER','DT_SAIDA','DIAS_PERM','MORTE','CAR_INT','DIAG_PRINC','DIAG_SECUN','CID_MORTE','CID_ASSO',
        'UTI_MES_TO','MARCA_UTI','VAL_UTI','VAL_TOT','NAT_JUR','COMPLEX','ESPEC'] + [f'DIAGSEC{i}' for i in range(1,10)]
COVID = ('U071','U072','B342')

def one(path, tries=6):
    for k in range(tries):
        try:
            return _one(path)
        except Exception as e:
            if k == tries - 1: return path, f'FAILED {e}'
            time.sleep(10 * (k + 1))

def _one(path):
    ano, mes, uf = [p.split('=')[1] for p in path.split('/')[-4:-1]]
    tag = f"{ano}_{mes}_{uf}"
    outd = f"{ROOT}/data/raw_deaths/{tag}.parquet"
    if os.path.exists(f"{ROOT}/data/agg/{tag}.json"): return tag, 'skip'
    sch = pq.read_schema(path, filesystem=S3)
    cols = [c for c in COLS if c in sch.names]
    d = pq.read_table(path, filesystem=S3, columns=cols).to_pandas()
    for c in COLS:
        if c not in d: d[c] = None
    s = lambda c: d[c].astype(str).str.strip()
    age = pd.to_numeric(d.IDADE, errors='coerce')
    d['adult'] = (s('COD_IDADE') == '4') & (age >= 18)
    diagcols = ['DIAG_PRINC','DIAG_SECUN','CID_MORTE','CID_ASSO'] + [f'DIAGSEC{i}' for i in range(1,10)]
    cov = np.zeros(len(d), bool)
    for c in diagcols:
        cov |= s(c).str.upper().str[:4].isin(COVID).values
    d['covid'] = cov
    d['death'] = s('MORTE') == '1'
    d['icu_days'] = pd.to_numeric(d.UTI_MES_TO, errors='coerce').fillna(0).astype('int32')
    d['val_uti'] = pd.to_numeric(d.VAL_UTI, errors='coerce').fillna(0.0)
    d['val_tot'] = pd.to_numeric(d.VAL_TOT, errors='coerce').fillna(0.0)
    d['file_uf'] = uf
    keep = ['N_AIH','IDENT','ANO_CMPT','MES_CMPT','CNES','MUNIC_MOV','MUNIC_RES','SEXO','IDADE','DT_INTER','DT_SAIDA',
            'DIAS_PERM','CAR_INT','DIAG_PRINC','CID_MORTE','MARCA_UTI','NAT_JUR','COMPLEX','ESPEC',
            'icu_days','val_uti','val_tot','adult','covid','death','file_uf']
    d[d.death & d.adult][keep].astype({k:str for k in keep if d[k].dtype==object}).to_parquet(outd, index=False)
    d[d.icu_days > 0][['N_AIH','IDENT','CNES','DT_SAIDA','icu_days','MARCA_UTI','val_uti','adult','covid','death','file_uf']
        ].astype({'N_AIH':str,'IDENT':str,'CNES':str,'DT_SAIDA':str,'MARCA_UTI':str}).to_parquet(f"{ROOT}/data/raw_icu/{tag}.parquet", index=False)
    # hospital-level aggregates (all ages, adults) for this file
    d['icu'] = d.icu_days > 0
    d['CNES'] = s('CNES')
    g = d[d.adult].groupby(['CNES']).agg(n=('icu','size'), n_icu=('icu','sum'), icu_days=('icu_days','sum'),
                                        deaths=('death','sum'), covid=('covid','sum')).reset_index()
    g['tag'] = tag
    g.to_parquet(f"{ROOT}/data/agg/{tag}_hosp.parquet", index=False)
    agg = {}
    for adult in (True, False):
        for death in (True, False):
            for covid in (True, False):
                m = (d.adult == adult) & (d.death == death) & (d.covid == covid)
                agg[f"a{int(adult)}d{int(death)}c{int(covid)}"] = dict(
                    n=int(m.sum()), n_icu=int((m & d.icu).sum()), icu_days=int(d.icu_days[m].sum()),
                    val_uti=float(d.val_uti[m].sum()), val_tot=float(d.val_tot[m].sum()))
    agg['n_rows'] = len(d); agg['ident5'] = int((s('IDENT')=='5').sum())
    json.dump(agg, open(f"{ROOT}/data/agg/{tag}.json", 'w'))
    return tag, len(d)

def listing(years):
    out = []
    for y in years:
        fi = S3.get_file_info(fs.FileSelector(f"healthbr-data/sih/rd/ano={y}", recursive=True))
        out += [i.path for i in fi if i.type == fs.FileType.File and i.path.endswith('.parquet')]
    return sorted(out)

if __name__ == '__main__':
    for sub in ('raw_deaths', 'raw_icu', 'agg'):
        os.makedirs(f"{ROOT}/data/{sub}", exist_ok=True)
    years = range(int(sys.argv[1]), int(sys.argv[2]) + 1)
    paths = listing(years)
    print(len(paths), 'files', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for i, (tag, r) in enumerate(ex.map(one, paths)):
            if i % 25 == 0 or str(r).startswith('FAILED'): print(i, tag, r, f"{time.time()-t0:.0f}s", flush=True)
    print('done', f"{time.time()-t0:.0f}s")
