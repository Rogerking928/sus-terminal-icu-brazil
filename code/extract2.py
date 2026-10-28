"""Second pass for the outcome analysis: every urgent admission of an adult with a principal diagnosis in which
intensive care is most often needed (sepsis A40-A41, stroke I60-I64, acute myocardial infarction I21-I22,
pneumonia J12-J18), competence years 2015-2025. One small Parquet per partition in data/cohort/.
COBRANCA (reason for discharge): 31 = transferred to another hospital; 41-43 = death."""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from extract import S3, listing, COVID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = f"{ROOT}/data/cohort2"   # v2 (2026-09-25): adds municipality of residence and of the hospital
COLS = ['N_AIH', 'IDENT', 'CNES', 'MUNIC_MOV', 'SEXO', 'COD_IDADE', 'IDADE', 'DT_SAIDA', 'DIAS_PERM', 'MORTE',
        'COBRANCA', 'CAR_INT', 'MUNIC_RES', 'DIAG_PRINC', 'DIAG_SECUN', 'CID_MORTE', 'CID_ASSO', 'UTI_MES_TO'] + \
       [f'DIAGSEC{i}' for i in range(1, 10)]
CONDS = {'Sepsis': ('A40', 'A41'), 'Stroke': ('I60', 'I61', 'I62', 'I63', 'I64'), 'Myocardial infarction': ('I21', 'I22'),
         'Pneumonia': tuple(f'J{i}' for i in range(12, 19))}
C3 = {c: k for k, v in CONDS.items() for c in v}


def _one(path):
    ano, mes, uf = [p.split('=')[1] for p in path.split('/')[-4:-1]]
    out = f"{OUTD}/{ano}_{mes}_{uf}.parquet"
    if os.path.exists(out): return 'skip'
    names = pq.read_schema(path, filesystem=S3).names
    d = pq.read_table(path, filesystem=S3, columns=[c for c in COLS if c in names]).to_pandas()
    for c in COLS:
        if c not in d: d[c] = None
    s = lambda c: d[c].astype(str).str.strip()
    age = pd.to_numeric(d.IDADE, errors='coerce')
    cond = s('DIAG_PRINC').str[:3].map(C3)
    keep = (s('COD_IDADE') == '4') & (age >= 18) & (s('CAR_INT') != '01') & cond.notna()
    d, age, cond = d[keep], age[keep], cond[keep]
    cov = np.zeros(len(d), bool)
    for c in ['DIAG_PRINC', 'DIAG_SECUN', 'CID_MORTE', 'CID_ASSO'] + [f'DIAGSEC{i}' for i in range(1, 10)]:
        cov |= d[c].astype(str).str.strip().str.upper().str[:4].isin(COVID).values
    y = pd.DataFrame({
        'N_AIH': pd.to_numeric(d.N_AIH, errors='coerce').astype('int64'),
        'IDENT': pd.to_numeric(d.IDENT, errors='coerce').astype('float32'),
        'CNES': pd.to_numeric(d.CNES, errors='coerce').astype('float64'),
        'uf2': d.MUNIC_MOV.astype(str).str[:2],
        'mov6': pd.to_numeric(d.MUNIC_MOV, errors='coerce').astype('float32'),
        'res6': pd.to_numeric(d.MUNIC_RES, errors='coerce').astype('float32'),
        'sex': pd.to_numeric(d.SEXO, errors='coerce').astype('float32'),
        'age': age.astype('float32'),
        'year': pd.to_numeric(d.DT_SAIDA.astype(str).str[:4], errors='coerce').astype('float32'),
        'los': pd.to_numeric(d.DIAS_PERM, errors='coerce').astype('float32'),
        'death': d.MORTE.astype(str).str.strip() == '1',
        'transfer': d.COBRANCA.astype(str).str.strip() == '31',
        'icu_days': pd.to_numeric(d.UTI_MES_TO, errors='coerce').fillna(0).astype('int32'),
        'cond': cond.values, 'covid': cov})
    y.to_parquet(out, index=False)
    return len(y)


def one(path, tries=6):
    for k in range(tries):
        try:
            return path, _one(path)
        except Exception as e:
            if k == tries - 1: return path, f'FAILED {e}'
            time.sleep(10 * (k + 1))


if __name__ == '__main__':
    os.makedirs(OUTD, exist_ok=True)
    paths = listing(range(2015, 2026))
    print(len(paths), 'files', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        for i, (p, r) in enumerate(ex.map(one, paths)):
            if i % 100 == 0 or str(r).startswith('FAILED'): print(i, p[-30:], r, f"{time.time()-t0:.0f}s", flush=True)
    print('done', f"{time.time()-t0:.0f}s")
