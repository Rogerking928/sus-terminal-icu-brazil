"""Check the Parquet mirror against the official DATASUS TabNet tabulation (sih/cnv/nibr.def):
for every processing month 2015-2025, national counts of approved AIHs, admissions (AIH type 1)
and deaths from TabNet vs the same counts from the extracted partitions. Writes out/t_s_validation.csv."""
import glob, json, os, re, urllib.parse, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tabnet(yymm, dfn='sih/cnv/nibr.def', pre='nibr'):
    data = [('Linha', 'Ano_processamento'), ('Coluna', '--Não-Ativa--')]
    data += [('Incremento', x) for x in ['Internações', 'AIH_aprovadas', 'Óbitos']]
    data += [('Arquivos', f'{pre}{yymm}.dbf')]
    data += [(k, 'TODAS_AS_CATEGORIAS__') for k in ['SMunicípio', 'SCapital', 'SRegião_de_Saúde_(CIR)']]
    data += [('formato', 'prn'), ('mostre', 'Mostra')]
    body = urllib.parse.urlencode(data, encoding='latin1').encode()
    r = urllib.request.urlopen(urllib.request.Request(
        'http://tabnet.datasus.gov.br/cgi/tabcgi.exe?' + dfn, body), timeout=180).read().decode('latin1')
    m = re.search(r'"Total";(\d+);(\d+);(\d+)', r)
    return tuple(int(x) for x in m.groups())


def mirror(y, m):
    rows = ident5 = deaths = 0
    fs = glob.glob(f"{ROOT}/data/agg/{y}_{m:02d}_*.json")
    for f in fs:
        a = json.load(open(f))
        rows += a['n_rows']; ident5 += a['ident5']
        deaths += sum(v['n'] for k, v in a.items() if isinstance(v, dict) and 'd1' in k)
    return len(fs), rows, rows - ident5, deaths


if __name__ == '__main__':
    os.makedirs(f"{ROOT}/out", exist_ok=True)
    out = []
    for y in range(2015, 2026):
        for m in range(1, 13):
            nf, rows, adm, dth = mirror(y, m)
            if nf == 0: continue
            try:
                t_adm, t_rows, t_dth = tabnet(f"{y % 100:02d}{m:02d}")
            except Exception as e:
                print(y, m, 'tabnet failed', e); continue
            src = 'by place of admission'
            if abs(rows - t_rows) > 0.01 * t_rows:
                # the by-place-of-admission tabulation is incomplete for at least one month (2020-12);
                # the by-place-of-residence tabulation of the same files is the second official check
                t_adm, t_rows, t_dth = tabnet(f"{y % 100:02d}{m:02d}", 'sih/cnv/nrbr.def', 'nrbr')
                src = 'by place of residence'
            out.append(dict(year=y, month=m, states=nf, source=src, tabnet_aih=t_rows, mirror_aih=rows, tabnet_adm=t_adm,
                            mirror_adm=adm, tabnet_deaths=t_dth, mirror_deaths=dth))
            print(out[-1], flush=True)
    d = pd.DataFrame(out)
    for k in ['aih', 'adm', 'deaths']:
        d[f'diff_{k}_pct'] = (100 * (d[f'mirror_{k}'] - d[f'tabnet_{k}']) / d[f'tabnet_{k}']).round(3)
    d.to_csv(f"{ROOT}/out/t_s_validation.csv", index=False)
    print(d.describe().T[['min', 'max']]); print(d.source.value_counts())
