"""Download the IBGE files used by the analysis into data/ibge/, check each MD5 against the copy analysed
(downloaded 25 September 2026), and build data/municipios_ibge2022.csv (seats of the 5,571 municipalities, including
the State District of Fernando de Noronha) from IBGE Localidades 2022.

The SIH/SUS admission files are not downloaded here: extract.py and extract2.py read them directly from the
healthbr-data mirror of DATASUS. A mismatch is reported, not overwritten silently: IBGE occasionally revises files.

  python3 fetch_data.py
"""
import hashlib
import io
import os
import sqlite3
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = [
    ("data/ibge/ipca_1737.json", "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/201501-202412", "88e0a88d39b61a237dc16258a16bf4bc"),
    ("data/ibge/br_uf.geojson", "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo%2Bjson&intrarregiao=UF&qualidade=minima", "538640c1b8916e14b7018b83dc6bfa31"),
    ("data/ibge/Localidades_Brasil_gpkg.zip", "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/localidades/Localidades_do_Brasil/2022/Localidades_Brasil_gpkg.zip", "86b99d20a3e9f58027b00a1af662a04f"),
]
MUN_MD5 = "d4e6d6bf793c8daefee71138580bc304"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def municipalities():
    """seat of each municipality: localities of category 'Cidade' plus Fernando de Noronha, one per municipality code"""
    zf = zipfile.ZipFile(os.path.join(ROOT, "data/ibge/Localidades_Brasil_gpkg.zip"))
    name = next(n for n in zf.namelist() if n.endswith(".gpkg"))
    gpkg = os.path.join(ROOT, "data/ibge", os.path.basename(name))
    open(gpkg, "wb").write(zf.read(name))
    con = sqlite3.connect(gpkg)
    d = pd.read_sql("select CD_MUN, NM_MUN, LAT_LOCALIDADE, LONG_LOCALIDADE from BR_localidades_2022 where CT_LOCALIDADE "
                    "in ('Cidade', 'Distrito Estadual de Fernando de Noronha') order by id", con)
    d = d.drop_duplicates("CD_MUN").sort_values("CD_MUN")
    out = pd.DataFrame({"codigo_ibge": d.CD_MUN.astype(int), "nome": d.NM_MUN,
                        "latitude": d.LAT_LOCALIDADE.astype(np.float32).astype(float),
                        "longitude": d.LONG_LOCALIDADE.astype(np.float32).astype(float)})
    path = os.path.join(ROOT, "data/municipios_ibge2022.csv")
    out.to_csv(path, index=False)
    return path


def main():
    os.makedirs(os.path.join(ROOT, "data/ibge"), exist_ok=True)
    bad = []
    for rel, url, want in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
                f.write(r.read())
        ok = md5(path) == want
        print(("ok      " if ok else "CHANGED ") + rel)
        if not ok:
            bad.append(rel)
    path = municipalities()
    ok = md5(path) == MUN_MD5
    print(("ok      " if ok else "CHANGED ") + "data/municipios_ibge2022.csv")
    if not ok:
        bad.append("data/municipios_ibge2022.csv")
    if bad:
        sys.exit(f"{len(bad)} file(s) differ from the versions analysed: {', '.join(bad)}")


if __name__ == "__main__":
    main()
