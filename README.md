# Intensive care in the terminal admission of adults who die in SUS-funded hospital stays, Brazil 2015-2024

Analysis code for a national cross-sectional study of the hospital admission records of Brazil's Unified Health System
(SIH/SUS): how often adults whose SUS-funded hospital stay ended in death received intensive care during that
terminal admission, how this varied over time, between states and between hospitals, and, in urgent admissions for
sepsis, pneumonia, stroke or acute myocardial infarction, how distance to the nearest hospital with intensive care
billing was related to the care received.

## Running

```
pip install -r requirements.txt
python3 fetch_data.py                    # IBGE price index, state boundaries and municipal seats -> data/ (MD5-checked)
cd code
python3 extract.py 2015 2025             # adult deaths and ICU AIHs from the SIH/SUS reduced files (about 80 min)
python3 validate.py                      # analysed files against the official DATASUS TabNet tabulation (Table S1)
python3 analysis.py                      # decedents, trends, states, resources, sensitivity -> out/
python3 analysis2.py a                   # variation between hospitals (Figure 2, Table S4)
python3 extract2.py                      # urgent admissions for the four conditions
python3 analysis3.py                     # distance to the nearest hospital with ICU billing (Table 3, Table S7)
python3 analysis5.py cluster ipca hosp state   # clustered model, deflated payments, hospitals, state ratios
python3 figs.py                          # Figures 1-2 and S1-S7 -> out/
python3 exhibits.py                      # Tables 1-3 with Figures 1-2, and the supplementary material -> manuscript/
python3 make_source_data.py              # Source_data.xlsx (one sheet per figure and table) -> manuscript/
```

The order matters: later steps read the outputs of earlier ones, and `analysis.py` rewrites `out/numbers.json`.
`analysis.py` and `analysis2.py` load every partition at once and need about 8 GB of memory.
Comments in the code are partly in Chinese.

## Data

- SIH/SUS reduced admission files (RD), DATASUS, Ministry of Health of Brazil, read from healthbr-data
  (https://github.com/SidneyBissoli/healthbr-data), an openly licensed (CC BY 4.0) Parquet redistribution of the
  DATASUS files synchronised weekly with the Ministry of Health; the analysis used the files as of 24 September 2026.
  The mirror is updated as DATASUS revises its files, so later downloads may differ slightly; `validate.py` compares
  them month by month with the official TabNet tabulation. A complete rerun from a clean directory on 28 October 2026
  reproduced every result of the article; the only differences were outside the study period: DATASUS had added late
  records to processing months September-December 2025 (23 more adult death records, the same increase in TabNet and
  in the mirror), which changes the first two counts of Figure S1 and the 2025 totals of Table S1 but no stay of
  2015-2024.
- IBGE: IPCA consumer price index (SIDRA table 1737), state boundaries (malhas API v3) and Localidades do Brasil 2022,
  downloaded by `fetch_data.py`.

No data are redistributed in this repository.

## Software

Python 3.12.3; package versions in `requirements.txt`.

The analysis was written and run by the author in Python 3, with Claude (Anthropic) and ChatGPT (OpenAI) used as
assistants under the author's direction; the author reviewed and tested all code and takes full responsibility for
the results it produces.

## Licence

MIT.
