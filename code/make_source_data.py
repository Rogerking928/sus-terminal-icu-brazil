"""Source data for every figure and table, one sheet each, in one Excel workbook (2026-09-25).

Every sheet is written from the CSV that the figure or table is drawn from (out/*.csv, numbers.json), so the
workbook and the exhibits cannot disagree. Usage: python3 make_source_data.py [output folder]
"""
import json, os, sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"
csv = lambda f: pd.read_csv(f"{OUT}/{f}.csv")
num = json.load(open(f"{OUT}/numbers.json"))

FONT = Font(name='Arial', size=10)
BOLD = Font(name='Arial', size=10, bold=True)
TITLE = Font(name='Arial', size=11, bold=True)
HEAD_FILL = PatternFill('solid', start_color='E8ECF0')


def sheet(wb, name, title, source, blocks):
    """blocks: list of (subtitle or None, DataFrame)"""
    ws = wb.create_sheet(name)
    ws['A1'] = title; ws['A1'].font = TITLE
    ws['A2'] = f"Source: {source}"; ws['A2'].font = Font(name='Arial', size=9, italic=True)
    r = 4
    widths = {}
    for sub, df in blocks:
        if sub:
            ws.cell(r, 1, sub).font = BOLD; r += 1
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(r, j, str(c)); cell.font = BOLD; cell.fill = HEAD_FILL
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            widths[j] = max(widths.get(j, 0), min(len(str(c)), 28))
        r += 1
        for row in df.itertuples(index=False):
            for j, v in enumerate(row, 1):
                if isinstance(v, float) and v.is_integer() and abs(v) > 1900 and abs(v) < 2100:
                    v = int(v)                      # years stored as floats in some CSVs
                cell = ws.cell(r, j, v.item() if hasattr(v, 'item') else v); cell.font = FONT
                widths[j] = max(widths.get(j, 0), min(len(str(v)), 28))
            r += 1
        r += 1
    for j, w in widths.items():
        ws.column_dimensions[get_column_letter(j)].width = max(10, w + 2)
    ws.freeze_panes = 'A4'
    return ws, r


def main(od):
    wb = Workbook(); wb.remove(wb.active)
    contents = []
    add = lambda name, title, source, blocks: (contents.append((name, title)), sheet(wb, name, title, source, blocks))[1]

    # ---------------- numbering of the Cadernos de Saude Publica version (2026-10-28), in order of citation
    ta = csv('t_trend_year_age'); ty = csv('t_trend_year')
    st = csv('t3_state'); hv = csv('t_hosp_variation'); q = num['hospital_variation']
    t2 = pd.read_csv(f"{OUT}/t2_model_or_cluster.csv").rename(columns={'Unnamed: 0': 'term'})
    add('Table 1', 'Table 1. Decedents, ICU care in the terminal admission and adjusted odds ratios, by characteristic '
        'and year', 'out/t1_characteristics.csv, out/t_trend_year.csv (analysis.py); out/t2_model_or_cluster.csv '
        '(analysis5.py): odds ratios, 95% CI clustered by hospital',
        [('Characteristics', csv('t1_characteristics')), ('Year of death', ty), ('Adjusted odds ratios', t2)])
    add('Fig 1', 'Figure 1. Intensive care during the terminal admission by state, crude and standardised',
        'out/t3_state.csv (analysis.py, analysis5.py): n decedents, k with ICU care, exp expected, oe observed/expected '
        '(Byar 95% CI oe_lo-oe_hi), std_pct = oe x national %', [(None, st)])
    add('Fig 2', 'Figure 2. Hospitals: observed and expected decedents with ICU care (funnel plot)',
        'out/t_hosp_variation.csv (analysis2.py part A): CNES facility code, n deaths, k with ICU care, e expected, '
        'oe = k / e', [('Summary', pd.DataFrame([{k: q[k] for k in ['hospitals', 'deaths_in_them', 'p10', 'p50', 'p90',
                                                                     'p90_over_p10', 'pct_var_within_state',
                                                                     'overdispersion', 'mor_hospital',
                                                                     'mor_hospital_within_state']}])),
                      ('Hospitals', hv[['CNES', 'uf', 'n', 'k', 'e', 'oe']])])
    rs = csv('t4_resource')
    add('Table 2', 'Table 2. Deaths in hospitals without ICU billing and adult ICU care used by stays ending in death, '
        'by year', 'out/t_access_year.csv, out/t4_resource.csv (analysis.py, analysis5.py)',
        [('Decedents in hospitals without ICU billing', csv('t_access_year')),
         ('Adult ICU stays, days and payments', rs[['year', 'hospitals_icu', 'stays', 'pct_stays_dying', 'days',
                                                    'pct_days_dying', 'val', 'pct_val_dying']])])
    ad = csv('t_iv_adjusted')
    add('Table 3', 'Table 3. Care received in urgent admissions for four conditions by distance: crude % and adjusted '
        'odds ratios', 'out/t_iv_crude.csv, out/t_iv_adjusted.csv (analysis3.py); OR against 0 km, 95% CI clustered by '
        'municipality', [('Crude percentages', csv('t_iv_crude')), ('Adjusted odds ratios', ad)])
    f = num['flow']
    add('Fig S1', 'Figure S1. Selection of adult decedents', 'out/numbers.json (analysis.py)',
        [(None, pd.DataFrame([('AIH records of adults ending in death, processing months 2015-2025', f['death_records']),
                              ('Hospital stays (AIHs sharing a number merged)', f['stays']),
                              ('Stays ending in death in 2015-2024', f['in_years']),
                              ('Without a COVID-19 code (analysed)', f['analysed'])], columns=['step', 'n']))])
    add('Table S1', 'Table S1. Analysed files against the official DATASUS TabNet tabulation, by processing month',
        'out/t_s_validation.csv (validate.py)', [(None, csv('t_s_validation'))])
    add('Fig S2', 'Figure S2. Share of adult decedents with ICU care in the terminal admission, by year and age',
        'out/t_trend_year_age.csv, out/t_trend_year.csv (analysis.py)',
        [('By age group (pct = n_icu / n x 100)', ta), ('All adults', ty)])
    add('Table S2', 'Table S2. ICU care during the terminal admission by year and age',
        'out/t_trend_year_age.csv (analysis.py)', [(None, ta)])
    add('Fig S3', 'Figure S3. Share with ICU care by principal diagnosis (ICD-10 chapter) and age',
        'out/t_cause_age.csv (analysis.py)', [(None, csv('t_cause_age'))])
    add('Table S3', 'Table S3. ICU care during the terminal admission by state', 'out/t3_state.csv (analysis.py)',
        [(None, st)])
    add('Fig S4', 'Figure S4. Standardised share with ICU care by state (map)', 'out/t3_state.csv: std_pct',
        [(None, st[['uf', 'region', 'std_pct']])])
    add('Table S4', 'Table S4. Spread of ICU care between hospitals within each state',
        'out/t_hosp_by_state.csv (analysis2.py): 10th and 90th percentiles of hospital observed/expected ratios',
        [(None, csv('t_hosp_by_state'))])
    add('Table S5', 'Table S5. Deaths in hospitals without ICU billing, by age', 'out/t_age_access.csv (analysis.py)',
        [(None, csv('t_age_access'))])
    add('Table S6', 'Table S6. Adult ICU stays, days and payments in stays that ended in death, by year',
        'out/t4_resource.csv (analysis.py, analysis5.py): val in BRL nominal, val_real in December 2024 BRL (IPCA)',
        [(None, rs), ('IPCA factors to December 2024', csv('t_ipca_factor'))])
    ri = rs[['year', 'stays', 'hospitals_icu']].copy()
    # stored as values (index = value / 2015 value x 100), so any viewer shows them without recalculation
    ri['stays_index'] = (100 * ri.stays / ri.stays.iloc[0]).round(1)
    ri['hospitals_index'] = (100 * ri.hospitals_icu / ri.hospitals_icu.iloc[0]).round(1)
    add('Fig S5', 'Figure S5. Adult ICU stays billed and hospitals billing adult ICU care, indexed to 2015 = 100',
        'out/t4_resource.csv (analysis.py, analysis5.py); index = value / 2015 value x 100', [(None, ri)])
    bc = csv('t_iv_by_condition')
    add('Fig S6', 'Figure S6. Urgent admissions for four conditions by distance to the nearest hospital with ICU billing',
        'out/t_iv_by_condition.csv, out/t_iv_crude.csv (analysis3.py); percentages',
        [('A: ICU care (icu, %) by condition', bc),
         ('B: admission to a hospital without ICU billing and transfer, all four conditions', csv('t_iv_crude'))])
    add('Table S7', 'Table S7. ICU care during urgent admissions by condition and distance',
        'out/t_iv_by_condition.csv (analysis3.py)', [(None, bc)])
    add('Table S8', 'Table S8. Sensitivity analyses', 'out/t5_sensitivity.csv (analysis.py)',
        [(None, csv('t5_sensitivity'))])
    add('Fig S7', 'Figure S7. Sensitivity analyses', 'out/t5_sensitivity.csv (analysis.py)',
        [(None, csv('t5_sensitivity'))])

    # ---------------- contents
    ws = wb.create_sheet('Contents', 0)
    ws['A1'] = 'Source data for all figures and tables'; ws['A1'].font = TITLE
    ws['A2'] = ('Intensive care in the terminal admission of adults who die in hospital stays funded by Brazil\'s '
                'Unified Health System, 2015-2024. SIH/SUS (DATASUS, via healthbr-data) and IBGE. '
                f'Generated {pd.Timestamp.today():%Y-%m-%d} by code/make_source_data.py.')
    ws['A2'].font = FONT
    ws['A4'] = 'Sheet'; ws['B4'] = 'Contents'
    for c in ('A4', 'B4'):
        ws[c].font = BOLD; ws[c].fill = HEAD_FILL
    for i, (name, title) in enumerate(contents, 5):
        ws.cell(i, 1, name).font = FONT; ws.cell(i, 2, title).font = FONT
        ws.cell(i, 1).hyperlink = f"#'{name}'!A1"
    ws.column_dimensions['A'].width = 12; ws.column_dimensions['B'].width = 110
    path = f"{od}/Source_data.xlsx"
    wb.save(path)
    print(path, len(contents), 'sheets')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/manuscript")
