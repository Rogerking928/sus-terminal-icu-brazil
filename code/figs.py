"""Figures at print width (174 mm), so the point sizes written here are the point sizes on the page.
PNG at 700 dpi plus a vector PDF of each. Reads out/*.csv written by analysis.py."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"
W = 174 / 25.4
# house style of the asthma and Enterococcus papers (their figstyle.py): DejaVu Sans, 9 pt text, 8.5 pt ticks and
# legends, ink #1f2933, blue-grey axes, no grid lines
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
                     'axes.edgecolor': '#5b6b7a', 'axes.labelcolor': '#1f2933', 'text.color': '#1f2933',
                     'xtick.color': '#5b6b7a', 'ytick.color': '#5b6b7a', 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
INK, INK2, GRID = '#1f2933', '#5b6b7a', '#e4e3df'
# categorical slots 1-5 of the reference palette, fixed order
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
REG_ORDER = ['North', 'Northeast', 'Centre-West', 'Southeast', 'South']
REG_COL = dict(zip(REG_ORDER, CAT))
AGE_LAB = ['18-44', '45-64', '65-74', '75-84', '85+']
BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=700, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def overlap_check(fig):
    """every pair of text boxes on the figure must not intersect (> 0.6 pt both ways)"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text() and t.get_visible():
                boxes.append((t.get_text(), t.get_window_extent(r)))
    k = 72 / fig.dpi
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            w = (min(a.x1, b.x1) - max(a.x0, b.x0)) * k
            h = (min(a.y1, b.y1) - max(a.y0, b.y0)) * k
            if w > 0.6 and h > 0.6:
                bad.append((boxes[i][0], boxes[j][0]))
    return bad


def tag(ax, letter, dx=None):
    """bold panel letter above the top-left corner of the axes (offset points, as in the house figstyle)"""
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-4, 10), textcoords='offset points',
                fontsize=13, fontweight='bold', ha='right', va='bottom', color=INK)


# ------------------------------------------------------------------ panels
def p_trend(ax):
    t = pd.read_csv(f"{OUT}/t_trend_year_age.csv")
    a = pd.read_csv(f"{OUT}/t_trend_year.csv")
    for c, g in zip(CAT, AGE_LAB):
        s = t[t.age_grp == g]
        ax.plot(s.year, s.pct, color=c, lw=1.4, marker='o', ms=2.8)
    ax.plot(a.year, a.pct, color=INK, lw=1.8, marker='o', ms=3.0)
    ax.axvspan(2019.5, 2021.5, color=GRID, lw=0, zorder=0)
    ends = [(t[(t.age_grp == g) & (t.year == t.year.max())].pct.iloc[0], f"{g} y", c) for c, g in zip(CAT, AGE_LAB)]
    ends.append((a.pct.iloc[-1], 'All adults', INK))
    ends.sort()
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 3.6)
    for (y0, lab, c), y in zip(ends, ys):
        ax.text(a.year.max() + 0.3, y, lab, color=INK, va='center', fontsize=8)
        ax.plot([a.year.max() + 0.05, a.year.max() + 0.27], [y0, y], color=c, lw=0.8)
    ax.set_xlim(a.year.min() - 0.3, a.year.max() + 2.9)
    ax.set_xticks(range(int(a.year.min()), int(a.year.max()) + 1, 3))
    ax.set_ylabel('Decedents with ICU care in the\nterminal admission (%)')
    ax.set_xlabel('Year of death')
    ax.set_axisbelow(True)
    lo = np.floor((min(t.pct.min(), a.pct.min()) - 1) / 5) * 5
    ax.set_ylim(lo, np.ceil(max(t.pct.max(), a.pct.max(), ys[-1]) / 5) * 5 + 3)
    ax.text(2020.5, ax.get_ylim()[1] - 0.4, 'COVID-19\nyears', ha='center', va='top', fontsize=8, color=INK2)


def p_heat(ax, fig):
    t = pd.read_csv(f"{OUT}/t_cause_age.csv")
    order = t.groupby('cause').n.sum().sort_values(ascending=False).index.tolist()
    m = t.pivot(index='cause', columns='age_grp', values='pct').reindex(index=order, columns=AGE_LAB)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list('b', BLUE)
    vmax = np.ceil(np.nanmax(m.values) / 10) * 10
    im = ax.imshow(m.values, cmap=cm, vmin=0, vmax=vmax, aspect='auto')
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v): continue
            # WCAG relative luminance of the cell; pick whichever of white or ink has the higher contrast ratio
            lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in cm(v / vmax)[:3]]
            L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
            c_white, c_ink = 1.05 / (L + 0.05), (L + 0.05) / (0.0036 + 0.05)
            ax.text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=8,
                    color='white' if c_white > c_ink else INK)
    ax.set_xticks(range(len(AGE_LAB))); ax.set_xticklabels([f"{a}" for a in AGE_LAB])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel('Age at death (years)'); ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label('With ICU care (%)'); cb.outline.set_visible(False)


def p_index(ax):
    """adult ICU stays and hospitals billing adult ICU care, each indexed to 2015 = 100 (one axis, common base)"""
    r = pd.read_csv(f"{OUT}/t4_resource.csv")
    y = r.year.astype(int)
    for col, lab, c, ls in (('stays', 'ICU stays', CAT[0], '-'),
                            ('hospitals_icu', 'Hospitals with ICU billing', CAT[1], '--')):
        v = 100 * r[col] / r[col].iloc[0]
        ax.plot(y, v, color=c, lw=1.4, ls=ls, marker='o', ms=2.8, label=lab)
    ax.axhline(100, color=INK2, lw=0.6)
    ax.set_xticks(range(int(y.min()), int(y.max()) + 1, 3))
    ax.set_xlabel('Year'); ax.set_ylabel('Index (2015 = 100)')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, handlelength=2.2)
    ax.set_ylim(90, float((100 * r.stays / r.stays.iloc[0]).max()) + 55)
    ax.axvspan(2019.5, 2021.5, ymax=0.72, color=GRID, lw=0, zorder=0)


def p_map(ax, fig):
    import json
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import analysis as A
    st = pd.read_csv(f"{OUT}/t3_state.csv").set_index('uf')
    g = json.load(open(f"{ROOT}/data/ibge/br_uf.geojson"))
    cm = LinearSegmentedColormap.from_list('b', BLUE)
    norm = Normalize(vmin=10, vmax=50)
    patches, vals = [], []
    for f in g['features']:
        uf = A.UF[f['properties']['codarea']]
        geom = f['geometry']
        polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        for p in polys:
            patches.append(Polygon(np.array(p[0]), closed=True)); vals.append(st.loc[uf, 'std_pct'])
    pc = PatchCollection(patches, cmap=cm, norm=norm, edgecolor='white', linewidth=0.4)
    pc.set_array(np.array(vals)); ax.add_collection(pc)
    ax.autoscale_view(); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
    cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.01, orientation='horizontal')
    cb.set_label('Standardised % with ICU care'); cb.outline.set_visible(False)


def p_states(ax):
    s = pd.read_csv(f"{OUT}/t3_state.csv").sort_values('std_pct')
    y = np.arange(len(s))
    lo = s.std_pct * s.oe_lo / s.oe
    hi = s.std_pct * s.oe_hi / s.oe
    for i, r in enumerate(s.itertuples()):
        c = REG_COL[r.region]
        ax.plot([r.crude_pct, r.std_pct], [i, i], color=INK2, lw=0.5, ls=':', zorder=1)
        ax.plot([lo.iloc[i], hi.iloc[i]], [i, i], color=c, lw=1.2)
        ax.plot(r.std_pct, i, 'o', color=c, ms=4.2, mec='white', mew=0.7, zorder=3)
        ax.plot(r.crude_pct, i, 'o', mfc='white', mec=INK2, ms=3.4, mew=0.7, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(s.uf, fontsize=7.5)
    nat = pd.read_csv(f"{OUT}/t_trend_year.csv"); p = 100 * nat.n_icu.sum() / nat.n.sum()
    ax.axvline(p, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('Decedents with ICU care (%)')
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(s) - 0.3)
    h = [plt.Line2D([], [], marker='o', color=REG_COL[r], ls='', ms=4.2, label=r) for r in REG_ORDER]
    h += [plt.Line2D([], [], marker='o', mfc='white', mec=INK2, ls='', ms=3.4, label='Crude'),
          plt.Line2D([], [], color=INK2, ls='--', lw=0.8, label='National')]
    ax.legend(handles=h, loc='upper left', frameon=False, handletextpad=0.3, borderaxespad=0.2, fontsize=8)


def p_funnel(ax):
    import json
    h = pd.read_csv(f"{OUT}/t_hosp_variation.csv")
    q = json.load(open(f"{OUT}/numbers.json"))['hospital_variation']
    ax.scatter(h.e, h.oe, s=5, color=CAT[0], alpha=0.55, lw=0, zorder=2)
    e = np.logspace(np.log10(h.e.min() * 0.9), np.log10(h.e.max() * 1.1), 200)
    for z, ls, lab in ((1.96, '--', '95% limits'), (3.09, ':', '99.8% limits')):
        ax.plot(e, 1 + z / np.sqrt(e), color=INK2, lw=0.8, ls=ls, label=lab)
        ax.plot(e, np.clip(1 - z / np.sqrt(e), 0, None), color=INK2, lw=0.8, ls=ls)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xscale('log')
    ax.set_xlim(e.min(), e.max()); ax.set_ylim(0, 3.6); ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks([100, 1000, 10000])
    ax.set_xlabel('Expected decedents with ICU care\nper hospital (log scale)')
    ax.set_ylabel('Observed/expected ratio')
    ax.set_axisbelow(True)
    ax.text(0.01, 0.99, f"{q['hospitals']:,} hospitals\nvariance {q['overdispersion']:.0f}x chance, "
            f"{q['pct_var_within_state']:.1f}% within states",
            transform=ax.transAxes, ha='left', va='top', fontsize=8, color=INK)


BANDLAB = ['0', '<25', '25-49', '50-99', '100+']


COND = [('Sepsis', CAT[0], 'o'), ('Myocardial infarction', CAT[1], 's'), ('Stroke', CAT[2], '^'),
        ('Pneumonia', CAT[3], 'D')]


def p_by_condition(ax, col, ylabel, ymax):
    """one line per condition (colour and marker), identified by the shared legend of Figure 3"""
    b = pd.read_csv(f"{OUT}/t_iv_by_condition.csv")
    bands = list(pd.read_csv(f"{OUT}/t_iv_crude.csv").band)
    x = np.arange(len(bands))
    for cond, c, mk in COND:
        y = b[b.cond == cond].set_index('band').reindex(bands)[col].values
        ax.plot(x, y, color=c, lw=1.4, marker=mk, ms=3.6, label=cond)
    ax.set_xticks(x); ax.set_xticklabels(BANDLAB, fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.4, len(bands) - 0.6)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)


def p_access_icu(ax):
    p_by_condition(ax, 'icu', 'Admissions with ICU care (%)', 50)




def p_access_flow(ax):
    c = pd.read_csv(f"{OUT}/t_iv_crude.csv")
    x = np.arange(len(c))
    # neutral ink and grey, so that panel C is not read against the condition colours of panels A and B
    for col, lab, colr, ls, dy, va, mk in (('noicu', 'No ICU billing', INK, '-', 6, 'bottom', 'o'),
                                           ('transfer', 'Transferred', '#8a939c', '--', 5, 'bottom', 'v')):
        ax.plot(x, c[col], color=colr, lw=1.4, ls=ls, marker=mk, ms=3.6)
        if col == 'noicu':
            ax.text(x[2] + 0.6, c[col].iloc[2] + 16, lab, ha='right', va='bottom', fontsize=8, color=INK)
        else:
            ax.text(x[2], c[col].iloc[2] + dy, lab, ha='center', va=va, fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(BANDLAB, fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.4, len(c) - 0.6)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Urgent admissions (%)')
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ figures
# 2026-10-28, Cadernos de Saude Publica: at most 5 illustrations and every panel of a composite figure counts as one,
# so the main figures are single panels (states, hospitals); the rest is supplementary material.
def fig1():
    fig, ax = plt.subplots(figsize=(W * 0.62, 5.4))
    p_states(ax)
    bad = overlap_check(fig); save(fig, 'figure1'); return bad


def fig2():
    fig, ax = plt.subplots(figsize=(W * 0.85, 3.4))
    p_funnel(ax)
    bad = overlap_check(fig); save(fig, 'figure2'); return bad


def figS2():
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.0))
    p_trend(ax)
    bad = overlap_check(fig); save(fig, 'figureS2'); return bad


def figS3():
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.2))
    p_heat(ax, fig)
    bad = overlap_check(fig); save(fig, 'figureS3'); return bad


def figS4():
    fig, ax = plt.subplots(figsize=(W * 0.6, 4.2))
    p_map(ax, fig)
    bad = overlap_check(fig); save(fig, 'figureS4'); return bad


def figS5():
    fig, ax = plt.subplots(figsize=(W * 0.5, 2.6))
    p_index(ax)
    bad = overlap_check(fig); save(fig, 'figureS5'); return bad


def figS6():
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.75, 3.0), gridspec_kw=dict(wspace=0.45))
    p_access_icu(axs[0]); tag(axs[0], 'A')
    p_access_flow(axs[1]); tag(axs[1], 'B')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.3, 1.13), handlelength=2.2,
               columnspacing=1.2, fontsize=8)
    bad = overlap_check(fig); save(fig, 'figureS6'); return bad


def figS7():
    s = pd.read_csv(f"{OUT}/t5_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.2))
    y = np.arange(len(s))[::-1]
    ax.axvline(s.pct_icu.iloc[0], color=INK2, lw=0.8, ls='--')
    ax.plot(s.pct_icu, y, 'o', color=CAT[0], ms=5)
    for yy, v in zip(y, s.pct_icu):
        ax.text(v + 0.4, yy, f"{v:.1f}", va='center', fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(s.analysis)
    ax.set_xlabel('Decedents with ICU care in the terminal admission (%)')
    ax.set_xlim(30, 46)
    ax.set_axisbelow(True)
    bad = overlap_check(fig); save(fig, 'figureS7'); return bad


def figS1():
    """flow of adult decedents into the analysis (RECORD)"""
    import json
    f = json.load(open(f"{OUT}/numbers.json"))['flow']
    fig, ax = plt.subplots(figsize=(W * 0.9, 3.6))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 10); ax.axis('off')
    box = dict(boxstyle='square,pad=0.5', fc='white', ec=INK, lw=0.7)
    main = [(9.0, f"AIH records of adults ending in death,\nprocessing months 2015-2025\nn = {f['death_records']:,}"),
            (6.2, f"Hospital stays (AIHs sharing a number merged)\nn = {f['stays']:,}"),
            (3.4, f"Stays ending in death in 2015-2024\nn = {f['in_years']:,}"),
            (0.6, f"Analysed adult decedents\nn = {f['analysed']:,}")]
    side = [(7.6, f"Duplicate death records of the\nsame stay (resubmissions)\nn = {f['death_records'] - f['stays']:,}"),
            (4.8, f"Discharge outside 2015-2024\nn = {f['stays'] - f['in_years']:,}"),
            (2.0, f"COVID-19 code in any\ndiagnosis field\nn = {f['in_years'] - f['non_covid']:,}")]
    for y, t in main:
        ax.text(3.2, y, t, ha='center', va='center', fontsize=7.5, bbox=box)
    half = [0.95, 0.65, 0.65, 0.65]   # half-heights of the boxes in data units (3- and 2-line boxes)
    for k, ((y0, _), (y1, _)) in enumerate(zip(main[:-1], main[1:])):
        ax.annotate('', xy=(3.2, y1 + half[k + 1]), xytext=(3.2, y0 - half[k]),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.7))
    for y, t in side:
        ax.annotate('', xy=(6.3, y), xytext=(3.2, y), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.7))
        ax.text(6.4, y, t, ha='left', va='center', fontsize=7.5, bbox=box)
    # the tight bounding box counts the text but not its frame: an invisible point keeps the widest frame inside
    ax.plot([11.3], [5], alpha=0)
    save(fig, 'figureS1')
    return []


if __name__ == '__main__':
    for f in (fig1, fig2, figS1, figS2, figS3, figS4, figS5, figS6, figS7):
        print(f.__name__, 'overlaps:', f())
