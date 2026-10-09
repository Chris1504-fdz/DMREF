"""Turn the ICDD PDF card XML (00-041-1107, PdO) into human-readable files and
compare it with a measured film scan from the 2026-08-27 delivery.

Outputs (output/pdo/experiment02_xrd_crystallinity/):
    pdf_card_00-041-1107_PdO.xlsx        card_info / peak_list / atoms / symmetry_ops / what_is_this
    pdf_card_00-041-1107_PdO.md          same content as plain text
    pdf_card_00-041-1107_icdd_chart.png  the stick chart embedded in the XML, decoded
    pdf_card_vs_measured_film_0827.png   card sticks next to the measured film 03_14
"""
import base64, os, sys, textwrap
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils import xrd_crystallinity as xc

CARD = f"{REPO}/raw_data_received/pdo/08_27_26/xrd_reference_09_10_26/PdO_PDF Card - 00-041-1107.xml"
XLSX = f"{REPO}/raw_data_received/pdo/08_27_26/PdO_data_0827.xlsx"
OUT = f"{REPO}/output/pdo/experiment02_xrd_crystallinity"
STEM = "pdf_card_00-041-1107_PdO"

root = ET.parse(CARD).getroot()
pdf = root.find("pdf_data")

# ---------------------------------------------------------------- card_info
NICE = {  # ICDD field name -> plain-English label
    "pdf_number": "PDF card number", "status": "Card status", "quality_mark": "Quality mark (Star = best)",
    "pressure_temperature": "Conditions", "temperature": "Temperature", "chemical_formula": "Chemical formula",
    "empirical_formula": "Empirical formula", "weight_percent": "Weight %", "atomic_percent": "Atomic %",
    "chemical_name": "Chemical name", "mineral_name": "Mineral name", "ima_symbol": "IMA mineral symbol",
    "entry_date": "Entry date", "patrad": "Radiation used for the reference pattern",
    "filttyp": "Monochromator / filter", "internal_standard": "Internal standard used", "instrument": "Instrument",
    "lower": "Lowest d-spacing scanned", "source": "Data source", "xstal_system": "Crystal system",
    "spgr": "Space group", "cell_a": "Cell a (Angstrom)", "cell_c": "Cell c (Angstrom)",
    "cell_vol": "Cell volume (Angstrom^3)", "z": "Formula units per cell (Z)", "cell_mv": "Molar volume per Z",
    "authcovera": "c/a ratio (author)", "xtldx": "Calculated density (g/cm^3)", "color": "Colour",
    "ss": "Smith-Snyder figure of merit", "xtlmolwt": "Molecular weight", "pearson": "Pearson symbol",
    "rep_struc": "Structure prototype", "cross_ref_pdf_numbers": "Alternate PDF cards for the same phase",
    "database_comments": "Database comments", "crystal_properties": "Crystal properties",
    "subfile_indicator": "ICDD subfiles", "strunz_class": "Strunz class",
}
SKIP = {"sg_sym_ops", "atomic_coords", "references", "graphs"}
info_rows = []
for el in pdf:
    if el.tag in SKIP:
        continue
    if el.tag.startswith(("xtl", "redcell", "lpf_", "rep_struc_alpha")) and el.tag not in NICE:
        continue  # duplicated / derived crystallographic bookkeeping
    txt = (el.text or "").strip()
    if txt:
        info_rows.append(dict(field=el.tag, meaning=NICE.get(el.tag, el.tag), value=txt))
for rg in pdf.iter("reference_group"):
    info_rows.append(dict(field="reference", meaning=rg.findtext("type"), value=rg.findtext("reference").strip()))
info = pd.DataFrame(info_rows)

# ---------------------------------------------------------------- peak_list
rows = []
for it in root.iter("intensity"):
    if it.find("theta") is None:
        continue
    I_txt = it.findtext("intensity").strip()
    I_num = 0.5 if I_txt.startswith("<") else float(I_txt)
    h, k, l = (it.findtext(t) for t in "hkl")
    rows.append(dict(two_theta_deg=float(it.findtext("theta")), d_spacing_A=float(it.findtext("da")),
                     relative_intensity=I_txt, relative_intensity_num=I_num, h=int(h), k=int(k), l=int(l),
                     hkl=f"({h}{k}{l})"))
peaks = pd.DataFrame(rows).sort_values("two_theta_deg").reset_index(drop=True)

def where(tt):
    cfg = xc.DEFAULT_CFG
    if cfg["window"][0] <= tt <= cfg["window"][1]:
        return "inside the 33-35 deg objective window (merged into the analysed peak)"
    if cfg["fit_window"][0] <= tt <= cfg["fit_window"][1]:
        return "inside the 31.5-36.5 deg fit window"
    if 41.3 <= tt <= 42.8:
        return "hidden under the sapphire (0006) substrate line at 41.7 deg"
    if tt > 80.6:
        return "outside the measured 2theta range (20.6-80.6 deg)"
    return ""
peaks["position_in_our_scans"] = peaks.two_theta_deg.map(where)

atoms = pd.DataFrame([{c.tag: (c.text or "").strip() for c in a} for a in root.iter("atomic_coord")])
symops = pd.DataFrame([{c.tag: (c.text or "").strip() for c in s} for s in root.iter("sg_sym_op")])

# ---------------------------------------------------------------- explanation
explain = textwrap.dedent(f"""
    WHAT THIS FILE IS
    -----------------
    An entry ("card") from the ICDD Powder Diffraction File, exported by the PDF-5+ 2026 software.
    It is the reference fingerprint of crystalline palladium oxide (PdO, mineral palladinite):
    the list of Bragg reflections that a randomly oriented PdO POWDER produces with Cu K-alpha X-rays.
    Each reflection is given as a 2-theta position, a d-spacing, Miller indices (hkl) and a
    RELATIVE intensity, scaled so that the strongest line, (101), equals 100.

    It is a DATABASE RECORD, not a measurement file:
      - there is no intensity-versus-2theta trace, no counts, no background, no peak widths;
      - the intensities are ratios between PdO's own lines (dimensionless), valid for a bulk
        powder with random grain orientation;
      - the source measurement was a powder mixed with a Si internal standard on a diffractometer
        in 1989 (Welton-Holzer & McCarthy, ICDD Grant-in-Aid), i.e. not our instrument.

    Its purpose in XRD practice is PHASE IDENTIFICATION: you compare the peak positions of an
    unknown pattern with the card to say "these peaks are PdO, this one is the (101) line".
    Software such as Jade or HighScore uses exactly these cards for that.

    WHY IT CANNOT BE THE 100 % REFERENCE OF OUR CRYSTALLINITY INDEX
    ----------------------------------------------------------------
    Our index is  crystallinity_pct = 100 x (measured PdO(101) area of a film) / (reference area),
    where both areas are in counts x degree measured on the same instrument with the same settings.
    The reference must therefore be the area that a FILM of comparable thickness gives when it is
    fully crystalline. The card offers no such number:
      1. No absolute scale. The card's "100" is a normalisation, not counts. 334 counts.deg / 100
         is not a percentage of anything.
      2. No thickness. A Bragg peak area is proportional to the diffracting volume; the card describes
         an effectively infinitely thick powder, our films are 3-16 nm. Even a measured powder scan
         would give an area thousands of times larger than any film, set by volume, not crystallinity.
      3. No instrument. Tube power, optics, step size, counting time and geometry all change the
         absolute intensity; the card's data were taken on a different diffractometer in 1989.
      4. No texture. The card assumes randomly oriented grains; a thin film is usually textured, which
         changes the ratio between (101), (002), (110) ... in an unknown way.
    Turning a reference pattern into a crystalline fraction is done in powder work with the
    Reference Intensity Ratio method, which needs a known amount of an internal standard mixed into
    the sample. That is impossible for a film on a substrate.

    WHAT IT IS USEFUL FOR
    ---------------------
      - Confirming the peak assignment: card (101) at {peaks.loc[peaks.hkl=='(101)','two_theta_deg'].iloc[0]:.3f} deg; the detected films fit at
        33.61-33.86 deg, so the 33-35 deg window is correct.
      - Showing that the analysed peak is really (101)+(002): the card's (002) line at
        {peaks.loc[peaks.hkl=='(002)','two_theta_deg'].iloc[0]:.3f} deg (relative intensity 22) is only 0.29 deg below (101) and merges with it
        at our ~1 deg FWHM. The intensity-weighted centroid of the two card lines is 33.78 deg;
        the mean fitted centre of the films is 33.79 deg.
      - Explaining why no other film line is usable: the next strongest lines, (112) at 54.7 deg (20)
        and (110) at 41.9 deg (14), are at the noise floor or under the sapphire (0006) line.
      - Citing the phase in reports and figure labels (PDF 00-041-1107).
""").strip()

# ---------------------------------------------------------------- write xlsx + md
os.makedirs(OUT, exist_ok=True)
with pd.ExcelWriter(f"{OUT}/{STEM}.xlsx") as w:
    pd.DataFrame({"what_is_this": explain.splitlines()}).to_excel(w, sheet_name="what_is_this", index=False)
    info.to_excel(w, sheet_name="card_info", index=False)
    peaks.to_excel(w, sheet_name="peak_list", index=False)
    atoms.to_excel(w, sheet_name="atoms", index=False)
    symops.to_excel(w, sheet_name="symmetry_ops", index=False)
    widths = {"what_is_this": [110], "card_info": [22, 42, 60], "peak_list": [14, 14, 18, 22, 6, 6, 6, 8, 70]}
    for sh, ws in w.sheets.items():
        for j, wdt in enumerate(widths.get(sh, [])):
            if hasattr(ws, "set_column"):                       # xlsxwriter
                ws.set_column(j, j, wdt)
            else:                                               # openpyxl
                ws.column_dimensions[chr(65 + j)].width = wdt

with open(f"{OUT}/{STEM}.md", "w") as f:
    f.write(f"# ICDD PDF card 00-041-1107 - PdO (palladinite), human-readable export\n\n")
    f.write(f"Source: `raw_data_received/pdo/08_27_26/xrd_reference_09_10_26/PdO_PDF Card - 00-041-1107.xml`\n\n")
    f.write("```\n" + explain + "\n```\n\n")
    f.write("## Card information\n\n" + info[["meaning", "value"]].to_markdown(index=False) + "\n\n")
    f.write("## Peak list (Cu K-alpha, relative intensity: strongest line = 100)\n\n")
    f.write(peaks[["two_theta_deg", "d_spacing_A", "relative_intensity", "hkl", "position_in_our_scans"]]
            .to_markdown(index=False) + "\n\n")
    f.write("## Atoms in the unit cell\n\n" + atoms.to_markdown(index=False) + "\n")

img = root.find("graphs/chart_image")
with open(f"{OUT}/pdf_card_00-041-1107_icdd_chart.png", "wb") as f:
    f.write(base64.b64decode(img.text.strip()))

# ---------------------------------------------------------------- figure: card vs measured film
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
xc._style(); P = xc.PALETTE
BLUE, ORANGE, MUTED = P["series"][0], P["series"][1], P["muted"]

sob = xc.read_xrd_sheet(XLSX, "sobol")
FILM = "03_14"
tt, ii = sob[FILM]
cfg = xc.DEFAULT_CFG
bm = xc.baseline_metrics(tt, ii, cfg); fp = xc.fit_peak(tt, ii, cfg)
strong = peaks[peaks.relative_intensity_num >= 3]

fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.15], width_ratios=[1.55, 1], hspace=0.38, wspace=0.18)
ax_meas = fig.add_subplot(gs[0, 0]); ax_card = fig.add_subplot(gs[1, 0], sharex=ax_meas); ax_zoom = fig.add_subplot(gs[:, 1])

m = (tt >= 20.5) & (tt <= 80.6)
ax_meas.semilogy(tt[m], ii[m], lw=0.6, color=ORANGE)
ax_meas.axvspan(*cfg["window"], color=P["window"], lw=0, zorder=0)
ax_meas.set_ylabel("measured counts (log)")
ax_meas.set_title(f"a) what we measure: film {FILM} (425 °C, 13 nm PdO on sapphire) — counts vs 2θ, "
                  f"substrate lines dominate", loc="left", color=P["ink2"])
ax_meas.text(41.7, 6e5, "Al₂O₃(0006)\nsubstrate", ha="center", va="bottom", fontsize=7, color=P["ink2"])
ax_meas.text(34.0, 1.6e3, "PdO peak\n(film)", ha="center", va="bottom", fontsize=7, color=P["ink2"])
ax_meas.set_ylim(300, 3e6); ax_meas.tick_params(labelbottom=False)

in_range = strong[strong.two_theta_deg <= 80.6]
ax_card.vlines(in_range.two_theta_deg, 0, in_range.relative_intensity_num, color=BLUE, lw=1.6)
groups, cur = [], []
for _, r in in_range.iterrows():                      # merge labels of lines closer than 0.8 deg
    if cur and r.two_theta_deg - cur[-1].two_theta_deg > 0.8:
        groups.append(cur); cur = []
    cur.append(r)
groups.append(cur)
for g in groups:
    x = np.mean([r.two_theta_deg for r in g]); y = max(r.relative_intensity_num for r in g)
    if y > 60:   # the tall (002)/(101) pair: label beside the stick, not above it
        ax_card.text(x + 1.0, y, "/".join(r.hkl for r in g), ha="left", va="top", fontsize=6.5, color=P["ink2"])
    else:
        ax_card.text(x, y + 5, "/".join(r.hkl for r in g), ha="center", va="bottom", fontsize=6.5,
                     color=P["ink2"], rotation=90)
ax_card.axvspan(*cfg["window"], color=P["window"], lw=0, zorder=0)
ax_card.set_ylim(0, 165); ax_card.set_yticks([0, 50, 100]); ax_card.set_xlim(20.5, 80.6)
ax_card.set_ylabel("relative intensity\n(strongest = 100)"); ax_card.set_xlabel("2θ (deg)")
ax_card.set_title("b) what the card contains: PDF 00-041-1107 — line positions and relative heights only\n"
                  "    (no counts, no peak widths, no background)", loc="left", color=P["ink2"])

# zoom: measured hump above the flank baseline, card sticks scaled so (101) = measured peak height
z = (tt >= 30.5) & (tt <= 37.5)
x1, x2 = 32.0, 36.0
y1 = np.median(ii[(tt > 31.5) & (tt < 32.5)]); y2 = np.median(ii[(tt > 35.5) & (tt < 36.5)])
base = y1 + (y2 - y1) * (tt - x1) / (x2 - x1)
ax_zoom.plot(tt[z], (ii - base)[z], lw=0.8, color=ORANGE, label=f"measured film {FILM}, counts above baseline")
peak_h = fp["fit_amp"]
two = peaks[peaks.hkl.isin(["(002)", "(101)"])]
ax_zoom.vlines(two.two_theta_deg, 0, two.relative_intensity_num / 100 * peak_h, color=BLUE, lw=2.2,
               label="card lines, scaled so (101) matches the measured peak height")
lab_y = -0.30 * peak_h
for _, r in two.iterrows():
    left = r.hkl == "(002)"
    ax_zoom.annotate(f"card {r.hkl}  {r.two_theta_deg:.2f}°,  I = {r.relative_intensity}",
                     xy=(r.two_theta_deg, 0), xytext=(r.two_theta_deg + (-0.12 if left else 0.12), lab_y),
                     ha="right" if left else "left", va="center", fontsize=7.5, color=P["ink2"],
                     arrowprops=dict(arrowstyle="-", color=P["axis"], lw=0.7))
ax_zoom.axvspan(*cfg["window"], color=P["window"], lw=0, zorder=0)
ax_zoom.axhline(0, color=P["axis"], lw=0.8)
ax_zoom.set_xlim(30.5, 37.0); ax_zoom.set_ylim(-0.42 * peak_h, peak_h * 1.45)
ax_zoom.set_xlabel("2θ (deg)"); ax_zoom.set_ylabel("counts above baseline")
ax_zoom.set_title(f"c) the 33–35° window: measured hump (area {bm['auc_101_net']:.0f} counts·deg, FWHM "
                  f"{fp['fit_fwhm']:.2f}°)\n    vs the card's two lines — the card fixes where and in what ratio, "
                  f"not how many counts", loc="left", color=P["ink2"])
ax_zoom.legend(loc="upper left", fontsize=7.5)
fig.suptitle("ICDD card vs measured film: the card is a fingerprint (positions + relative heights), "
             "the index needs a measured area in counts·deg", x=0.01, ha="left", fontsize=10, color=P["ink"])
fig.savefig(f"{OUT}/pdf_card_vs_measured_film_0827.png", dpi=170, bbox_inches="tight")

print(f"wrote {OUT}/{STEM}.xlsx, .md, pdf_card_00-041-1107_icdd_chart.png, pdf_card_vs_measured_film_0827.png")
print(f"\nfilm {FILM}: peak amplitude {peak_h:.0f} counts, area {bm['auc_101_net']:.0f} counts.deg, FWHM {fp['fit_fwhm']:.2f} deg, "
      f"background ~{np.median(ii[(tt>31.5)&(tt<32.5)]):.0f} counts")
print(f"card: {len(peaks)} lines, {len(strong)} with I >= 3")
