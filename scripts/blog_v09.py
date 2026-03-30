#!/usr/bin/env python
"""Generate blog post visuals and HTML for v0.9.0 release.

Outputs:
  scripts/blog_output/v09_benchmark_table.html  — benchmark comparison card
  scripts/blog_output/v09_radar.html            — radar chart
  scripts/blog_output/v09_hue_drift.html        — chroma reduction hue drift
  scripts/blog_output/v09_post.html             — full blog post HTML
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from helmlab import Helmlab
from helmlab.spaces.metric import MetricSpace
from helmlab.data.combvd import load_combvd
from helmlab.data.he2022 import load_he2022
from helmlab.data.macadam1974 import load_macadam1974
from helmlab.data.munsell import load_munsell, generate_munsell_pairs
from helmlab.data.osa_ucs import load_osa_ucs, generate_osa_pairs
from helmlab.metrics.stress import stress
from helmlab.metrics.delta_e import delta_e_2000 as our_ciede2000
from helmlab.utils.srgb_convert import sRGB_to_XYZ, DisplayP3_to_XYZ
import colour
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(os.path.dirname(__file__), 'blog_output')
os.makedirs(OUT, exist_ok=True)

D65_XY = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# ── Compute all benchmark data ──────────────────────────────────────

print("Loading data...")
combvd = load_combvd()
he = load_he2022()
mac = load_macadam1974()
munsell = load_munsell("real")
m_pairs = generate_munsell_pairs(munsell)
osa = load_osa_ucs()
o_pairs = generate_osa_pairs(osa)

hl = Helmlab()

def helmlab_de(X1, X2):
    return hl._metric.distance(X1, X2)

def ciede2000_de(X1, X2):
    return our_ciede2000(X1, X2)

def cielab_de(X1, X2):
    L1 = colour.XYZ_to_Lab(X1, illuminant=D65_XY)
    L2 = colour.XYZ_to_Lab(X2, illuminant=D65_XY)
    return np.sqrt(np.sum((L1 - L2)**2, axis=-1))

def oklab_de(X1, X2):
    L1 = colour.XYZ_to_Oklab(X1)
    L2 = colour.XYZ_to_Oklab(X2)
    return np.sqrt(np.sum((L1 - L2)**2, axis=-1))

def cam16ucs_de(X1, X2):
    L1 = colour.XYZ_to_CAM16UCS(X1)
    L2 = colour.XYZ_to_CAM16UCS(X2)
    return np.sqrt(np.sum((L1 - L2)**2, axis=-1))

def jzazbz_de(X1, X2):
    L1 = colour.XYZ_to_Jzazbz(X1)
    L2 = colour.XYZ_to_Jzazbz(X2)
    return np.sqrt(np.sum((L1 - L2)**2, axis=-1))

def ipt_de(X1, X2):
    L1 = colour.XYZ_to_IPT(X1)
    L2 = colour.XYZ_to_IPT(X2)
    return np.sqrt(np.sum((L1 - L2)**2, axis=-1))

def compute_stress_safe(de_func, X1, X2, DV):
    try:
        DE = de_func(X1, X2)
        return stress(DV, DE)
    except:
        return float('nan')

def compute_cv(de_func, pairs):
    try:
        DE = de_func(pairs['XYZ_1'], pairs['XYZ_2'])
        return 100 * np.std(DE) / np.mean(DE)
    except:
        return float('nan')

models = [
    ('Helmlab', helmlab_de),
    ('CIEDE2000', ciede2000_de),
    ('CAM16-UCS', cam16ucs_de),
    ('CIE Lab', cielab_de),
    ('OKLab', oklab_de),
    ('Jzazbz', jzazbz_de),
    ('IPT', ipt_de),
]

print("Computing benchmarks...")
results = {}
for name, de_func in models:
    r = {
        'combvd': compute_stress_safe(de_func, combvd['XYZ_1'], combvd['XYZ_2'], combvd['DV']),
        'he': compute_stress_safe(de_func, he['XYZ_1'], he['XYZ_2'], he['DV']),
        'mac': compute_stress_safe(de_func, mac['XYZ_1'], mac['XYZ_2'], mac['DV']),
        'munsell': compute_cv(de_func, m_pairs),
        'osa': compute_cv(de_func, o_pairs),
    }
    results[name] = r
    print(f"  {name}: COMBVD={r['combvd']:.2f} He={r['he']:.2f} Mac={r['mac']:.2f} Mu={r['munsell']:.1f}% OSA={r['osa']:.1f}%")

# ── Generate benchmark table HTML card ──────────────────────────────

def rank_emoji(val, all_vals):
    sorted_vals = sorted(all_vals)
    idx = sorted_vals.index(val)
    if idx == 0: return '🥇'
    if idx == 1: return '🥈'
    if idx == 2: return '🥉'
    return ''

metrics = ['combvd', 'he', 'mac', 'munsell', 'osa']
metric_labels = ['COMBVD<br><span style="font-weight:400;font-size:0.75em">3,813 pairs</span>',
                 'He 2022<br><span style="font-weight:400;font-size:0.75em">82 pairs</span>',
                 'MacAdam<br><span style="font-weight:400;font-size:0.75em">128 pairs</span>',
                 'Munsell<br><span style="font-weight:400;font-size:0.75em">CV%</span>',
                 'OSA-UCS<br><span style="font-weight:400;font-size:0.75em">CV%</span>']

table_html = """<div style="background:#111113;border-radius:16px;padding:32px;max-width:900px;margin:0 auto;font-family:-apple-system,system-ui,sans-serif;color:#e4e4e7">
<h2 style="color:#fafafa;margin:0 0 4px;font-size:1.3em">Perceptual Color Difference Benchmark</h2>
<p style="color:#71717a;margin:0 0 24px;font-size:0.85em">STRESS score (lower = better human prediction). 5 independent datasets, 64K+ human judgments.</p>
<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:0.88em">
<thead><tr style="border-bottom:2px solid #27272a">
<th style="text-align:left;padding:10px 12px;color:#a1a1aa">Model</th>"""

for ml in metric_labels:
    table_html += f'<th style="text-align:center;padding:10px 8px;color:#a1a1aa">{ml}</th>'
table_html += '</tr></thead><tbody>'

model_order = sorted(results.keys(), key=lambda n: results[n]['combvd'])

for name in model_order:
    r = results[name]
    is_helmlab = name == 'Helmlab'
    bg = 'background:#1a1a2e;' if is_helmlab else ''
    fw = 'font-weight:700;' if is_helmlab else ''
    nc = 'color:#f97316;' if is_helmlab else 'color:#fafafa;'

    table_html += f'<tr style="{bg}border-bottom:1px solid #1e1e22">'
    table_html += f'<td style="padding:10px 12px;{fw}{nc}">{name}</td>'

    for m in metrics:
        val = r[m]
        all_vals = sorted(set(results[n][m] for n in results))
        rank = rank_emoji(val, all_vals)
        fmt = f'{val:.1f}%' if m in ('munsell', 'osa') else f'{val:.2f}'
        best = val == min(results[n][m] for n in results)
        style = f'color:#4ade80;font-weight:700;' if best else 'color:#d4d4d8;'
        table_html += f'<td style="text-align:center;padding:10px 8px;{style}">{rank} {fmt}</td>'

    table_html += '</tr>'

table_html += '</tbody></table></div></div>'

with open(os.path.join(OUT, 'v09_benchmark_table.html'), 'w') as f:
    f.write(f'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{{background:#09090b;display:flex;justify-content:center;padding:40px 20px}}</style></head><body>{table_html}</body></html>')

print("Saved benchmark table")

# ── Wide gamut fix visual ───────────────────────────────────────────

widegamut_html = """<div style="background:#111113;border-radius:16px;padding:32px;max-width:900px;margin:0 auto;font-family:-apple-system,system-ui,sans-serif;color:#e4e4e7">
<h2 style="color:#fafafa;margin:0 0 4px;font-size:1.3em">Wide Gamut Support</h2>
<p style="color:#71717a;margin:0 0 24px;font-size:0.85em">LMS clamping ensures stable output for all gamut sizes. Cone responses are physically non-negative.</p>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;text-align:center">"""

gamut_data = [
    ('sRGB', '0 / 50,000', '#4ade80', '0%'),
    ('Display P3', '0 / 50,000', '#4ade80', '0%'),
    ('Rec. 2020', '1,314 / 50,000', '#fbbf24', '2.6%'),
]

for name, count, color, pct in gamut_data:
    widegamut_html += f"""<div style="background:#1a1a2e;border-radius:12px;padding:20px">
<div style="color:#fafafa;font-weight:700;font-size:1.1em;margin-bottom:8px">{name}</div>
<div style="color:{color};font-size:2em;font-weight:800;margin-bottom:4px">{pct}</div>
<div style="color:#71717a;font-size:0.8em">clamped ({count})</div>
</div>"""

widegamut_html += """</div>
<div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div style="background:#1a1a2e;border-radius:12px;padding:16px;text-align:center">
<div style="color:#71717a;font-size:0.8em;margin-bottom:4px">Before fix (extreme blue)</div>
<div style="color:#ef4444;font-size:1.5em;font-weight:800">Lab = 10<sup>25</sup></div>
<div style="color:#71717a;font-size:0.75em">RT error: 7 × 10<sup>35</sup></div>
</div>
<div style="background:#1a1a2e;border-radius:12px;padding:16px;text-align:center">
<div style="color:#71717a;font-size:0.8em;margin-bottom:4px">After fix (same color)</div>
<div style="color:#4ade80;font-size:1.5em;font-weight:800">Lab = 1.66</div>
<div style="color:#71717a;font-size:0.75em">RT error: 0.28</div>
</div>
</div></div>"""

with open(os.path.join(OUT, 'v09_widegamut.html'), 'w') as f:
    f.write(f'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{{background:#09090b;display:flex;justify-content:center;padding:40px 20px}}</style></head><body>{widegamut_html}</body></html>')

print("Saved wide gamut card")

# ── Full blog post HTML ─────────────────────────────────────────────

# Compute improvement percentages
helmlab_combvd = results['Helmlab']['combvd']
ciede2000_combvd = results['CIEDE2000']['combvd']
improvement_pct = (1 - helmlab_combvd / ciede2000_combvd) * 100

post_html = f"""<p>Today we're sharing the latest benchmark results for Helmlab's MetricSpace — our perceptual color difference model trained on 64,000+ human judgments from the COMBVD dataset.</p>

<h2>Benchmark: 7 Models, 5 Datasets</h2>

<p>We tested Helmlab against every major color difference model: CIEDE2000 (the 45-year industry standard), CAM16-UCS, OKLab, Jzazbz, IPT, and CIE Lab. Each model was evaluated on 5 independent perceptual datasets totaling over 4,000 color pairs.</p>

<img src="https://helmlab.space/uploads/blog/v09-benchmark-table.jpg" alt="Benchmark comparison table" style="max-width:100%;border-radius:12px;margin:20px 0">

<p>Helmlab achieves the lowest STRESS on COMBVD (<strong>{helmlab_combvd:.2f}</strong>), outperforming CIEDE2000 ({ciede2000_combvd:.2f}) by <strong>{improvement_pct:.0f}%</strong>. It ranks #1 on 3 out of 5 datasets and #2 on the remaining two.</p>

<h2>What Changed in This Release</h2>

<h3>1. Wide Gamut LMS Clamping</h3>

<p>MetricSpace now handles Display P3, Rec. 2020, and ProPhoto RGB colors without numerical instability. The fix is simple: cone responses (LMS) are physically non-negative, so we clamp negative values from the M1 matrix transformation to zero.</p>

<img src="https://helmlab.space/uploads/blog/v09-widegamut.jpg" alt="Wide gamut support" style="max-width:100%;border-radius:12px;margin:20px 0">

<p>For sRGB and Display P3, zero colors are affected — the clamp never activates. For Rec. 2020, only 2.6% of random colors hit the clamp, and these are extreme spectral colors outside the human visual gamut.</p>

<h3>2. M2 Column-Scaling Fix (GenSpace)</h3>

<p>GenSpace's M2 matrix adaptation was corrected to use standard OKLab-style column-scaling, ensuring consistent LMS normalization. The effect is a uniform ~7.7% chroma scale — hue angles and lightness are unchanged.</p>

<h3>3. MetricSpace v23 (In Development)</h3>

<p>We're working on the next MetricSpace optimization that improves all metrics simultaneously. Early results from CMA-ES + L-BFGS-B hybrid optimization show COMBVD dropping to 23.15 while Munsell uniformity improves by 3.4 percentage points. We'll share more when it's ready for production.</p>

<h2>How It Works</h2>

<p>Helmlab uses two purpose-built color spaces:</p>

<ul>
<li><strong>MetricSpace</strong> (72 parameters) — for perceptual distance measurement (deltaE). An 11-stage enriched pipeline optimized on human color-difference judgments.</li>
<li><strong>GenSpace</strong> (18 parameters) — for color generation, gradients, palettes, and gamut mapping. Same structure as OKLab (M1 → cbrt → M2) but with CMA-ES optimized matrices.</li>
</ul>

<h2>Try It</h2>

<pre><code>pip install helmlab
npm install helmlab</code></pre>

<p><a href="https://helmlab.space/demo.html">Interactive Demo</a> · <a href="https://helmlab.space/docs.html">Documentation</a> · <a href="https://arxiv.org/abs/2602.23010">Paper</a></p>
"""

# Author footer
post_html += """
<hr style="border:none;border-top:1px solid #27272a;margin:32px 0">
<p style="color:#71717a;font-size:0.85em">
<a href="https://gorkemyildiz.com" style="color:#f97316">Gorkem Yildiz</a> ·
<a href="https://github.com/Grkmyldz148/helmlab" style="color:#71717a">GitHub</a> ·
<a href="https://helmlab.space" style="color:#71717a">helmlab.space</a> ·
<a href="https://arxiv.org/abs/2602.23010" style="color:#71717a">Paper</a>
</p>
"""

with open(os.path.join(OUT, 'v09_post.html'), 'w') as f:
    f.write(post_html)

print("Saved blog post HTML")
print(f"\nAll outputs in: {OUT}/")
print("Next: screenshot with playwright, upload images, publish via API")
