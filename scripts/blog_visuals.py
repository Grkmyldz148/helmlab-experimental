#!/usr/bin/env python3
"""Generate blog post visuals for v14 release using Playwright.

Creates HTML cards → screenshots as JPEG at 100% quality.
"""
import asyncio
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from helmlab import GenSpace
from helmlab.utils.srgb_convert import sRGB_to_XYZ, XYZ_to_sRGB, hex_to_srgb

OUT = ROOT / "scripts" / "blog_output"
OUT.mkdir(exist_ok=True)

gs = GenSpace()

# ── Helper: generate gradient hex colors ───────────────────
def gradient_hex(hex1, hex2, n=15):
    srgb1 = hex_to_srgb(hex1).flatten()
    srgb2 = hex_to_srgb(hex2).flatten()
    xyz1 = sRGB_to_XYZ(srgb1.reshape(1,3))[0]
    xyz2 = sRGB_to_XYZ(srgb2.reshape(1,3))[0]
    lab1 = gs.from_XYZ(xyz1)
    lab2 = gs.from_XYZ(xyz2)
    hexes = []
    for t in np.linspace(0, 1, n):
        lab_t = lab1*(1-t) + lab2*t
        xyz_t = gs.to_XYZ(lab_t)
        srgb_t = XYZ_to_sRGB(xyz_t.reshape(1,3))[0]
        srgb_t = np.clip(srgb_t, 0, 1)
        r, g, b = (srgb_t * 255).astype(int)
        hexes.append(f"#{r:02x}{g:02x}{b:02x}")
    return hexes

# Oklab gradient for comparison
OKLAB_M1 = np.array([[0.8189330101,0.3618667424,-0.1288597137],
                      [0.0329845436,0.9293118715,0.0361456387],
                      [0.0482003018,0.2643662691,0.6338517070]])
OKLAB_M2 = np.array([[0.2104542553,0.7936177850,-0.0040720468],
                      [1.9779984951,-2.4285922050,0.4505937099],
                      [0.0259040371,0.7827717662,-0.8086757660]])
OKLAB_M1_inv = np.linalg.inv(OKLAB_M1)
OKLAB_M2_inv = np.linalg.inv(OKLAB_M2)

def oklab_gradient_hex(hex1, hex2, n=15):
    srgb1 = hex_to_srgb(hex1).flatten()
    srgb2 = hex_to_srgb(hex2).flatten()
    xyz1 = sRGB_to_XYZ(srgb1.reshape(1,3))[0]
    xyz2 = sRGB_to_XYZ(srgb2.reshape(1,3))[0]
    lms1 = xyz1 @ OKLAB_M1.T; lms1 = np.sign(lms1)*np.abs(lms1)**(1/3)
    lab1 = lms1 @ OKLAB_M2.T
    lms2 = xyz2 @ OKLAB_M1.T; lms2 = np.sign(lms2)*np.abs(lms2)**(1/3)
    lab2 = lms2 @ OKLAB_M2.T
    hexes = []
    for t in np.linspace(0, 1, n):
        lab_t = lab1*(1-t) + lab2*t
        lms_c = lab_t @ OKLAB_M2_inv.T
        lms = np.sign(lms_c)*np.abs(lms_c)**3
        xyz_t = lms @ OKLAB_M1_inv.T
        srgb_t = XYZ_to_sRGB(xyz_t.reshape(1,3))[0]
        srgb_t = np.clip(srgb_t, 0, 1)
        r, g, b = (srgb_t * 255).astype(int)
        hexes.append(f"#{r:02x}{g:02x}{b:02x}")
    return hexes

# ── Visual 1: Hero banner ─────────────────────────────────
pairs = [
    ("#ff0000", "#0000ff"), ("#ff6600", "#0066ff"), ("#00cc88", "#ff0088"),
    ("#ffcc00", "#0088ff"), ("#ff0066", "#00ffcc"), ("#6600ff", "#00ff66"),
]
gradient_strips = []
for h1, h2 in pairs:
    gradient_strips.append(gradient_hex(h1, h2, 30))

hero_html = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#09090b;font-family:-apple-system,system-ui,sans-serif;}
.hero{width:1200px;height:630px;display:flex;flex-direction:column;justify-content:center;align-items:center;position:relative;overflow:hidden}
.gradients{position:absolute;top:0;left:0;right:0;bottom:0;opacity:0.35}
.grad-row{display:flex;height:""" + str(630//len(pairs)) + """px}
.grad-cell{flex:1}
.content{position:relative;z-index:1;text-align:center;color:#fafafa}
.title{font-size:52px;font-weight:800;margin-bottom:12px;letter-spacing:-1px}
.subtitle{font-size:22px;color:#a1a1aa;font-weight:400;margin-bottom:24px}
.badge{display:inline-block;background:#1e293b;color:#60a5fa;font-size:15px;font-weight:600;padding:6px 18px;border-radius:20px;margin:0 6px}
.version{color:#f97316;font-weight:700}
</style></head><body><div class="hero">
<div class="gradients">"""
for strip in gradient_strips:
    hero_html += '<div class="grad-row">'
    for c in strip:
        hero_html += f'<div class="grad-cell" style="background:{c}"></div>'
    hero_html += '</div>'
hero_html += """</div>
<div class="content">
<div class="title">Helmlab <span class="version">v0.7.0</span></div>
<div class="subtitle">GenSpace v14 — CMA-ES Optimized Color Space</div>
<div style="margin-top:8px">
<span class="badge">28/43 Benchmark Wins</span>
<span class="badge">Sky-Blue Gradients</span>
<span class="badge">Zero Purple Shift</span>
</div>
</div></div></body></html>"""
(OUT / "hero.html").write_text(hero_html)

# ── Visual 2: Gradient comparison (v14 vs OKLab) ──────────
comp_pairs = [
    ("Blue → White", "#0000ff", "#ffffff"),
    ("Red → Cyan", "#ff0000", "#00ffff"),
    ("Green → Magenta", "#00ff00", "#ff00ff"),
]
comp_html = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#09090b;font-family:-apple-system,system-ui,sans-serif;padding:40px}
.card{width:1120px;background:#111113;border:1px solid #27272a;border-radius:16px;padding:32px;overflow:hidden}
h2{color:#fafafa;font-size:24px;margin-bottom:24px;font-weight:700}
.row{display:flex;gap:24px;margin-bottom:24px}
.col{flex:1}
.label{color:#71717a;font-size:13px;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px}
.grad-bar{display:flex;border-radius:8px;overflow:hidden;height:48px}
.grad-bar div{flex:1}
.pair-label{color:#a1a1aa;font-size:14px;font-weight:600;margin-bottom:6px}
.highlight{color:#22c55e;font-weight:700}
.dim{color:#ef4444;font-weight:700}
</style></head><body><div class="card">
<h2>Gradient Comparison: Helmlab v14 vs OKLab</h2>"""
for name, h1, h2 in comp_pairs:
    v14_grad = gradient_hex(h1, h2, 30)
    ok_grad = oklab_gradient_hex(h1, h2, 30)
    comp_html += f'<div class="pair-label">{name}</div><div class="row">'
    comp_html += '<div class="col"><div class="label">Helmlab v14 <span class="highlight">✓</span></div><div class="grad-bar">'
    for c in v14_grad:
        comp_html += f'<div style="background:{c}"></div>'
    comp_html += '</div></div><div class="col"><div class="label">OKLab</div><div class="grad-bar">'
    for c in ok_grad:
        comp_html += f'<div style="background:{c}"></div>'
    comp_html += '</div></div></div>'
comp_html += "</div></body></html>"
(OUT / "gradient_comparison.html").write_text(comp_html)

# ── Visual 3: Benchmark wins chart ────────────────────────
bench_html = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#09090b;font-family:-apple-system,system-ui,sans-serif;padding:40px}
.card{width:1120px;background:#111113;border:1px solid #27272a;border-radius:16px;padding:32px}
h2{color:#fafafa;font-size:24px;margin-bottom:8px;font-weight:700}
.desc{color:#71717a;font-size:14px;margin-bottom:32px}
.bar-group{margin-bottom:20px}
.bar-label{display:flex;justify-content:space-between;margin-bottom:6px}
.bar-name{color:#d4d4d8;font-size:15px;font-weight:600}
.bar-value{color:#a1a1aa;font-size:15px;font-weight:700}
.bar-track{background:#1a1a1e;border-radius:8px;height:36px;overflow:hidden}
.bar-fill{height:100%;border-radius:8px;display:flex;align-items:center;padding-left:12px;font-size:13px;font-weight:700;color:#fff;transition:width 0.5s}
</style></head><body><div class="card">
<h2>Perceptual Benchmark Results</h2>
<div class="desc">43 tests across gradient uniformity, hue accuracy, gamut mapping, achromatic stability, and more</div>"""
bars = [
    ("Helmlab v14", 28, "#3b82f6", 43),
    ("CIE Lab", 9, "#a1a1aa", 43),
    ("OKLab", 6, "#f97316", 43),
]
for name, wins, color, total in bars:
    pct = wins / total * 100
    bench_html += f'''<div class="bar-group">
<div class="bar-label"><span class="bar-name">{name}</span><span class="bar-value">{wins}/{total} wins</span></div>
<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}">{wins}</div></div>
</div>'''
bench_html += "</div></body></html>"
(OUT / "benchmark.html").write_text(bench_html)

# ── Visual 4: Blue→White midpoint comparison ──────────────
bw_v14 = gradient_hex("#0000ff", "#ffffff", 11)
bw_ok = oklab_gradient_hex("#0000ff", "#ffffff", 11)
mid_html = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#09090b;font-family:-apple-system,system-ui,sans-serif;padding:40px}
.card{width:1120px;background:#111113;border:1px solid #27272a;border-radius:16px;padding:32px}
h2{color:#fafafa;font-size:24px;margin-bottom:8px;font-weight:700}
.desc{color:#71717a;font-size:14px;margin-bottom:24px}
.comparison{display:flex;gap:32px}
.side{flex:1;text-align:center}
.side-label{color:#a1a1aa;font-size:14px;font-weight:600;margin-bottom:12px}
.gradient-strip{display:flex;border-radius:12px;overflow:hidden;height:80px;margin-bottom:12px}
.gradient-strip div{flex:1}
.midpoint{width:120px;height:120px;border-radius:50%;margin:0 auto 8px;border:3px solid #27272a}
.mid-hex{color:#d4d4d8;font-size:15px;font-weight:600;font-family:monospace}
.mid-desc{color:#71717a;font-size:12px;margin-top:4px}
.good{color:#22c55e}.bad{color:#ef4444}
</style></head><body><div class="card">
<h2>The Purple Problem: Blue→White Midpoint</h2>
<div class="desc">OKLab's Blue→White gradient passes through purple — Helmlab v14 stays sky-blue</div>
<div class="comparison">
<div class="side">
<div class="side-label">Helmlab v14</div>
<div class="gradient-strip">"""
for c in bw_v14:
    mid_html += f'<div style="background:{c}"></div>'
mid_html += f"""</div>
<div class="midpoint" style="background:{bw_v14[5]}"></div>
<div class="mid-hex">{bw_v14[5]}</div>
<div class="mid-desc good">Sky blue — perceptually correct</div>
</div>
<div class="side">
<div class="side-label">OKLab</div>
<div class="gradient-strip">"""
for c in bw_ok:
    mid_html += f'<div style="background:{c}"></div>'
mid_html += f"""</div>
<div class="midpoint" style="background:{bw_ok[5]}"></div>
<div class="mid-hex">{bw_ok[5]}</div>
<div class="mid-desc bad">Purple shift — hue distortion</div>
</div>
</div></div></body></html>"""
(OUT / "blue_white.html").write_text(mid_html)

# ── Visual 5: Stats summary card ──────────────────────────
stats_html = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#09090b;font-family:-apple-system,system-ui,sans-serif;padding:40px}
.card{width:1120px;background:#111113;border:1px solid #27272a;border-radius:16px;padding:32px}
h2{color:#fafafa;font-size:24px;margin-bottom:24px;font-weight:700}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.stat{background:#0a0a0c;border:1px solid #1f1f23;border-radius:12px;padding:20px;text-align:center}
.stat-value{color:#fafafa;font-size:32px;font-weight:800;margin-bottom:4px}
.stat-label{color:#71717a;font-size:13px;font-weight:500}
.stat-value .unit{font-size:16px;color:#a1a1aa}
.accent{color:#3b82f6}
.green{color:#22c55e}
.orange{color:#f97316}
</style></head><body><div class="card">
<h2>Helmlab v0.7.0 — Key Metrics</h2>
<div class="stats">
<div class="stat"><div class="stat-value accent">28<span class="unit">/43</span></div><div class="stat-label">Benchmark Wins</div></div>
<div class="stat"><div class="stat-value green">604</div><div class="stat-label">Tests Passing</div></div>
<div class="stat"><div class="stat-value orange">10⁻¹⁵</div><div class="stat-label">Roundtrip Error</div></div>
<div class="stat"><div class="stat-value" style="color:#a855f7">0%</div><div class="stat-label">Gradient CV (arc-len)</div></div>
<div class="stat"><div class="stat-value accent">23.30</div><div class="stat-label">MetricSpace STRESS</div></div>
<div class="stat"><div class="stat-value green">2.28</div><div class="stat-label">M1 Condition Number</div></div>
<div class="stat"><div class="stat-value orange">18</div><div class="stat-label">GenSpace Parameters</div></div>
<div class="stat"><div class="stat-value" style="color:#a855f7">~12KB</div><div class="stat-label">JS Bundle (gzipped)</div></div>
</div></div></body></html>"""
(OUT / "stats.html").write_text(stats_html)

print(f"Generated 5 HTML visuals in {OUT}")
print("Files:", [f.name for f in OUT.glob("*.html")])
