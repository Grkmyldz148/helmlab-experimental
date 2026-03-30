"""v31: Post-M2 hue-dependent L correction for yellow cusp.

KEY INSIGHT: Don't touch M1/M2 (they're good for CV). Add a small
hue-dependent correction AFTER M2 that compresses high-L yellows.

Pipeline: XYZ → M1 → cbrt → M2 → hue_L_correction → Lab

Correction: L_out = L - amp * w(h,C) * max(0, L - L_knee)
where w = exp(-(dh/h_width)²) * C/(C+0.01)

4 params: amp, h_center, h_width, L_knee
M1/M2 fixed from v14. Only 4 DOF to optimize!

Analytically invertible (no Newton):
  L_in = (L_out - amp*w*L_knee) / (1 - amp*w)  when L_out >= L_knee
"""
import json, time, math, numpy as np, torch, subprocess, sys
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
import cma

D65 = np.array([0.95047, 1.0, 1.08883])
M_S_np = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
M_Si_np = np.linalg.inv(M_S_np)
M_S = torch.tensor(M_S_np, device=device)
M_Si = torch.linalg.inv(M_S)
D65_T = torch.tensor(D65, device=device)

# v14 matrices — FIXED, not optimized
V14_M1 = np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2 = np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])
V14_M1i = np.linalg.inv(V14_M1)
V14_M2i = np.linalg.inv(V14_M2)
M1t = torch.tensor(V14_M1, device=device)
M2t = torch.tensor(V14_M2, device=device)
M1it = torch.linalg.inv(M1t)
M2it = torch.linalg.inv(M2t)

def scbrt_np(x): return np.sign(x)*np.abs(x)**(1/3)
def s2l_np(c): return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s_np(c): return np.where(c<=0.0031308,c*12.92,1.055*np.maximum(c,1e-10)**(1/2.4)-0.055)

# ── Hue-dependent L correction ──
def apply_hue_L_correction(L, a, b, amp, h_center, h_width, L_knee):
    """Forward: compress high-L at yellow hues."""
    C = np.sqrt(a**2 + b**2)
    if C < 1e-10:
        return L  # achromatic: no correction
    h = np.arctan2(b, a)
    dh = np.arctan2(np.sin(h - h_center), np.cos(h - h_center))
    w = np.exp(-(dh / max(h_width, 0.01))**2) * C / (C + 0.01)
    excess = max(0.0, L - L_knee)
    return L - amp * w * excess

def invert_hue_L_correction(L_out, a, b, amp, h_center, h_width, L_knee):
    """Inverse: analytically exact."""
    C = np.sqrt(a**2 + b**2)
    if C < 1e-10:
        return L_out
    h = np.arctan2(b, a)
    dh = np.arctan2(np.sin(h - h_center), np.cos(h - h_center))
    w = np.exp(-(dh / max(h_width, 0.01))**2) * C / (C + 0.01)
    aw = amp * w
    if aw > 0.99: aw = 0.99  # safety
    # Forward: L_out = L_in - aw * max(0, L_in - L_knee)
    # Case L_in > L_knee: L_out = L_in*(1-aw) + aw*L_knee
    #   → L_in = (L_out - aw*L_knee) / (1-aw)
    # Case L_in <= L_knee: L_out = L_in (no correction)
    # Try the corrected case first:
    L_candidate = (L_out - aw * L_knee) / (1 - aw)
    if L_candidate > L_knee:
        return L_candidate  # was corrected
    else:
        return L_out  # was below knee, no correction applied

# ── Forward/inverse with enrichment ──
def fwd_enriched(xyz, amp, h_center, h_width, L_knee):
    lab = V14_M2 @ scbrt_np(V14_M1 @ xyz)
    L_new = apply_hue_L_correction(lab[0], lab[1], lab[2], amp, h_center, h_width, L_knee)
    return np.array([L_new, lab[1], lab[2]])

def inv_enriched(lab, amp, h_center, h_width, L_knee):
    L_orig = invert_hue_L_correction(lab[0], lab[1], lab[2], amp, h_center, h_width, L_knee)
    lab_orig = np.array([L_orig, lab[1], lab[2]])
    lc = V14_M2i @ lab_orig
    return V14_M1i @ (np.sign(lc)*np.abs(lc)**3)

def srgb2xyz(rgb): return M_S_np @ s2l_np(np.array(rgb, dtype=float))
def xyz2cl(xyz):
    r = xyz / D65
    f = np.where(r > 0.008856, r**(1/3), 7.787*r + 16/116)
    return np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])
def de00(l1,l2):
    dL=l2[0]-l1[0];C1=np.sqrt(l1[1]**2+l1[2]**2);C2=np.sqrt(l2[1]**2+l2[2]**2)
    dC=C2-C1;dH=np.sqrt(max(0,(l2[1]-l1[1])**2+(l2[2]-l1[2])**2-dC**2))
    SL=1+0.015*(l1[0]-50)**2/np.sqrt(20+(l1[0]-50)**2);SC=1+0.045*C1;SH=1+0.015*C1
    return np.sqrt((dL/SL)**2+(dC/SC)**2+(dH/SH)**2)

# ── Training pairs ──
pairs = []
prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
for i in range(len(prims)):
    for j in range(i+1,len(prims)): pairs.append((prims[i],prims[j]))
for g1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
    for g2 in [g1+0.2,g1+0.4]:
        if g2<=1.0: pairs.append(([g1]*3,[g2]*3))
rng=np.random.RandomState(42)
for _ in range(80): pairs.append((rng.rand(3).tolist(),rng.rand(3).tolist()))
pair_xyz = [(srgb2xyz(c1), srgb2xyz(c2)) for c1, c2 in pairs]

# ── Metrics ──
def compute_cv(amp, h_center, h_width, L_knee):
    cvs = []
    for x1, x2 in pair_xyz:
        l1 = fwd_enriched(x1, amp, h_center, h_width, L_knee)
        l2 = fwd_enriched(x2, amp, h_center, h_width, L_knee)
        ds = []; prev = None
        for t in np.linspace(0, 1, 26):
            lab = l1 + t * (l2 - l1)
            xyz = inv_enriched(lab, amp, h_center, h_width, L_knee)
            s8 = np.round(l2s_np(np.clip(M_Si_np @ xyz, 0, 1)) * 255) / 255
            cl = xyz2cl(np.maximum(M_S_np @ s2l_np(s8), 1e-10))
            if prev is not None: ds.append(de00(prev, cl))
            prev = cl
        if ds:
            a = np.array(ds); m = np.mean(a)
            if m > 0.001: cvs.append(np.std(a)/m)
    return np.mean(cvs) if cvs else 999

def compute_cusp_L(amp, h_center, h_width, L_knee, hue_deg=85):
    """Find cusp L at given hue with enrichment."""
    hr = hue_deg * np.pi / 180
    ch, sh = np.cos(hr), np.sin(hr)
    best_C, best_L = 0, 0.5
    for Li in range(500, 1000, 2):
        L = Li / 1000
        lo, hi = 0.0, 0.4
        for _ in range(40):
            mid = (lo + hi) / 2
            lab = np.array([L, mid*ch, mid*sh])
            xyz = inv_enriched(lab, amp, h_center, h_width, L_knee)
            rgb = M_Si_np @ xyz
            if np.all(rgb >= -0.001) and np.all(rgb <= 1.001):
                lo = mid
            else:
                hi = mid
        if lo > best_C:
            best_C = lo; best_L = L
    return best_L, best_C

def compute_roundtrip_err(amp, h_center, h_width, L_knee):
    max_err = 0
    rng2 = np.random.RandomState(42)
    for _ in range(500):
        rgb = rng2.rand(3)
        xyz = M_S_np @ s2l_np(rgb)
        lab = fwd_enriched(xyz, amp, h_center, h_width, L_knee)
        xyz2 = inv_enriched(lab, amp, h_center, h_width, L_knee)
        rgb2 = l2s_np(np.clip(M_Si_np @ xyz2, 0, 1))
        e = np.max(np.abs(rgb - rgb2))
        if e > max_err: max_err = e
    return max_err

def compute_achromatic(amp, h_center, h_width, L_knee):
    max_ab = 0
    for i in range(257):
        g = i/256
        xyz = srgb2xyz([g,g,g])
        lab = fwd_enriched(xyz, amp, h_center, h_width, L_knee)
        c = np.sqrt(lab[1]**2 + lab[2]**2)
        if c > max_ab: max_ab = c
    return max_ab

# ── Objective (only 4 params!) ──
def objective(x):
    amp = x[0]        # [0, 1] range
    h_center = x[1]   # radians (~1.48 for 85°)
    h_width = x[2]    # radians (~0.26 for 15°)
    L_knee = x[3]     # [0.7, 0.95]

    # Bounds
    if amp < 0 or amp > 0.95: return 999
    if h_width < 0.05 or h_width > 1.0: return 999
    if L_knee < 0.5 or L_knee > 0.98: return 999

    try:
        # CV (main quality metric)
        cv = compute_cv(amp, h_center, h_width, L_knee)

        # Cusp L at yellow hues + cliff steepness
        cusp_pen = 0
        for hd in [80, 85, 90]:
            cL, cC = compute_cusp_L(amp, h_center, h_width, L_knee, hd)
            # Cusp L target: 0.83-0.87 (OKLab is ~0.84)
            if cL > 0.87: cusp_pen += (cL - 0.87)**2 * 50
            if cL < 0.80: cusp_pen += (0.80 - cL)**2 * 50
            if cC < 0.05: cusp_pen += (0.05 - cC)**2 * 500
            # Cliff steepness: check chroma 0.03 L after cusp
            if cC > 0.01:
                hr2 = hd * np.pi / 180
                ch2, sh2 = np.cos(hr2), np.sin(hr2)
                lo2, hi2 = 0.0, 0.4
                for _ in range(40):
                    mid2 = (lo2+hi2)/2
                    lab2 = np.array([cL+0.03, mid2*ch2, mid2*sh2])
                    xyz2 = inv_enriched(lab2, amp, h_center, h_width, L_knee)
                    rgb2 = M_Si_np @ xyz2
                    if np.all(rgb2 >= -0.001) and np.all(rgb2 <= 1.001): lo2 = mid2
                    else: hi2 = mid2
                drop = (cC - lo2) / cC
                # OKLab drop is ~15%. Penalize above 40%
                if drop > 0.40: cusp_pen += (drop - 0.40)**2 * 20

        # Round-trip (must stay perfect)
        rt = compute_roundtrip_err(amp, h_center, h_width, L_knee)
        if rt > 1e-8: return 50 + rt * 1e6

        # Achromatic (must stay perfect)
        ach = compute_achromatic(amp, h_center, h_width, L_knee)
        if ach > 1e-4: return 50 + ach * 1e4

        loss = cv + cusp_pen
        return loss
    except:
        return 999

# ── Run ──
print(f"\n{'='*60}", flush=True)
print("  v31: Post-M2 hue-dependent L correction", flush=True)
print("  M1/M2 fixed (v14). Only 4 enrichment params.", flush=True)
print(f"{'='*60}\n", flush=True)

# Baseline (no correction = amp=0)
print("--- Baseline (no enrichment) ---", flush=True)
cv0 = compute_cv(0, 0, 1, 0.9)
cL0, cC0 = compute_cusp_L(0, 0, 1, 0.9, 85)
rt0 = compute_roundtrip_err(0, 0, 1, 0.9)
ach0 = compute_achromatic(0, 0, 1, 0.9)
print(f"  CV={cv0*100:.2f}% cusp_L={cL0:.3f} cusp_C={cC0:.4f} RT={rt0:.2e} ach={ach0:.2e}", flush=True)

# CMA-ES on 4 params
# x0: amp=0.3, h_center=85°=1.484rad, h_width=15°=0.262rad, L_knee=0.85
x0 = np.array([0.366, 1.537, 0.882, 0.682])  # start from v31 converged params
sigma = 0.1

print(f"\n--- CMA-ES: 4 params, 300 gen x 32 pop ---", flush=True)
best_loss = 999.; best_x = x0.copy()
t0 = time.time(); ev = [0]; lp = [0]

def obj_fn(x):
    global best_loss, best_x
    loss = objective(x); ev[0] += 1
    if loss < best_loss:
        best_loss = loss; best_x = x.copy()
        now = time.time()
        if now - lp[0] > 10:
            lp[0] = now
            amp, hc, hw, lk = x
            cv = compute_cv(amp, hc, hw, lk)
            cL, cC = compute_cusp_L(amp, hc, hw, lk, 85)
            print(f"  #{ev[0]:>5d} [{now-t0:4.0f}s] loss={loss:.4f} CV={cv*100:.1f}% cusp_L={cL:.3f} cusp_C={cC:.4f} amp={amp:.3f} h={math.degrees(hc):.0f}° w={math.degrees(hw):.0f}° knee={lk:.3f}", flush=True)
    return loss

opts = cma.CMAOptions()
opts.set("maxiter", 300); opts.set("popsize", 32); opts.set("tolfun", 1e-11); opts.set("verbose", -1)
es = cma.CMAEvolutionStrategy(x0, sigma, opts)
while not es.stop():
    sols = es.ask(); fits = [obj_fn(x) for x in sols]; es.tell(sols, fits)
el = time.time() - t0

amp, h_center, h_width, L_knee = best_x
print(f"\n  DONE: {ev[0]} evals {el:.0f}s", flush=True)
print(f"  amp={amp:.4f} h_center={math.degrees(h_center):.1f}° h_width={math.degrees(h_width):.1f}° L_knee={L_knee:.4f}", flush=True)

# Final metrics
cv = compute_cv(amp, h_center, h_width, L_knee)
rt = compute_roundtrip_err(amp, h_center, h_width, L_knee)
ach = compute_achromatic(amp, h_center, h_width, L_knee)
print(f"  CV={cv*100:.2f}% RT={rt:.2e} ach={ach:.2e}", flush=True)

print(f"\n  Yellow cusp profile (h=85°):", flush=True)
hr = 85 * np.pi / 180; ch, sh = np.cos(hr), np.sin(hr)
prev_c = None
for Li in range(700, 1000, 10):
    L = Li / 1000
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo + hi) / 2
        lab = np.array([L, mid*ch, mid*sh])
        xyz = inv_enriched(lab, amp, h_center, h_width, L_knee)
        rgb = M_Si_np @ xyz
        if np.all(rgb >= -0.001) and np.all(rgb <= 1.001): lo = mid
        else: hi = mid
    arrow = ""
    if prev_c is not None:
        arrow = " ↑" if lo > prev_c + 0.001 else " ↓" if lo < prev_c - 0.001 else " ="
    prev_c = lo
    print(f"    L={L:.2f} C={lo:.4f}{arrow}", flush=True)

# Cusp at multiple hues
print(f"\n  Cusp L at key hues:", flush=True)
for hd in [60, 75, 80, 85, 90, 95, 100, 120, 180, 240, 300]:
    cL, cC = compute_cusp_L(amp, h_center, h_width, L_knee, hd)
    flag = " ← YELLOW" if 75 <= hd <= 95 else ""
    print(f"    h={hd:>3}°: cusp_L={cL:.3f} cusp_C={cC:.4f}{flag}", flush=True)

# Save enrichment params
enrichment = {
    "version": "v31-enrichment",
    "M1": V14_M1.tolist(), "M2": V14_M2.tolist(),
    "M1_inv": V14_M1i.tolist(), "M2_inv": V14_M2i.tolist(),
    "enrichment": {
        "type": "hue_L_correction",
        "amp": float(amp),
        "h_center": float(h_center),
        "h_width": float(h_width),
        "L_knee": float(L_knee)
    }
}
with open("/root/gen_v31_enrichment.json", "w") as f:
    json.dump(enrichment, f, indent=2)
print(f"\nSaved: gen_v31_enrichment.json", flush=True)

# Production test (using enriched forward/inverse as M1/M2 equivalent)
# We can't directly use production_test_gpu.py since it doesn't know about enrichment.
# Instead, print key metrics for manual comparison.
print(f"\n{'='*60}")
print(f"  KEY METRICS vs v14 BASELINE")
print(f"{'='*60}")
print(f"  {'Metric':<25} {'v14':>10} {'v31 (enriched)':>15}")
print(f"  {'-'*50}")
print(f"  {'Gradient CV':<25} {'22.73%':>10} {f'{cv*100:.2f}%':>15}")

cL85, cC85 = compute_cusp_L(amp, h_center, h_width, L_knee, 85)
print(f"  {'Yellow Cusp L (h=85°)':<25} {'0.988':>10} {f'{cL85:.3f}':>15}")
print(f"  {'Yellow Cusp C (h=85°)':<25} {'0.203':>10} {f'{cC85:.4f}':>15}")
print(f"  {'Round-trip error':<25} {'1.74e-14':>10} {f'{rt:.2e}':>15}")
print(f"  {'Achromatic max C':<25} {'4.09e-08':>10} {f'{ach:.2e}':>15}")

# Cliff steepness
cL85_raw = 0.988  # v14 without enrichment
hr85 = 85 * np.pi / 180
ch85, sh85 = np.cos(hr85), np.sin(hr85)
# Post-cusp chroma (2 L steps = 0.02 after cusp)
lab_post = np.array([cL85 + 0.01, 0.15*ch85, 0.15*sh85])
xyz_post = inv_enriched(lab_post, amp, h_center, h_width, L_knee)
rgb_post = M_Si_np @ xyz_post
# Compare with v14
lab_post_v14 = np.array([0.990, 0.15*ch85, 0.15*sh85])
lc_v14 = V14_M2i @ lab_post_v14; xyz_v14 = V14_M1i @ (np.sign(lc_v14)*np.abs(lc_v14)**3)
rgb_v14 = M_Si_np @ xyz_v14

print(f"\n  Enrichment params:")
print(f"    amp     = {amp:.4f}")
print(f"    h_center= {math.degrees(h_center):.1f}°")
print(f"    h_width = {math.degrees(h_width):.1f}°")
print(f"    L_knee  = {L_knee:.4f}")
