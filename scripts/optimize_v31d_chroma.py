"""v31d: L correction + chroma scaling for smooth cliff.

L correction moves cusp down (v31 proved this works).
Chroma scaling smooths the cliff by gradually reducing C near cusp.

Pipeline: XYZ → M1 → cbrt → M2 → L_correction → C_scaling → Lab

6 params total:
  L correction: amp_L, h_center, h_width, L_knee  (from v31)
  C scaling:    amp_C, C_knee                      (NEW)

C_scaling: C_out = C * (1 - amp_C * w(h) * max(0, L_raw - C_knee))
  - w(h) same Gaussian as L correction (reuse h_center, h_width)
  - L_raw is pre-L-correction L (so C scaling happens before L shift)
  - Reduces chroma gradually as L approaches 1.0 near yellow

Inverse: solve C_in from C_out analytically, then L inverse as before.
"""
import json, time, math, numpy as np, subprocess, sys

D65 = np.array([0.95047, 1.0, 1.08883])
M_S_np = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
M_Si_np = np.linalg.inv(M_S_np)

V14_M1 = np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2 = np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])
V14_M1i = np.linalg.inv(V14_M1)
V14_M2i = np.linalg.inv(V14_M2)

def scbrt(x): return np.sign(x)*np.abs(x)**(1/3)
def s2l(c): return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(c): return np.where(c<=0.0031308,c*12.92,1.055*np.maximum(c,1e-10)**(1/2.4)-0.055)
def srgb2xyz(rgb): return M_S_np @ s2l(np.array(rgb, dtype=float))
def xyz2cl(xyz):
    r = xyz/D65; f = np.where(r>0.008856, r**(1/3), 7.787*r+16/116)
    return np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])
def de00(l1, l2):
    dL=l2[0]-l1[0]; C1=np.sqrt(l1[1]**2+l1[2]**2); C2=np.sqrt(l2[1]**2+l2[2]**2)
    dC=C2-C1; dH=np.sqrt(max(0,(l2[1]-l1[1])**2+(l2[2]-l1[2])**2-dC**2))
    SL=1+0.015*(l1[0]-50)**2/np.sqrt(20+(l1[0]-50)**2); SC=1+0.045*C1; SH=1+0.015*C1
    return np.sqrt((dL/SL)**2+(dC/SC)**2+(dH/SH)**2)

def hue_weight(h, a, b, h_center, h_width):
    """Gaussian hue weight, proportional to chroma."""
    C = np.sqrt(a**2 + b**2)
    if C < 1e-10: return 0.0
    dh = np.arctan2(np.sin(h - h_center), np.cos(h - h_center))
    return np.exp(-(dh / max(h_width, 0.01))**2) * C / (C + 0.01)

# ── Forward: M2 → C_scale → L_correct → Lab ──
def fwd_enriched(xyz, amp_L, h_center, h_width, L_knee, amp_C, C_knee):
    lab = (V14_M2 @ scbrt(V14_M1 @ xyz)).copy()
    L, a, b = lab[0], lab[1], lab[2]
    C = np.sqrt(a**2 + b**2)
    if C < 1e-10: return lab
    h = np.arctan2(b, a)
    w = hue_weight(h, a, b, h_center, h_width)

    # 1. Chroma scaling (applied to raw L, before L correction)
    C_excess = max(0.0, L - C_knee)
    C_scale = 1.0 - amp_C * w * C_excess
    C_scale = max(C_scale, 0.01)  # safety
    lab[1] *= C_scale
    lab[2] *= C_scale

    # 2. L correction (same as v31)
    L_excess = max(0.0, L - L_knee)
    lab[0] = L - amp_L * w * L_excess

    return lab

# ── Inverse: L_uncorrect → C_unscale → M2_inv ──
def inv_enriched(lab, amp_L, h_center, h_width, L_knee, amp_C, C_knee):
    L_out, a_out, b_out = lab[0], lab[1], lab[2]
    C_out = np.sqrt(a_out**2 + b_out**2)

    if C_out < 1e-10:
        lc = V14_M2i @ np.array([L_out, a_out, b_out])
        return V14_M1i @ (np.sign(lc)*np.abs(lc)**3)

    h = np.arctan2(b_out, a_out)

    # We need L_raw (pre-correction L) to undo both corrections.
    # But C_scale depends on L_raw, and L correction depends on L_raw.
    # Order was: C_scale(L_raw), then L_correct(L_raw).
    # So L_out = L_raw - amp_L * w * max(0, L_raw - L_knee)
    # And a_out = a_raw * C_scale, where C_scale = 1 - amp_C * w * max(0, L_raw - C_knee)
    #
    # Step 1: recover L_raw from L_out (same as v31)
    # Note: w depends on C_out (post-scale chroma), not C_raw.
    # But w uses C/(C+0.01) which is ~1 for any reasonable C.
    # For the inverse, we compute w from the OUTPUT (a_out, b_out).
    # This is an approximation — w should use C_raw, but C_raw ≈ C_out/C_scale ≈ C_out * (1+small).
    # The error is tiny for small amp_C.

    # Use output chroma for w (approximation, verified by round-trip test)
    w = hue_weight(h, a_out, b_out, h_center, h_width)
    aw_L = min(amp_L * w, 0.99)

    # Recover L_raw
    L_cand = (L_out - aw_L * L_knee) / (1 - aw_L)
    L_raw = L_cand if L_cand > L_knee else L_out

    # Step 2: recover C_scale from L_raw
    C_excess = max(0.0, L_raw - C_knee)
    # w should use C_raw, but we approximate with w from output
    C_scale = 1.0 - amp_C * w * C_excess
    C_scale = max(C_scale, 0.01)

    # Step 3: un-scale chroma
    a_raw = a_out / C_scale
    b_raw = b_out / C_scale

    # Step 4: standard M2 inverse
    lc = V14_M2i @ np.array([L_raw, a_raw, b_raw])
    return V14_M1i @ (np.sign(lc)*np.abs(lc)**3)

# ── Training pairs ──
pairs = []
prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
for i in range(len(prims)):
    for j in range(i+1, len(prims)): pairs.append((prims[i], prims[j]))
rng = np.random.RandomState(42)
for _ in range(80): pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))

def compute_cv(params):
    amp_L, hc, hw, lk, amp_C, ck = params
    cvs = []
    for c1, c2 in pairs:
        x1, x2 = srgb2xyz(c1), srgb2xyz(c2)
        l1 = fwd_enriched(x1, amp_L, hc, hw, lk, amp_C, ck)
        l2 = fwd_enriched(x2, amp_L, hc, hw, lk, amp_C, ck)
        ds = []; prev = None
        for t in np.linspace(0, 1, 26):
            lab = l1 + t * (l2 - l1)
            xyz = inv_enriched(lab, amp_L, hc, hw, lk, amp_C, ck)
            s8 = np.round(l2s(np.clip(M_Si_np @ xyz, 0, 1))*255)/255
            cl = xyz2cl(np.maximum(M_S_np @ s2l(s8), 1e-10))
            if prev is not None: ds.append(de00(prev, cl))
            prev = cl
        if ds:
            a = np.array(ds); m = np.mean(a)
            if m > 0.001: cvs.append(np.std(a)/m)
    return np.mean(cvs) if cvs else 999

def compute_cusp_and_cliff(params, hue_deg=85):
    amp_L, hc, hw, lk, amp_C, ck = params
    hr = hue_deg * np.pi / 180
    ch, sh = np.cos(hr), np.sin(hr)
    best_C, best_L = 0, 0.5
    profile = []
    for Li in range(500, 1000, 2):
        L = Li/1000
        lo, hi = 0.0, 0.4
        for _ in range(40):
            mid = (lo+hi)/2
            lab = np.array([L, mid*ch, mid*sh])
            xyz = inv_enriched(lab, amp_L, hc, hw, lk, amp_C, ck)
            rgb = M_Si_np @ xyz
            if np.all(rgb >= -0.001) and np.all(rgb <= 1.001): lo = mid
            else: hi = mid
        profile.append((L, lo))
        if lo > best_C: best_C = lo; best_L = L

    # Cliff: chroma drop 0.03 L after cusp
    ci = next((i for i,(L,C) in enumerate(profile) if L >= best_L), len(profile)-1)
    post_i = min(ci + 3, len(profile)-1)  # 3 steps = 0.006 L
    post_C = profile[post_i][1]
    drop = (best_C - post_C) / best_C if best_C > 0.01 else 0
    return best_L, best_C, drop

def compute_rt(params):
    amp_L, hc, hw, lk, amp_C, ck = params
    max_err = 0
    rng2 = np.random.RandomState(42)
    for _ in range(500):
        rgb = rng2.rand(3)
        xyz = M_S_np @ s2l(rgb)
        lab = fwd_enriched(xyz, amp_L, hc, hw, lk, amp_C, ck)
        xyz2 = inv_enriched(lab, amp_L, hc, hw, lk, amp_C, ck)
        rgb2 = l2s(np.clip(M_Si_np @ xyz2, 0, 1))
        e = np.max(np.abs(rgb - rgb2))
        if e > max_err: max_err = e
    return max_err

# ── Objective ──
def objective(x):
    amp_L = x[0]; hc = x[1]; hw = x[2]; lk = x[3]
    amp_C = x[4]; ck = x[5]
    if amp_L < 0 or amp_L > 0.95: return 999
    if amp_C < 0 or amp_C > 0.95: return 999
    if hw < 0.1 or hw > 1.5: return 999
    if lk < 0.4 or lk > 0.95: return 999
    if ck < 0.4 or ck > 0.98: return 999

    params = (amp_L, hc, hw, lk, amp_C, ck)
    try:
        cv = compute_cv(params)
        cusp_pen = 0
        for hd in [80, 85, 90]:
            cL, cC, drop = compute_cusp_and_cliff(params, hd)
            if cL > 0.88: cusp_pen += (cL - 0.88)**2 * 30
            if cL < 0.78: cusp_pen += (0.78 - cL)**2 * 30
            if cC < 0.05: cusp_pen += (0.05 - cC)**2 * 500
            # Cliff: penalize drop > 35%
            if drop > 0.35: cusp_pen += (drop - 0.35)**2 * 20

        rt = compute_rt(params)
        if rt > 1e-6: return 50 + rt * 1e4  # relaxed for approximation

        loss = cv + cusp_pen
        return loss
    except: return 999

# ── Run ──
import cma
print("="*60, flush=True)
print("  v31d: L correction + chroma scaling", flush=True)
print("  6 params: amp_L, h_center, h_width, L_knee, amp_C, C_knee", flush=True)
print("="*60, flush=True)

# Baseline
p0 = (0, 0, 1, 0.9, 0, 0.9)
cv0 = compute_cv(p0)
cL0, cC0, drop0 = compute_cusp_and_cliff(p0)
print(f"\n  Baseline: CV={cv0*100:.2f}% cusp_L={cL0:.3f} cliff={drop0*100:.1f}%", flush=True)

# Start from v31 + small C scaling
# v31 converged: amp_L=0.366, hc=88°=1.536rad, hw=51°=0.890rad, lk=0.682
x0 = np.array([0.366, 1.536, 0.890, 0.682, 0.15, 0.82])
print(f"\n--- CMA-ES: 6 params, 300 gen x 40 pop ---", flush=True)

best_loss = 999.; best_x = x0.copy()
t0 = time.time(); ev = [0]; lp = [0]

def obj_fn(x):
    global best_loss, best_x
    loss = objective(x); ev[0] += 1
    if loss < best_loss:
        best_loss = loss; best_x = x.copy()
        now = time.time()
        if now - lp[0] > 15:
            lp[0] = now
            params = tuple(x)
            cv = compute_cv(params)
            cL, cC, drop = compute_cusp_and_cliff(params, 85)
            print(f"  #{ev[0]:>5d} [{now-t0:4.0f}s] loss={loss:.4f} CV={cv*100:.1f}% cusp_L={cL:.3f} cliff={drop*100:.0f}% ampL={x[0]:.3f} ampC={x[4]:.3f} ck={x[5]:.3f}", flush=True)
    return loss

opts = cma.CMAOptions()
opts.set("maxiter", 300); opts.set("popsize", 40); opts.set("tolfun", 1e-11); opts.set("verbose", -1)
es = cma.CMAEvolutionStrategy(x0, 0.1, opts)
while not es.stop():
    sols = es.ask(); fits = [obj_fn(x) for x in sols]; es.tell(sols, fits)
el = time.time() - t0

params = tuple(best_x)
cv = compute_cv(params)
rt = compute_rt(params)
print(f"\n  DONE: {ev[0]} evals {el:.0f}s", flush=True)
print(f"  amp_L={best_x[0]:.4f} h={math.degrees(best_x[1]):.1f} w={math.degrees(best_x[2]):.1f} Lknee={best_x[3]:.4f}", flush=True)
print(f"  amp_C={best_x[4]:.4f} Cknee={best_x[5]:.4f}", flush=True)
print(f"  CV={cv*100:.2f}% RT={rt:.2e}", flush=True)

# Cusp profile
print(f"\n  Yellow cusp profile (h=85):", flush=True)
hr = 85*np.pi/180; ch, sh = np.cos(hr), np.sin(hr)
for Li in range(700, 1000, 10):
    L = Li/1000
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo+hi)/2
        lab = np.array([L, mid*ch, mid*sh])
        xyz = inv_enriched(lab, *params)
        rgb = M_Si_np @ xyz
        if np.all(rgb >= -0.001) and np.all(rgb <= 1.001): lo = mid
        else: hi = mid
    print(f"    L={L:.2f} C={lo:.4f}", flush=True)

# Cusp at key hues
print(f"\n  Cusp L at key hues:", flush=True)
for hd in [60, 75, 80, 85, 90, 95, 120, 180, 240]:
    cL, cC, drop = compute_cusp_and_cliff(params, hd)
    tag = " YELLOW" if 75 <= hd <= 95 else ""
    print(f"    h={hd:>3}: cusp_L={cL:.3f} C={cC:.4f} cliff={drop*100:.0f}%{tag}", flush=True)

# Save
enrichment = {
    "version": "v31d-chroma",
    "M1": V14_M1.tolist(), "M2": V14_M2.tolist(),
    "M1_inv": V14_M1i.tolist(), "M2_inv": V14_M2i.tolist(),
    "enrichment": {
        "type": "hue_LC_correction",
        "amp_L": float(best_x[0]), "h_center": float(best_x[1]),
        "h_width": float(best_x[2]), "L_knee": float(best_x[3]),
        "amp_C": float(best_x[4]), "C_knee": float(best_x[5])
    }
}
with open("/root/gen_v31d_chroma.json", "w") as f:
    json.dump(enrichment, f, indent=2)
print(f"\n  Saved: gen_v31d_chroma.json", flush=True)
