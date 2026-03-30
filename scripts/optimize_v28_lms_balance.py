"""v28: LMS Channel Balance constraint — targeting ROOT CAUSE of cusp shelf.

Problem: v14's M1 maps yellow→white delta to LMS as [0.073, 0.116, 0.616].
S channel is 8x dominant. After cbrt, L/M channels barely change →
M2 produces constant chroma → shelf/cliff at yellow cusp.

Fix: Penalize imbalanced LMS channel response for critical transitions.
max(|delta_lms|)/min(|delta_lms|) > threshold → penalty.

Based on v27_fixed.py with proven soft-constraint architecture.
"""
import json, time, numpy as np, torch, os, sys
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
import cma

D65 = np.array([0.95047, 1.0, 1.08883])
M_S = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], device=device)
M_Si = torch.linalg.inv(M_S)
D65_T = torch.tensor(D65, device=device)

# References
V14_M1 = np.array([
    [0.7583761294836658, 0.38380162590825084, -0.09608055040602373],
    [0.12671393631532843, 0.8421628149123207, 0.03434823621506485],
    [0.07639223722200054, 0.258943526275451, 0.6139139663787314]
])
V14_M2 = np.array([
    [0.10058070589596230, 1.01558970993941444, -0.11617041583537688],
    [2.36157646996164416, -2.44099737506293479, 0.07942090510129070],
    [0.04565327074453784, 0.81875488445424471, -0.86440815519878267]
])
OK_M1s = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
])
OK_M1 = OK_M1s @ np.linalg.inv(np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
]))
OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
])

def scbrt(x): return torch.sign(x) * torch.abs(x).pow(1./3.)
def s2l(c): return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))
def l2s(c): return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-10).pow(1./2.4) - 0.055)

# ── Training pairs ──
def build_pairs():
    pairs = []
    prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1, len(prims)):
            pairs.append((prims[i], prims[j]))
    for g1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for g2 in [g1+0.2, g1+0.4]:
            if g2 <= 1.0: pairs.append(([g1]*3, [g2]*3))
    rng = np.random.RandomState(42)
    for _ in range(80):
        pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))
    pt = torch.zeros(len(pairs), 2, 3, device=device)
    for i, (c1, c2) in enumerate(pairs):
        pt[i,0] = M_S @ s2l(torch.tensor(c1, device=device))
        pt[i,1] = M_S @ s2l(torch.tensor(c2, device=device))
    return pt

N_ST = 25
T_ST = torch.linspace(0, 1, N_ST+1, device=device)

# ── GPU metrics ──
def gpu_cv(M1, M2, pairs):
    try:
        M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    except: return 0.99
    N = pairs.shape[0]
    l1 = scbrt(pairs[:,0] @ M1.T) @ M2.T
    l2 = scbrt(pairs[:,1] @ M1.T) @ M2.T
    t = T_ST.view(1,-1,1)
    labs = l1.unsqueeze(1) + t * (l2 - l1).unsqueeze(1)
    lf = labs.reshape(-1, 3)
    lc = lf @ M2i.T
    lm = torch.sign(lc) * torch.abs(lc).pow(3.)
    lin = (lm @ M1i.T) @ M_Si.T
    s8 = (l2s(lin.clamp(0,1)) * 255).round() / 255.
    xb = s2l(s8) @ M_S.T
    r = xb.clamp(min=1e-10) / D65_T
    f = torch.where(r > 0.008856, r.pow(1./3.), 7.787*r + 16./116.)
    cl = torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1).reshape(N, N_ST+1, 3)
    c1, c2 = cl[:,:-1], cl[:,1:]
    dL = c2[...,0] - c1[...,0]
    C1 = (c1[...,1]**2 + c1[...,2]**2).sqrt()
    C2 = (c2[...,1]**2 + c2[...,2]**2).sqrt()
    dC = C2 - C1
    dH = ((c2[...,1]-c1[...,1])**2 + (c2[...,2]-c1[...,2])**2 - dC**2).clamp(min=0).sqrt()
    SL = 1 + 0.015*(c1[...,0]-50)**2 / (20+(c1[...,0]-50)**2).sqrt()
    SC = 1 + 0.045*C1
    SH = 1 + 0.015*C1
    de = ((dL/SL)**2 + (dC/SC)**2 + (dH/SH)**2).sqrt()
    md = de.mean(1); sd = de.std(1); v = md > 0.001
    cvs = torch.where(v, sd/md, torch.zeros_like(md))
    return cvs[v].mean().item() if v.any() else 0.99

Lf = torch.linspace(0.80, 0.998, 50, device=device)
Cf = torch.linspace(0.001, 0.4, 50, device=device)
Le = Lf.view(50,1).expand(50,50)
Ce = Cf.view(1,50).expand(50,50)
Cf_e = Cf.view(1,50).expand(50,50)
MONO_H = [h * 3.14159265 / 180 for h in range(60, 121, 5)]

def gpu_mono(M1, M2):
    try:
        M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    except: return 99.
    pen = 0.
    for hr in MONO_H:
        ch, sh = np.cos(hr), np.sin(hr)
        lab = torch.stack([Le, Ce*ch, Ce*sh], dim=-1).reshape(-1, 3)
        lc = lab @ M2i.T
        lm = torch.sign(lc) * torch.abs(lc).pow(3.)
        lin = (lm @ M1i.T) @ M_Si.T
        ok = ((lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)).reshape(50, 50)
        mc, _ = torch.where(ok, Cf_e, torch.zeros(50,50, device=device)).max(dim=1)
        ci = mc.argmax().item()
        cL = Lf[ci].item()
        if cL > 0.95: pen += (cL - 0.95)**2 * 200
        d = mc[1:] - mc[:-1]
        p = d[d > 0.002]
        if p.numel() > 0: pen += p.pow(2).sum().item() * 100
    return pen / len(MONO_H)

def gpu_hue(M1, M2):
    prims = torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]], dtype=torch.float64, device=device)
    exp = torch.tensor([0, 60, 120, 180, 240, 300], dtype=torch.float64, device=device)
    lab = scbrt(s2l(prims) @ M_S.T @ M1.T) @ M2.T
    h = torch.atan2(lab[:,2], lab[:,1]) * (180/3.14159265) % 360
    dh = h - exp
    dh = torch.where(dh > 180, dh - 360, dh)
    dh = torch.where(dh < -180, dh + 360, dh)
    return (dh**2).mean().item()

def gpu_info(M1, M2):
    M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    # Yellow
    yl = scbrt((M_S @ torch.tensor([1.,1.,0.], device=device)) @ M1.T) @ M2.T
    yL, yC = yl[0].item(), (yl[1]**2 + yl[2]**2).sqrt().item()
    # Blue->White midpoint
    bx = M_S @ s2l(torch.tensor([0.,0.,1.], device=device))
    wx = M_S @ s2l(torch.tensor([1.,1.,1.], device=device))
    bl = scbrt(bx @ M1.T) @ M2.T
    wl = scbrt(wx @ M1.T) @ M2.T
    ml = (bl + wl) / 2
    lc = ml @ M2i.T
    lm = torch.sign(lc) * torch.abs(lc).pow(3.)
    mx = lm @ M1i.T
    ms = l2s((M_Si @ mx).clamp(0,1))
    bw = ms[1].item() / max(ms[0].item(), 0.01)
    # Primary L range
    ps = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]], dtype=torch.float64, device=device)
    pl = scbrt(s2l(ps) @ M_S.T @ M1.T) @ M2.T
    plr = (pl[:,0].max() - pl[:,0].min()).item()
    c1, c2 = torch.linalg.cond(M1).item(), torch.linalg.cond(M2).item()
    return {'yL': yL, 'yC': yC, 'bw': bw, 'plr': plr, 'c1': c1, 'c2': c2}

# ══════════════════════════════════════════════════════════════
#  NEW: LMS Channel Balance — ROOT CAUSE constraint
# ══════════════════════════════════════════════════════════════
def gpu_lms_balance(M1):
    """Penalize imbalanced LMS channel response for critical transitions.

    Root cause: If M1 maps a color→white delta to LMS with one channel
    8x dominant, after cbrt the other channels barely change, producing
    a shelf/cliff in the gamut boundary.

    We check yellow→white and nearby hues (the problem region).
    Target: max/min ratio < 3.0 (OKLab is ~2.5).
    """
    # Critical XYZ vectors: colors that transition through the cusp region
    # Yellow (#FFFF00) linear sRGB → XYZ
    yellow_xyz = M_S @ s2l(torch.tensor([1., 1., 0.], device=device))
    # White (#FFFFFF) linear sRGB → XYZ
    white_xyz = M_S @ s2l(torch.tensor([1., 1., 1.], device=device))
    # Orange-ish (#FFcc00) → XYZ
    orange_xyz = M_S @ s2l(torch.tensor([1., 0.8, 0.], device=device))
    # Lime (#ccFF00) → XYZ
    lime_xyz = M_S @ s2l(torch.tensor([0.8, 1., 0.], device=device))

    penalty = 0.0

    for color_xyz in [yellow_xyz, orange_xyz, lime_xyz]:
        # LMS at color and white
        lms_color = color_xyz @ M1.T
        lms_white = white_xyz @ M1.T

        # Delta LMS (what changes from color→white)
        delta = torch.abs(lms_white - lms_color)

        # Avoid division by zero: use max(delta_i, epsilon)
        delta_safe = delta.clamp(min=1e-6)

        # Ratio: max channel delta / min channel delta
        ratio = delta_safe.max() / delta_safe.min()

        # Soft penalty for ratio > 3.0
        # OKLab achieves ~2.5, v14 has ~8.4
        threshold = 3.0
        if ratio.item() > threshold:
            penalty += (ratio.item() - threshold) ** 2

    return penalty / 3.0  # average over checked transitions


def gpu_cusp_position(M1, M2):
    """Penalize cusp L too far from OKLab's cusp L.

    OKLab cusp at h≈85° is around L=0.84-0.85.
    v14 cusp is at L=0.988 — way too high, causing the cliff.
    Target: cusp L within 0.05 of OKLab.
    """
    try:
        M2i = torch.linalg.inv(M2)
        M1i = torch.linalg.inv(M1)
    except: return 99.

    # OKLab cusp L values at key hues (pre-computed reference)
    # These are approximate OKLab sRGB gamut cusp lightness values
    ok_cusps = {
        70: 0.79, 75: 0.82, 80: 0.84, 85: 0.84,
        90: 0.85, 95: 0.86, 100: 0.83, 105: 0.80,
        110: 0.77, 115: 0.75, 120: 0.73
    }

    penalty = 0.0
    n = 0

    for hue_deg, ok_cusp_L in ok_cusps.items():
        hr = hue_deg * 3.14159265 / 180
        ch, sh = np.cos(hr), np.sin(hr)

        # Find cusp: scan L values, find max chroma in gamut
        Ls = torch.linspace(0.3, 0.999, 100, device=device)
        Cs = torch.linspace(0.001, 0.4, 80, device=device)

        best_L = 0.5
        best_C = 0.0

        for Li in range(100):
            Lv = Ls[Li].item()
            lab = torch.stack([
                torch.full((80,), Lv, device=device),
                Cs * ch,
                Cs * sh
            ], dim=1)
            lc = lab @ M2i.T
            lm = torch.sign(lc) * torch.abs(lc).pow(3.)
            lin = (lm @ M1i.T) @ M_Si.T
            ok = (lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)
            if ok.any():
                mc = Cs[ok].max().item()
                if mc > best_C:
                    best_C = mc
                    best_L = Lv

        # Penalize difference from OKLab cusp
        diff = abs(best_L - ok_cusp_L)
        if diff > 0.03:  # allow 0.03 tolerance
            penalty += (diff - 0.03) ** 2
        n += 1

    return penalty / max(n, 1)


# ── Parameterization ──
def ortho(s):
    sn = s / np.linalg.norm(s)
    v = np.array([1,0,0.]) if abs(sn[0]) < 0.9 else np.array([0,1,0.])
    e1 = v - np.dot(v, sn) * sn
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(sn, e1)
    return e1, e2

def unpack(x):
    M1 = np.zeros((3, 3))
    for i in range(3):
        M1[i,0] = x[2*i]
        M1[i,1] = x[2*i+1]
        M1[i,2] = (1 - M1[i,0]*D65[0] - M1[i,1]*D65[1]) / D65[2]
    lms = M1 @ D65
    if np.any(lms <= 0): return None, None
    s = lms ** (1/3)
    if np.linalg.norm(s) < 1e-10: return None, None
    e1, e2 = ortho(s)
    M2 = np.zeros((3, 3))
    M2[0] = x[6:9]
    Lw = M2[0] @ s
    if abs(Lw) < 1e-10: return None, None
    M2[0] /= Lw
    M2[1] = x[9]*e1 + x[10]*e2
    M2[2] = x[11]*e1 + x[12]*e2
    return M1, M2

def pack(M1, M2):
    x = np.zeros(13)
    for i in range(3):
        x[2*i] = M1[i,0]
        x[2*i+1] = M1[i,1]
    x[6:9] = M2[0]
    lms = M1 @ D65
    s = lms ** (1/3)
    e1, e2 = ortho(s)
    x[9] = M2[1] @ e1; x[10] = M2[1] @ e2
    x[11] = M2[2] @ e1; x[12] = M2[2] @ e2
    return x

def write_progress(phase, data):
    with open(f"progress_phase{phase}.json", "w") as f:
        json.dump(data, f, indent=2)

# ── Objective ──
def make_objective(pairs, lms_weight=2.0, cusp_weight=5.0):
    """Soft constraints + LMS balance + cusp position."""

    # Pre-compute OKLab LMS balance for reference
    OK_M1t = torch.tensor(OK_M1, device=device)
    ok_bal = gpu_lms_balance(OK_M1t)
    print(f"  OKLab LMS balance penalty: {ok_bal:.4f}")

    V14_M1t = torch.tensor(V14_M1, device=device)
    v14_bal = gpu_lms_balance(V14_M1t)
    print(f"  v14   LMS balance penalty: {v14_bal:.4f}")

    call_count = [0]

    def objective(x):
        try:
            M1n, M2n = unpack(x)
            if M1n is None: return 999.
            M1t = torch.tensor(M1n, device=device)
            M2t = torch.tensor(M2n, device=device)

            with torch.no_grad():
                c1, c2 = torch.linalg.cond(M1t).item(), torch.linalg.cond(M2t).item()
                if c1 > 20 or c2 > 30: return 999.

                info = gpu_info(M1t, M2t)
                cv = gpu_cv(M1t, M2t, pairs)
                mono = gpu_mono(M1t, M2t)
                hue = gpu_hue(M1t, M2t)

                # NEW: LMS balance (cheap, only M1 needed)
                lms_bal = gpu_lms_balance(M1t)

                # Soft penalties
                pen = 0.
                if info['yC'] < 0.15: pen += (0.15 - info['yC'])**2 * 500
                if info['yL'] < 0.90: pen += (0.90 - info['yL'])**2 * 500
                if info['bw'] < 1.15: pen += (1.15 - info['bw'])**2 * 500
                if info['plr'] < 0.40: pen += (0.40 - info['plr'])**2 * 500
                if c1 > 4: pen += (c1 - 4)**2 * 5
                if c2 > 12: pen += (c2 - 12)**2 * 5
                if cv > 0.25: pen += (cv - 0.25)**2 * 50

                # Cusp position (expensive — only check every 20th eval initially)
                call_count[0] += 1
                cusp_pen = 0.0
                if call_count[0] % 10 == 0 or pen == 0:
                    cusp_pen = gpu_cusp_position(M1t, M2t)

                loss = (cv + 0.3*cv
                        + 3.0*mono
                        + 0.01*hue
                        + lms_weight * lms_bal
                        + cusp_weight * cusp_pen
                        + pen)
                return loss
        except: return 999.
    return objective


def run_seed(name, x0, sigma, gens, popsize, pairs, obj_fn):
    best = {'loss': 999, 'x': x0.copy()}
    ev = [0]
    t0 = time.time()
    last_print = [0]

    def obj(x):
        loss = obj_fn(x)
        ev[0] += 1
        if loss < best['loss']:
            best['loss'] = loss
            best['x'] = x.copy()
            now = time.time()
            if now - last_print[0] > 10:
                last_print[0] = now
                M1n, M2n = unpack(x)
                if M1n is not None:
                    M1t = torch.tensor(M1n, device=device)
                    M2t = torch.tensor(M2n, device=device)
                    with torch.no_grad():
                        cv = gpu_cv(M1t, M2t, pairs)
                        mono = gpu_mono(M1t, M2t)
                        inf = gpu_info(M1t, M2t)
                        bal = gpu_lms_balance(M1t)
                    print(f"    #{ev[0]:>6d} [{now-t0:5.0f}s] loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f} bal={bal:.3f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} cond=({inf['c1']:.1f},{inf['c2']:.1f})", flush=True)
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", gens)
    opts.set("popsize", popsize)
    opts.set("tolfun", 1e-11)
    opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        sols = es.ask()
        fits = [obj(x) for x in sols]
        es.tell(sols, fits)

    el = time.time() - t0
    M1n, M2n = unpack(best['x'])
    M1t = torch.tensor(M1n, device=device)
    M2t = torch.tensor(M2n, device=device)
    with torch.no_grad():
        cv = gpu_cv(M1t, M2t, pairs)
        mono = gpu_mono(M1t, M2t)
        hue = gpu_hue(M1t, M2t)
        inf = gpu_info(M1t, M2t)
        bal = gpu_lms_balance(M1t)

    result = {
        'name': name, 'evals': ev[0], 'time': el, 'loss': best['loss'],
        'cv': cv, 'mono': mono, 'hue': hue, 'lms_bal': bal, **inf
    }
    print(f"  {name}: {ev[0]} evals {el:.0f}s | loss={best['loss']:.4f} CV={cv*100:.2f}% mono={mono:.4f} bal={bal:.3f} hue={hue:.1f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} cond=({inf['c1']:.1f},{inf['c2']:.1f})", flush=True)
    return best['x'], best['loss'], result


def save_checkpoint(name, x, info):
    M1f, M2f = unpack(x)
    M1i, M2i = np.linalg.inv(M1f), np.linalg.inv(M2f)
    ckpt = {
        "version": "v28-lms-balance",
        "M1": M1f.tolist(), "M2": M2f.tolist(),
        "M1_inv": M1i.tolist(), "M2_inv": M2i.tolist(),
        "metrics": info
    }
    fname = f"gen_v28_{name}.json"
    with open(fname, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"  Saved: {fname}", flush=True)
    return fname


def print_yellow_boundary(M1f, M2f):
    M1t = torch.tensor(M1f, device=device)
    M2t = torch.tensor(M2f, device=device)
    M1it, M2it = torch.linalg.inv(M1t), torch.linalg.inv(M2t)

    print(f"\n  Yellow boundary (h=85deg):")
    h = 85 * 3.14159265 / 180
    ch, sh = np.cos(h), np.sin(h)
    prev = None
    for Lv in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]:
        Cs = torch.linspace(0.001, 0.4, 80, device=device)
        Le2 = torch.full((80,), Lv, device=device)
        lab = torch.stack([Le2, Cs*ch, Cs*sh], dim=1)
        lc = lab @ M2it.T
        lm = torch.sign(lc) * torch.abs(lc).pow(3.)
        lin = (lm @ M1it.T) @ M_Si.T
        ok = (lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)
        mc = Cs[ok].max().item() if ok.any() else 0
        arrow = ""
        if prev is not None:
            arrow = " UP" if mc > prev + 0.001 else " DN" if mc < prev - 0.001 else " =="
        prev = mc
        print(f"    L={Lv:.2f} C={mc:.4f}{arrow}")


def print_lms_analysis(M1f):
    """Print LMS channel balance analysis."""
    M1t = torch.tensor(M1f, device=device)
    yellow_xyz = M_S @ s2l(torch.tensor([1., 1., 0.], device=device))
    white_xyz = M_S @ s2l(torch.tensor([1., 1., 1.], device=device))

    lms_y = yellow_xyz @ M1t.T
    lms_w = white_xyz @ M1t.T
    delta = torch.abs(lms_w - lms_y)

    print(f"\n  LMS Channel Analysis (Yellow→White):")
    print(f"    LMS_yellow = [{lms_y[0].item():.4f}, {lms_y[1].item():.4f}, {lms_y[2].item():.4f}]")
    print(f"    LMS_white  = [{lms_w[0].item():.4f}, {lms_w[1].item():.4f}, {lms_w[2].item():.4f}]")
    print(f"    delta_LMS  = [{delta[0].item():.4f}, {delta[1].item():.4f}, {delta[2].item():.4f}]")
    ratio = delta.max().item() / max(delta.min().item(), 1e-6)
    print(f"    max/min ratio = {ratio:.2f} (target: <3.0, v14: ~8.4, OKLab: ~2.5)")


def main():
    print(f"\n{'='*60}")
    print("  v28: LMS CHANNEL BALANCE — ROOT CAUSE FIX")
    print("  Targets imbalanced M1 channel response")
    print(f"{'='*60}\n", flush=True)

    pairs = build_pairs()
    print(f"Training pairs: {pairs.shape[0]}")

    # ── Baseline analysis ──
    print("\n--- LMS Balance Analysis: Baselines ---")
    print_lms_analysis(V14_M1)
    print("\n  --- OKLab ---")
    print_lms_analysis(OK_M1)

    obj_fn = make_objective(pairs, lms_weight=2.0, cusp_weight=5.0)

    # Verify baselines
    print("\n--- Baseline verification ---", flush=True)
    x_v14 = pack(V14_M1, V14_M2)
    loss0 = obj_fn(x_v14)
    M1c, M2c = unpack(x_v14)
    M1t, M2t = torch.tensor(M1c, device=device), torch.tensor(M2c, device=device)
    with torch.no_grad():
        cv0 = gpu_cv(M1t, M2t, pairs)
        mono0 = gpu_mono(M1t, M2t)
        inf0 = gpu_info(M1t, M2t)
        bal0 = gpu_lms_balance(M1t)
    print(f"  v14: CV={cv0*100:.2f}% mono={mono0:.4f} bal={bal0:.3f} yL={inf0['yL']:.3f} yC={inf0['yC']:.3f} loss={loss0:.4f}")

    x_ok = pack(OK_M1, OK_M2)
    loss_ok = obj_fn(x_ok)
    M1c2, M2c2 = unpack(x_ok)
    M1t2, M2t2 = torch.tensor(M1c2, device=device), torch.tensor(M2c2, device=device)
    with torch.no_grad():
        cv_ok = gpu_cv(M1t2, M2t2, pairs)
        bal_ok = gpu_lms_balance(M1t2)
    print(f"  OKLab: CV={cv_ok*100:.2f}% bal={bal_ok:.3f} loss={loss_ok:.4f}")
    print(flush=True)

    # ── Phase 1: 12 seeds × 300 gen × 64 pop ──
    print("--- Phase 1: 12 seeds x 300 gen x 64 pop ---", flush=True)
    seeds = [
        ("v14", pack(V14_M1, V14_M2), 0.02),
        ("OKLab", pack(OK_M1, OK_M2), 0.02),
        ("mid", pack((V14_M1+OK_M1)/2, (V14_M2+OK_M2)/2), 0.03),
    ]

    rng = np.random.RandomState(2028)
    for i in range(9):
        x = np.zeros(13)
        # Bias toward OKLab-like M1 (better LMS balance)
        base_M1 = 0.3*V14_M1 + 0.7*OK_M1  # lean toward OKLab structure
        for r in range(3):
            x[2*r] = base_M1[r,0] + rng.randn() * 0.15
            x[2*r+1] = base_M1[r,1] + rng.randn() * 0.15
        x[6:9] = (V14_M2[0] + OK_M2[0]) / 2 + rng.randn(3) * 0.1
        x[9:13] = rng.randn(4) * 0.8
        seeds.append((f"rnd{i}", x, 0.05))

    p1_results = []
    for name, x0, sigma in seeds:
        xb, loss, info = run_seed(name, x0, sigma, gens=300, popsize=64, pairs=pairs, obj_fn=obj_fn)
        p1_results.append((name, xb, loss, info))

    p1_results.sort(key=lambda r: r[2])
    print(f"\n--- Phase 1 ranking ---", flush=True)
    for i, (name, _, loss, info) in enumerate(p1_results):
        flag = "***" if i < 3 else ""
        print(f"  {i+1:>2}. {name:>6}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} bal={info['lms_bal']:.3f} hue={info['hue']:.1f} yL={info['yL']:.3f} yC={info['yC']:.3f} {flag}")
    write_progress(1, [{'name': n, 'loss': l, 'metrics': m} for n, _, l, m in p1_results])

    # Save phase 1 best
    save_checkpoint("p1_best", p1_results[0][1], p1_results[0][3])
    print(flush=True)

    # ── Phase 2: Top 3 × 1500 gen × 96 pop ──
    print("--- Phase 2: Top 3 x 1500 gen x 96 pop ---", flush=True)
    p2_results = []
    for name, x0, _, _ in p1_results[:3]:
        xb, loss, info = run_seed(f"{name}+", x0, sigma=0.005, gens=1500, popsize=96, pairs=pairs, obj_fn=obj_fn)
        p2_results.append((f"{name}+", xb, loss, info))

    p2_results.sort(key=lambda r: r[2])
    print(f"\n--- Phase 2 ranking ---", flush=True)
    for i, (name, _, loss, info) in enumerate(p2_results):
        print(f"  {i+1}. {name}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} bal={info['lms_bal']:.3f} hue={info['hue']:.1f} yL={info['yL']:.3f} yC={info['yC']:.3f}")
    write_progress(2, [{'name': n, 'loss': l, 'metrics': m} for n, _, l, m in p2_results])

    # Save phase 2 best
    save_checkpoint("p2_best", p2_results[0][1], p2_results[0][3])
    print(flush=True)

    # ── Phase 3: Winner polish ──
    wn, wx, _, _ = p2_results[0]
    print(f"--- Phase 3: '{wn}' x 800 gen x 128 pop, sigma=0.002 ---", flush=True)
    fx, fl, fi = run_seed("FINAL", wx, sigma=0.002, gens=800, popsize=128, pairs=pairs, obj_fn=obj_fn)

    M1f, M2f = unpack(fx)
    M1i, M2i = np.linalg.inv(M1f), np.linalg.inv(M2f)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT")
    print(f"{'='*60}")
    print(f"  CV={fi['cv']*100:.2f}% mono={fi['mono']:.4f} hue={fi['hue']:.1f}")
    print(f"  LMS balance={fi['lms_bal']:.4f} (v14: ~5.0, OKLab: ~0.0)")
    print(f"  yL={fi['yL']:.3f} yC={fi['yC']:.3f} bw={fi['bw']:.2f} plr={fi['plr']:.3f}")
    print(f"  cond=({fi['c1']:.1f},{fi['c2']:.1f})")

    print_yellow_boundary(M1f, M2f)
    print_lms_analysis(M1f)

    print(f"\nM1 =")
    for r in M1f: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print(f"M2 =")
    for r in M2f: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print(f"M1_inv =")
    for r in M1i: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print(f"M2_inv =")
    for r in M2i: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")

    save_checkpoint("final", fx, fi)
    write_progress(3, fi)


if __name__ == "__main__":
    main()
