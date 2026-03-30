"""v28b: LMS Balance ONLY (no cusp_position — it's the bottleneck).

LMS balance alone is sufficient to fix cusp position.
v28 showed: bal constraint drives yL from 0.988→0.904 naturally.
This version runs 10x faster by removing cusp_position scan.
"""
import json, time, numpy as np, torch, os
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

def gpu_cv(M1, M2, pairs):
    try: M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
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
    SC = 1 + 0.045*C1; SH = 1 + 0.015*C1
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
    try: M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
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
    yl = scbrt((M_S @ torch.tensor([1.,1.,0.], device=device)) @ M1.T) @ M2.T
    yL, yC = yl[0].item(), (yl[1]**2 + yl[2]**2).sqrt().item()
    bx = M_S @ s2l(torch.tensor([0.,0.,1.], device=device))
    wx = M_S @ s2l(torch.tensor([1.,1.,1.], device=device))
    bl = scbrt(bx @ M1.T) @ M2.T; wl = scbrt(wx @ M1.T) @ M2.T
    ml = (bl + wl) / 2
    lc = ml @ M2i.T; lm = torch.sign(lc) * torch.abs(lc).pow(3.); mx = lm @ M1i.T
    ms = l2s((M_Si @ mx).clamp(0,1))
    bw = ms[1].item() / max(ms[0].item(), 0.01)
    ps = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]], dtype=torch.float64, device=device)
    pl = scbrt(s2l(ps) @ M_S.T @ M1.T) @ M2.T
    plr = (pl[:,0].max() - pl[:,0].min()).item()
    c1, c2 = torch.linalg.cond(M1).item(), torch.linalg.cond(M2).item()
    return {'yL': yL, 'yC': yC, 'bw': bw, 'plr': plr, 'c1': c1, 'c2': c2}

# ══════════════════════════════════════════════════════════
#  LMS Channel Balance — ROOT CAUSE constraint (CHEAP!)
# ══════════════════════════════════════════════════════════
# Pre-compute critical XYZ vectors (computed once on device)
_WHITE_XYZ = None
_YELLOW_XYZ = None
_ORANGE_XYZ = None
_LIME_XYZ = None

def init_reference_colors():
    global _WHITE_XYZ, _YELLOW_XYZ, _ORANGE_XYZ, _LIME_XYZ
    _WHITE_XYZ = M_S @ s2l(torch.tensor([1., 1., 1.], device=device))
    _YELLOW_XYZ = M_S @ s2l(torch.tensor([1., 1., 0.], device=device))
    _ORANGE_XYZ = M_S @ s2l(torch.tensor([1., 0.8, 0.], device=device))
    _LIME_XYZ = M_S @ s2l(torch.tensor([0.8, 1., 0.], device=device))

def gpu_lms_balance(M1):
    """Penalize max/min LMS channel ratio > threshold for critical transitions.
    This is the ROOT CAUSE fix: ensures cbrt compression acts evenly on all channels.
    Cost: ~3 matrix multiplies — negligible vs cv/mono."""
    penalty = 0.0
    for color_xyz in [_YELLOW_XYZ, _ORANGE_XYZ, _LIME_XYZ]:
        lms_c = color_xyz @ M1.T
        lms_w = _WHITE_XYZ @ M1.T
        delta = torch.abs(lms_w - lms_c).clamp(min=1e-6)
        ratio = delta.max() / delta.min()
        thr = 3.0
        if ratio.item() > thr:
            penalty += (ratio.item() - thr) ** 2
    return penalty / 3.0

# ── Parameterization ──
def ortho(s):
    sn = s / np.linalg.norm(s)
    v = np.array([1,0,0.]) if abs(sn[0]) < 0.9 else np.array([0,1,0.])
    e1 = v - np.dot(v, sn) * sn; e1 /= np.linalg.norm(e1)
    e2 = np.cross(sn, e1)
    return e1, e2

def unpack(x):
    M1 = np.zeros((3, 3))
    for i in range(3):
        M1[i,0] = x[2*i]; M1[i,1] = x[2*i+1]
        M1[i,2] = (1 - M1[i,0]*D65[0] - M1[i,1]*D65[1]) / D65[2]
    lms = M1 @ D65
    if np.any(lms <= 0): return None, None
    s = lms ** (1/3)
    if np.linalg.norm(s) < 1e-10: return None, None
    e1, e2 = ortho(s)
    M2 = np.zeros((3, 3)); M2[0] = x[6:9]
    Lw = M2[0] @ s
    if abs(Lw) < 1e-10: return None, None
    M2[0] /= Lw
    M2[1] = x[9]*e1 + x[10]*e2
    M2[2] = x[11]*e1 + x[12]*e2
    return M1, M2

def pack(M1, M2):
    x = np.zeros(13)
    for i in range(3): x[2*i] = M1[i,0]; x[2*i+1] = M1[i,1]
    x[6:9] = M2[0]
    lms = M1 @ D65; s = lms ** (1/3)
    e1, e2 = ortho(s)
    x[9] = M2[1] @ e1; x[10] = M2[1] @ e2
    x[11] = M2[2] @ e1; x[12] = M2[2] @ e2
    return x

def write_progress(phase, data):
    with open(f"progress_v28b_phase{phase}.json", "w") as f:
        json.dump(data, f, indent=2)

# ── Objective ──
def make_objective(pairs, lms_w=2.0):
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
                bal = gpu_lms_balance(M1t)

                pen = 0.
                if info['yC'] < 0.15: pen += (0.15 - info['yC'])**2 * 500
                if info['yL'] < 0.85: pen += (0.85 - info['yL'])**2 * 500  # don't go TOO low
                if info['bw'] < 1.15: pen += (1.15 - info['bw'])**2 * 500
                if info['plr'] < 0.40: pen += (0.40 - info['plr'])**2 * 500
                if c1 > 4: pen += (c1 - 4)**2 * 5
                if c2 > 12: pen += (c2 - 12)**2 * 5
                if cv > 0.25: pen += (cv - 0.25)**2 * 50

                return cv + 0.3*cv + 3.0*mono + 0.01*hue + lms_w*bal + pen
        except: return 999.
    return objective

def run_seed(name, x0, sigma, gens, popsize, pairs, obj_fn):
    best = {'loss': 999, 'x': x0.copy()}; ev = [0]; t0 = time.time(); lp = [0]
    def obj(x):
        loss = obj_fn(x); ev[0] += 1
        if loss < best['loss']:
            best['loss'] = loss; best['x'] = x.copy()
            now = time.time()
            if now - lp[0] > 15:
                lp[0] = now
                M1n, M2n = unpack(x)
                if M1n is not None:
                    M1t = torch.tensor(M1n, device=device)
                    M2t = torch.tensor(M2n, device=device)
                    with torch.no_grad():
                        cv = gpu_cv(M1t, M2t, pairs); mono = gpu_mono(M1t, M2t)
                        inf = gpu_info(M1t, M2t); bal = gpu_lms_balance(M1t)
                    print(f"    #{ev[0]:>6d} [{now-t0:5.0f}s] loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f} bal={bal:.3f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} cond=({inf['c1']:.1f},{inf['c2']:.1f})", flush=True)
        return loss
    opts = cma.CMAOptions()
    opts.set("maxiter", gens); opts.set("popsize", popsize); opts.set("tolfun", 1e-11); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj(x) for x in sols]; es.tell(sols, fits)
    el = time.time() - t0
    M1n, M2n = unpack(best['x'])
    M1t, M2t = torch.tensor(M1n, device=device), torch.tensor(M2n, device=device)
    with torch.no_grad():
        cv = gpu_cv(M1t, M2t, pairs); mono = gpu_mono(M1t, M2t)
        hue = gpu_hue(M1t, M2t); inf = gpu_info(M1t, M2t)
        bal = gpu_lms_balance(M1t)
    result = {'name': name, 'evals': ev[0], 'time': el, 'loss': best['loss'],
              'cv': cv, 'mono': mono, 'hue': hue, 'lms_bal': bal, **inf}
    print(f"  {name}: {ev[0]} evals {el:.0f}s | loss={best['loss']:.4f} CV={cv*100:.2f}% mono={mono:.4f} bal={bal:.3f} hue={hue:.1f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} cond=({inf['c1']:.1f},{inf['c2']:.1f})", flush=True)
    return best['x'], best['loss'], result

def save_ckpt(name, x, info):
    M1f, M2f = unpack(x)
    M1i, M2i = np.linalg.inv(M1f), np.linalg.inv(M2f)
    ckpt = {"version": f"v28b-{name}", "M1": M1f.tolist(), "M2": M2f.tolist(),
            "M1_inv": M1i.tolist(), "M2_inv": M2i.tolist(), "metrics": info}
    fn = f"gen_v28b_{name}.json"
    with open(fn, "w") as f: json.dump(ckpt, f, indent=2)
    print(f"  Saved: {fn}", flush=True)
    return fn

def print_yellow_boundary(M1f, M2f):
    M1t, M2t = torch.tensor(M1f, device=device), torch.tensor(M2f, device=device)
    M2it = torch.linalg.inv(M2t); M1it = torch.linalg.inv(M1t)
    print(f"\n  Yellow boundary (h=85deg):")
    h = 85 * 3.14159265 / 180; ch, sh = np.cos(h), np.sin(h)
    prev = None
    for Lv in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]:
        Cs = torch.linspace(0.001, 0.4, 80, device=device)
        lab = torch.stack([torch.full((80,), Lv, device=device), Cs*ch, Cs*sh], dim=1)
        lc = lab @ M2it.T; lm = torch.sign(lc) * torch.abs(lc).pow(3.)
        lin = (lm @ M1it.T) @ M_Si.T
        ok = (lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)
        mc = Cs[ok].max().item() if ok.any() else 0
        arrow = ""
        if prev is not None: arrow = " UP" if mc > prev + 0.001 else " DN" if mc < prev - 0.001 else " =="
        prev = mc
        print(f"    L={Lv:.2f} C={mc:.4f}{arrow}")

def print_lms(M1f):
    M1t = torch.tensor(M1f, device=device)
    delta = torch.abs(_WHITE_XYZ @ M1t.T - _YELLOW_XYZ @ M1t.T)
    r = delta.max().item() / max(delta.min().item(), 1e-6)
    print(f"\n  LMS delta (Y→W): [{delta[0].item():.4f}, {delta[1].item():.4f}, {delta[2].item():.4f}] ratio={r:.2f}")


def main():
    init_reference_colors()
    print(f"\n{'='*60}")
    print("  v28b: LMS BALANCE (fast, no cusp scan)")
    print(f"{'='*60}\n", flush=True)

    pairs = build_pairs()
    print(f"Training pairs: {pairs.shape[0]}")

    # Baseline LMS analysis
    print("\n--- LMS balance baselines ---")
    print_lms(V14_M1); print(f"  v14 bal penalty: {gpu_lms_balance(torch.tensor(V14_M1, device=device)):.4f}")
    print_lms(OK_M1); print(f"  OKLab bal penalty: {gpu_lms_balance(torch.tensor(OK_M1, device=device)):.4f}")

    # Sweep lms_weight to find best trade-off
    LMS_WEIGHTS = [1.0, 2.0, 4.0]
    all_best = []

    for lw in LMS_WEIGHTS:
        print(f"\n{'='*60}")
        print(f"  LMS_WEIGHT = {lw}")
        print(f"{'='*60}", flush=True)

        obj_fn = make_objective(pairs, lms_w=lw)

        # Verify baselines
        x_v14 = pack(V14_M1, V14_M2)
        l0 = obj_fn(x_v14)
        x_ok = pack(OK_M1, OK_M2)
        lok = obj_fn(x_ok)
        print(f"  v14 loss={l0:.4f}, OKLab loss={lok:.4f}")

        # Phase 1: 12 seeds x 200 gen x 64 pop (fast!)
        print(f"\n--- Phase 1: 12 seeds x 200 gen x 64 pop ---", flush=True)
        seeds = [
            ("v14", pack(V14_M1, V14_M2), 0.02),
            ("OKLab", pack(OK_M1, OK_M2), 0.02),
            ("mid", pack((V14_M1+OK_M1)/2, (V14_M2+OK_M2)/2), 0.03),
        ]
        rng = np.random.RandomState(2028)
        for i in range(9):
            x = np.zeros(13)
            base_M1 = 0.3*V14_M1 + 0.7*OK_M1
            for r in range(3):
                x[2*r] = base_M1[r,0] + rng.randn() * 0.15
                x[2*r+1] = base_M1[r,1] + rng.randn() * 0.15
            x[6:9] = (V14_M2[0] + OK_M2[0]) / 2 + rng.randn(3) * 0.1
            x[9:13] = rng.randn(4) * 0.8
            seeds.append((f"rnd{i}", x, 0.05))

        p1 = []
        for name, x0, sigma in seeds:
            xb, loss, info = run_seed(name, x0, sigma, gens=200, popsize=64, pairs=pairs, obj_fn=obj_fn)
            p1.append((name, xb, loss, info))

        p1.sort(key=lambda r: r[2])
        print(f"\n--- Phase 1 ranking (lms_w={lw}) ---", flush=True)
        for i, (name, _, loss, info) in enumerate(p1):
            flag = "***" if i < 3 else ""
            print(f"  {i+1:>2}. {name:>6}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} bal={info['lms_bal']:.3f} yL={info['yL']:.3f} yC={info['yC']:.3f} {flag}")

        # Phase 2: Top 3 x 800 gen x 96 pop
        print(f"\n--- Phase 2: Top 3 x 800 gen x 96 pop ---", flush=True)
        p2 = []
        for name, x0, _, _ in p1[:3]:
            xb, loss, info = run_seed(f"{name}+", x0, sigma=0.005, gens=800, popsize=96, pairs=pairs, obj_fn=obj_fn)
            p2.append((f"{name}+", xb, loss, info))

        p2.sort(key=lambda r: r[2])
        print(f"\n--- Phase 2 ranking (lms_w={lw}) ---", flush=True)
        for i, (name, _, loss, info) in enumerate(p2):
            print(f"  {i+1}. {name}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} bal={info['lms_bal']:.3f} yL={info['yL']:.3f} yC={info['yC']:.3f}")

        # Phase 3: Winner polish
        wn, wx, _, _ = p2[0]
        print(f"\n--- Phase 3: '{wn}' x 500 gen x 128 pop ---", flush=True)
        fx, fl, fi = run_seed("FINAL", wx, sigma=0.002, gens=500, popsize=128, pairs=pairs, obj_fn=obj_fn)

        save_ckpt(f"lw{lw:.0f}", fx, fi)
        all_best.append((lw, fx, fl, fi))

        M1f, M2f = unpack(fx)
        print_yellow_boundary(M1f, M2f)
        print_lms(M1f)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  WEIGHT SWEEP SUMMARY")
    print(f"{'='*60}")
    for lw, _, loss, info in all_best:
        print(f"  lms_w={lw:.1f}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} bal={info['lms_bal']:.3f} yL={info['yL']:.3f} yC={info['yC']:.3f} bw={info['bw']:.2f}")

    # Pick best overall (lowest cv+mono with bal<0.01)
    viable = [(lw, x, l, i) for lw, x, l, i in all_best if i['lms_bal'] < 0.01]
    if viable:
        viable.sort(key=lambda r: r[3]['cv'])
        best_lw, best_x, best_l, best_i = viable[0]
        print(f"\n  WINNER: lms_w={best_lw} CV={best_i['cv']*100:.2f}% yL={best_i['yL']:.3f}")
        save_ckpt("winner", best_x, best_i)
        M1f, M2f = unpack(best_x)
        M1i, M2i = np.linalg.inv(M1f), np.linalg.inv(M2f)
        print(f"\nM1 =")
        for r in M1f: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
        print(f"M2 =")
        for r in M2f: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    else:
        print("\n  No viable solution with bal < 0.01!")

    write_progress("final", [{'lms_w': lw, 'loss': l, 'metrics': i} for lw, _, l, i in all_best])


if __name__ == "__main__":
    main()
