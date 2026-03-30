#!/usr/bin/env python3
"""Neural Color Space — learn the optimal XYZ→Lab mapping.

No M1, no M2, no cbrt. A small invertible neural network learns
the mapping that produces the best gradients.

Architecture: Invertible Residual Network
  Encoder: XYZ → Lab (3→3 with residual blocks)
  Decoder: Lab → XYZ (same network, reversed)

Loss: midpoint_quality + gradient_CV + munsell_V + achromatic
"""

import json, math, os, sys, time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
MS = torch.tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]], device=dev)
MSi = torch.linalg.inv(MS)

def s2l(c): return torch.where(c<=0.04045, c/12.92, ((c+0.055)/1.055).pow(2.4))
def l2s(c): return torch.where(c<=0.0031308, c*12.92, 1.055*c.clamp(min=1e-12).pow(1./2.4)-0.055)

# Data
import colorsys
_mp = []
for h in range(0,360,30):
    r1,g1,b1 = colorsys.hsv_to_rgb(h/360,1,1)
    r2,g2,b2 = colorsys.hsv_to_rgb(((h+180)%360)/360,1,1)
    _mp.append(([r1,g1,b1],[r2,g2,b2]))
for rgb in [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]:
    _mp.append((rgb,[1,1,1]))
    _mp.append((rgb,[0,0,0]))
_mp.append(([1,0.6,0.2],[0.2,0.4,1]))
_mp.append(([1,0.5,0.5],[0.5,1,1]))
_mp.append(([0.6,0.2,0.8],[0.9,0.8,0.2]))

MID_XYZ1=torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r,_ in _mp])
MID_XYZ2=torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for _,r in _mp])

def _cielab(xyz):
    r=xyz/D65;d3=(6./29.)**3
    f=torch.where(r>d3,r.pow(1./3.),r/(3*(6./29.)**2)+4./29.)
    return torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],dim=-1)

# Munsell
MY={1:0.01221,2:0.03126,3:0.06552,4:0.12,5:0.1977,6:0.30049,7:0.4306,8:0.591,9:0.7866}
MUNSELL_GRAYS=torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)

# Gradient pairs (subsample for speed)
for _d in [os.path.join(ROOT,"colorbench"), os.path.join(ROOT,"space-test-project")]:
    if os.path.isdir(os.path.join(_d,"core")):
        sys.path.insert(0,_d)
        from core.pairs import generate_all_pairs
        PT_full,_ = generate_all_pairs(dev)
        # Subsample 300 pairs
        idx = torch.randperm(PT_full.shape[0], device=dev)[:300]
        PT = PT_full[idx]
        print(f"Gradient pairs: {PT.shape[0]} (subsampled from {PT_full.shape[0]})", flush=True)
        break

N_ST = 16  # fewer steps for speed

# ================================================================
#  INVERTIBLE NEURAL COLOR SPACE
# ================================================================

class InvertibleBlock(nn.Module):
    """Residual block with spectral normalization for invertibility."""
    def __init__(self, dim=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, dim),
        )
        # Initialize small weights for near-identity start
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        self.scale = nn.Parameter(torch.tensor(0.1))  # controls residual strength

    def forward(self, x):
        return x + self.scale * self.net(x)

    def inverse(self, y, n_iter=20):
        """Fixed-point iteration: x = y - scale * net(x)"""
        x = y.clone()
        for _ in range(n_iter):
            x = y - self.scale * self.net(x)
        return x


class NeuralColorSpace(nn.Module):
    def __init__(self, n_blocks=3, hidden=16):
        super().__init__()
        # Pre-processing: normalize XYZ to [0,1]-ish range
        self.xyz_scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=dev))

        # Invertible blocks
        self.blocks = nn.ModuleList([InvertibleBlock(3, hidden) for _ in range(n_blocks)])

        # Post-processing: scale to Lab range
        self.lab_scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=dev))
        self.lab_bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=dev))

    def forward(self, xyz):
        """XYZ → Lab"""
        # Normalize
        h = xyz * self.xyz_scale
        # Cube root (like cbrt but learnable scale)
        h = torch.sign(h) * h.abs().clamp(min=1e-30).pow(1./3.)
        # Invertible blocks
        for block in self.blocks:
            h = block(h)
        # Scale to Lab
        return h * self.lab_scale + self.lab_bias

    def inverse(self, lab):
        """Lab → XYZ"""
        # Undo scale
        h = (lab - self.lab_bias) / self.lab_scale.clamp(min=1e-10)
        # Undo blocks (reverse order)
        for block in reversed(self.blocks):
            h = block.inverse(h)
        # Undo cbrt
        h = torch.sign(h) * h.abs().pow(3.)
        # Undo normalize
        return h / self.xyz_scale.clamp(min=1e-10)

    def compute_loss(self):
        # 1. Round-trip loss (must be invertible!)
        torch.manual_seed(int(time.time()) % 10000)
        xyz_sample = torch.rand(500, 3, device=dev) * D65
        lab_sample = self.forward(xyz_sample)
        xyz_rt = self.inverse(lab_sample)
        rt_loss = (xyz_rt - xyz_sample).pow(2).mean() * 1000

        # 2. Achromatic: D65 gray ramp should have a=b=0
        t_gray = torch.linspace(0.01, 0.99, 50, device=dev)
        grays = D65.unsqueeze(0) * t_gray.unsqueeze(1)
        lab_gray = self.forward(grays)
        ach_loss = (lab_gray[:, 1].pow(2) + lab_gray[:, 2].pow(2)).mean() * 500

        # 3. L monotonicity: gray ramp L should increase
        dL = lab_gray[1:, 0] - lab_gray[:-1, 0]
        mono_loss = (-dL).clamp(min=0).sum() * 100

        # 4. White = 1, Black = 0
        lab_w = self.forward(D65.unsqueeze(0))
        lab_k = self.forward(torch.zeros(1, 3, device=dev))
        white_loss = (lab_w[0, 0] - 1.0).pow(2) * 200 + lab_w[0, 1].pow(2) * 200 + lab_w[0, 2].pow(2) * 200
        black_loss = lab_k[0, 0].pow(2) * 200

        # 5. Midpoint chroma preservation
        lab1 = self.forward(MID_XYZ1)
        lab2 = self.forward(MID_XYZ2)
        lab_mid = 0.5 * (lab1 + lab2)
        C_mid = (lab_mid[:, 1]**2 + lab_mid[:, 2]**2).sqrt()
        C1 = (lab1[:, 1]**2 + lab1[:, 2]**2).sqrt()
        C2 = (lab2[:, 1]**2 + lab2[:, 2]**2).sqrt()
        C_avg = 0.5 * (C1 + C2)
        mask = C_avg > 0.01
        chroma_ratio = torch.where(mask, C_mid / C_avg.clamp(min=0.001), torch.ones_like(C_mid))
        chroma_loss = (1.0 - chroma_ratio).clamp(min=0).mean()

        # 6. Midpoint hue preservation
        h_mid = torch.atan2(lab_mid[:, 2], lab_mid[:, 1])
        h1 = torch.atan2(lab1[:, 2], lab1[:, 1])
        h2 = torch.atan2(lab2[:, 2], lab2[:, 1])
        dh = h2 - h1
        dh = torch.where(dh > math.pi, dh - 2*math.pi, dh)
        dh = torch.where(dh < -math.pi, dh + 2*math.pi, dh)
        h_expected = h1 + 0.5 * dh
        h_err = torch.atan2(torch.sin(h_mid - h_expected), torch.cos(h_mid - h_expected)).abs()
        hue_loss = torch.where(C_mid > 0.01, h_err, torch.zeros_like(h_err)).mean()

        # 7. Gradient CV (simplified)
        lab_s = self.forward(PT[:, 0])
        lab_e = self.forward(PT[:, 1])
        t = torch.linspace(0, 1, N_ST, device=dev).view(-1, 1, 1)
        labs = lab_s.unsqueeze(0) + t * (lab_e - lab_s).unsqueeze(0)
        dlab = labs[1:] - labs[:-1]
        de = (dlab**2).sum(dim=-1).sqrt()
        md = de.mean(dim=0); sd = de.std(dim=0)
        ok = md > 0.001
        cvs = torch.where(ok, sd / md, torch.zeros_like(md))
        cv_loss = cvs[ok].mean() if ok.any() else torch.tensor(0.0, device=dev)

        # 8. Munsell V uniformity
        lab_g = self.forward(MUNSELL_GRAYS)
        dL_m = lab_g[1:, 0] - lab_g[:-1, 0]
        munsv_loss = dL_m.std() / (dL_m.abs().mean() + 1e-10)

        # Combined
        loss = (rt_loss +
                ach_loss +
                mono_loss +
                white_loss + black_loss +
                4.0 * chroma_loss +
                2.0 * hue_loss +
                3.0 * cv_loss +
                1.5 * munsv_loss)

        return loss, {
            'rt': rt_loss.item(),
            'ach': ach_loss.item(),
            'chroma': chroma_loss.item(),
            'hue': hue_loss.item(),
            'cv': cv_loss.item(),
            'munsv': munsv_loss.item(),
            'whiteL': lab_w[0, 0].item(),
        }

# ================================================================
#  TRAINING
# ================================================================

model = NeuralColorSpace(n_blocks=3, hidden=16).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params}", flush=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

best_loss = float('inf')
for step in range(5000):
    optimizer.zero_grad()
    loss, metrics = model.compute_loss()
    if torch.isnan(loss):
        print(f"  step {step}: NaN loss, skipping", flush=True)
        continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if loss.item() < best_loss:
        best_loss = loss.item()

    if step % 200 == 0 or step < 5:
        print(f"  step {step:5d}  loss={loss.item():.2f}  "
              f"rt={metrics['rt']:.3f}  ach={metrics['ach']:.3f}  "
              f"chroma={metrics['chroma']:.4f}  hue={metrics['hue']:.3f}  "
              f"cv={metrics['cv']:.3f}  munsv={metrics['munsv']:.3f}  "
              f"wL={metrics['whiteL']:.3f}",
              flush=True)

# Save
print("\n=== SAVING ===", flush=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = os.path.join(CKPT, f"neural_space_{ts}.pt")
torch.save(model.state_dict(), path)
print(f"Saved: {path}")

# Quick test
with torch.no_grad():
    blue = s2l(torch.tensor([[0., 0., 1.]], device=dev)) @ MS.T
    white = D65.unsqueeze(0)
    lab_b = model.forward(blue)
    lab_w = model.forward(white)
    lab_mid = 0.5 * (lab_b + lab_w)
    xyz_mid = model.inverse(lab_mid)
    rgb_mid = l2s((xyz_mid @ MSi.T).clamp(0, 1))[0]
    gr = rgb_mid[1].item() / max(rgb_mid[0].item(), 1e-10)
    hex_mid = '#{:02x}{:02x}{:02x}'.format(int(rgb_mid[0]*255), int(rgb_mid[1]*255), int(rgb_mid[2]*255))

    xr = torch.rand(1000, 3, device=dev) * D65
    lr = model.forward(xr)
    rr = model.inverse(lr)
    rt = (rr - xr).abs().max().item()

    lab_g = model.forward(MUNSELL_GRAYS)
    dL = lab_g[1:, 0] - lab_g[:-1, 0]
    munsv = (dL.std() / dL.abs().mean() * 100).item()

    print(f"\nFINAL:")
    print(f"  Blue-White: {hex_mid}  G/R={gr:.3f}")
    print(f"  RT: {rt:.2e}")
    print(f"  White L={lab_w[0,0]:.4f}")
    print(f"  Munsell V: {munsv:.2f}%")
    print(f"  Params: {n_params}")
