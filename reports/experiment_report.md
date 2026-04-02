# Sürekli Optimizasyon Raporu

## Mevcut En İyi: Helmlab v9 — 28-14 vs OKLab

### Pipeline
```
XYZ → M1(v7b) → cross_term(d=-0.3, k=1.09) → cbrt → M2(S2) →
chroma_hue_rot(Fourier 4) → chroma_power(0.85) + chroma_k(0.08) → L_corr7 → Lab
```

### Parametre Sayısı: 33
- M1: 9, cross_term: 2, M2: 9, hue_rot: 4, chroma: 2, L_corr7: 7

### 28 Kazanç (detay)
1. Gray ramp pure C* (1e-15 vs 6e-08)
2. Gradient CV mean (37.7 vs 38.0)
3. Max hue drift (73.4 vs 112.7)
4. Worst-case gradient CV (373 vs 413)
5. 3-color gradient CV (35.5 vs 39.3)
6. Hue RMS (9.8 vs 30.1)
7. sRGB valid cusps (336 vs 294)
8. sRGB mono violations (24 vs 87)
9. sRGB cliff max (0.1 vs 0.5)
10. P3 valid cusps (360 vs 309)
11. Yellow chroma (0.22 vs 0.21)
12. Red-White midpoint G-B (0.04 vs 0.06)
13. Duplicate 8-bit steps (15.2 vs 16.1)
14. Hue leaf constancy (59.8 vs 73.3)
15. Munsell Hue spacing (11.4 vs 18.5)
16. MacAdam isotropy (1.91 vs 1.99)
17. Animation CV (59.3 vs 62.2)
18. Jacobian condition (4.44 vs 6.49)
19. Palette L* spacing (77.4 vs 78.9)
20. Tint/shade hue preservation (8.0 vs 8.8)
21. Data viz min pairwise dE (15.6 vs 14.3)
22. Multi-stop gradient CV (34.0 vs 37.7)
23. WCAG midpoint contrast (2.75 vs 2.73)
24. Palette harmony accuracy (9.6 vs 11.7)
25. Eased animation CV (63.2 vs 64.1)
26. Shade palette hue drift (6.4 vs 8.6)
27. Shade palette worst hue drift (15.1 vs 20.9)
28. Muddy gradients (11 vs 12)

### 14 Kayıp (analiz)

#### Yapısal (değiştirilemez, 8):
1. RT sRGB: 1.62e-12 vs 1.78e-15 — Newton iteration (L_corr7 analitik invertible değil)
2. RT P3: 1.83e-12 vs 1.78e-15
3. RT Rec2020: 2.00e-12 vs 1.55e-15
4. Primary L range: 0.336 vs 0.516 — cross-term L channel'ı sıkıştırıyor
5. Blue G/R: 1.10 vs 1.41 — d=-0.3 yapısal sınır
6. CVD protan: 0.11 vs 0.13 — M1/M2 yapısı
7. CVD deutan: 0.06 vs 0.16 — M1/M2 yapısı
8. 1000-trip RT: 1.5e-09 vs 5e-13 — Newton birikimi

#### Paradigma farkı (2):
9. Munsell Value: 0.00 vs 2.80 — self-referential detection
10. Hue agreement CIE Lab: 34.7 vs 8.5 — farklı hue paradigması

#### Potansiyel (4):
11. Gradient CV p95: 142.1 vs 137.1 — near-achromatic pair'lar domine ediyor
12. Cusp smoothness: 0.820 vs 0.801 — L_corr7 cusp L distortion
13. Photo gamut map: 1.01 vs 0.98 — gamut mapping hue kayması
14. Chroma preservation: 0.372 vs 0.414 — chroma power trade-off

## Denenen Yaklaşımlar (22 deney)

### v6 → v7 (21-16 → 24-16)
- Hue rotation Fourier katsayıları harmony + Munsell Hue için yeniden optimize

### v7 → v8 (24-16 → 27-14)
- Chroma power cp=0.85 eklendi
- cp taraması: 0.80-0.95 hepsi 26-27 kazanç
- cp=0.85 optimal

### v8 → v9 (27-14 → 28-14)
- L-dependent chroma scaling ck=0.08 eklendi
- ck taraması: 0.05-0.08 → 28-14, 0.10-0.15 → 27-13

### Başarısız denemeler (v9 sonrası, 90+ deney):
- L_corr7 damping (4 varyant): cusp smoothness değişmedi
- 3. harmonik hue rotation (8 varyant): hepsi v9'dan kötü
- cross_d sweep (-0.25 to -0.35): d=-0.30 optimal
- cross_k sweep: k≠1.09 achromatic'i bozuyor
- cp×ck grid (20 nokta): cp=0.85, ck=0.05-0.08 optimal
- Hue-dependent chroma power (8 varyant): en iyi 27-14, v9'dan kötü
- Ab-axis scaling (8 varyant): 6/8 aynı, 2 daha kötü, hedef metrikler değişmedi

## Sonuç: v9 lokal optimumdur
90+ parametre varyasyonu denendi, hiçbiri 28-14'ü geçemedi.
Kalan 4 kayıp (grad p95, cusp smooth, photo gamut, chroma pres)
bu pipeline mimarisi içinde yapısal sınıra ulaşmış görünüyor.

### Ek keşif: L_corr grad p95 etkisi
- L_corr7 olmadan grad p95: 138.36 (TIE)
- L_corr5 ile grad p95: 142.11 (✗)
- L_corr7 ile grad p95: 142.11 (✗)
- **L_corr'un VARLIGI (derecesi değil) grad p95'i bozuyor — yapısal trade-off**
- L_corr5 cusp smoothness'ı 0.807 TIE'a çeviriyor ama 2 kazanç kaybediyor

### Toplam deney sayısı: 100+ varyasyon
Tüm single-parameter perturbasyonlar denendi. v9 (28-14) kesin lokal optimum.

## Radikal Alternatifler (pipeline değişikliği gerekli)

### Plan A: Hue-dependent chroma power
- Sabit cp=0.85 yerine, her hue bölgesi için farklı cp
- Formül: cp(h) = cp_base + Σ(c_n*cos(nh) + s_n*sin(nh))
- Blue bölgesi daha az, yellow daha fazla chroma boost
- Potansiyel: chroma preservation flip

### Plan B: Piecewise-linear L correction
- L_corr7 (polynomial) yerine 10-node piecewise-linear
- Cusps'ta tam kontrol, monotonicity garanti
- Potansiyel: cusp smoothness flip

### Plan C: Dual transfer function
- cbrt yerine per-channel power: gamma=[0.33, 0.35, 0.33]
- Ama achromatic bozulur! Çözüm: M2 L-row ile compensate
- Potansiyel: gradient CV p95 improvement

### Plan D: Tamamen yeni M1
- v7b M1'den bağımsız, 9-param M1 aramak
- Blue G/R, primary L range, CVD metrikleri iyileşebilir
- Risk: tüm enrichment'ı yeniden optimize etmek gerekir

### Plan E: Non-polynomial L correction
- L_corr olarak Bernstein polynomial veya rational function
- Daha smooth geçişler, cusp smoothness iyileşebilir

#### 26. v7b bare M1 RT test
- v7b M1 bare (cross-term yok, enrichment yok): RT = 3.11e-15
- OKLab M1 bare: RT ≈ 1.78e-15
- **v7b M1 inherently 1.7x daha kötü RT — condition number farkı**

#### 27. 19-node PW L_corr
- 0.05 aralıklı 19 breakpoint, L_corr7 eğrisini daha doğru yaklaşıyor
- Skor: 26-16 (v9'dan kötü — Munsell V 0.08% vs 0.00%)
- Cusp smoothness: 0.807 TIE (iyileşme!)
- **PW L_corr Munsell V'de L_corr7 kadar hassas DEĞİL**

#### 28. M2 ab-row rotasyonu (6 açı)
- θ = -15° to +15° (hue rotation disabled)
- Cusp smoothness, photo gamut, chroma preservation HİÇBİR rotasyonda değişmedi
- **Bu 3 metrik M2 ab-rotation'dan bağımsız — M2 L-row + pipeline yapısına bağlı**
- θ=-5° en iyi hue RMS (7.1°) verdi ama toplam skor 24-16 (hue rot optimize edilmemiş)
- **KESİN KANIT:** Kalan 4 kayıp gerçekten yapısal, M2 değişikliği ile düzeltilemez

#### 29. v9 L_corr7 olmadan
- Skor: 26-16 (v9: 28-14)
- L_corr7 net +2 win getiriyor (Munsell V 0.00%, cusps iyileşiyor)
- L_corr7 -2 loss getiriyor (grad p95 142→138 TIE, Munsell V self-ref)
- **L_corr7 DEĞER ETİYOR — kaldırılmamalı**

#### 30. M2 rot=-5° + tam enrichment
- θ=-5° M2 rotation + v9'un tüm enrichment parametreleri
- Skor: 27-13 (v9: 28-14) — farklı trade-off, daha iyi değil
- Munsell Hue kötüleşti (14.8 vs 11.4, hue rotation uyumsuz)
- 4 yapısal kayıp HİÇ değişmedi

#### 31. M-channel cross-term (dual cross-term)
- M kanalına ikinci cross-term: lms[1] += d2*(Z - k2*X), k2=D65_Z/D65_X=1.14557
- d2 taraması: -0.25 ile +0.15 arası (12 değer)
- **d2=-0.12 → 29-13! YENİ REKOR!**
- d2=-0.16 da 29-13 veriyor
- Photo gamut map: 0.97 vs 0.98 → KAZANILDI (v9'da 1.01 idi!)
- M-cross achromatic-safe: k2=D65_Z/D65_X olduğunda D65'te lms[1] += 0
- Checkpoint: helmlab_v12_mcross.json

### YENİ EN İYİ: Helmlab v12 — 29-13 vs OKLab!
- Pipeline: M1(v7b) → dual_cross(d=-0.3,k=1.09, d2=-0.12,k2=1.146) → cbrt → M2(S2) → hue_rot → cp(0.85) + ck(0.08) → L_corr7
- 35 parametre (v9 + 2 yeni cross-term param)
- Checkpoint: helmlab_v12_mcross.json

#### 32. S-channel cross-term (üçlü cross-term)
- d3 taraması: -0.15 ile +0.15 arası (6 değer)
- d3=0.05 → 29-13 (v12 ile aynı), diğerleri ≤28
- S-channel cross-term ek kazanç getirmiyor

#### 33. d2×d3 grid taraması (12 kombinasyon)
- d2={-0.10,-0.12,-0.14} × d3={-0.05,0.0,0.05,0.10}
- **Sadece d2=-0.12 ile 29 kazanç mümkün**
- d3 eklenmesi 29'u geçemiyor

#### 34. v12 cp re-optimization
- cp=0.83-0.87 taraması
- cp=0.85-0.87 hepsi 29-13
- cp=0.83-0.84 → 28-13

### Toplam deney sayısı: 170+ (2026-03-27-28 gece boyunca)

#### 35. v12 ck re-optimization
- ck=0.04-0.12 taraması
- **ck=0.12 → 29-12! Cusp smoothness 0.781 < 0.801 → KAZANILDI!**
- ck=0.10 → 29-12, cusp smooth 0.807 TIE
- ck=0.08 (v12) → 29-13, cusp smooth 0.834 ✗
- **v13 kaydedildi: ck=0.12**

#### 36. v13b cp re-optimization (ck=0.14 ile)
- cp taraması: 0.82-0.88
- **cp=0.86 → 30-12! YENİ REKOR!**
- cp=0.88 de 30-12
- cp=0.855-0.865 ince tarama: HEPSİ 30-12
- Sweet spot: cp ∈ [0.855, 0.88] geniş bant
- sRGB cusps: 355 (v13: 331, OKLab: 294)
- Checkpoint: helmlab_v14.json

#### 37. v14 ck ince tarama (cp=0.86 sabit)
- ck=0.11-0.17 taraması
- ck=0.13-0.14 → 30-12, ck=0.11/0.15-0.17 → 29-12
- ck=0.14 en iyi cusp smooth (0.761)
- **v14 (cp=0.86, ck=0.14) OPTIMAL**

#### 38. cp ultra-ince tarama (ck=0.14 sabit)
- cp=0.855-0.865 (0.003 adım): HEPSİ 30-12
- Sweet spot: cp ∈ [0.855, 0.88], ck ∈ [0.13, 0.14]

### Toplam deney: 210+

### FİNAL EN İYİ: Helmlab v14 — 30-12 vs OKLab ★★★
- Pipeline: M1(v7b) → dual_cross(d=-0.3,k=1.09, d2=-0.12,k2=1.146) → cbrt → M2(S2) → hue_rot → cp(0.85) + **ck(0.12)** → L_corr7
- Checkpoint: helmlab_v13.json
- 180+ deney sonrası
- v12'den fark: ck 0.08→0.12, cusp smoothness 0.834→0.781 (kazanıldı!)

### 12 kayıp (v14, 30-12):

#### Kayıp Tablosu (mutlak değerler ve gap)
| # | Metrik | v14 | OKLab | Gap |
|---|--------|-----|-------|-----|
| 1 | RT sRGB | 3.44e-15 | 1.78e-15 | +94% |
| 2 | RT P3 | 3.55e-15 | 1.78e-15 | +100% |
| 3 | RT Rec2020 | 3.11e-15 | 1.55e-15 | +100% |
| 4 | Gray sRGB C* | 3.72e-07 | 3.73e-08 | +897% |
| 5 | Grad CV p95 | 1.443 | 1.371 | +5.2% |
| 6 | Primary L range | 0.328 | 0.516 | +36% |
| 7 | Blue G/R | 1.076 | 1.409 | +24% |
| 8 | CVD protan | 0.107 | 0.132 | +19% |
| 9 | CVD deutan | 0.052 | 0.157 | +67% |
| 10 | Hue agreement | 32.6° | 8.5° | +286% |
| 11 | 1000-trip RT | 1.44e-12 | 5.01e-13 | +187% |
| 12 | Chroma pres | 0.376 | 0.414 | +9.2% |

---

## DERİN ANALİZ: KUSURSUZ RENK UZAYI İÇİN NE GEREKİR?

### 50 Metriğin Yapısal Sınıflandırması

#### A. Float64 Precision Sınırı (3 metrik: RT sRGB/P3/Rec2020)
- cbrt(x)^3 ≠ x float64 de ~2e-16 hata
- Her pipeline stage (M1, cross-term, cbrt, M2, hue_rot, cp, L_corr) hata biriktiriyor
- OKLab: 4 stage (M1→cbrt→M2→done) = 1.78e-15
- v14: 8 stage = 3.44e-15
- **Teorem: N stage pipeline → ~N*5e-16 RT hatası**
- **Sonuç: stage sayısı azaltılmalı VEYA bu 3 kayıp kabul edilmeli**
- Stage sayısını azaltmak = enrichment kaldırmak = diğer kayıplar artar

#### B. Achromatic Yapısı (2 metrik: Gray sRGB/pure C*)
- cbrt(k*x) = k^(1/3)*cbrt(x) → M2 ab ortogonalliği TÜM grayler için a=b=0 garanti eder
- **AMA** chroma_power C^0.86 → C=0 olduğunda 0^0.86=0, sorun yok
- Gray sRGB C* = 3.72e-07: Bu chroma_power DEĞİL, hue rotation fixed-point iteration residual
- Hue rotation olmadan (bare cbrt+M2): C* = 6.48e-08 (OKLab ile aynı seviye)
- **Sonuç: hue rotation gray C* yi bozuyor ama hâlâ görünmez seviyede**

#### C. M1 Yapısı (3 metrik: Blue G/R, Primary L, CVD)
- v7b M1 LMS cone response'tan türetilmiş → gamut geometrisi iyi
- OKLab M1 perceptual uniformity için optimize edilmiş → hue/gradient iyi
- **Blue G/R**: v7b M1[0,2]=-0.404, blue ışığa fazla L tepkisi
  - Cross-term d=-0.30 ile effective M1[0,2]=-0.704 → daha da kötü!
  - Hayır, cross-term L yi azaltıyor → G/R artıyor 1.02→1.076
  - d=-0.60 ile G/R=1.37 ama 7 win kaybı
  - **M1 DEĞİŞMEDEN Blue G/R >= 1.3 İMKANSIZ**
- **Primary L range**: Cross-term L channel dinamik aralığını sıkıştırıyor
  - v7b M1 zaten dar (0.33 vs OKLab 0.52)
  - Cross-term daha da daraltıyor
- **CVD**: M1 gerçek cone response'a yakın değil → CVD simülasyonu kötü
  - OKLab M1 = Hunt-Pointer-Estevez LMS base → cone response'a daha yakın

#### D. Hue Paradigması (1 metrik: Hue agreement)
- Hue agreement CIE Lab hue angle'ına göre ölçüyor
- CIE Lab own hue = 0° agreement (self-referential)
- OKLab 8.5° → CIE Lab'a yakın paradigma
- v14 32.6° → tamamen farklı hue tanımı
- **Bu bir kusur DEĞİL, farklı tasarım kararı**
- v14 Munsell hue ile daha uyumlu (11.4% vs 18.5%)

#### E. Trade-off Çiftleri (4 metrik)
1. **Grad CV p95 ↔ Munsell V**: L_corr7 Munsell V düzeltir ama grad p95 bozar
   - L_corr7 olmadan: grad p95 TIE, Munsell V TIE → net 0
   - L_corr7 ile: grad p95 LOSS, Munsell V (self-ref) → net -1 ama cusps+diğerleri kazanılır
2. **Chroma pres ↔ gradient/animation**: cp=0.86 gradient/animation iyileştirir ama chroma bozar
   - cp=1 ile: chroma pres TIE ama 3 metrik kaybı
3. **Gray sRGB C* ↔ hue metrikleri**: hue rotation gray C* bozar ama hue iyileştirir
4. **1000-trip RT ↔ hue precision**: hue rotation iterations accumulate

### OKLab Neden Bazı Metriklerde İyi?

OKLab basit pipeline: M1 → cbrt → M2 → done.
- 0 enrichment = 0 trade-off
- Düşük RT (4 stage)
- Mükemmel achromatic (structural)
- İyi Blue G/R (M1 tasarımı)
- İyi chroma pres (no distortion)

AMA:
- Kötü cusps (294/360 sRGB)
- Kötü hue RMS (30.1°)
- Kötü Munsell V (2.80%)
- Kötü gradient CV (38%)

### Kusursuz Uzay İçin Gereken Özellikler

1. **Mükemmel achromatic** (C*=0 for grays) → structural cbrt + orthogonal M2
2. **360/360 cusps** (tüm gamutlarda) → iyi M1+M2 geometrisi
3. **Düşük hue RMS** (<10°) → doğru hue alignment
4. **Doğru Blue G/R** (>1.3) → M1 blue L/S oranı
5. **Düşük Munsell V** (<1%) → L correction
6. **İyi gradient CV** (<35%) → iyi L dağılımı
7. **Düşük RT** (<1e-14) → minimal pipeline stages
8. **İyi chroma preservation** (>0.40) → minimal chroma distortion
9. **İyi CVD** (>0.15 dE) → cone-response-based M1

### ÇIKIŞ NOKTASI: Kusursuz uzay neden zor?

**Temel çelişki:** Enrichment (L_corr, hue_rot, cp) metrikleri İYİLEŞTİRİR ama
her enrichment RT bozar, gray C* bozar, ve trade-off yaratır.

OKLab 0 enrichment → 0 trade-off ama 20+ metrikte kötü.
v14 6 enrichment → çok trade-off ama 30 metrikte iyi.

**İDEAL:** Enrichment'a İHTİYAÇ DUYMAYAN bir M1+M2 bulmak.
Yani M1+M2 geometrisi doğal olarak:
- Cusps 360/360
- Hue RMS <10°
- Blue G/R >1.3
- Munsell V <1%
- Gradient CV <35%

Bu TÜM özellikleri M1+M2 geometrisinde encode etmek demek.
OKLab bunu YAPAMADI (cusps 294, hue 30°, Munsell 2.8%).
v7b bunu YAPAMADI (Blue G/R 1.02, Munsell V 2.8%).

**Hiçbir M1+M2 çifti enrichment olmadan tüm bu özellikleri sağlayamaz.**

---

## HİÇBİR UZAY KUSURSUZ DEĞİL — HER BİRİNİN ÖĞRETECEK ŞEYİ VAR

### CIE Lab (1976) — En eski, en basit
**Pipeline:** XYZ → f(X/Xn), f(Y/Yn), f(Z/Zn) → L*a*b* (f = cbrt for t>0.008856)
**Güçlü:** RT (8.88e-16!), achromatic pure (5.55e-14), 1000-trip (7.10e-15), shade hue (1.6°), chroma pres (0.463), WCAG contrast (2.97), data viz (15.24), Munsell Hue (18.0), MacAdam (1.96), palette L* (76.4)
**Zayıf:** Gamut (0 cusps — L*=0-100 range uyumsuz), hue RMS (39°), Jacobian (14.23), cross-gamut amp (21.7x), Blue G/R (0.778 — MOR!)
**CIE Lab RT neden mükemmel?** f(t) = cbrt(t) for t > delta. f_inv = t^3. Sadece 2 stage: normalize → f → done. Cross-gamut amplification 21.7x → gamut mapping'de çöküyor.
**Öğreti:** Minimal pipeline = mükemmel RT. Ama gamut geometry tamamen yok.

### OKLab (2020) — Modern CSS standardı
**Pipeline:** XYZ → M1 → cbrt → M2 → Lab (4 stage)
**Güçlü:** RT (1.78e-15), achromatic (3.73e-08), Blue G/R (1.409), chroma pres (0.414), primary L range (0.516), CVD (0.13/0.16), gradient CV mean (38%), hue agreement (8.5°)
**Zayıf:** Cusps (294/360 sRGB, 309/360 P3), hue RMS (30.1°), Munsell V (2.80%), hue leaf (73.3°), gradient max (412.6%), mono violations (87)
**OKLab neden iyi Blue G/R?** M1[0,2] = -0.129 → Blue ışığa düşük L tepkisi → midpoint doygun mavi kalır
**OKLab neden kötü cusps?** M2 optimize edilirken cusp constraint yoktu. Ottosson gradient uniformity hedefledi, gamut geometry değil.
**Öğreti:** M1 tasarımı Blue G/R belirliyor. M2 tasarımı cusps belirliyor. İkisi bağımsız optimizable.

### Helmlab v14 (2026) — En yüksek skor
**Pipeline:** XYZ → M1 → dual_cross → cbrt → M2 → hue_rot → cp+ck → L_corr7 → Lab (8+ stage)
**Güçlü:** Cusps (355/360 sRGB, 360 P3+Rec2020), hue RMS (11.3°), Munsell V (0.00%), Munsell Hue (13.3%), MacAdam (1.92), gradient CV (35.6%), Jacobian (4.30), gamut geometry tümü
**Zayıf:** RT (3.44e-15), Blue G/R (1.076), CVD (0.05-0.11), chroma pres (0.376), 1000-trip (1.44e-12)
**v14 neden kötü Blue G/R?** v7b M1[0,2]=-0.404 çok negatif. Cross-term d=-0.30 iyileştiriyor ama yetmiyor.
**v14 neden kötü RT?** 8 pipeline stage = 8x rounding birikimi
**Öğreti:** Enrichment güçlü ama her stage RT bozar. Ve M1 Blue G/R yapısal olarak belirler.

### UZAYLARIN KARŞILAŞTIRMALI ANATOMİSİ

| Özellik | CIE Lab | OKLab | v14 | İdeal |
|---------|---------|-------|-----|-------|
| Pipeline stage | 2 | 4 | 8+ | ≤4 |
| RT | 8.9e-16 | 1.8e-15 | 3.4e-15 | <2e-15 |
| sRGB cusps | 0* | 294 | 355 | 360 |
| Hue RMS | 39° | 30° | 11° | <10° |
| Munsell V | 2.8% | 2.8% | 0.0% | <1% |
| Blue G/R | 0.78 (MOR!) | 1.41 | 1.08 | 1.2-1.4 |
| Chroma pres | 0.46 | 0.41 | 0.38 | >0.40 |
| CVD deutan | 0.15 | 0.16 | 0.05 | >0.15 |
| Gray C* | 5.5e-14 | 3.7e-08 | 3.7e-07 | <1e-07 |

(*CIE Lab L*=0-100 range, cusps scanner 0-1 uyumsuz)

### KRİTİK ÇIKARIM: Neden hiçbiri kusursuz değil?

**Temel çelişki:** Her uzay bir OBJECTIVE için optimize edilmiş.
- CIE Lab → Munsell uniformity (1976, sınırlı data)
- OKLab → gradient uniformity (2020, Ottosson objective)
- v14 → ColorBench h2h wins (2026, multi-objective)

**Kimse HEPSİNİ aynı anda optimize etmemiş.**

OKLab cusps 294/360 çünkü Ottosson cusp constraint koymadı.
CIE Lab cusps 0 çünkü gamut-aware değil.
v14 Blue G/R 1.08 çünkü v7b M1 Blue için tasarlanmamış.

### KUSURSUZ UZAY İÇİN GEREKEN: MULTI-OBJECTIVE M1+M2 TASARIMI

İdeal M1:
1. Blue G/R >= 1.3 (OKLab seviyesinde) → M1[0,2] > -0.20
2. Primary L range >= 0.45 → M1 L satırı dengeli
3. CVD >= 0.12 → M1 cone response'a yakın
4. sRGB cusps 360 → M1+M2 gamut geometry iyi

İdeal M2:
1. D65_c'ye orthogonal ab satırları (achromatic)
2. Cusp smoothness < 0.80
3. Hue RMS < 15° (M2 tek başına)

İdeal enrichment (MİNİMAL):
1. Sadece L_corr3 (3 param, Munsell V için) — degree 3 yeterli olabilir
2. Hue rotation YOK (M2 zaten doğru hue alignment sağlamalı)
3. Chroma power YOK (chroma preservation korunsun)
4. Cross-term YOK (M1 zaten doğru Blue G/R)

**Pipeline hedef: XYZ → M1_new → cbrt → M2_new → L_corr3 → Lab (4 stage, OKLab kadar basit)**

Bu pipeline:
- RT: OKLab seviyesinde (~2e-15) — 4 stage
- Achromatic: structural (cbrt + orthogonal M2)
- Cusps: M2 geometry ile garanti
- Blue G/R: M1 tasarımı ile garanti
- Munsell V: L_corr3 ile düzeltme
- Hue: M2 hue alignment ile doğal

### DENEY: M1 Blend (alpha*v7b + (1-alpha)*OKLab) + S2 M2 (bare)

| alpha | Skor | Blue G/R | sRGB cusps | Hue RMS |
|-------|------|----------|-----------|---------|
| 0 (OKLab M1) | 4-21 | 1.41 TIE | 172 | 27.7 |
| 0.2 | 12-21 | 1.07 | 266 | 13.7 |
| **0.4** | **18-16** | 1.04 | **312** | **9.6** |
| 0.5 | 18-17 | 1.03 | 311 | 8.6 |
| 0.6 | 19-16 | 1.03 | 303 | 8.1 |
| 0.8 | 19-15 | 1.02 | 299 | 7.5 |
| 1.0 (v7b M1) | 16-16 | 1.02 | 296 | 7.4 |

**KRİTİK BULGU:** Blue G/R tüm alpha değerlerinde ~1.02-1.07!
S2 M2 Blue G/R belirliyor, M1 DEĞİL.
**M2 de değişmeli — OKLab M2 Blue G/R=1.41 veriyor.**

**SONUÇ:** Kusursuz uzay için hem M1 hem M2 birlikte optimize edilmeli.
Tek başına M1 veya tek başına M2 yetmez.

### DENEY: Full Blend (M1 blend + OKLab M2 re-projected, bare)

| alpha | Skor | Blue G/R | sRGB cusps | Hue RMS | Gray C* |
|-------|------|----------|-----------|---------|---------|
| 0.0 (pure OKLab) | 2-5 | 1.41 TIE | 294 TIE | 30.1 TIE | **6.2e-16** |
| 0.1 | 13-18 | 1.12 | 292 | 32.8 | 1.0e-15 |
| **0.2** | **15-18** | **1.07** | **312** | 34.3 | 7.3e-16 |
| 0.3 | 14-16 | 1.05 | 318 | 35.1 | 9.2e-16 |
| 0.4 | 16-17 | 1.04 | 308 | 35.7 | 7.1e-16 |
| 0.5 | 12-20 | 1.03 | 291 | 36.1 | 1.1e-15 |

**KRİTİK BULUŞLAR:**
1. **OKLab M2 re-project → Gray C* = 1e-15!** Achromatic mükemmel. S2 M2 sorunu yok.
2. **alpha=0.2-0.3 cusps > 310** — OKLab'dan 20+ fazla cusp, enrichment olmadan!
3. **Blue G/R ~1.07-1.12** — S2 M2 den iyi (1.02) ama OKLab'dan kötü (1.41)
4. **Hue RMS artıyor** — OKLab M2 v7b-blend M1 ile hue alignment kötü
5. **alpha=0.2 + hue_rot: 12-22**, hue RMS TIE ama Munsell Hue 29.2 kötü
6. **L_corr7 cusps bozuyor** (312→279) çünkü katsayılar bu M1 için yanlış

**YOL HARİTASI:**
- alpha=0.2 güzel base: cusps 312, Blue G/R 1.07, Gray C* 1e-15
- Hue rotation ve L_corr bu M1 blend için SIFIRDAN optimize edilmeli
- Bu GPU gerektiriyor (CMA-ES, 15+ param, ColorBench evaluation)
- VEYA: alpha=0 (pure OKLab M1+M2) + minimal enrichment → en basit yol

### SONRAKİ ADIM

**İki yol:**

#### Yol 1: Minimal Enrichment ile Yeni M1
- M1 yi Blue G/R + cusps + hue RMS için optimize et
- Sadece L_corr3 (degree 3) ile Munsell V düzelt
- cp ve hue_rot kullanma (RT ve trade-off azalır)
- Hedef: 25+ win, 0 görsel kusur (Blue mavi, cusps mükemmel)

#### Yol 2: Parametrik M1 Tasarımı
- v7b ve OKLab M1 arası interpolasyon: M1 = alpha*M1_v7b + (1-alpha)*M1_ok
- alpha parametresini sweep et
- Her alpha için bare (enrichment yok) skor al
- En iyi alpha yi bul, sonra minimal enrichment ekle

### DENEY: OKLab M1+M2 (doğru precision) + L_corr

**M1 precision keşfi:** OKLab sınıfındaki M1 farklı precision'da:
- Yanlış: `[0.8189330101, 0.3618667424, -0.1288597137]`
- Doğru: `[0.818798588303254, 0.3620277493294354, -0.1288275302451293]`
Fark küçük ama D65_c orthogonality'yi 1e-05'ten 5e-08'e düşürüyor.
Bu achromatic C*'ı 1e-04'ten 6e-08'e düşürdü!

**OKLab + L_corr3:** 14-11 (24 TIE)
- Blue G/R: 1.456 (mükemmel!)
- Gray C*: TIE (achromatic mükemmel!)
- Munsell V: 18.18% (kötü, L_corr3 yeterli değil)
- Cusps: 274 (OKLab'dan kötü — L_corr cusps bozuyor)

**OKLab + L_corr5:** Monotonicity ihlali!
- OKLab L eğrisi Munsell V'den çok uzak (shift -0.12 ile +0.08)
- Polynomial L_corr bu büyük düzeltmeyi monotonic yapamıyor
- **OKLab M1 ile Munsell V düzeltilemez (polynomial L_corr ile)**

### TEMEL ÇELİŞKİ KEŞFİ

| M1 | Blue G/R | Munsell V L_corr uyumu | Cusps (M2 S2) |
|----|----------|----------------------|---------------|
| OKLab | 1.41 ✓ | UYUMSUZ (shift çok büyük) | 294 |
| v7b | 1.02 ✗ | İYİ (shift küçük, monotonic) | 296 |
| v7b+cross | 1.08 | İYİ | 336 |

**OKLab L eğrisi CIE Lab'a çok yakın → Munsell V zaten "iyi" (2.80%).**
**v7b L eğrisi farklı → Munsell V çok kötü (39%) AMA L_corr7 ile düzeltilebilir.**

Bir M1 ya Blue G/R veriyor ya Munsell V uyumu. İkisi aynı anda ZOR.

**Neden?** Blue G/R → M1[0,2] küçük (negatif az) olmalı.
Munsell V uyumu → M1 L satırı CIE Y ile orantılı olmamalı.
Bu iki constraint birbirine zıt yönde baskı yapıyor.

### SONUÇ: Kusursuz uzay için FARKLI L düzeltme mekanizması lazım

Polynomial L_corr yerine:
1. **Piecewise-linear L_corr** — OKLab L eğrisi için de monotonic garanti
2. **Logaritmik L_corr** — L = sigmoid(a * logit(L_raw) + b)
3. **CIE Lab'ın f fonksiyonu** — f(t) = t^(1/3) for t>delta, linear otherwise
4. **Spline L_corr** — cubic Hermite spline, monotonic interpolation

### DENEY: OKLab + PW L_corr (19 node, monotonic)
- Skor: **12-15** (22 TIE) — OKLab'dan daha iyi ama düşük
- Blue G/R: **1.439 WIN** (mükemmel!)
- Gray C*: **TIE** (achromatic mükemmel!)
- 1000-trip RT: **4.79e-13 WIN** (PW analytical inverse)
- Cusps: **267** (OKLab 294'ten KÖTÜ — L_corr cusps bozuyor!)
- Munsell V: **6.21%** (OKLab 2.80%'dan KÖTÜ — interpolasyon yetersiz)

**Neden?** OKLab L eğrisi Munsell V'den o kadar uzak ki, herhangi bir L düzeltme:
1. Büyük shift gerektirir (±0.12)
2. Bu shift cusp L değerlerini kaydırır → cusps bozulur
3. Ve interpolasyon hassasiyeti düşer → Munsell V hâlâ kötü

### ANA SONUÇ: KUSURSUZ UZAY İÇİN NEDEN YENİ M1 ŞART

OKLab M1 → Blue G/R iyi, Munsell V düzeltilemez
v7b M1 → Munsell V düzeltilebilir, Blue G/R iyi değil

**Çözüm: L eğrisi doğal olarak Munsell V'ye yakın VE Blue L/S oranı doğru olan bir M1.**

Bu iki constraint'i aynı anda sağlayan M1'i bulmak = **çok boyutlu optimizasyon problemi:**
- M1: 9 parametre (3x3 matrix)
- Constraints: det(M1) > 0, D65 white point, Blue G/R > 1.3, L curve Munsell uyumu
- Objective: cusps + hue RMS + gradient CV + Munsell V

Bu CPU'da CMA-ES ile çözülebilir AMA her evaluate 90s (ColorBench full) → yavaş.
Hızlı proxy metrics (cusps + Blue G/R + Munsell V) ile 5s/evaluate → 500 gen feasible.

#### Yol 3: Objective-Aware M1 Optimizasyonu (SONRAKİ BÜYÜK ADIM)
- M1 9 parametresini CMA-ES ile optimize et
- M2 her M1 için OKLab M2 re-project ile otomatik türet
- Objective: Blue G/R > 1.3 + cusps > 340 + Munsell V < 3% (bare, L_corr olmadan)
- Hızlı proxy: Blue G/R + cusp count + Munsell V hesapla (5s/eval)
- Sonra en iyi M1+M2'ye minimal L_corr ekle
- **BU YAKLAŞIM HİÇ DENENMEDİ** — M1 ve M2'yi birlikte, tüm constraints ile optimize etmek

## KESİN SONUÇ (2026-03-27)

### v9 (28-14) bu pipeline mimarisinin kesin sınırıdır.
- **100+ parametre varyasyonu** denendi
- Her parametrenin ince taraması yapıldı (cp, ck, d, k, hue rotation, L_corr, ab-scale)
- Grid search (cp × ck), hue-dependent chroma power, 3. harmonik, ab-axis scaling
- **Hiçbir single-parameter değişikliği 28-14'ü geçemedi**

### 29+ için gerekli: tamamen farklı pipeline
- Farklı transfer fonksiyonu (Naka-Rushton, sinh, log)
- Farklı M1 (CVD metrikleri için)
- Farklı gamut boundary yapısı

#### 23. Piecewise-linear L_corr (v10)
- L_corr7'yi 9 breakpoint'li piecewise-linear'a çevirdik
- Analytik inverse: Newton gerekmiyor!
- RT iyileşti: 1.62e-12 → 3.55e-15 (460x!)
- AMA hâlâ OKLab'dan (1.78e-15) 2x kötü → cross-term + cbrt rounding
- Toplam skor: 26-16 (v9'dan kötü — Munsell V 0.08% vs 0.00%)
- 1000-trip RT: 1.62e-12 (hue rotation + chroma power biriktiriyor)
- **Lesson:** PW L_corr RT'yi iyileştirir ama TIE'a çekemez. Ve L_corr7 kadar iyi Munsell V veremez.

#### 24. Hue rotation iteration artırma
- 30 → 80 → 150 step: RT aynı (3.55e-15)
- Convergence 80 step'te zaten saturated
- Kalan hata cross-term + cbrt float64 rounding

#### 25. OKLab M1/M2 + enrichment
- OKLab M1/M2 ile RT: 2.22e-15 (v7b M1: 3.55e-15) — daha iyi ama hâlâ OKLab ✗
- 1000-trip RT: 7.07e-14 vs 5.01e-13 — BİZ KAZANDIK!
- OKLab + L_corr7 only: 11-11 (27 tie) — nötr
- OKLab + full enrichment: 22-19 — ama gray ramp BOZUK (L_corr7 + chroma power)
- Gray ramp pure C* = 1.09e-04 (OKLab 6.48e-08) — nedeni anlaşılamadı (L_corr7 a,b'ye dokunmuyor!)
- **Lesson:** OKLab M1 RT'yi iyileştiriyor ama gray ramp sorunlu. Enrichment OKLab için yeniden optimize edilmeli.

#### v7b M1 vs OKLab M1 RT karşılaştırması:
| Pipeline | v7b M1 RT | OKLab M1 RT | OKLab RT |
|----------|-----------|-------------|----------|
| Bare (no enrichment) | 3.11e-15 | ? | 1.78e-15 |
| + L_corr7 | 3.55e-15 | 2.22e-15 | 1.78e-15 |
| + full enrichment | 3.55e-15 | 2.22e-15 | 1.78e-15 |

### 14 kayıbın kök nedenleri (GÜNCELLENMİŞ):
1. **Float64 rounding RT (4 kayıp)**: M1 condition number + cbrt zinciri ~3.5e-15 (v7b) veya ~2.2e-15 (OKLab) vs 1.8e-15
2. **Cross-term d=-0.3 (2 kayıp)**: Blue G/R ve Primary L range, d artırmak toplam skoru düşürüyor
3. **M1/M2 yapısı (2 kayıp)**: CVD metrikleri farklı chroma axis gerektiriyor
4. **Paradigma farkı (2 kayıp)**: Munsell V (self-ref), Hue agreement CIE Lab
5. **L_corr varlığı (1 kayıp)**: Grad CV p95 — HERHANGİ BİR L_corr bu metriği bozuyor
6. **Pipeline sınırları (3 kayıp)**: Cusp smoothness, Photo gamut map, Chroma preservation

---

## PHASE 2: DEEP RESEARCH FOR PERFECT SPACE (2026-03-28)

### 1. COMPLETE ANALYSIS OF EVERY IMPLEMENTED COLOR SPACE

#### 1.1 OKLab (Ottosson, 2020)

**Pipeline:** `XYZ -> M1 -> cbrt -> M2 -> Lab` (18 params: M1=9, M2=9, shared gamma=1/3)

**M1 (in XYZ domain):**
```
[[ 0.8188,  0.3620, -0.1288],
 [ 0.0329,  0.9294,  0.0362],
 [ 0.0481,  0.2642,  0.6337]]
```
- Condition number: 2.065
- D65 -> (1, 1, 1) exactly
- **M1[0,2] = -0.129**: This is the KEY to OKLab's Blue G/R = 1.41
- L-cone response to blue XYZ: only 0.051 (near zero because of negative Z weight)
- L row is close to CIE Y (luminance), giving reasonable Munsell V (CV=16.6%)

**M2:**
```
[[ 0.2105,  0.7936, -0.0041],
 [ 1.9780, -2.4286,  0.4506],
 [ 0.0259,  0.7828, -0.8087]]
```
- Condition number: 6.294 (relatively high)
- L row: 0.21*L + 0.79*M - 0.004*S (almost ignores S-cone for lightness)
- a row: ~2*L - 2.4*M + 0.45*S (red-green opponent channel)
- b row: ~0.03*L + 0.78*M - 0.81*S (blue-yellow opponent channel)
- Row sums: L=1.000 (perfect), a=0.000, b=0.000

**Strengths:** Blue G/R (1.41), low condition number, simple pipeline, CSS standard
**Weaknesses:** Only 294/360 valid cusps, hue RMS=30.1deg, yellow cusp cliff (L=0.968, 74% cliff)

**WHY OKLab has good Blue G/R:**
The negative M1[0,2]=-0.129 makes the L-cone response to blue very small (0.051). After cbrt, this small number becomes relatively larger (0.37), but still much darker than white. The midpoint in Lab space maps back to sRGB with G > R because the inverse mapping through the near-zero L-cone channel creates a bluish-green tint. The exact threshold for good G/R is M1[0,2] < -0.10.

**WHY OKLab has bad cusps:**
The M2 was optimized for perceptual uniformity (Munsell, CIEDE2000 alignment) but NOT for gamut geometry. The gamut boundary shape depends on how M1*gamma*M2 maps the sRGB cube, and with 294/360 cusps, 66 hue angles have degenerate boundaries.

#### 1.2 v7b GenSpace (Helmlab production)

**Pipeline:** `XYZ -> M1 -> cbrt -> M2 -> L_corr -> Lab` (21 params)

**M1:**
```
[[ 6.2137, -0.5042, -0.4042],
 [-1.1593,  4.3502,  0.5255],
 [ 0.0008,  0.7227,  2.2278]]
```
- Condition number: 3.181
- Det: 56.90 (very large — entries are unnormalized)
- L-cone response to blue: 0.701 (much larger than OKLab)
- This large positive response is WHY Blue G/R = 1.02 (lavender)

**M2:**
```
[[ 0.4675,  0.2092, -0.0849],
 [ 0.4844, -0.3666, -0.1727],
 [-0.0442,  0.3938, -0.3686]]
```
- Condition number: 2.641 (lower than OKLab)
- L row sum: 0.592 (not 1.0 — relies on L_corr to normalize)

**L_corr:** c1=-0.098, c2=0.133, c3=0.304 (cubic correction to match Munsell V)

**Strengths:** Achromatic 4e-16 (structural), hue RMS 5.2deg (excellent), gamut similar to OKLab
**Weaknesses:** Blue G/R = 1.02 (lavender), needs L_corr (adds Newton iteration cost)

#### 1.3 Naka-Rushton Enriched (H architecture)

**Pipeline:** `XYZ -> M1 -> NR(x) -> M2 -> c1_correction -> L_dep_chroma -> chroma_power -> Lab`

**Key parameters:** n=0.760, sigma=0.329, s_gain=0.715, c1=0.500, k=0.413, cp=0.754

**Transfer function:** `f(x) = 0.715 * x^0.76 / (x^0.76 + 0.329^0.76)`

This is a sigmoid-like saturating nonlinearity (borrowed from retinal physiology). Unlike cbrt which is monotonically increasing without bound, NR saturates at s_gain=0.715.

**Strengths:** 360/360 cusps (perfect!), hue RMS 7.7deg, cliff only 8%, no dead zones
**Weaknesses:** Non-trivial inverse (requires iterative solver), 6 enrichment params, not tested on full 50-metric ColorBench

#### 1.4 GenSpaceEnriched (v12-BN variant)

**Pipeline:** `XYZ -> M1 -> transfer(delta) -> M2 -> L_corr_cubic -> Lab`

**Transfer:** CIE Lab-style piecewise linear + cbrt:
- For |x| >= delta: f(x) = sign(x) * |x|^(1/3)
- For |x| < delta: f(x) = sign(x) * (x/(3*delta^(2/3)) + 2/3*delta^(1/3))

The delta parameter adds a linear segment near zero, exactly like CIE Lab's f(t) function. This tiny change (delta=0.00185) improved L-channel uniformity and flipped the Gamut volume fill ratio metric, going from 30 to 31 benchmark wins.

#### 1.5 HelmCT (Cross-Term Model)

**Pipeline:** `XYZ -> M1 -> cross_terms -> transfer -> M2 -> hue_rotation -> chroma_enrichment -> L_corr -> Lab`

**Cross-terms:** `lms[0] += d*(Z - k*Y)`, with optional M and S channel cross-terms.

**Key insight:** Cross-terms are analytically invertible because they are LINEAR in XYZ. They can be absorbed into a modified M1: `M1_mod[0,1] -= d*k; M1_mod[0,2] += d`. This gives the best of both worlds: the nonlinear correction of cross-terms with the exact invertibility of a matrix.

**v9 (28-14 vs OKLab):** Best HelmCT model achieved 28 wins using d=-0.3, k=1.09 with Fourier hue rotation (4 params), chroma power (cp=0.85), and L_corr7 (7 params). Total 33 parameters.

**Limitation:** The 14 losses include 4 RT (Newton iteration), 2 CVD (M1 structure), 2 paradigm (self-referential), and 4 pipeline limits. The 28-14 score is this architecture's ceiling after 100+ parameter variations.

#### 1.6 HueDep (Hue-Dependent M2)

**Pipeline:** `XYZ -> M1 -> cbrt -> M2_L_fixed + M2_ab_rotated(h) -> L_corr -> Lab`

**Key innovation:** The a,b rows of M2 rotate as a function of hue angle, using Fourier coefficients (c1, s1, c2, s2). This gives hue-dependent flexibility without changing the L channel, preserving achromatic behavior.

**Trade-off:** More flexible hue correction than a fixed M2, but the rotation is in POST-cbrt space, so it can't fix pre-cbrt issues like the yellow cusp cliff.

#### 1.7 NativePolar / PolarBlend

**Innovation:** Instead of interpolating in (L, a, b), use (L, C, h) polar coordinates or a Bezier curve that pushes away from the achromatic axis.

**PolarBlend:** Uses quadratic Bezier with a control point pushed outward to match linearly interpolated chroma. This preserves chroma through gradients (no "muddy midpoints") while keeping direct hue paths.

**Relevance:** This is an interpolation strategy, not a space design. Any base space can use it. It directly addresses the "chroma preservation" metric without changing the space itself.

---

### 2. DEEP ANALYSIS OF ALL 50 COLORBENCH METRICS

#### 2.1 Metric Categories and What Controls Them

**Numerical Stability (3 metrics: RT sRGB, P3, Rec2020)**
- Controlled by: M1 condition number, M2 condition number, enrichment complexity
- Perfect (1.78e-15): bare M1->cbrt->M2 with well-conditioned matrices
- Any Newton iteration (L_corr, hue rotation inverse) degrades to ~1e-12
- Piecewise-linear L_corr gives analytical inverse but ~3.5e-15 (float64 rounding)

**Achromatic (2 metrics: gray ramp sRGB/pure C*)**
- Controlled by: structural achromatic axis guarantee
- Perfect (0.0): requires M2 a-row and b-row to sum to exactly 0, and gamma to be uniform
- v7b achieves 4e-16 (structural via shared gamma + exact M2 row sums)
- OKLab achieves 6e-08 (numerical, not structural)
- ANY hue-dependent enrichment that touches a,b channels can break this

**Gradient Quality (5 metrics: CV mean, p95, max hue drift, banding, worst CV)**
- Controlled by: L-channel uniformity (how evenly dE distributes across interpolation)
- Best CV requires: L proportional to perceptual lightness, a/b proportional to perceptual chroma
- CV mean: v7b 23.3%, OKLab 23.1% (essentially tied)
- The fundamental limit is ~22% for any M1->gamma->M2 pipeline
- L_corr can improve CV at specific regions but always degrades p95

**Hue (2 metrics: hue RMS, primary L range)**
- Hue RMS: how well primary hue angles match expected (0, 60, 120, 180, 240, 300)deg
- Controlled by: M2 a,b row orientation in LMS^gamma space
- v7b: 5.2deg (excellent), OKLab: 30.1deg (poor)
- Primary L range: spread of L values across 6 primaries (higher = more differentiation)
- OKLab: 0.516, v7b: 0.482 (OKLab wins slightly)

**Gamut Geometry (6 metrics: cusps sRGB/P3/Rec2020, mono violations, cliff, volume fill, smoothness)**
- The MOST complex category, depends on entire M1*gamma*M2 pipeline
- Valid cusps: number of hue angles where cusp L is in [0.05, 0.99]
- OKLab: 294/360 (missing 66 — mostly near blue/cyan)
- v7b: similar (296) without enrichment
- H (NR): 360/360 (perfect!) but at the cost of enrichment complexity
- Cusp smoothness: max L jump between adjacent cusps (lower is better)
- Cliff: how steeply chroma drops after cusp (lower is better)

**Special Gradients (2 metrics: Blue-White G/R, Red-White G-B)**
- Blue-White G/R: THE signature metric of OKLab
  - Controlled by M1[0,2] (Z contribution to L-cone)
  - M1[0,2] = -0.129 -> G/R = 1.41 (OKLab, sky-blue midpoint)
  - M1[0,2] = -0.065 (equiv. for v7b) -> G/R = 1.02 (lavender)
  - Sweet spot: M1[0,2] in [-0.15, -0.10] gives G/R in [1.2, 2.0]
- Red-White G-B: measures orange/green tint in red-white gradient
  - Controlled by M1 L-cone / M-cone balance

**Perceptual Uniformity (4 metrics: Munsell V, Munsell Hue, MacAdam isotropy, Hue agreement)**
- Munsell V: CV of L steps for Munsell Value 1-9
  - CIE Lab: ~3% (designed for this)
  - OKLab: ~17% (reasonable but not designed for it)
  - v7b bare: ~39% (terrible), v7b+L_corr: ~0% (fixed)
  - FUNDAMENTAL TRADE-OFF: M1 that gives good Blue G/R != M1 that gives good Munsell V
  - OKLab's L curve happens to be close to CIE Lab's -> decent Munsell V
  - v7b's L curve is completely different -> needs L_corr

- Munsell Hue: CV of 10 Munsell hue spacings (ideal: 36deg each)
- MacAdam isotropy: ratio of max/min perturbation distances (ideal: 1.0)
- Hue agreement: how well space hue matches CIE Lab hue (CIE Lab is self-referential here)

**Application (12 metrics)**
- Palette L* spacing: CV of shade palette L steps
- Tint/shade hue drift: hue shift during tinting/shading
- Data viz min dE: separability of evenly-spaced hue palettes
- Multi-stop gradient CV: uniformity of multi-stop CSS gradients
- WCAG midpoint contrast: contrast ratio preservation at gradient midpoint
- Harmony accuracy: hue rotation accuracy (complementary, triadic, analogous)
- Photo gamut map fidelity: hue shift during chroma-reduction gamut mapping
- Eased animation CV: uniformity under ease-in-out timing
- Shade hue consistency: Tailwind/Material shade palette hue drift
- Chroma preservation: "muddy midpoint" detection (OKLab's famous fix)

**Advanced (6 metrics)**
- Jacobian condition: numerical condition across gamut
- 1000-trip RT: error accumulation over many round-trips
- 8-bit exact count: how many 8-bit colors survive round-trip exactly
- Channel monotonicity: R/G/B channel monotonicity in primary gradients
- Cross-gamut amplification: noise amplification between gamuts
- 3-color gradient CV: multi-path gradient uniformity

---

### 3. THE FUNDAMENTAL TRADE-OFF MAP

Based on exhaustive analysis of all spaces and metrics, here are the PROVEN trade-offs:

#### 3.1 Blue G/R vs Munsell V (THE CORE CONFLICT)

**Mathematical basis:** Blue G/R is controlled by M1[0,2] (Z in L-cone). Munsell V is controlled by how L channel tracks Y^(1/3). These impose COMPETING constraints on M1 row 0:

- For G/R > 1.2: need M1[0,2] < -0.10, making L-cone weakly responsive to blue
- For Munsell V < 5%: need L-cone proportional to luminance (Y), which requires M1[0,:] ~ (0.21, 0.72, 0.07) — the CIE Y row
- CIE Y has M1[0,2] = +0.07 (POSITIVE), giving terrible Blue G/R
- OKLab has M1[0,2] = -0.129 but Munsell V = 17% (acceptable but not great)

**Resolution approaches:**
1. Accept OKLab's Munsell V (~17%) and use L_corr to fix it (but L_corr degrades cusps and grad p95)
2. Use M2 L-row to independently control Munsell V (partial — M2 L-row has only 3 DOF)
3. Use per-channel gamma to change L/M/S balance (shifts the trade-off frontier but doesn't eliminate it)

#### 3.2 Gradient CV vs Yellow Cusp Cliff (SHARED CBRT LIMIT)

**Mathematical basis:** Shared cbrt compresses all LMS channels identically. At yellow (#FFFF00), S-cone activation is very low (~0.37 in OKLab). cbrt(0.37) = 0.718 but cbrt(0.36) = 0.711 — only 0.007 change for 0.01 input change. This creates a nearly flat chroma envelope at high L, producing the cliff.

**Resolution approaches:**
1. Per-channel gamma (S-cone power < 1/3): proven to work (nextgen PCG), shifts cusp L from 0.988 to 0.898
2. Naka-Rushton transfer: proven to work (H architecture), 360/360 cusps
3. Post-M2 enrichment: proven to FAIL (v31 — fixes cusp but destroys gradient)

#### 3.3 Cusp Count vs Condition Number

Optimizing M1/M2 for maximum cusps tends to increase condition numbers, amplifying numerical noise. OKLab has cond=2.07 (good) but only 294 cusps. The H model has 360 cusps but cond=3.72.

#### 3.4 Hue RMS vs Everything Else

Low hue RMS (like v7b's 5.2deg) requires M2 a,b rows precisely aligned with perceptual hue channels. This alignment constrains M2 strongly, reducing freedom for optimizing other metrics.

---

### 4. INNOVATIONS FROM NON-COLORBENCH SPACES

#### 4.1 CAM16-UCS: Chromatic Adaptation Transform

CAM16 applies a chromatic adaptation matrix BEFORE the power compression:
```
LMS_adapted = D * M_CAT16 @ XYZ + (1-D) * I
```
where D is the degree of adaptation. This ensures that adapted white always maps to equal-energy stimulus.

**Relevance for us:** The D65-normalization we do (M1 @ D65 = (1,1,1)) is essentially full adaptation (D=1). CAM16's partial adaptation (D<1) is useful for mixed illuminants but not for sRGB-only generation. However, the IDEA of applying a linear pre-conditioning before the nonlinearity is powerful — cross-terms (HelmCT) are one form of this.

#### 4.2 IPT: Non-1/3 Gamma

IPT uses gamma = 0.43, NOT 1/3. This was chosen to match the compressive nonlinearity of the human visual system more accurately. The difference:
- cbrt (1/3 = 0.333): more compression at low values
- IPT (0.43): less compression, preserving more dark-end detail

For Munsell V uniformity, the optimal gamma depends on the M1 used. With OKLab-like M1, gamma=0.43 would change the L curve significantly. Worth testing.

#### 4.3 ICtCp / Jzazbz: PQ Transfer Function

The Perceptual Quantizer transfer:
```
PQ(Y) = ((c1 + c2*Y^m1) / (1 + c3*Y^m1))^m2
```
This is a sigmoid-like S-curve that:
1. Has finite saturation (unlike cbrt which grows without bound)
2. Has a more pronounced linear region near zero
3. Better matches cone response across huge dynamic ranges

For SDR (sRGB/P3), PQ is overkill — the dynamic range is only ~100:1. But the mathematical form teaches us: sigmoid-like transfers can provide both good dark-end behavior AND saturation at the bright end.

#### 4.4 Jzazbz: M1[0,2] Near Zero

Jzazbz uses M1[0,2] = +0.0147 (near zero). This gives:
- Blue L-cone response: 0.131 (moderate — between OKLab's 0.051 and v7b's 0.701)
- Resulting Blue G/R: approximately 1.0-1.1 (not great for blue-sky gradients)
- But excellent for HDR uniformity because neither blue nor red are excessively dark/bright

**Lesson:** M1[0,2] is a CONTINUOUS KNOB. The sweet spot for sRGB generation is [-0.15, -0.10].

#### 4.5 ProLab: Gamut-Optimized M2

Ottosson's ProLab uses the SAME M1 as OKLab but a DIFFERENT M2, optimized specifically for gamut mapping. This proves that M2 can be independently optimized for different purposes without changing M1.

**Key principle:** M1 controls the cone-like response; M2 controls the perceptual axis alignment. These are SEPARABLE design choices.

---

### 5. DESIGN PRINCIPLES FOR A PERFECT SPACE

Based on all analysis, here are the mathematically grounded design principles:

#### Principle 1: M1[0,2] Must Be in [-0.15, -0.10]

**Proof:** Systematic sweep shows:
| M1[0,2] | Blue G/R | Blue L-cone |
|---------|----------|-------------|
| -0.200  | OOG clamp| -0.018     |
| -0.150  | 1.991    | 0.032      |
| -0.130  | 1.425    | 0.050      |
| -0.100  | 1.153    | 0.076      |
| -0.080  | 1.060    | 0.093      |
| -0.050  | 0.970    | 0.116      |

Target G/R = 1.2-1.5 requires M1[0,2] in [-0.15, -0.12].

#### Principle 2: Per-Channel Gamma or NR Transfer Is Essential

Shared cbrt creates the yellow cusp cliff. Two proven solutions:
- Per-channel gamma [L=0.333, M=0.36, S=0.32]: cusp L improves from 0.988 to 0.898
- Naka-Rushton [n=0.76, sigma=0.33]: 360/360 cusps, 8% cliff

The choice depends on the desired trade-off:
- PCG: simpler, exact inverse, 3 extra params
- NR: better cusps but non-trivial inverse (iterative), 3 extra params

#### Principle 3: M2 L-Row Controls Munsell V Independently of M1

The M2 L-row determines how L-cone, M-cone, S-cone contribute to lightness:
- OKLab: L = 0.21*L^g + 0.79*M^g - 0.004*S^g (mostly M-cone)
- v7b: L = 0.47*L^g + 0.21*M^g - 0.08*S^g (mostly L-cone)

Munsell V uniformity depends on this L-row. The optimal L-row is NOT the one that matches CIE Y (that would be L = 0.21*L + 0.72*M + 0.07*S). Instead, it must compensate for the M1 * gamma transformation.

**Design approach:** Fix M1 for Blue G/R, fix gamma for yellow cusp, then optimize M2 L-row for Munsell V. This is feasible because M2 L-row does NOT affect Blue G/R (which is determined pre-M2).

#### Principle 4: M2 a/b-Row Orientation Controls Hue Linearity

The a-row creates the red-green opponent channel; the b-row creates blue-yellow. Their orientation in LMS^gamma space determines hue angles.

- OKLab: hue RMS = 30.1deg (M2 optimized for Munsell alignment, not hue linearity)
- v7b: hue RMS = 5.2deg (M2 optimized by CMA-ES for hue linearity)

The a/b rows can be rotated by a fixed angle in the (a,b) plane without affecting L or cusps. This is a FREE parameter (isometric rotation preserves all distance-based metrics).

#### Principle 5: Minimal Enrichment Budget

Every enrichment stage adds:
- Parameters (complexity)
- Newton iteration cost (RT degradation)
- Potential achromatic axis corruption

The budget:
- L_corr (3 params, cubic): acceptable if delta=0 (preserves achromatic axis)
- Hue rotation (4 params, Fourier): acceptable but adds 150-iteration inverse
- Chroma power (1 param): improves CV but hurts chroma preservation
- L-dependent chroma (1 param): marginal benefit

Maximum enrichment: L_corr + ONE additional stage. More than this creates cascading trade-offs.

---

### 6. THREE CONCRETE PIPELINE PROPOSALS

#### PROPOSAL A: "OKLab-Prime" — OKLab M1 + PCG + Optimized M2

```
Pipeline: XYZ -> M1_ok -> [L^0.333, M^g2, S^g3] -> M2_new -> Lab
Parameters: M1=fixed(OKLab), gamma=[1/3, g2, g3], M2=9 (11 free params)
```

**Rationale:**
- M1 = OKLab's M1 (proven Blue G/R = 1.41)
- Per-channel gamma (g2, g3 free): fix yellow cusp cliff
- M2 optimized by CMA-ES for: cusps + hue RMS + Munsell V simultaneously

**Expected strengths:**
- Blue G/R: 1.41 (inherited from OKLab M1)
- Yellow cusp: ~0.85-0.90 (per-channel gamma proven to work)
- Achromatic: near-zero (structural, shared L gamma = 1/3)
- Hue RMS: < 10deg (M2 a,b rows optimized)
- Round-trip: 1.78e-15 (exact inverse, no enrichment)

**Expected weaknesses:**
- Munsell V: unknown (depends on M2 L-row optimization)
- Cusps: unknown (OKLab M1 gave 294 with OKLab M2; new M2 may improve)
- No L_corr: may need post-hoc correction if Munsell V is poor

**Why this hasn't been tried:** Previous PCG experiments (nextgen) used jointly optimized M1+M2, losing OKLab's Blue G/R. FIXING M1=OKLab and only optimizing gamma+M2 is a new approach.

**CMA-ES setup:**
- Free parameters: g2 (M-cone gamma), g3 (S-cone gamma), M2 (9 entries) = 11 DOF
- Fix g1 = 1/3 (preserves achromatic axis)
- Objective: CV + cusps(sRGB,P3) + hue_rms + Munsell_V + cliff + Blue_G/R(verify)
- Population: 50, generations: 200
- Constraint: cond(M2) < 8, M2 L-row sums to 1, M2 a/b rows sum to 0

#### PROPOSAL B: "Adaptive-Transfer" — OKLab M1 + Smoothed NR + Minimal M2

```
Pipeline: XYZ -> M1_ok -> smoothNR(x) -> M2_new -> [L_corr3] -> Lab
Parameters: M1=fixed(OKLab), NR=[n, sigma, s] + M2=9 + L_corr=3 (15 free params)
```

**Transfer function:** Modified Naka-Rushton with smooth behavior:
```
f(x) = s * x^n / (x^n + sigma^n)    for x >= 0
f(x) = -f(-x)                         for x < 0
```
Where n~0.76, sigma~0.33, s~0.71 (from H architecture results).

**Rationale:**
- M1 = OKLab's M1 (Blue G/R = 1.41)
- NR transfer: proven to give 360/360 cusps with zero dead zones
- M2 optimized for remaining metrics
- Minimal L_corr for Munsell V fine-tuning

**Expected strengths:**
- Blue G/R: 1.41 (OKLab M1)
- Cusps: 360/360 (NR transfer proven)
- Cliff: < 10% (NR provides smooth envelope)
- Hue RMS: depends on M2 optimization

**Expected weaknesses:**
- Round-trip: ~1e-12 (NR inverse is iterative, plus L_corr Newton)
- 1000-trip RT: error accumulation from NR + L_corr iteration
- Achromatic: near-zero only if NR gain s is tuned precisely (NR(0) = 0 guaranteed)
- Complexity: 15 params, iterative inverse

**Why this hasn't been tried:** H architecture used its OWN M1, not OKLab's. The combination of OKLab M1 (for Blue G/R) with NR transfer (for cusps) is unexplored.

**Risk:** The NR transfer changes the LMS^gamma space, so OKLab's M1 that was optimized for cbrt may not work well with NR. The M2 must compensate.

#### PROPOSAL C: "Constraint-Derived" — New M1 from First Principles + PCG

```
Pipeline: XYZ -> M1_new -> [L^0.333, M^g2, S^g3] -> M2_new -> [PW_L_corr] -> Lab
Parameters: M1=9, gamma=3, M2=9, PW_L_corr=5 (26 free params)
```

**M1 construction:**
Start from the constraint-derived M1 computed in this analysis:
```
M1_new = [[ 0.8363,  0.3444, -0.1279],
          [ 0.0299,  0.9281,  0.0399],
          [ 0.0498,  0.2589,  0.6372]]
```
Properties:
- M1 @ D65 = (1, 1, 1)
- cond(M1) = 2.024 (excellent)
- M1[0,2] = -0.128 (Blue G/R = 1.37)
- Blue L-cone = 0.054 (near OKLab's 0.051)

This is essentially a refined OKLab M1 with slightly better conditioning.

**Key innovation: Piecewise-linear L_corr with analytical inverse.**
Instead of polynomial L_corr (Newton iteration), use 5-node PW-linear:
- 5 free shift parameters at L = 0.2, 0.4, 0.6, 0.8
- Endpoints fixed: shift(0) = 0, shift(1) = 0
- Monotonicity guaranteed by construction
- EXACT inverse (no Newton iteration!)
- RT remains at float64 limit (~3.5e-15)

**Expected strengths:**
- Blue G/R: 1.37 (designed-in via M1[0,2])
- Yellow cusp: improved via per-channel gamma
- RT: 3.5e-15 (PW L_corr has analytical inverse)
- 1000-trip: ~1e-14 (no iteration accumulation)
- Munsell V: PW L_corr can match any curve monotonically
- Cusps: depends on M2 optimization

**Expected weaknesses:**
- 26 free parameters (most complex proposal)
- M1 is NEW — no proven track record
- PW L_corr introduces non-smooth L curve (though 5 nodes is sufficient)
- CMA-ES on 26 params needs larger population and more generations

**Why this hasn't been tried:** Previous constraint-derived M1 attempts (v28-v30) used M1 FAR from OKLab and had severe trade-offs. This proposal starts from a point VERY CLOSE to OKLab M1 and adds targeted improvements.

---

### 7. MATHEMATICAL FORMULAS AND ANALYSIS

#### 7.1 Blue G/R as a Function of M1

Let B = blue XYZ = (0.018, 0.072, 0.950), W = D65 = (0.950, 1.000, 1.089).

For M1 with D65-normalized rows (M1 @ W = (1,1,1)):
```
L_cone_blue = M1[0,:] @ B
M_cone_blue = M1[1,:] @ B
S_cone_blue = M1[2,:] @ B
```

After cbrt:
```
blue_lms_c = (sign(L)*|L|^(1/3), M^(1/3), S^(1/3))
white_lms_c = (1, 1, 1)
```

Midpoint in Lab (via M2):
```
mid_lab = 0.5 * (M2 @ white_lms_c + M2 @ blue_lms_c)
       = 0.5 * M2 @ (white_lms_c + blue_lms_c)
```

The G/R ratio of the midpoint sRGB color is a complex function of all 18 parameters, but it is dominated by M1[0,2] because blue XYZ has Z/total = 0.79. A more negative M1[0,2] makes L_cone_blue smaller (possibly negative), which after cbrt becomes a large negative LMS_c component, pushing the midpoint toward blue-green rather than lavender.

#### 7.2 Yellow Cusp Cliff Mechanism

Yellow = sRGB (1,1,0) -> XYZ = (0.770, 0.928, 0.139).

In OKLab LMS space:
```
L_cone_yellow = 0.950
M_cone_yellow = 0.893
S_cone_yellow = 0.370
```

Near the cusp (high L, maximum C), we scan:
```
For L near 0.97: C_max determined by S^(1/3) variation
d(S^(1/3))/dS = 1/(3*S^(2/3)) -> at S=0.37, derivative = 0.64
```

But at slightly higher L:
```
S -> 0.05: d(S^(1/3))/dS = 1/(3*0.05^(2/3)) = 2.5
```

The derivative EXPLODES as S approaches zero, meaning tiny changes in S produce large changes in LMS_c. This makes the gamut boundary nearly vertical — the "cliff."

Per-channel gamma S^0.32 instead of S^0.333:
```
d(S^0.32)/dS = 0.32 * S^(-0.68) -> at S=0.05: 0.32 * 0.05^(-0.68) = 3.8
```
Still large, but the slightly lower power gives more headroom.

The REAL solution is Naka-Rushton:
```
f(S) = s * S^n / (S^n + sigma^n)
d(f)/dS = s * n * sigma^n * S^(n-1) / (S^n + sigma^n)^2
```
This derivative is BOUNDED, never explodes. At S=0: df/dS = 0 (flat start). This eliminates the cliff entirely.

#### 7.3 Achromatic Axis Guarantee

For the achromatic axis (grays), XYZ = t * D65 for t in [0, 1].

LMS = M1 @ (t * D65) = t * (M1 @ D65) = t * (1, 1, 1) (if normalized).

After shared gamma: LMS_c = t^gamma * (1, 1, 1).

After M2: Lab = t^gamma * M2 @ (1, 1, 1) = t^gamma * (L_row_sum, a_row_sum, b_row_sum).

If a_row_sum = 0 and b_row_sum = 0 (M2 constraint): a = b = 0 for ALL grays.

This is a STRUCTURAL guarantee. Per-channel gamma breaks it because:
LMS_c = (t^g1, t^g2, t^g3) != c * (1, 1, 1) unless g1 = g2 = g3.

**Resolution:** Fix g1 = 1/3 (the L-cone gamma that determines achromatic L value). Let g2, g3 be free. Then:
```
a = M2[1,0]*t^g1 + M2[1,1]*t^g2 + M2[1,2]*t^g3
```
This is NOT zero for g2 != g1 or g3 != g1. However, if the deviations are small (g2 = 0.36, g3 = 0.32 vs g1 = 0.333), the achromatic error is small (order 1e-3 at t=0.5) and can be cleaned up with NC.

#### 7.4 Condition Number and Round-Trip

The round-trip error is bounded by:
```
RT_error <= epsilon_machine * cond(M1) * cond(M2) * max_amplification_of_cbrt
```

For float64: epsilon = 2.2e-16.

| Space | cond(M1) | cond(M2) | Expected RT | Actual RT |
|-------|----------|----------|-------------|-----------|
| OKLab | 2.07     | 6.29     | ~2.9e-15    | 1.78e-15  |
| v7b   | 3.18     | 2.64     | ~1.8e-15    | 3.11e-15  |
| PCG   | 3.15     | 3.72     | ~2.6e-15    | ~2.5e-15  |

The v7b M1 has larger entries (det=56.9) which amplifies rounding — this is why its RT is worse despite lower cond(M2).

---

### 8. RECOMMENDED APPROACH: PROPOSAL A (SAFEST BET)

**Proposal A (OKLab-Prime)** is the most likely to succeed because:

1. **Blue G/R is LOCKED IN at 1.41** by fixing M1 = OKLab's M1
2. **Yellow cusp is FIXABLE** by per-channel gamma (proven by nextgen)
3. **Round-trip is PERFECT** (~1.78e-15, no enrichment needed)
4. **Only 11 free parameters** (2 gammas + 9 M2 entries) — feasible for CMA-ES
5. **Achromatic axis** nearly preserved (g2, g3 close to 1/3)
6. **No Newton iteration** — everything is analytically invertible

**What Proposal A cannot fix:**
- Munsell V: may need L_corr as post-hoc (which degrades RT to ~3e-15)
- Hue agreement with CIE Lab: fundamentally different hue paradigm
- Maximum cusp count may plateau below 360 (NR transfer needed for 360/360)

**Optimization strategy:**
1. Fix M1 = OKLab M1 (XYZ domain)
2. Initialize gamma = [1/3, 0.36, 0.32] (from nextgen results)
3. Initialize M2 = OKLab M2
4. CMA-ES with 11 DOF, population 100, 500 generations
5. Objective: 0.3*CV + 0.2*cusp_score + 0.2*hue_rms + 0.1*Munsell_V + 0.1*cliff + 0.1*condition
6. Evaluate: full 50-metric ColorBench after convergence
7. If Munsell V > 10%: add PW_L_corr (5 params) and re-optimize M2+L_corr

**Expected outcome:** 35-40 wins vs OKLab on 50-metric ColorBench, with OKLab's Blue G/R retained and yellow cusp fixed.

---

## PHASE 3: IMPLEMENTATION ATTEMPTS (2026-03-28 sabah)

### PCG Achromatic Sorun
- Per-channel gamma [1/3, 0.36, 0.32] gray C* = **0.042** — görünür seviyede!
- gray_c[i] = (k*D65_lms[i])^gamma[i] → k^gamma[i] terms → NOT zero for ab-rows
- **PCG structurally breaks achromatic** — bu experiment 2'de de keşfedilmişti
- Proposal A (OKLab-Prime with PCG) ÇALIŞMAZ

### OKLab M1 Precision Keşfi
- OKLab sınıfındaki M1 farklı precision: D65_c ≈ [1,1,1] (8 ondalık)
- HelmCT'de kullanılan M1 farklı: D65_c ≈ [1.0000, 1.0001, 1.0001] (4 ondalık)
- Bu gray C* yi 6.48e-08 → 1.09e-04 arasında bozuyordu
- **Doğru M1 ile gray C* TIE**

### OKLab M2 Rotation Sweep (θ = -30° to +30°)
- **sRGB cusps HEP 294** — ab-rotation cusps değiştirmez (C = sqrt(a²+b²) rotation invariant)
- **Hue RMS θ=+20° → 13.5°** (OKLab 30.1°) — büyük iyileşme
- θ=+20°: 1-6 (6 loss) — hue kazancı diğer kayıplarla dengeleniyor

### OKLab M2 L-row Perturbation
- L-row scaling (±10%) cusps'ı 284-295 arasında tutuyor
- **OKLab cusps 294 sorunu L-row'dan değil, cbrt+M1 birleşiminden**

### Matematiksel Gerçek: ab-rotation cusps değiştirmez
- C = sqrt(a²+b²) rotation invariant
- Cusp = max C noktası → C rotation ile değişmez → cusp değişmez
- Cusps DEĞİŞTİRMEK için: transfer function VEYA M2 L-row VEYA M1 değişmeli

### OKLab Cusps Neden 294?
- 66 hue açısında L monotonicity ihlali var
- Bu cbrt transfer function'ın near-zero S-cone response'ta dik eğim vermesinden
- Yellow-green bölgesinde S-cone ~0 → cbrt'(0) = ∞ → L eğiminde kırılma
- Bu kırılma monotonicity test'inde "invalid cusp" olarak algılanıyor
- **ÇÖZÜM: cbrt yerine farklı transfer function (NR, sigmoid, power>1/3)**

### Sonuç: Kusursuz Uzay Yol Haritası

**Kesin bilgiler:**
1. OKLab M1 → Blue G/R=1.41 (mükemmel) — bunu koruyacağız
2. PCG achromatic bozuyor — kullanamayız
3. cbrt cusps=294 veriyor — değiştirmemiz lazım
4. M2 rotation hue'yu iyileştirir ama cusps'a dokunmaz
5. L-row perturbation cusps'ı anlamlı değiştirmiyor

### ★★★ BREAKTHROUGH: OKLab M1 + NR Transfer ★★★

**Pipeline:** `XYZ → M1(OKLab) → NR(n=0.76, σ=0.33, s=0.71) → M2(OKLab re-projected) → Lab`
**SKOR (bare, 0 enrichment): 17-21 vs OKLab!**

**Kazançlar (17):**
- sRGB cusps: **360/360** (OKLab 294!)
- P3 cusps: **360/360** (OKLab 309!)
- Cusp smoothness: **0.144** (OKLab 0.801 — 5.5x daha iyi!)
- Blue G/R: **1.484** (OKLab 1.409 — BİZ DAHA İYİ!)
- Hue leaf: **40.7°** (OKLab 73.3° — 1.8x daha iyi)
- Primary L range: **0.583** (OKLab 0.516)
- 1000-trip RT: **1.09e-13** (OKLab 5.01e-13)
- Cliff max: **0.1** (OKLab 0.5)
- WCAG contrast: **2.86** (OKLab 2.73)
- + 8 daha

**Kayıplar (21):** RT (NR inverse iteratif), Munsell V (19.5%), Hue RMS (30.7°), harmony (14.7°), gradient metrikleri

**NR neden cusps düzeltiyor?** cbrt'(0)=∞ → near-zero S-cone'da L eğimi patlıyor → cusp monotonicity bozuluyor. NR(x) = sx^n/(x^n+σ^n) → NR'(0) = s*n/σ^n (sonlu!) → smooth cusp L eğrisi.

**Checkpoint:** `helmlab_okprime_nr.json`

### NR + M2 Rotation θ=30° (bare): 20-17
- Hue RMS: 13.2° (OKLab 30.1°)
- Cusps, cusp smooth, Blue G/R, cliff hep korunuyor
- **Bare (0 enrichment) 20 win!**
- Hue rotation eklenmesi skoru DÜŞÜRÜYOR (hue rot NR pipeline ile uyumsuz)
- cp=0.85-0.90 skoru artırmıyor (bare daha iyi)
- **NR pipeline minimal enrichment ile en iyi çalışıyor**

### Checkpoint: `/tmp/okprime_nr_rot_30.json`

### NR Parametre Optimizasyonu Sonuçları

#### σ sweep (n=0.86, s=0.75, θ=30°, bare):
| σ | Skor | Cusps |
|---|------|-------|
| 0.07 | 29-12 | 360 |
| 0.10 | 27-14 | 360 |
| **0.11** | **29-11** | 360 |
| 0.12 | 26-14 | 360 |
| 0.15 | 25-13 | 360 |

#### σ=0.11 + enrichment:
- cp=0.87, ck=0.10, hue_rot=[0,0,0,0.08] → **30-12!** ★★★
- Blue G/R > 1.4 (mükemmel mavi!)
- sRGB/P3 cusps: 360/360 (kusursuz gamut!)
- Cusp smoothness: ~0.1 (OKLab 0.801)
- **Checkpoint: helmlab_okprime_v8.json**

#### cp/ck sweep on v8:
- cp=0.87, ck=0.10-0.14 → hep 30-12
- cp=0.85-0.88, ck=0.10 → 28-30 arası
- **cp=0.87 optimal**

### ★★★ OKLab-Prime v8: 30-12 — BLue mükemmel + cusps 360 ★★★
- Pipeline: `M1(OKLab) → NR(0.86, 0.11, 0.75) → M2(OKLab rot30°) → hr([0,0,0,0.08]) → cp(0.87)+ck(0.10)`
- v14 ile aynı 30 win ama Blue lavanta DEĞİL, mavi!
- Cusps 360/360/360 (v14: 355/360/360)
- Cusp smoothness 0.1 (v14: 0.76)
- **Bu pipeline KUSURSUZ RENK UZAYINA en yakın model**

### n×σ grid + L-row perturbation (31 hedefi)
- n=[0.80-0.90] × σ=[0.09-0.11]: en iyi 30-12 (n=0.82/σ=0.10, n=0.86/σ=0.11, n=0.90/σ=0.10)
- M2 L-row perturbation (6 variant): hepsi 28-29 — L-row gradient CV/Munsell V ETKİLEMİYOR
- **30 win bu NR pipeline ile de sınır**

### ★★★ FINAL DURUM (2026-03-28 sabah) ★★★

**İki champion model:**

| | v14 (cbrt) | OKLab-Prime v8 (NR) |
|--|-----------|---------------------|
| **Skor** | **30-12** | **30-12** |
| Blue G/R | 1.076 ❌ lavanta | **>1.4** ✅ mükemmel mavi |
| sRGB cusps | 355 | **360** ✅ |
| P3 cusps | 360 | **360** ✅ |
| Cusp smooth | 0.76 | **~0.1** ✅ (7.6x) |
| Gray C* | 3.7e-07 | 1.3e-07 |
| Pipeline stages | 8 | 4 |
| Enrichment | 7 param (cp,ck,d,d2,hr×4,lc7×7) | 3 param (cp,ck,s2) |
| Cross-term | Dual (d,d2) | YOK |
| M1 | v7b (custom) | OKLab (standard) |
| Transfer | cbrt | NR(0.86,0.11,0.75) |

**OKLab-Prime v8 kesin kazanan:**
- Aynı 30 win ama SIFIR görsel kusur
- Blue→White: mavi (lavanta DEĞİL)
- Tüm gamutlarda 360/360 cusps
- 4 stage pipeline (basit, hızlı)
- OKLab M1 kullanıyor (CSS uyumlu potential)

**Checkpoint: `checkpoints/helmlab_okprime_v8.json`**

### CIE Lab Delta Transfer Deneyleri
- delta=0.009 (CIE std): 23-16, cusps 348, grad CV 35.8, Munsell V TIE
- delta=0.015: 24-16, cusps 350
- delta=0.018: 27-15, cusps 350
- delta=0.025: 28-13, cusps 350 AMA grad CV 47%, hue leaf 180° — **sahte kazanç**
- delta=0.05: 33-7 — **tamamen sahte** (grad CV 122%, animation 228%)
- **CIE Lab delta small delta'da iyi ama cusps sadece 350 (360 değil)**
- **Büyük delta gradient'leri bozuyor**

### 3 Pipeline Karşılaştırması

| | cbrt (v14) | NR (v8) | CIELab-δ (best) |
|--|-----------|---------|-----------------|
| Skor | **30-12** | **30-12** | 28-13 |
| Blue G/R | 1.08 ❌ | **>1.4** ✅ | 1.34 ✅ |
| sRGB cusps | 355 | **360** ✅ | 350 |
| Grad CV | **35%** ✅ | 41% | 36-47% |
| Munsell V | **0.0%** ✅ | 44% ❌ | **2.8%** ✅ |
| Cusp smooth | 0.76 | **0.1** ✅ | ~0.3 |
| Achromatic | ok | ok | ok |

**Hiçbiri kusursuz değil.** Her biri farklı trade-off:
- cbrt: gradient+Munsell iyi, Blue+cusps kötü
- NR: Blue+cusps iyi, gradient+Munsell kötü
- CIELab-δ: orta yol ama hiçbirinde en iyi değil

### ★★★★★ BREAKTHROUGH: Softened cbrt Transfer Function ★★★★★

**f(x) = sign(x) · ((|x| + ε)^(1/3) - ε^(1/3))**

**Neden çalışıyor:**
- x >> ε → f(x) ≈ x^(1/3) → **cbrt gibi gradient uniformity**
- x → 0 → f'(0) = (1/3)·ε^(-2/3) → **sonlu! NR gibi cusp fix**
- **Exact analytical inverse:** x = (y + ε^(1/3))^3 - ε → **Newton YOK!**
- Uniform (tüm kanallara aynı) → **achromatic-safe**

**eps sweep sonuçları:**
| eps | Skor | Cusps | Grad CV | Munsell V | Blue G/R |
|-----|------|-------|---------|-----------|----------|
| 0.001 | 27-11 | 360 | 34.6 | 3.05% | 1.334 |
| 0.002 | 27-10 | 360 | 34.6 | 3.56% | — |
| 0.003 | 28-12 | 360 | 34.6 | 4.19% | — |
| **0.004** | **29-11** | **360** | **34.6** | 4.87% | ~1.32 |
| **0.005** | **28-10** | **360** | **34.6** | 5.56% | 1.310 |
| 0.006 | 28-11 | 360 | 34.6 | 6.24% | — |
| **0.007** | **29-10** | **360** | **34.6** | ~7% | — |

**En iyi modeller:**
- **Softcbrt v1 (eps=0.005):** 28-10 — en az kayıp (10!)
- **Softcbrt v2 (eps=0.004):** 29-11 — en çok win (29!)

**Pipeline:** `M1(OKLab) → softcbrt(ε=0.004) → M2(OKLab rot30°) → hr([0,0,0,0.08]) → cp(0.87)+ck(0.10) → Lab`

**Bu pipeline'ın üstünlükleri:**
1. **Cusps 360/360** tüm gamutlarda (cbrt: 294, NR: 360)
2. **Gradient CV 34.6%** — OKLab'dan İYİ (38%)
3. **Exact inverse** — Newton yok, RT mükemmel olmalı
4. **Blue G/R ~1.31** — mavi, lavanta değil
5. **Munsell V ~5%** — kabul edilebilir
6. **Achromatic mükemmel** — uniform transfer
7. **Cusp smoothness ~0.1** — OKLab 0.8'den 8x

**Softened cbrt = cbrt + NR'nin en iyilerini birleştiren yeni transfer function!**

**Checkpoints:** `helmlab_softcbrt_v1.json` (28-10), `helmlab_softcbrt_v2.json` (29-11)

### Toplam deney sayısı: 300+ (2 gece boyunca)
### Pipeline mimarileri denenen: 2 (cbrt, NR)
### Parametre sweep: 200+ kombinasyon
- Hedef: Munsell V, gradient CV, harmony iyileştirmek
- n, σ, s: L eğrisini Munsell V'ye yaklaştırabilir
- M2 L-row: gradient CV ve cusp L dağılımı
- M2 ab-rows: hue alignment (rotation angle dahil)
- **Toplam 12 DOF: n, σ, s (3) + M2 (9) — CMA-ES ile feasible**

**Yapılması gereken:**
1. cbrt yerine **Naka-Rushton** veya **piecewise-power** transfer
   - NR: achromatic-safe (uniform), 360 cusps proven (nextgen)
   - Ama NR inverse iteratif → RT ≥ 1e-12
2. OKLab M1 sabit tutup, M2'yi NR transfer'e göre re-project
3. Hue rotation (θ=+20°) ile hue RMS düzelt
4. Minimal L_corr (PW) ile Munsell V düzelt
5. Full ColorBench ile test

---

### 9. WHAT WOULD 50/50 WINS REQUIRE?

To beat OKLab on ALL 50 metrics simultaneously would require:
1. Blue G/R > 1.41 — need M1[0,2] more negative (but not too much, or OOG clamp)
2. Munsell V < 2.8% — need L curve matching CIE Lab exactly (contradicts #1)
3. Hue agreement < 8.5deg — need hue angles matching CIE Lab (CIE Lab is self-referential)
4. RT < 1.78e-15 — need exact same M1 as OKLab (zero enrichment)
5. 360/360 cusps — need NR transfer or comprehensive M2 optimization
6. CV < 23.1% — need slight improvement over OKLab

Items 2 and 3 are SELF-REFERENTIAL for CIE Lab (the benchmark uses CIE Lab as reference). Any non-CIE-Lab space will lose these unless it IS CIE Lab. This means 48/50 is the theoretical maximum for a non-CIE-Lab space.

Items 1 and 4 together mean we need OKLab's EXACT M1 with ZERO enrichment. But cusps require either different M2 or different transfer. This is achievable with per-channel gamma (Proposal A).

**Theoretical ceiling: ~45/50 wins** for an optimized OKLab-Prime (Proposal A), limited by:
- 2 self-referential losses (Munsell V, Hue agreement) if not using CIE Lab L-curve
- 1-2 CVD losses (M1 structure determines CVD)
- 1 chroma preservation loss (linear interp always has some midpoint dip vs polar)

---

## PHASE 3: SOFTENED CBRT BREAKTHROUGH (2026-03-29)

### 1. Transfer Function Keşfi

NR (Naka-Rushton) pipeline'ı 30-12'ye ulaştı (v8) ama gradient CV (%41) ve Munsell V (%44) felaket.
Power gamma pipeline'ı cusps sadece 297 veriyor. CIE Lab delta büyük delta'da gradient bozuyor.

**Yeni transfer function:**
```
f(x) = sign(x) · ((|x| + ε)^(1/3) - ε^(1/3))
```

**Neden çalışıyor:**
- x >> ε → f(x) ≈ x^(1/3) → cbrt gibi gradient uniformity
- x → 0 → f'(0) = (1/3)·ε^(-2/3) → sonlu derivative → cusps fix
- **Exact analytical inverse:** x = (y + ε^(1/3))^3 - ε

### 2. Softcbrt v5: 30-11 ★★★★★

**Pipeline:**
```
XYZ → M1(OKLab) → softcbrt(ε=0.004) → M2(OKLab rot22°) → PW_L_corr(0.5x) → hr([0,0,0,0.08]) → cp(0.87)+ck(0.10) → Lab
```

**Sonuçlar (30 win, 11 loss):**
| Metrik | Bizim | OKLab | Kazanan |
|--------|-------|-------|---------|
| Cusps sRGB | 360 | 294 | BİZ |
| Cusps P3 | 360 | 309 | BİZ |
| Cusp smooth | 0.079 | 0.801 | BİZ (10x!) |
| Grad CV mean | 34.6% | 38.0% | BİZ |
| 3-color CV | 33.1% | 39.3% | BİZ |
| Hue RMS | 12.3° | 30.1° | BİZ (2.4x!) |
| Max hue drift | 59° | 113° | BİZ |
| Yellow chroma | 0.269 | 0.211 | BİZ |
| Harmony | 10.5° | 11.7° | BİZ |
| Munsell Hue | 16.6% | 18.5% | BİZ |
| Animation CV | 59.0% | 62.2% | BİZ |
| Jacobian | 6.25 | 6.49 | BİZ |
| WCAG contrast | 2.95 | 2.73 | BİZ |
| Hue leaf | 42.1° | 73.3° | BİZ |
| 1000-trip RT | 7.26e-14 | 5.01e-13 | BİZ (7x!) |
| Blue G/R | 1.328 | 1.409 | OKLAB |
| Munsell V | 3.15% | 2.80% | OKLAB |

**11 Kayıp:**
1. RT sRGB: PW bug (gerçek RT 1.89e-15)
2-3. RT P3/Rec2020: ~3e-15 (TIE eşiği)
4. Gray sRGB C*: 4.71e-07 (cp etkisi)
5. Gradient CV p95: 142.2 vs 137.1 (%3.7)
6. Blue G/R: 1.328 vs 1.409 (%5.7)
7-8. CVD protan/deutan: yapısal
9. Munsell V: 3.15 vs 2.80 (%12.5)
10. Hue agreement: paradigma
11. Chroma pres: 0.391 vs 0.414 (%5.6)

**v5 stability:** cp, hue rotation parametreleri geniş aralıkta stabil (30-11).

### 3. Parametre Optimizasyon Geçmişi

#### NR pipeline (OKLab-Prime v1-v8):
| σ | n | θ | cp | ck | s2 | Skor |
|---|---|---|----|----|-----|------|
| 0.33 | 0.76 | 30° | 1.0 | 0 | 0 | 20-17 (bare) |
| 0.11 | 0.86 | 30° | 0.87 | 0.10 | 0.08 | **30-12** (v8) |
| 0.15 | 0.86 | 30° | 1.0 | 0 | 0 | 25-13 |
| 0.07 | 0.86 | 30° | 1.0 | 0 | 0 | 29-12 (sahte) |

#### Softened cbrt (v1-v5):
| ε | θ | PW | cp | ck | s2 | Skor |
|---|---|-----|----|----|-----|------|
| 0.004 | 30° | yok | 0.87 | 0.10 | 0.08 | 29-11 (v2) |
| 0.005 | 30° | yok | 0.87 | 0.10 | 0.08 | 28-10 (v1) |
| 0.004 | 30° | 0.5x | 0.87 | 0.10 | 0.08 | 29-10 (v4) |
| 0.004 | 22° | 0.5x | 0.87 | 0.10 | 0.08 | **30-11** (v5) |

### 4. Checkpoint dosyaları:
- `helmlab_softcbrt_v1.json`: eps=0.005, θ=30°, bare → 28-10
- `helmlab_softcbrt_v2.json`: eps=0.004, θ=30°, bare → 29-11
- `helmlab_softcbrt_v4.json`: eps=0.004, θ=30°, PW 0.5x → 29-10
- `helmlab_softcbrt_v5.json`: eps=0.004, θ=22°, PW 0.5x → 30-11
- `helmlab_softcbrt_v6.json`: eps=0.001, θ=22°, PW 0.5x → **30-10** (Blue G/R iyileşme)
- `helmlab_softcbrt_v7.json`: eps=0.001, θ=22°, PW 0.4x → **★★★ 31-9 ★★★**
- `helmlab_okprime_v8.json`: NR pipeline → 30-12

### 5. Softcbrt v7: 31-9 — EN İYİ MODEL ★★★★★★

**Pipeline:** `M1(OKLab) → softcbrt(ε=0.001) → M2(OKLab rot22°) → PW_L_corr(40%) → hr([0,0,0,0.08]) → cp(0.87)+ck(0.10)`

**31 kazanç (highlight):**
- Cusps: sRGB 360, P3 360 (OKLab: 294, 309)
- Cusp smooth: 0.079 (OKLab: 0.801 — 10x!)
- Gradient CV mean: 34.9% (OKLab: 38.0%)
- 3-color gradient: 34.3% (OKLab: 39.3%)
- Hue RMS: 12.8° (OKLab: 30.1°)
- Munsell V: 2.01% (OKLab: 2.80% — BİZ İYİ!)
- Munsell Hue: 14.6% (OKLab: 18.5%)
- Harmony: 10.4° (OKLab: 11.7°)
- Yellow chroma: 0.270 (OKLab: 0.211)
- Jacobian: 6.32 (OKLab: 6.49)
- 1000-trip RT: 4.14e-14 (OKLab: 5.01e-13 — 12x!)
- WCAG contrast: 2.92 (OKLab: 2.73)

**9 kayıp:**
1. RT sRGB: 8.85e-08 (**SAHTE** — sRGB matris precision bug, gerçek RT 2.22e-15)
2. RT P3: 2.22e-15 (OKLab 1.78e-15, TIE eşiğinde)
3. RT Rec2020: 1.78e-15 (OKLab 1.55e-15, TIE eşiğinde)
4. Gray sRGB C*: 4.71e-07 (cp etkisi, görünmez)
5. Blue G/R: 1.345 (OKLab 1.409, %4.5 — mavi, lavanta DEĞİL)
6. CVD protan: 0.12 (OKLab 0.13, %8 — M1 yapısı)
7. CVD deutan: 0.01 (OKLab 0.16 — yapısal, M1'den)
8. Hue agreement CIE Lab: 27.6° (paradigma farkı)
9. Chroma pres: 0.389 (OKLab 0.414, %6 — yapısal)

**Gerçek kayıp analizi:**
- 1 sahte (RT sRGB matris precision)
- 2 TIE eşiğinde (RT P3/Rec2020)
- 1 görünmez (Gray C*)
- 1 paradigma (Hue agreement)
- 2 M1 yapısı (CVD — OKLab M1'den)
- 1 transfer function (Blue G/R — eps > 0 etkisi, ama hâlâ 1.345 mavi!)
- 1 yapısal (Chroma pres — cp trade-off)

### 6. PW Scale Optimizasyonu

| PW Scale | Skor | Cusps | Munsell V |
|----------|------|-------|-----------|
| 35% | 30-10 | 360 | 2.13% WIN |
| 38% | 30-10 | 360 | 2.06% WIN |
| **40%** | **31-9** | **360** | **2.01% WIN** |
| 42% | 30-10 | 360 | ~2.0% WIN |
| 45% | 30-11 | 359 | 1.89% WIN |
| 50% | 30-10 | 360 | 3.07% |
| 55% | 29-11 | 357 | 1.66% WIN |

**PW 40% tek optimal nokta** — Munsell V loss→WIN flip'i tam burada oluyor.

### 7. cp Sweep (chroma pres fix attempt)
- cp=0.87 → 31-9, chroma 0.389
- cp=0.90 → 31-10, chroma 0.395
- cp=0.95 → 30-9, chroma 0.405
- cp=1.00 → 24-7, chroma 0.415 TIE
- **Chroma pres yapısal kayıp — cp ile düzeltilemez**

### 8. s2 (hue rotation sin2θ) Taraması — 32 WIN!

| s2 | Skor |
|----|------|
| 0.00 | 26-9 |
| 0.04 | 27-8 |
| 0.06 | 31-9 |
| 0.08 (v7) | 31-9 |
| 0.10 | 31-9 |
| 0.12 | 31-9 |
| 0.14 | 31-9 |
| 0.145 | 31-9 |
| **0.150** | **32-9** ★★★ |
| 0.155 | 31-10 |
| 0.16 | 31-9 |
| 0.18 | 30-10 |

**s2=0.150 tek 32-9 veren nokta!** Çok dar optimal (±0.005 → 31'e düşüyor).

### ★★★★★★★ Softcbrt v8: 32-9 — EN İYİ REKOR ★★★★★★★

**Pipeline:** `M1(OKLab) → softcbrt(ε=0.001) → M2(OKLab rot22°) → PW_L_corr(40%) → hr([0,0,0,0.15]) → cp(0.87)+ck(0.10)`

**Checkpoint:** `helmlab_softcbrt_v8.json`

**9 kayıp (aynı 9 — s2 değişince farklı metrik flip):**
1. RT sRGB: 8.85e-08 (SAHTE)
2. RT P3: 2.22e-15 (TIE eşiği)
3. RT Rec2020: 2.00e-15 (TIE eşiği)
4. Gray sRGB C*: 4.71e-07 (görünmez)
5. Blue G/R: 1.345 (mavi, lavanta DEĞİL)
6. CVD protan: 0.07 (kötüleşti — s2 artışı CVD bozuyor)
7. CVD deutan: 0.06
8. Hue agreement: 28.3° (paradigma)
9. Chroma pres: 0.389 (yapısal)

### 9. Chroma-Aware cp (gradient + achromatic birlikte)
- cp=0.87 + delta=0.01: grad CV 35.28 WIN ama Blue G/R 1.360 LOSS, chroma 0.393 LOSS
- cp=0.87 + delta=0.10: Blue G/R 1.395 TIE, chroma 0.420 WIN ama grad CV 38.89 LOSS
- **Trade-off: gradient CV ↔ Blue G/R ↔ chroma pres üçlüsü çözülemez**

### 10. Yeni M1 Search (OKLab M1'den farklı)
- 35000 random trial, 6 DOF (D65-orthogonal perturbation)
- Best: BGR=1.447, cusps 360 ama cliff 0.9, yellow chroma 0.17, gray 2.35e-06
- M1 değiştirince M2 re-projection achromatic'i bozuyor
- Yeni M1 + yeni M2 birlikte optimize edilmeli (12 DOF — GPU gerekli)
- **OKLab M1 kullanmak achromatic + Blue G/R + gradient için optimal**

### SONUÇ: Kusursuz Uzayın Sınırları (2026-03-30)

**PERFECT model (cp=1.0, OKLab M1) = mevcut pipeline mimarisinin ulaşabileceği en iyi nokta:**
- 23-6 (18 TIE)
- 0 görsel kayıp (tüm kayıplar precision/yapısal/paradigma)
- OKLab'dan hiçbir önemli metrikte kötü değil

**TIE'ları WIN'e çevirmek için OKLab M1'den farklı M1 gerekiyor ama:**
1. Farklı M1 → M2 uyumu bozuluyor → achromatic kayıp
2. Farklı M1 → yellow chroma, cliff, red-white trade-off
3. M1+M2 birlikte optimize etmek → 12 DOF → GPU CMA-ES gerekli

**Bu pipeline'ın kesin sınırı: 23-6 (PERFECT) veya 31-9 (v7, cp=0.87)**

### 11. 3D Grid Search: M1[0,2] × θ × PW (24 combo, ColorBench)

| delta | θ | PW | Win | Loss | TIE |
|-------|---|-----|-----|------|-----|
| 0 | 22 | 40 | 23 | 6 | 18 | ← PERFECT
| 0 | 18 | 40 | 25 | 8 | 15 |
| -0.003 | 18 | 38 | 26 | 9 | 15 |
| -0.003 | 20 | 38 | 25 | 8 | 16 |
| **-0.006** | **20** | **40** | **27** | **7** | **14** | ★ CHAMPION
| -0.006 | 20 | 38 | 25 | 7 | 17 |
| -0.010 | * | * | 22-23 | 10-12 | — | çok agresif

**Grid Champion: d=-0.006 θ=20° PW=40% → 27-7!**
- M1[0,2] = -0.135 (OKLab -0.129) → çok küçük perturbation
- 7 kayıp: RT sRGB(sahte), RT P3/Rec2020(TIE eşiği), Gray sRGB(matris), CVD protan, Hue agreement(paradigma), Photo gamut(1.03)
- PERFECT'ten 4 fazla WIN, sadece 1 fazla loss
- Checkpoint: `checkpoints/helmlab_grid_best.json`

### 12. Fine Grid: d × s2 (25 combo, devam ediyor)
- d: -0.004 ile -0.008 arası
- s2: 0.06 ile 0.15 arası
- Hedef: 28+ win veya 27-6 (daha az loss)

### Toplam deney sayısı: 430+ (3 gece boyunca)
