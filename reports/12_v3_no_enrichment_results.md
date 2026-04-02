# V3 No-Enrichment Test Results

**Date**: 2026-04-01
**Score**: 31-11-19 (vs production 36-6-19)

## Summary
Removing enrichment+PW gains:
- RT sRGB: ~5e-8 (better but still LOSS — HelmCT class has sRGB matrix issue)
- Analytical invertibility: PASS
- Cost: ~1.5x PASS

But loses:
- Blue G/R: 1.52→1.38 (now LOSS)
- Munsell Value: 0.03%→4.00% (now LOSS)
- Grad CV p95: 138→142 (now LOSS)
- Worst CV: now LOSS
- Jacobian: 6.37→6.83 (now LOSS)

## The Dilemma

| Feature | With Enrichment+PW | Without |
|---------|-------------------|---------|
| Blue G/R | 1.52 WIN | 1.38 LOSS |
| Munsell V | 0.03% WIN | 4.00% LOSS |
| RT sRGB | 1e-7 LOSS | ~5e-8 LOSS (still!) |
| Invertibility | Newton FAIL | Exact PASS |
| Cost | 2.5x FAIL | 1.5x PASS |
| Score | 36-6-19 | 31-11-19 |

KEY INSIGHT: RT sRGB is STILL 5.64e-8 even without enrichment!
This means the RT issue is NOT from enrichment Newton — it's from HelmCT's
cross_term/hue_correction stages that still exist in the HelmCT class.

## Conclusion

Pure removal of enrichment is NET NEGATIVE (-5 WINs).
Need to either:
1. Keep enrichment but fix its invertibility (maybe Fourier instead of sin²)
2. Or find analytically invertible alternatives for Blue G/R and Munsell
3. The RQ_L correction (5-knot spline) could replace PW for Munsell

## Next Step
- Add RQ_L correction to V3 to fix Munsell Value
- Find analytically invertible hue correction for Blue G/R
- Run ColorBench on V3 + RQ_L + analytical hue correction
