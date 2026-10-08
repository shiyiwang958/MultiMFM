"""python compare.py orig.pt port.pt -> per-tensor max |orig - port| (CPU fp32 parity; see docs/provenance/dna/verification.md)."""
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
assert set(a) == set(b), set(a) ^ set(b)
for k in sorted(a):
    d = float((a[k].double() - b[k].double()).abs().max())
    print(f"{k:28s} max|orig-port|={d:.3e} rel={d / max(float(a[k].double().abs().max()), 1e-30):.2e}")
