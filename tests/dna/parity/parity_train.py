"""dMFM training-objective parity: one GaussianMfmDiagDistiller.step_both on a fixed batch, original vs port.
    python parity_train.py {orig,port} out.pt      (orig: seq_h100 env with ORIG=<tree> on PYTHONPATH)"""
import os, sys, json
from pathlib import Path
import torch
impl, out = sys.argv[1], sys.argv[2]
REPO = os.environ.get("MULTIMFM_ROOT", str(Path(__file__).resolve().parents[3]))
if impl == "orig":
    O = os.environ["ORIG"]; sys.path[:0] = [O, O + "/scripts"]
    from probe_dna_mfm_glass import load_args_json
    from model.dna_models import DNAMFMStudent
    from utils.mfm_diag_distill import GaussianMfmDiagDistiller
else:
    from dmfm.utils.model_loading import load_args_json
    from dmfm.models.dna_models import DNAMFMStudent
    from dmfm.utils.mfm_diag_distill import GaussianMfmDiagDistiller
torch.set_num_threads(8)
res = {}
for kind, L in (("dmfm", 50), ("dmfm_4step", 50), ("dmfm", 200)):
    args = load_args_json(f"{REPO}/checkpoints/dna/{kind}/L{L}/args.json")
    args.mfm_teacher_ckpt = f"{REPO}/checkpoints/dna/base/L{L}/best.pt"
    args.mfm_teacher_ckpt_hparams = f"{REPO}/checkpoints/dna/base/L{L}/args.json"
    torch.manual_seed(0)
    student = DNAMFMStudent(args, alphabet_size=4)
    distiller = GaussianMfmDiagDistiller(args, alphabet_size=4, device=torch.device("cpu"))
    torch.manual_seed(1)
    seq = torch.randint(4, (4, L))
    torch.manual_seed(2)
    loss, logs = distiller.step_both(seq=seq, student_model=student, global_step=10_000)
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in student.parameters() if p.grad is not None])
    res[f"{kind}{L}_loss"] = loss.detach().reshape(1)
    res[f"{kind}{L}_grad"] = g.detach()
    for k in ("mfm_diag_loss_mean", "mfm_esd_loss_mean", "mfm_s", "mfm_u", "mfm_tcond"):
        if isinstance(logs.get(k), torch.Tensor):
            res[f"{kind}{L}_{k}"] = logs[k].detach().reshape(-1).float()
torch.save(res, out); print(impl, {k: float(v.flatten()[0]) for k, v in res.items() if k.endswith("_loss")})
