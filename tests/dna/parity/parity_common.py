"""Shared forward-pass parity computation; `G`/`D`/`H`/`C0` are the original or the ported modules."""
import torch

import os
REPO = os.environ.get("MULTIMFM_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[3]))
DMFM = {50: "epoch=95-step=49000.ckpt", 100: "epoch=80-step=39000.ckpt", 200: "epoch=51-step=45000.ckpt", 400: "epoch=33-step=45750-v1.ckpt"}
DMFM4 = {50: "epoch=29-step=15000.ckpt", 100: "epoch=306-step=149000.ckpt", 200: "epoch=39-step=34000.ckpt", 400: "epoch=96-step=129000.ckpt"}


def rnd(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


def run(G, D, H, build_c0, flow_step):
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    out = {}
    motif = G.motif_tensor("TTTTTC", dev)
    motif2 = G.motif_tensor("AAAATT", dev)
    kw = dict(score_center=0.5, score_std=0.1, z_target=1.0, reward_beta=1.0, reward_scale=1.0,
              motif=motif, motif2=motif2, motif_tau=0.1, conjunction_tau=0.1)
    for L in (50, 100, 200, 400):
        model, cfg = G.load_model(f"{REPO}/checkpoints/dna/base/L{L}/best.pt", dev)
        x = rnd((2, L, 4), L)
        t = torch.full((2,), 0.4)
        with torch.no_grad():
            out[f"base{L}_logits"] = model.psi_st(x, t, t, return_logits=True)
            out[f"base{L}_v"] = model.v(t, t, x)
            out[f"base{L}_step"] = flow_step(cfg, model, x, t, torch.full((2,), 0.45))[0]
            eps = rnd((6, L, 4), L + 1)
            xr, tr = x.repeat_interleave(3, 0), torch.full((6,), 0.5)
            out[f"glass{L}_euler"] = G.glass_integrate_diff(model, eps, xr, tr, 4, end_time=0.999, solver="euler")
            out[f"glass{L}_rk4"] = G.glass_integrate_diff(model, eps, xr, tr, 4, end_time=0.999, solver="rk4")
        pool = rnd((2, 3, L, 4), L + 2)
        for reward in ("gc", "motif", "conjunction"):
            v, g = G.estimate_value_gradient(model, x, t_eval=0.5, eps_pool=pool, reward=reward, nfe_value=4, mc_chunk=2, **kw)
            out[f"glass{L}_{reward}_V"], out[f"glass{L}_{reward}_grad"] = v, g
        for kind, files in (("dmfm", DMFM), ("dmfm_4step", DMFM4)):
            student = D.load_dmfm(f"{REPO}/checkpoints/dna/{kind}/L{L}/{files[L]}", dev)
            with torch.no_grad():
                s0, s1 = torch.full((6,), 0.1), torch.full((6,), 0.6)
                out[f"{kind}{L}_v"] = student.v(s0, s1, eps, tr, xr)
                out[f"{kind}{L}_map"] = student(s0, s1, eps, tr, xr)
                out[f"{kind}{L}_onestep"] = D.one_step_dmfm(student, eps, xr, tr)
                out[f"{kind}{L}_flow4"] = D.flow_map_dmfm(student, eps, xr, tr, n_steps=4, end_time=0.999)
            v, g = D.estimate_value_gradient(student, x, t_eval=0.5, eps_pool=pool, reward="motif", mc_chunk=2,
                                             dmfm_sampler="flow_map", dmfm_steps=4, dmfm_end_time=0.999, **kw)
            out[f"{kind}{L}_motif_V"], out[f"{kind}{L}_motif_grad"] = v, g
    guide = build_c0("park_cnn")
    guide.load_state_dict(torch.load(f"{REPO}/checkpoints/dna/c0/guide/best_state.pt", map_location="cpu"))
    oracle = build_c0("park_cnn")
    oracle.load_state_dict(torch.load(f"{REPO}/checkpoints/dna/c0/oracle/best_state.pt", map_location="cpu"))
    probs = torch.softmax(rnd((5, 50, 4), 7), -1)
    with torch.no_grad():
        out["c0_guide"], out["c0_oracle"] = guide(probs), oracle(probs)
    model, cfg = G.load_model(f"{REPO}/checkpoints/dna/base/L50/best.pt", dev)
    x = rnd((2, 50, 4), 11)
    pool = rnd((2, 8, 50, 4), 12)
    v, g = H.estimate_v_and_grad(model, guide, x, 0.6, pool, 1.0, None, 0.15, 0.5, 4, 8)
    out["table1_V"], out["table1_grad"] = v, g
    return {k: v.detach().clone() for k, v in out.items()}
