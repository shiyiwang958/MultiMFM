"""Unconditional yeast DNA window dataset.

Ported from DNA-MFM@187fe7b ``utils/dataset.py``; only the default path changed
(the old ``data/yeast_mid50.pt`` default referred to the purged pre-split data).
"""
import torch

from dmfm.utils.torch_io import torch_load
class YeastMiddleDataset(torch.utils.data.Dataset):
    """
    Unconditional DNA dataset backed by a torch file produced by `dmfm.experiments.prepare_yeast_parent_splits`.

    Returns (seq, cls) to match the DNAModule interface, where cls is a dummy zero label.
    """

    def __init__(self, args, path: str):
        payload = torch_load(path, map_location="cpu")
        if isinstance(payload, dict) and "seqs" in payload:
            seqs = payload["seqs"]
        else:
            # allow passing the tensor directly
            seqs = payload

        if not isinstance(seqs, torch.Tensor):
            raise TypeError(f"Expected 'seqs' to be a torch.Tensor, got {type(seqs)}")
        if seqs.ndim != 2:
            raise ValueError(f"Expected seqs to have shape [N, L], got {tuple(seqs.shape)}")
        if seqs.dtype != torch.long:
            seqs = seqs.long()
        if seqs.min().item() < 0 or seqs.max().item() > 3:
            raise ValueError("Expected integer-encoded DNA with values in {0,1,2,3}.")

        self.seqs = seqs
        self.clss = torch.zeros((len(self.seqs),), dtype=torch.long)
        self.num_cls = 1
        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.clss[idx]
