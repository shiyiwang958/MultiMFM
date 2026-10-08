#!/bin/bash
# Assemble the ORIGINAL DNA code (DNA-MFM@187fe7b library from the 2026-09-25 reconstruction +
# the surviving July files/scripts) into $1, from the read-only holylabs backup.
#   bash tests/dna/parity/build_orig_tree.sh /tmp/orig
#   ORIG=/tmp/orig PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/orig:$BK/scratch_dfm_recon_20260925/pydeps \
#     ~/micromamba/envs/seq_h100/bin/python tests/dna/parity/parity_orig.py orig.pt
#   python tests/dna/parity/parity_port.py port.pt && python tests/dna/parity/compare.py orig.pt port.pt
set -euo pipefail
O="$1"; BK="${BK:-/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929}"
W="${BK}/scratch_dfm_recon_20260925"; R="${BK}/dirichlet-flow-matching"
mkdir -p "${O}/scripts"
cp -r "${W}/model" "${W}/utils" "${W}/mfm" "${W}/lightning_modules" "${W}/dinko" "${O}/"
cp "${R}/dinko/c0_regressors.py" "${O}/dinko/"
cp "${R}/utils/mfm_diag_distill.py" "${R}/utils/parsing.py" "${O}/utils/"
for s in ablate_glass_gradient_mc ablate_dmfm_one_step_gradient_mc probe_dna_mfm_glass; do cp "${R}/scripts/${s}.py" "${O}/scripts/"; done
cp "${W}/scripts/sample_yeast_c0_guidance_hist.py" "${O}/scripts/"
find "${O}" -name __pycache__ -prune -exec rm -rf {} +
echo "original tree in ${O}"
