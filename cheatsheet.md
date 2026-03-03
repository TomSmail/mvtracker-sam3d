# MVTracker + SAM3D Data Pipeline Cheatsheet

Quick command reference for this project (Panoptic + human-track generation).

---

## 0) One-time setup

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

If running on Euler nodes, load modules first:

```bash
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source /cluster/home/tsmail/mvtracker-sam3d/venv/bin/activate
```

---

## 1) Data + checkpoint download

Purpose: download Panoptic data to scratch and SAM3D checkpoint to home.

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
bash scripts/data_engine/download_panoptic.sh
```

Use a custom full-dataset archive URL (if your default URL is only a subset):

```bash
PANOPTIC_TARBALL_URL="<full_panoptic_archive_url>" \
PANOPTIC_EXPECTED_MIN_SEQS=65 \
bash scripts/data_engine/download_panoptic.sh
```

Scratch location used by pipeline:

```bash
/cluster/scratch/tsmail/datasets/panoptic-multiview
```

---

## 2) Hugging Face auth (for gated SAM3D assets)

Login once per environment:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

Check auth:

```bash
huggingface-cli whoami
```

---

## 3) SAM3D inference

### SLURM array (recommended)

Runs all sequences, all views, all frames:

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
bash scripts/data_engine/submit_slurm_inference_all.sh
```

Manual submission with custom array range:

```bash
sbatch --array=0-64 scripts/data_engine/slurm_inference.sh
```

### Single-sequence debug run

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
source venv/bin/activate
PYTHONPATH=$PWD python scripts/data_engine/run_sam3d_inference.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq_index 0 \
  --detector_path "$HOME/checkpoints/vitdet"
```

---

## 4) Generate fused human tracks

### SLURM array (recommended)

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
bash scripts/data_engine/submit_slurm_generate_all.sh
```

### Force full regeneration (don’t skip existing `.npz`)

```bash
rm /cluster/scratch/tsmail/datasets/panoptic-multiview/*/human_tracks.npz
bash scripts/data_engine/submit_slurm_generate_all.sh
```

### Single-sequence debug run

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
source venv/bin/activate
PYTHONPATH=$PWD python scripts/data_engine/generate_human_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq_index 0 \
  --n_vertex_samples 500 \
  --smooth_kernel 3
```

Optional person limit for known single-person sequences:

```bash
... --max_persons 1
```

---

## 5) Visualize outputs

### Raw SAM3D predictions

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
source venv/bin/activate
python scripts/data_engine/visualize_sam3d.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq basketball --view 1 --output viz_basketball_v1.mp4
```

### Fused tracks (from `human_tracks.npz`)

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
source venv/bin/activate
python scripts/data_engine/visualize_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq basketball --view 1 --trails --output tracks_basketball_v1.mp4
```

Single frame:

```bash
python scripts/data_engine/visualize_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq basketball --view 1 --frame 20 --output tracks_basketball_v1_f20.jpg
```

Side-by-side (raw vs fused):

```bash
python scripts/data_engine/visualize_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq basketball --view 1 --output compare.mp4 --side_by_side
```

---

## 6) Data quality + volume report

Console report:

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
source venv/bin/activate
python scripts/data_engine/report_human_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview
```

Save JSON + Markdown:

```bash
python scripts/data_engine/report_human_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --save_json reports/human_tracks_report.json \
  --save_markdown reports/human_tracks_report.md
```

Single sequence:

```bash
python scripts/data_engine/report_human_tracks.py \
  --data_root /cluster/scratch/tsmail/datasets/panoptic-multiview \
  --seq basketball
```

---

## 7) Train/eval with human-track data

Example eval run:

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
source venv/bin/activate
PYTHONPATH=$PWD python mvtracker/cli/eval.py experiment=mvtracker_human
```

---

## 8) SLURM monitoring + logs

Check queue/jobs:

```bash
squeue -u tsmail
```

Job details:

```bash
scontrol show job <job_id>
```

Watch a specific log:

```bash
tail -f logs/slurm_logs/gen-human-tracks-<jobid>_<task>.out
```

List recent logs:

```bash
ls -lt logs/slurm_logs | head
```

---

## 9) Useful troubleshooting commands

Count predicted frames per view:

```bash
ls /cluster/scratch/tsmail/datasets/panoptic-multiview/basketball/sam3d_predictions/view_01 | wc -l
```

Check whether tracks exist for all sequences:

```bash
ls /cluster/scratch/tsmail/datasets/panoptic-multiview/*/human_tracks.npz
```

Inspect keys in a tracks file:

```bash
python - <<'PY'
import numpy as np
d = np.load('/cluster/scratch/tsmail/datasets/panoptic-multiview/basketball/human_tracks.npz')
print(d.files)
for k in d.files:
    v = d[k]
    print(k, getattr(v, 'shape', None), getattr(v, 'dtype', None))
PY
```

---

## 10) Notes

- Scratch data (`/cluster/scratch/tsmail`) can be wiped periodically. Re-run download + generation scripts when needed.
- Typical pipeline order:
  1. `download_panoptic.sh`
  2. `slurm_inference.sh`
  3. `slurm_generate.sh`
  4. `visualize_tracks.py` + `report_human_tracks.py`
- Keep this file updated whenever new stable commands are added.

---

## 11) Full Panoptic Studio source (official, not subset)

Official dataset portal:

```text
http://domedb.perception.cs.cmu.edu/
```

Official toolbox + downloader (`getData.sh`):

```text
https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
```

Quick start for downloading a full sequence from CMU Panoptic:

```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git
cd panoptic-toolbox/scripts

# Usage:
# ./getData.sh <sequence_name> <num_vga_views> <num_hd_views>

# Example: download one sequence with all 31 HD cams:
./getData.sh 160422_ultimatum1 0 31
```

No-clone version (download only `getData.sh` and run it):

```bash
mkdir -p /cluster/scratch/tsmail/datasets/panoptic-raw
cd /cluster/scratch/tsmail/datasets/panoptic-raw
wget -O getData.sh https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/master/scripts/getData.sh
chmod +x getData.sh

# Download one sequence (all 31 HD cams, no VGA):
./getData.sh 160422_ultimatum1 0 31
```

Using your local copied scripts (`getPanopticDataLatest.sh` + `getData.sh`) with scratch output:

```bash
cd /cluster/home/tsmail/mvtracker-sam3d
PANOPTIC_RAW_ROOT=/cluster/scratch/tsmail/datasets/panoptic-raw \
PANOPTIC_VGA_VIEWS=0 \
PANOPTIC_HD_VIEWS=31 \
bash scripts/data_engine/getPanopticDataLatest.sh
```

If you hit `Permission denied` on copied helper scripts:

```bash
chmod +x scripts/data_engine/getData.sh scripts/data_engine/getPanopticDataLatest.sh
```

Panoptic tools page (download usage details):

```text
http://domedb.perception.cs.cmu.edu/develop/tools.html
```

