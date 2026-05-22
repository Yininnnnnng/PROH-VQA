# PROH-VQA: Pseudo-ROI Guided Head-Selective Attention for Medical Visual Question Answering

This repository accompanies the paper *PROH-VQA: Pseudo-ROI Guided
Head-Selective Attention for Medical Visual Question Answering*.

PROH-VQA is an annotation-free training pipeline for medical visual
question answering (Med-VQA). It (1) generates question-conditioned
pseudo-ROIs by coupling MedSAM with BiomedCLIP, (2) identifies which
cross-attention heads of a trained Med-VQA model are spatially aligned,
and (3) selectively regularizes only those heads.

This repository releases the **pseudo-ROI generation** code (Stage 1 of
the pipeline). The full training code (Stages A/B/C) will be released
upon acceptance.

## Repository contents
PROH-VQA/
├── README.md
├── requirements.txt
└── ROI_GitHub.ipynb        # MedSAM + BiomedCLIP pseudo-ROI pipeline
## Environment

Tested with Python 3.10 on Google Colab (single NVIDIA GPU).

```bash
pip install -r requirements.txt
```

The pseudo-ROI pipeline additionally requires a **MedSAM ViT-B
checkpoint** (`medsam_vit_b.pth`), which is not distributed via pip.
Download it from the official MedSAM repository and set its path in the
notebook (see below).

- MedSAM: https://github.com/bowang-lab/MedSAM
- BiomedCLIP is loaded automatically from the Hugging Face Hub
  (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) via `open_clip`.

## Data

We evaluate on two public Med-VQA benchmarks. No data is redistributed
in this repository.

- **SLAKE** (SLAKE-EN): https://huggingface.co/datasets/BoKelvin/SLAKE
- **VQA-RAD**: https://huggingface.co/datasets/flaviagiammarino/vqa-rad

Download SLAKE and point `SLAKE_PATH` to its root directory, which should
contain `imgs/` and `test.json`.

## Running the pseudo-ROI pipeline

Open `ROI_GitHub.ipynb` and set the two paths at the bottom of the notebook:

```python
SLAKE_PATH        = "/path/to/SLAKE/Slake1.0"
MEDSAM_CHECKPOINT = "/path/to/medsam_vit_b.pth"
```

Running the notebook produces, for each sample, the top-k scored bounding
boxes and a per-sample confidence weight, together with a visualization
of the candidate masks and the selected region.

### Pseudo-ROI configuration (paper Section 2.1)

| Paper symbol | Code variable | Value |
|---|---|---|
| η (negative-evidence strength, Eq. 1) | `neg_weight` | 0.5 |
| View weights w_v (tight/pad/soft, Eq. 1) | `window_weights` (per question type) | e.g. existence = (0.50, 0.30, 0.20) |
| α / β / γ (composite weights, Eq. 2) | `clip_weight` / `sam_weight` / `prior_weight` (+ per-type overrides in `_get_adaptive_weights`) | base 0.55 / 0.25 / 0.15 |
| IoU deduplication threshold | `iou_threshold` | 0.6 |
| MedSAM score threshold | `sam_score_threshold` | 0.20 |
| Minimum area ratio | `min_area_ratio` | 0.001 |
| AMG predicted-IoU threshold | `amg_pred_iou_thresh` | 0.65 |
| Soft-mask attenuation | `soft_mask_alpha` | 0.25 |
| Padding ratio (padded view) | `padding_ratio` | 0.25 |

All weights and thresholds are deterministic; no per-sample tuning is
performed. The pipeline runs once, offline, before training.

## VQA model and training configuration (paper Section 3.1)

The training code is not yet released, but the full configuration needed
to reproduce the model is given below.

**Architecture.**
- Vision encoder: BiomedCLIP ViT-B/16, producing N = 196 patch tokens at
  D = 768 after dropping the [CLS] token; only the last 20 parameter
  tensors are unfrozen.
- Text encoder: PubMedBERT, fully fine-tuned.
- Fusion: a single cross-attention layer with H = 8 heads and hidden
  dimension D = 768 (text as queries, visual tokens as keys/values),
  followed by a feed-forward network.
- Classifier: two-layer MLP, hidden dimension 768, 220 output classes
  for SLAKE-EN.

**Stage A (baseline training).**
30 epochs; AdamW; base learning rate 3e-5 for the new projection,
cross-attention, FFN, and classifier; pretrained-encoder parameters at
3e-6; cosine schedule with 10% warmup; weight decay 1e-4; label
smoothing 0.05; gradient clipping 1.0; batch size 32.

**Stage B (head functional analysis).**
A single forward pass over the SLAKE-EN validation set. The top-k
pseudo-ROI boxes (k = 3) are projected onto a 14×14 grid and converted
to a Gaussian soft mask; per-head [CLS]-row attention is correlated
(Pearson) with this mask, averaged over the validation set.
Parameter-free.

**Stage C (selective attention regularization).**
Resumes from the Stage A checkpoint; reduced learning rate 1e-5;
regularization weight λ = 0.15; binary ROI mask on the 14×14 grid;
confidence-weighted MSE attention loss applied only to the top-K = 3
spatially aligned heads. 25 epochs; other optimizer settings as in Stage A.

**Hardware.** All experiments use a single NVIDIA A100-SXM4-40GB GPU.

## Citation

(BibTeX will be added upon publication.)
