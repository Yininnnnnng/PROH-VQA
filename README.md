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
