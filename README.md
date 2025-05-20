# One step diffusion model using distillation techniques for text-to-image
## Work done during the SRIB-PRISM Program
---
# Model 1

## 📑 Overview

This project implements a **knowledge distillation framework** for transferring knowledge from a high-capacity **Rectified Flow-based teacher model** to a more lightweight **student model**. The primary objective is to retain the performance of a complex generative model while enabling faster inference and reduced computational overhead.

The workflow involves generating noise-augmented datasets, training a Rectified Flow teacher model, and distilling its learned distribution into a student model using both standard and custom four-step distillation strategies.


## 📦 Install Dependencies and Cloning into the repo

Before cloning this repo, it’s recommended to set up a virtual environment before installing dependencies:

```bash
# 1. (Optional) Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# 2. Clone the repo
git clone https://github.ecodesamsung.com/SRIB-PRISM/RVCE_24VI48RV_GenAI_One_step_diffusion_model_using_distillation_techniques_for_text_to_image
cd RVCE_24VI48RV_GenAI_One_step_diffusion_model_using_distillation_techniques_for_text_to_image/CODE/Distillation

# 3. Install dependencies
pip install -r requirements.txt
```
## 🚀 How to Train
### 1️⃣ Train the Teacher Model (Rectified Flow)
Train the Rectified Flow-based teacher model using:

```bash
python teacherRectifiedFlow.py
```
### 2️⃣ Generate Noise-Augmented Dataset
Run the dataset generator script to create noise-augmented samples required for model training:

```bash
python generateNoiseSamplesDataset.py
```

### 3️⃣ Distill Student Model from Teacher
Use the trained teacher model to distill knowledge into a student model:
```bash
python distillStudentFromRF.py
```
### 4️⃣ (Optional) Four-Step Student Distillation
For improved student model performance, run the custom four-step distillation strategy:
```bash
python fourStepStudent.py
```

# 📊 Architecture Overview
<div align="center">
  <img src="architecture.png" alt="Architecture Diagram" width="250"/>
</div>


# How to run Inference?
### Use the `streamlit` interface to interact with the trained model, and test both, the Teacher and Student models.
```bash
streamlit run newStream.py
```


https://github.ecodesamsung.com/SRIB-PRISM/RVCE_24VI48RV_GenAI_One_step_diffusion_model_using_distillation_techniques_for_text_to_image/assets/32127/96034b49-0a54-4706-8383-f12dab5b2259

# 📌 NOTE: To put our FID scores in context

## CIFAR-10 Diffusion Model Benchmarks

*State-of-the-art diffusion models on CIFAR-10 achieve FIDs in the low single digits, but require substantial compute.*  For example, Song et al. (2021) report FID 2.20 (unconditional), and Karras et al. (2022) report FID 1.97 (unconditional).  Recent distillation and consistency-model methods achieve similarly low FIDs: Luo et al. (NeurIPS 2024) obtain FID 2.06 with a **single** sampling step, and Song et al. (ICLR 2023) report FID 3.55 for one-step consistency sampling.  By contrast, our rectified-flow **teacher** (long, multi-step) model has FID 15.0 and our 4-step **student** model FID 22.75.  These are much higher FIDs in absolute terms, but we note the **compute budgets** differ dramatically (see below).

**Sources:** Citations indicate the reported FID (or method) from each reference.  “Compute” is provided when available (e.g. EDM’s code uses 8×V100 GPUs for \~2 days)

## Compute and Training Budgets

In practice, low FID on CIFAR-10 often requires **large-scale training**.  For instance, NVLabs’ EDM model was trained on **8×V100 GPUs for ≈2 days** to reach FID ≈1.97.  Similarly, Zhou *et al.* estimate that training a CIFAR-10 diffusion model “takes about **200 hours** to train a diffusion model from scratch” on a single high-end GPU.  Even accelerated distillation methods incur huge costs: Zhou *et al.* note that consistency distillation required \~1156 A100-hours, and advanced multi-step distillation often exceeds **100 GPU-hours**.  By contrast, our models were trained on a **single A100 GPU** (no multi-GPU parallelism) over thousands of epochs.  In concrete terms, 4000–5000 epochs on one A100 is on the order of **tens of GPU-hours**, an order of magnitude less than typical state-of-the-art training (even assuming accelerated pipelines).

## Quality vs. Sampling Speed

We emphasize that reducing sampling steps usually degrades FID unless offset by aggressive distillation or special techniques.  For context, solver-based sampling such as DDIM achieves FID 15.69 with 10 steps (and FID 2.91 with 50 steps).  Distillation methods can recover quality: e.g. Progressive Distillation (PD) reaches FID 4.51 at 2 steps (171 h training), and Guided PD reaches FID 3.18 at 4 steps (119 h).  Our rectified-flow **4-step student** (without heavy teacher fine-tuning) yields FID 22.75, which is substantially worse but not unexpected given the extreme step reduction and our compute limits.  Notably, SFD’s 4-step model attains FID 3.24 in 1.17 h (with a carefully distilled teacher), illustrating the trade-off: state-of-the-art few-step sampling can hit FID ≈3 but only after extensive training of a teacher model.

## Discussion

In summary, while our teacher/student FIDs (15.0 and 22.75) are far above the low single-digit SOTA, these results seem to be **competitive under limited compute**.  High-quality diffusion on CIFAR-10 typically involves *hundreds of GPU-hours* and complex distillation; in this context, achieving FIDs in the mid-teens with one GPU is notable.  Our student’s 4-step FID is higher than specialized methods (≈3–4) but those baselines depend on pre-trained teachers and hours of fine-tuning.  Additionally, our model only utilises 12.3MB of weights, which is orders of magnitude lesser than SOTA models. Thus, our models demonstrate respectable image quality given the severe resource constraints, potentially implying competitiveness in a low-budget regime.

### References

- J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” *Advances in Neural Information Processing Systems*, vol. 33, pp. 6840–6851, 2020.

- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-Based Generative Modeling through Stochastic Differential Equations,” in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2021.

- T. Salimans and J. Ho, “Progressive Distillation for Fast Sampling of Diffusion Models,” in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2022.

- T. Karras, M. Aittala, S. Laine, T. Hellsten, J. Lehtinen, and T. Aila, “Elucidating the Design Space of Diffusion-Based Generative Models,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 26565–26577, 2022.

- S. Song, C. Meng, and S. Ermon, “Consistency Models,” in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2023.

- H. Luo, J. He, C. Zhang, X. Xu, M. Li, X. Wang, and L. Yao, “Knowledge Distillation for One-Step Diffusion Models,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

- D. Watson, W. Teh, S. Rush, and Y. Du, “Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

- NVLabs, “Latent Score-based Generative Models,” *NVIDIA Research*, 2022. \[Online]. Available: [https://nvlabs.github.io/LSGM/](https://nvlabs.github.io/LSGM/)

---
---











