# KPG-Surv

## 1. Project Overview
**KPG-Surv** is a knowledge-guided multimodal survival prediction framework for **TCGA-KIRC** (kidney renal clear cell carcinoma). It integrates **whole-slide pathology images (WSIs)**, **KEGG pathway–level genomics**, and **clinical information**.

The framework introduces **pathway-gated cross-attention** to inject biological priors into multimodal fusion at the **attention-logit** level. In addition, a **cross-modal semantic alignment loss** is used to encourage semantic consistency across modalities, significantly improving prognostic accuracy and robustness. KPG-Surv consistently outperforms existing methods on **C-index**, **time-dependent AUC**, and **risk stratification**.

---

## 2. Project Structure

### 2.1 Repository Layout
```text
KPG-Surv/
├── train.py                     # Main training script: training + validation + saving the best model
├── vit_model_knowledge_gated.py  # Core model: pathway-gated multimodal Transformer
├── wsi_gene_dataset.py          # Multimodal dataset loader: WSI, genomics, clinical, hypoxia
├── utils_cox.py                 # Survival analysis utilities: Cox loss, C-index, plotting, evaluation
├── bootstrap_utils.py           # Bootstrap confidence interval computation
├── regularization.py            # Regularization module to mitigate overfitting
├── Genomics_Branch_20260113.py  # Genomics branch: KEGG pathway encoding + SNN feature extraction
└── Initial+Embedding.py         # Hypoxia-pathway initial embedding extraction (hypoxia prior features)
--use_gating True: Enable pathway-gated cross-attention (core contribution)
--use_context_tokens True: Enable clinical context tokens
--alpha 0.5: Weight of the semantic alignment loss
--num_classes 3: Output risk scores for 1/3/5-year horizons
```
## 3. Data Preprocessing Pipeline
WSI: Tile WSIs into 512×512 patches → 20× magnification → CTransPath extracts 768-dimensional embeddings
Genomics: Raw expression → log transform → KEGG pathway grouping → SNN encoding (64D per pathway)
Clinical variables (age, sex, stage, grade): standardization + one-hot encoding
Hypoxia: 16 hypoxia-related KEGG pathways → mean expression → standardization.

## 4. Environment
```text
torch==2.1.0
torchvision==0.16.0
torchaudio==2.0.1
torchmetrics>=1.7.0
numpy>=1.26.0
scipy>=1.10.0
pandas>=1.4.0
scikit-learn>=1.0
lifelines>=0.27.7
statsmodels>=0.14.0
opencv-python>=4.9.0
pillow>=11.0
albumentations>=1.4.0
openslide-python>=1.4.3
transformers>=4.47.0
sentence-transformers>=3.3.0
peft>=0.8.0
einops>=0.8.0
tqdm
pyyaml
matplotlib
seaborn
h5py
psutil
python-dotenv

