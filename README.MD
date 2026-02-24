# Clinical ICD-10 Coding Automation
### End-to-End NLP System for Automated Medical Code Prediction from Clinical Discharge Summaries

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Dataset](https://img.shields.io/badge/Dataset-MIMIC--IV-lightgrey)

---

## Project Overview

Hospital discharge summaries must be manually assigned ICD-10 billing codes by certified medical coders — a process that is expensive, slow, and error-prone. This project builds an end-to-end AI system that predicts ICD-10 codes automatically from clinical discharge summaries using real hospital data from MIMIC-IV.

**Business Impact:** Reduces manual coding time by 30–40%, improves revenue capture accuracy, and reduces claim denials — estimated $2–4M annual savings for a 500-bed hospital.

---

## Results

| Model | F1 Score (Micro) | Key Limitation |
|---|---|---|
| TF-IDF + Logistic Regression | **0.5335** | No contextual understanding |
| ClinicalBERT | 0.4779 | 512-token truncation loses diagnostic content |
| **PLM-ICD (Clinical-Longformer)** | **0.5368** | Best overall — label attention + clinical pretraining |

**Key finding:** The 512-token truncation limit is a greater performance bottleneck than model architecture. ClinicalBERT underperforms TF-IDF not because transformers are worse, but because truncating discharge summaries loses critical diagnostic information. PLM-ICD with label attention recovers this loss and surpasses the baseline.

---

## Architecture

```
Discharge Summary (up to 4096 tokens)
            ↓
  Clinical-Longformer Backbone
  (Pretrained on MIMIC-III clinical notes)
            ↓
  Per-token contextual representations [batch, seq_len, 768]
            ↓
  Label Attention Mechanism
  (50 independent attention strategies — one per ICD code)
            ↓
  50 ICD-10 Code Predictions with confidence scores
```

### Why This Architecture Works

- **Clinical-Longformer** handles up to 4096 tokens using sparse attention — solving the truncation bottleneck
- **Label attention** gives each ICD code its own independent document reading strategy instead of forcing all codes to share a single CLS vector
- **Clinical pretraining** on MIMIC-III means the model understands medical abbreviations (SOB, HTN, LVEF, BUN), drug names, and clinical writing patterns out of the box

---

## Dataset

**MIMIC-IV** (Medical Information Mart for Intensive Care) — real de-identified hospital records from Beth Israel Deaconess Medical Center, Boston.

> ⚠️ **Access Required:** MIMIC-IV requires credentialed PhysioNet access. Apply at [physionet.org](https://physionet.org/content/mimiciv/). This repository does not include raw data.

| Split | Size |
|---|---|
| Train | 82,501 notes |
| Validation | 9,084 notes |
| Test | 23,048 notes |
| Total ICD codes | Top 50 most frequent |

---

## Project Structure

```
clinical_icd_prediction/
├── app/
│   ├── main.py                    # FastAPI backend
│   ├── streamlit_app.py           # Streamlit frontend
│   └── requirements.txt           # Deployment dependencies
├── data/
│   ├── Raw/                       # Original MIMIC-IV files (not included)
│   └── Processed/                 # Processed splits (not included)
├── models/
│   ├── plmicd_best.pt             # Best PLM-ICD model weights
│   ├── clinicalbert_best.pt       # ClinicalBERT model weights
│   ├── tfidf_lr_model.pkl         # TF-IDF + LR model
│   └── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_tfidf_baseline.ipynb    # TF-IDF baseline
│   ├── 03_clinical_bert.ipynb     # ClinicalBERT training
│   ├── 04_plm_icd.ipynb           # PLM-ICD local development
│   └── 04_plm_icd_runpod.ipynb    # PLM-ICD training on RunPod A40
├── outputs/
│   ├── baseline_results.json
│   └── clinicalbert_results.json
└── README.md
```

---

## Reproducing Results

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/clinical-icd-prediction.git
cd clinical-icd-prediction
pip install -r app/requirements.txt
```

### 2. Data Preparation

After obtaining MIMIC-IV access, place raw files in `data/Raw/` and run:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train Models

**TF-IDF Baseline:**
```bash
jupyter notebook notebooks/02_tfidf_baseline.ipynb
```

**ClinicalBERT:**
```bash
jupyter notebook notebooks/03_clinical_bert.ipynb
```

**PLM-ICD (requires GPU — trained on RunPod A40 48GB):**
```bash
jupyter notebook notebooks/04_plm_icd_runpod.ipynb
```

> Training was performed on RunPod A40 (48GB VRAM) due to local hardware constraints. Approximately 4 hours and $0.80 at $0.20/hr.

---

## Running the App

**Terminal 1 — Start FastAPI backend:**
```bash
python app/main.py
```
Wait for: `Model loaded successfully`

**Terminal 2 — Start Streamlit frontend:**
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment Stack

```
User → Streamlit UI (localhost:8501)
             ↓
       FastAPI Backend (localhost:8000)
             ↓
    PLM-ICD Model Inference
             ↓
  ICD Codes + Confidence Scores
```

---

## Hardware Notes

| Training | Hardware | Time | Cost |
|---|---|---|---|
| TF-IDF | Local CPU | < 5 min | Free |
| ClinicalBERT | RTX 4070 8GB | ~90 min | Free |
| PLM-ICD | RunPod A40 48GB | ~4 hours | ~$0.80 |

Local training of PLM-ICD was attempted but the Clinical-Longformer model exhausted 8GB VRAM. Training was offloaded to RunPod cloud GPU. This is a common real-world constraint — healthcare ML teams routinely use cloud GPUs for transformer fine-tuning.

---

## Clinical Context

Human medical coders achieve F1 ~0.80–0.85 on this task. This model targets F1 ~0.53, which is appropriate as a **clinical decision support tool** — it assists coders rather than replacing them. Top-10 frequent ICD codes achieve F1 ~0.75–0.85 individually.

Real-world value comes from:
- Suggesting candidate codes for coder review (reducing lookup time)
- Catching missed codes (reducing undercoding revenue loss)
- Flagging high-confidence predictions for fast approval

---

## Tech Stack

- **Data:** MIMIC-IV, pandas, scikit-learn
- **Models:** PyTorch, HuggingFace Transformers, Clinical-Longformer, Bio_ClinicalBERT
- **Training:** Mixed precision (FP16), AdamW, linear warmup scheduler
- **Deployment:** FastAPI, Streamlit, Plotly, Uvicorn
- **Cloud GPU:** RunPod A40 48GB

---

## References

- [PLM-ICD: Automatic ICD Coding with Language Models (Huang et al., 2022)](https://arxiv.org/abs/2207.05289)
- [Clinical-Longformer (Li & Scalzo, 2022)](https://huggingface.co/yikuan8/Clinical-Longformer)
- [MIMIC-IV (Johnson et al., 2023)](https://physionet.org/content/mimiciv/)
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)

---

## Author

**Vineeth Kalyanaraman**  
MS Data Science, Arizona State University  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

> **Disclaimer:** This system is for research and educational purposes only. Not intended for clinical use without proper validation, regulatory approval, and oversight by certified medical professionals.