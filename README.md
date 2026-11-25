# Diagram Understanding with Vision-Language Models


---

## Objective

Investigate whether **context** and **annotation injection** improve the diagrammatic and structural reasoning capabilities of parameter-efficient Vision-Language Models (VLMs) on spatial placeholder queries using the AI2D dataset.

---

## Problem Statement

Multimodal Vision-Language Models (VLMs) that perform exceptionally well on natural image benchmarks often struggle with **diagram understanding**, particularly in dense scientific illustrations. This limitation is even more pronounced in compact VLMs intended for edge-device deployment.

Scientific diagrams contain fine-grained visual elements such as arrows, connector lines, labels, and placeholders (e.g., **A**, **B**, **C**) that are often lost during image patchification and downsampling. While conventional fine-tuning improves semantic understanding, it does not sufficiently enhance spatial reasoning, causing models to perform close to random guessing (~25%) on many diagram-based multiple-choice questions.

This project investigates whether providing structured annotation context during fine-tuning can improve reasoning over scientific diagrams.

---

# VLM Fine-Tuning on AI2D

## Hardware

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)

## Dataset

- **Dataset:** AI2D
- **Images:** 4,903
- **Questions:** 15,501
- **Categories:** 17

## Frameworks

- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- TRL
- Unsloth

---

# Project Overview

The project consists of **two phases**.

## Phase 1 — Zero-Shot Evaluation

Two Vision-Language Models are evaluated directly on AI2D without any training.

The better-performing model is selected for fine-tuning.

| Model | Zero-Shot Evaluation | Fine-Tuned? |
|--------|----------------------|-------------|
| Qwen2-VL-2B | ✅ | Only if selected |
| LLaVA-1.5-7B | ✅ | Only if selected |

**Winner Selection Criterion**

- Highest test-set accuracy

---

## Phase 2 — Fine-Tuning

Only the winning model is fine-tuned under two different conditions.

| Condition | Model Input | Purpose |
|------------|------------|---------|
| Base | Image | Baseline from Phase 1 |
| Tuning A | Image + Question | Evaluate the effect of AI2D fine-tuning |
| Tuning B | Image + Question + Annotation | Evaluate whether annotation context improves performance |

The final comparison is performed between:

- Base Model
- Tuning A
- Tuning B

---

# Repository Structure

```text
VLM_Final_Project/
├── ai2d/                          # Original dataset (read-only)
│   ├── images/
│   ├── annotations/
│   ├── questions/
│   └── categories.json
│
├── metadata_output/               # Metadata generated from preprocessing
│   ├── metadata_images.csv
│   ├── metadata_questions.csv
│   ├── metadata_integrity.csv
│   └── metadata_full.json
│
├── processed_data/
│   ├── splits/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── annotation_summaries/
│
├── checkpoints/
│   ├── <winner>_tuningA/
│   └── <winner>_tuningB/
│
├── logs/
│   ├── qwen2vl_2b_tuningA/
│   ├── ...
│   └── training_summary.csv
│
├── results/
│   ├── predictions/
│   └── comparison_report.csv
│
└── notebooks/
    ├── 01_eda_and_data_preprocessing.ipynb
    ├── 02_phase1_zeroshot.ipynb
    ├── 03_tuningA_n_B_qwen2vl_unsloth_kaggle.ipynb
    ├── 04_comparison_and_eval_metrics.ipynb
    └── 05_inference.ipynb
```

---

# Notebook Pipeline

## 01 — Exploratory Data Analysis & Data Preprocessing

- Explore the AI2D dataset
- Verify dataset integrity
- Generate metadata
- Create train/validation/test splits
- Generate annotation summaries

---

## 02 — Phase 1: Zero-Shot Evaluation

- Evaluate Qwen2-VL-2B
- Evaluate LLaVA-1.5-7B
- Compare performance
- Select the best-performing model

---

## 03 — Fine-Tuning

Fine-tune the selected model using:

- **Tuning A**
  - Image + Question

- **Tuning B**
  - Image + Question + Annotation Summary

Training uses:

- LoRA
- QLoRA (4-bit quantization)
- Unsloth
- Hugging Face TRL

---

## 04 — Evaluation & Comparison

Evaluate all three model variants using:

- Accuracy
- Macro F1 Score
- Weighted F1 Score
- Cohen's Kappa
- Classification Report
- Confusion Matrix
- McNemar's Statistical Significance Test

Generate the final comparison report between:

- Base
- Tuning A
- Tuning B

---

## 05 — Inference

Load the fine-tuned checkpoint and launch an interactive **Gradio** interface for inference.

The application allows users to:

- Upload a scientific diagram
- Enter a multiple-choice question
- (Optional) Provide annotation/context text
- Run inference using the fine-tuned model
- View the predicted answer in real time

---

# Usage

## 1. Clone the repository

```bash
git clone <repository-url>
cd AI2D-VQA-MLProject
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the notebooks

Execute the notebooks in the following order:

1. `01_eda_and_data_preprocessing.ipynb`
2. `02_phase1_zeroshot.ipynb`
3. `03_tuningA_n_B_qwen2vl_unsloth_kaggle.ipynb`
4. `04_comparison_and_eval_metrics.ipynb`
5. `05_inference.ipynb`

---

## Launch the Inference App

Open and run:

```text
notebooks/05_inference.ipynb
```

Execute all cells to launch the **Gradio** web interface.

After the interface starts:

1. Upload a diagram image.
2. Enter the multiple-choice question.
3. (Optional) Provide annotation/context text.
4. Click **Predict**.
5. View the model's predicted answer.

---

# Results

The project compares three model configurations:

- Base Model
- Fine-Tuning A
- Fine-Tuning B

Performance is evaluated using multiple classification metrics to determine whether annotation-based context injection improves diagram understanding.

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- QLoRA
- TRL
- Unsloth
- Gradio
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# License

This project was developed as part of the **Machine Learning** course at the **University of Europe for Applied Sciences (UE)**.
