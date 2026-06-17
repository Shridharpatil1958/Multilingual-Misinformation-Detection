# 🧠 Multilingual Misinformation Detection

### Cross-Lingual Fake News Detection using NLP, Evidence Retrieval & Natural Language Inference

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge\&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge\&logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green?style=for-the-badge\&logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

<h3>🌍 Detect misinformation across Hindi, Tamil, and English social media posts using state-of-the-art multilingual NLP models.</h3>

⭐ **Star this repository if you find it useful!**

</div>

---

## 📌 Overview

Misinformation spreads rapidly across social media platforms, especially in multilingual countries like India. This project introduces an **end-to-end multilingual misinformation detection pipeline** that combines:

* 🤖 Transformer-based classification
* 🔎 Evidence retrieval
* 🧠 Natural Language Inference (NLI)
* 🌍 Cross-lingual understanding
* 🖼️ OCR for WhatsApp images

The system supports **Hindi**, **Tamil**, **English**, and **code-mixed text (Hinglish)**.

---

## 🚀 Key Features

### 🌍 Multilingual Support

* English
* Hindi
* Tamil
* Hinglish / Code-Mixed Text

### 🤖 Claim Classification

* XLM-RoBERTa
* IndicBERT
* mBERT

### 🔎 Hybrid Retrieval

* BM25 Sparse Retrieval
* FAISS Dense Retrieval
* Hybrid Ranking

### 🧠 Fact Verification

* mDeBERTa NLI
* Cross-lingual entailment detection

### 🖼️ OCR Support

* WhatsApp screenshot analysis
* Image text extraction

### ⚡ Production Ready

* FastAPI backend
* Docker deployment
* REST APIs
* Gradio interface

---

# 🏗️ System Architecture

```text
Social Media Post
        │
        ▼
Language Detection
        │
        ▼
Text Preprocessing
        │
        ▼
Claim Classification (XLM-R)
        │
        ▼
Evidence Retrieval
  ┌──────────────┐
  │ BM25 Search  │
  └──────────────┘
        +
  ┌──────────────┐
  │ FAISS Search │
  └──────────────┘
        │
        ▼
Natural Language Inference
      (mDeBERTa)
        │
        ▼
Final Verdict
(True / False / Misleading)
```

---

# 📂 Project Structure

```bash
multilingual-misinfo/
├── configs/
│   ├── model_config.yaml
│   └── retrieval_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── preprocessor.py
│   │   └── hinglish_handler.py
│   │
│   ├── models/
│   │   ├── classifier.py
│   │   ├── nli_model.py
│   │   └── trainer.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── router.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── logger.py
│       └── ocr.py
│
├── scripts/
├── notebooks/
├── demo/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

| Category      | Technologies              |
| ------------- | ------------------------- |
| Programming   | Python 3.10+              |
| Deep Learning | PyTorch                   |
| NLP           | Hugging Face Transformers |
| Retrieval     | BM25, FAISS               |
| API           | FastAPI                   |
| Deployment    | Docker                    |
| UI            | Gradio                    |
| OCR           | EasyOCR / Tesseract       |

---

# 📊 Dataset Sources

| Dataset         | Purpose             |
| --------------- | ------------------- |
| LIAR Dataset    | Fake news detection |
| Factify         | Fact-checking       |
| IndicCorp       | Multilingual corpus |
| Twitter Dataset | Social media claims |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/multilingual-misinfo.git
cd multilingual-misinfo
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Download Dataset

```bash
python scripts/download_data.py
```

This downloads:

* LIAR Dataset
* Factify Dataset
* External Evidence Corpus

---

# 🔨 Build FAISS Index

```bash
python scripts/build_index.py
```

Generated files:

```bash
data/index/
├── faiss.index
└── metadata.pkl
```

---

# 🏋️ Model Training

## Phase 1: Claim Classification

```bash
python -m src.models.trainer \
    --config configs/model_config.yaml \
    --task classify
```

### Output

* Trained XLM-R model
* Checkpoints
* Metrics

---

## Phase 2: NLI Training

```bash
python -m src.models.trainer \
    --config configs/model_config.yaml \
    --task nli
```

### Output

* Fine-tuned mDeBERTa
* Entailment model

---

# 🔎 Evidence Retrieval

## Sparse Retrieval

```python
from src.retrieval.bm25_retriever import BM25Retriever

retriever = BM25Retriever()
```

## Dense Retrieval

```python
from src.retrieval.dense_retriever import DenseRetriever

retriever = DenseRetriever()
```

## Hybrid Retrieval

```python
from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()
```

---

# 🌐 Run API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Interactive API documentation is automatically generated.

---

# 🧪 API Example

## Request

```json
POST /predict

{
    "text": "COVID vaccines cause infertility."
}
```

## Response

```json
{
    "language": "English",
    "claim_label": "Potential Misinformation",
    "evidence": [
        "WHO states no evidence supports this claim."
    ],
    "verdict": "False"
}
```

---

# 🎨 Run Gradio Demo

```bash
python demo/app.py
```

Features:

* Real-time predictions
* Evidence display
* OCR image upload
* Multilingual support

---

# 📈 Evaluation

Run complete pipeline evaluation:

```bash
python scripts/evaluate.py \
    --config configs/model_config.yaml
```

### Metrics

| Metric    | Description                 |
| --------- | --------------------------- |
| Accuracy  | Classification accuracy     |
| Precision | Positive prediction quality |
| Recall    | Detection rate              |
| F1 Score  | Balanced score              |
| NDCG      | Retrieval quality           |
| MRR       | Ranking quality             |

---

# 🧪 Unit Tests

Run tests:

```bash
pytest tests/
```

Coverage includes:

* ✅ Classifier Tests
* ✅ Retrieval Tests
* ✅ API Tests

---

# 🐳 Docker Deployment

Build Docker image:

```bash
docker build -t multilingual-misinfo .
```

Run container:

```bash
docker run -p 8000:8000 multilingual-misinfo
```

---

# 📸 Demo Screenshots

Add your screenshots here:

```markdown
![Dashboard](images/dashboard.png)

![Prediction](images/prediction.png)

![OCR Demo](images/ocr_demo.png)
```

---

# 🔮 Future Enhancements

* [ ] Real-time social media monitoring
* [ ] Explainable AI (XAI)
* [ ] Knowledge Graph integration
* [ ] Mobile application
* [ ] LLM-based reasoning
* [ ] Browser extension

---

# 📚 Research Papers

```bibtex
@article{xlmr,
  title={Unsupervised Cross-lingual Representation Learning},
  author={Conneau et al.},
  year={2020}
}
```

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push branch

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

---

<div align="center">

## 👨‍💻 Author

### **Shridhar Patil**

📧 **[shridharpatil0513@gmail.com](mailto:shridharpatil0513@gmail.com)**

💼 **Data Analyst | Data Scientist | AI Engineer**

⭐ **If you like this project, give it a star!**

</div>
