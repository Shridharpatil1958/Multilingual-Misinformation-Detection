# 🧠 Multilingual Misinformation Detection

An end-to-end pipeline for detecting misinformation in Hindi, Tamil, and English social media posts using cross-lingual NLP, evidence retrieval, and NLI.

## Project Structure

```
multilingual-misinfo/
├── configs/                    # YAML config files
│   ├── model_config.yaml
│   └── retrieval_config.yaml
├── data/
│   ├── raw/                    # Downloaded raw datasets
│   ├── processed/              # Tokenized & cleaned data
│   └── index/                  # FAISS index files
├── src/
│   ├── data/
│   │   ├── dataset_loader.py   # Load LIAR, Factify, custom datasets
│   │   ├── preprocessor.py     # Clean, normalize, language detect
│   │   └── hinglish_handler.py # Code-mixed text handling
│   ├── models/
│   │   ├── classifier.py       # XLM-R claim classifier
│   │   ├── nli_model.py        # mDeBERTa NLI model
│   │   └── trainer.py          # Training loop
│   ├── retrieval/
│   │   ├── bm25_retriever.py   # Sparse BM25 retrieval
│   │   ├── dense_retriever.py  # FAISS dense retrieval
│   │   └── hybrid_retriever.py # Hybrid BM25 + FAISS
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── schemas.py          # Pydantic models
│   │   └── router.py           # API routes
│   └── utils/
│       ├── metrics.py          # Eval metrics
│       ├── logger.py           # Logging setup
│       └── ocr.py              # OCR for WhatsApp images
├── scripts/
│   ├── download_data.py        # Dataset download scripts
│   ├── build_index.py          # Build FAISS index
│   └── evaluate.py             # Full pipeline evaluation
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_retrieval_eval.ipynb
├── demo/
│   └── app.py                  # Gradio demo
├── tests/
│   ├── test_classifier.py
│   ├── test_retrieval.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
git clone https://github.com/yourname/multilingual-misinfo
cd multilingual-misinfo
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_index.py
```

## Train

```bash
# Phase 1: Claim classifier
python -m src.models.trainer --config configs/model_config.yaml --task classify

# Phase 2: NLI model
python -m src.models.trainer --config configs/model_config.yaml --task nli
```

## Run API

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Run Demo

```bash
python demo/app.py
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/model_config.yaml
```
