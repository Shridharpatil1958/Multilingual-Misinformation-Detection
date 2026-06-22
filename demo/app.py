# demo/app.py

"""
Gradio demo for Multilingual Misinformation Detector.
Run: python demo/app.py
"""

import sys
sys.path.insert(0, ".")

import gradio as gr
import httpx
import json
from pathlib import Path

API_URL = "http://localhost:8000"
LABEL_COLORS = {
    "SUPPORTS": "#22c55e",
    "REFUTES":  "#ef4444",
    "NOT_ENOUGH_INFO": "#f59e0b",
}
LABEL_EMOJI = {
    "SUPPORTS": "✅",
    "REFUTES":  "❌",
    "NOT_ENOUGH_INFO": "⚠️",
}


def verify_claim(claim: str, top_k: int, use_api: bool):
    """Call the FastAPI backend and format results for Gradio."""
    if not claim.strip():
        return "Please enter a claim.", "", ""

    if use_api:
        try:
            resp = httpx.post(
                f"{API_URL}/verify",
                json={"claim": claim, "top_k_evidence": int(top_k), "return_evidence": True},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"API Error: {e}", "", ""
    else:
        # Direct inference (no API server needed for demo)
        try:
            import yaml, torch
            from src.models.classifier import ClaimClassifier
            from src.data.preprocessor import TextPreprocessor

            config_path = "configs/model_config.yaml"
            if not Path(config_path).exists():
                return "Config not found. Run from project root.", "", ""

            with open(config_path) as f:
                config = yaml.safe_load(f)

            cls_path = config["classifier"]["output_dir"]
            if not Path(cls_path).exists():
                return "Model not trained yet. Run: python -m src.models.trainer --task classify", "", ""

            prep = TextPreprocessor(config["classifier"]["model_name"])
            model = ClaimClassifier.load(cls_path).eval()
            enc = prep.tokenize([claim])
            preds, probs = model.predict(enc["input_ids"], enc["attention_mask"])

            from src.models.classifier import ID2LABEL
            verdict = ID2LABEL[preds[0].item()]
            probs_list = probs[0].tolist()
            labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            data = {
                "verdict": verdict,
                "confidence": max(probs_list),
                "probabilities": {l: round(p, 4) for l, p in zip(labels, probs_list)},
                "language": "auto",
                "is_hinglish": False,
                "best_evidence": "(Direct inference — no retrieval)",
                "evidence_passages": [],
            }
        except Exception as e:
            return f"Error: {e}", "", ""

    verdict = data["verdict"]
    confidence = data["confidence"]
    probs = data["probabilities"]
    lang = data.get("language", "?")
    is_hinglish = data.get("is_hinglish", False)
    best_ev = data.get("best_evidence", "")

    # Verdict card
    emoji = LABEL_EMOJI.get(verdict, "?")
    result_md = f"""
## {emoji} {verdict}

**Confidence:** {confidence:.1%}  
**Language:** {lang}{' 🇮🇳 Hinglish detected' if is_hinglish else ''}

| Label | Probability |
|---|---|
| ✅ SUPPORTS | {probs.get('SUPPORTS', 0):.1%} |
| ❌ REFUTES | {probs.get('REFUTES', 0):.1%} |
| ⚠️ NOT ENOUGH INFO | {probs.get('NOT_ENOUGH_INFO', 0):.1%} |
"""

    evidence_md = f"**Best supporting evidence:**\n\n> {best_ev}" if best_ev else "No evidence retrieved."

    passages = data.get("evidence_passages") or []
    passages_text = ""
    for i, p in enumerate(passages[:5], 1):
        nli = p.get("nli_label", "")
        passages_text += f"**[{i}]** {LABEL_EMOJI.get(nli, '')} *{nli}*\n{p.get('text', '')[:300]}...\n\n---\n\n"

    return result_md, evidence_md, passages_text or "No passages retrieved."


with gr.Blocks(
    title="Multilingual Misinfo Detector",
    theme=gr.themes.Soft(),
    css=".verdict-card { border-radius: 12px; padding: 16px; }",
) as demo:

    gr.Markdown("""
# 🧠 Multilingual Misinformation Detector
Verify claims in **Hindi, Tamil, English, and Hinglish** using cross-lingual NLP + evidence retrieval.
""")

    with gr.Row():
        with gr.Column(scale=2):
            claim_input = gr.Textbox(
                label="Enter claim to verify",
                placeholder="e.g. COVID vaccine mein microchip hai / Onions cure fever / ...",
                lines=3,
            )
            with gr.Row():
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Evidence passages")
                use_api_checkbox = gr.Checkbox(label="Use API server", value=True)
            verify_btn = gr.Button("🔍 Verify Claim", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Try these examples:")
            examples = gr.Examples(
                examples=[
                    ["COVID vaccine contains microchips", 5, True],
                    ["COVID vaccine mein microchip hai", 5, True],
                    ["India launched Chandrayaan-3 in 2023", 5, True],
                    ["Drinking hot water kills COVID virus immediately", 5, True],
                    ["yeh news sach hai ya jhoot — onion juice COVID cure karta hai", 5, True],
                ],
                inputs=[claim_input, top_k_slider, use_api_checkbox],
            )

    with gr.Row():
        verdict_output = gr.Markdown(label="Verdict")

    with gr.Accordion("Evidence", open=False):
        best_evidence_output = gr.Markdown()
        passages_output = gr.Markdown()

    verify_btn.click(
        fn=verify_claim,
        inputs=[claim_input, top_k_slider, use_api_checkbox],
        outputs=[verdict_output, best_evidence_output, passages_output],
    )

    gr.Markdown("""
---
**Models:** XLM-RoBERTa (classifier) · mDeBERTa-v3 (NLI) · MiniLM-multilingual (retrieval)  
**Datasets:** LIAR · Factify · AltNews · Boom Live  
""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
