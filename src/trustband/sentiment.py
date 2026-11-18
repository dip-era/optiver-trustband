"""FinBERT sentiment for tweet text.

FinBERT (ProsusAI/finbert) is a BERT classifier fine-tuned on Financial PhraseBank. Unlike a modern
chat LLM it has no knowledge of what happened to these stocks in 2014-16, so scoring historical text
with it does not leak future outcomes.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def clean(t: str) -> str:
    return re.sub(r"https?://\S+", "", t).strip()


@torch.no_grad()
def score_texts(texts: list[str], model_name: str, batch_size: int = 256, max_length: int = 64) -> pd.DataFrame:
    """Return p_positive / p_negative / p_neutral and sent = p_pos - p_neg for each text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  {i:,}/{len(texts):,}", flush=True)
    p = pd.DataFrame(np.vstack(out), columns=[f"p_{label}" for label in labels])
    p["sent"] = p["p_positive"] - p["p_negative"]
    return p


def score_tweets(tw: pd.DataFrame, model_name: str, **kw) -> pd.DataFrame:
    """Score each unique link-stripped text once, then merge the scores back onto every tweet."""
    tw = tw.copy()
    tw["clean"] = tw["text"].map(clean)
    uniq = tw["clean"].drop_duplicates().reset_index(drop=True)
    print(f"scoring {len(uniq):,} unique texts on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    scores = score_texts(uniq.tolist(), model_name, **kw)
    scores["clean"] = uniq
    return tw.merge(scores, on="clean", how="left")
