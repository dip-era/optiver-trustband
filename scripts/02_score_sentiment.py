"""Step 2: FinBERT sentiment for every unique tweet text and for the attack templates."""
import pandas as pd

from trustband.attacks import template_frame
from trustband.config import load_config
from trustband.sentiment import score_texts, score_tweets


def main():
    cfg = load_config()
    proc = cfg["paths"]["processed_dir"]
    s = cfg["sentiment"]
    kw = dict(batch_size=s["batch_size"], max_length=s["max_length"])

    tw = pd.read_pickle(proc / "tweets.pkl")
    score_tweets(tw, s["model"], **kw).to_pickle(proc / "tweets_sent.pkl")

    templates = template_frame()
    scores = score_texts([t.replace("{T}", "XYZ") for t in templates.template], s["model"], **kw)
    templates = pd.concat([templates, scores], axis=1)
    templates.to_pickle(proc / "attack_templates.pkl")
    print(templates[["kind", "sent", "template"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
