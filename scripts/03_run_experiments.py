"""Step 3: experiments E1-E3.

  E1  Does static split conformal hold 90% coverage on non-exchangeable market data?
  E2  Does text add value? price-only vs naive fusion vs trust-weighted fusion.
  E3  Manipulation: inject coordinated pump/dump posts into 10% of test stock-days.
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trustband.attacks import inject_attack
from trustband.config import load_config
from trustband.conformal import online_conformal
from trustband.evaluate import attention_spike_uplift, coverage_stats, daily_rank_ic
from trustband.features import FEATURE_SETS, PRICE, build_panel, text_features
from trustband.models import fit_mean_and_scale, predict


def between(df, period):
    return df[(df.date >= period[0]) & (df.date <= period[1])].reset_index(drop=True)


def main():
    cfg = load_config()
    proc, out = cfg["paths"]["processed_dir"], cfg["paths"]["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    per, cc, tcfg = cfg["periods"], cfg["conformal"], cfg["trust"]
    stress = cfg["stress_windows"]
    conf_kw = dict(alpha=cc["alpha"], gamma=cc["gamma"], window=cc["window"])

    px = pd.read_pickle(proc / "prices.pkl")
    tw = pd.read_pickle(proc / "tweets_sent.pkl")
    templates = pd.read_pickle(proc / "attack_templates.pkl")

    data = build_panel(px, tw, per["train"][0], per["test"][1], tcfg)
    tr, ca, te = between(data, per["train"]), between(data, per["calibration"]), between(data, per["test"])
    stats = {"n_train": len(tr), "n_cal": len(ca), "n_test": len(te), "tweet_day_share_test": float(te["has_tw"].mean())}
    print(stats)

    models = {name: fit_mean_and_scale(tr, feats, cfg["lightgbm"]) for name, feats in FEATURE_SETS.items()}

    # ---------------- E1 + E2: coverage under regime shift, value of text
    rows, rolls = [], {}
    for name, feats in FEATURE_SETS.items():
        yh_c, sg_c = predict(models[name], ca, feats)
        yh, sg = predict(models[name], te, feats)
        df = te.assign(yhat=yh, sg=sg, ay=np.abs(te["y"]))
        cal_abs = np.abs(ca["y"].values - yh_c)
        ic, vol_ic = daily_rank_ic(df, "yhat", "y"), daily_rank_ic(df, "sg", "ay")
        variants = {
            "split-abs": online_conformal(te.date, te.y, yh, np.ones(len(te)), cal_abs, adaptive=False, **conf_kw),
            "split-norm": online_conformal(te.date, te.y, yh, sg, cal_abs / sg_c, adaptive=False, **conf_kw),
            "aci-norm": online_conformal(te.date, te.y, yh, sg, cal_abs / sg_c, adaptive=True, **conf_kw),
        }
        for label, half in variants.items():
            r, roll = coverage_stats(te.date, te.y, yh, half, stress, f"{name}/{label}")
            rows.append({**r, "rank_ic": ic, "vol_rank_ic": vol_ic})
            rolls[f"{name}/{label}"] = roll
    e12 = pd.DataFrame(rows)
    print("\nE1/E2\n", e12.round(3).to_string(index=False))

    spikes = {col: attention_spike_uplift(data, col) for col in ["attn", "tattn"]}
    t = data[data.has_tw == 1]
    sent_vs_y = float(t["sent_mean"].corr(t["y"], method="spearman"))
    print("\nE2b attention-spike uplift:", json.dumps(spikes), " sentiment-vs-return spearman:", round(sent_vs_y, 4))

    # ---------------- E3: manipulation
    acfg = cfg["attacks"]
    rng = np.random.default_rng(acfg["seed"])
    pidx = pd.MultiIndex.from_frame(data[["ticker", "date"]])
    price_cols = data.drop(columns=[c for c in data.columns if c in set(sum(FEATURE_SETS.values(), [])) - set(PRICE)])
    att_rows = []
    for kind in acfg["kinds"]:
        tw_att, tgt = inject_attack(tw, te, kind, templates, rng, acfg["fraction"], acfg["min_posts"], acfg["target_seed"])
        data_att = pd.concat([price_cols.set_index(["ticker", "date"]), text_features(tw_att, pidx, tcfg)], axis=1).reset_index()
        te_att = between(data_att, per["test"])
        key = te_att.merge(tgt, on=["ticker", "date"], how="left")
        is_att = key["dir"].notna().values
        sign = np.where(key["dir"] == "pump", 1, -1)
        for name in ["naive", "trust"]:
            feats = FEATURE_SETS[name]
            yh0, _ = predict(models[name], te, feats)
            yh1, sg1 = predict(models[name], te_att, feats)
            yh_c, sg_c = predict(models[name], ca, feats)
            cal_norm = np.abs(ca["y"].values - yh_c) / sg_c
            variants = [("aci", yh1, online_conformal(te_att.date, te_att.y, yh1, sg1, cal_norm, **conf_kw))]
            if name == "trust":
                # trust gate: low-trust days fall back to the price-only model and get their own bucket
                low = te_att["trust"].fillna(1).values < tcfg["gate_threshold"]
                ca_low = ca["trust"].fillna(1).values < tcfg["gate_threshold"]
                yp, sp = predict(models["price"], te_att, PRICE)
                ypc, spc = predict(models["price"], ca, PRICE)
                yg, sgg = np.where(low, yp, yh1), np.where(low, sp, sg1)
                cal_g = np.abs(ca["y"].values - np.where(ca_low, ypc, yh_c)) / np.where(ca_low, spc, sg_c)
                half = online_conformal(te_att.date, te_att.y, yg, sgg, cal_g, cal_groups=ca_low.astype(int),
                                        groups=low.astype(int), **conf_kw)
                variants.append(("gate+mondrian-aci", yg, half))
            for vlabel, yh, half in variants:
                hit = np.abs(te_att.y.values - yh) <= half
                shift = (yh - yh0) * sign  # movement in the attacker's intended direction
                hijack = (np.sign(yh) == sign) & (np.sign(yh0) != sign)
                att_rows.append({
                    "attack": kind, "model": f"{name}/{vlabel}",
                    "shift_bp_attacked": 100 * shift[is_att].mean(), "hijack_rate": hijack[is_att].mean(),
                    "cov_attacked": hit[is_att].mean(), "cov_clean": hit[~is_att].mean(),
                    "width_attacked": 2 * np.median(half[is_att]), "width_clean": 2 * np.median(half[~is_att]),
                    "trust_attacked_med": float(np.nanmedian(te_att["trust"].values[is_att])),
                })
    e3 = pd.DataFrame(att_rows)
    print("\nE3\n", e3.round(3).to_string(index=False))

    # ---------------- outputs
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for k, c in [("naive/split-abs", "#c0392b"), ("naive/split-norm", "#e67e22"), ("naive/aci-norm", "#1f6feb")]:
        rolls[k].plot(ax=ax, label=k.split("/")[1], color=c, lw=1.6)
    ax.axhline(1 - cc["alpha"], ls="--", c="k", lw=1)
    for a, b in stress.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.15)
    ax.set_ylabel("rolling 20-day coverage")
    ax.set_title("Target 90% coverage: static split conformal vs adaptive conformal (shaded = stress windows)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "coverage_over_time.png", dpi=160)

    e12.to_csv(out / "e1_e2.csv", index=False)
    e3.to_csv(out / "e3_attack.csv", index=False)
    tw_stats = {
        "tweets": int(len(tw)), "users": int(tw.user_id.nunique()),
        "automated_share": float(tw.automated_client.mean()),
        "near_dup_share": float(1 - tw.norm.nunique() / len(tw)),
        "new_account_share": float((tw.acct_age_days < tcfg["new_account_days"]).mean()),
        "spam_cashtag_share": float((tw.n_cashtags >= tcfg["cashtag_spam_min"]).mean()),
    }
    with open(out / "stats.json", "w") as f:
        json.dump({**stats, **tw_stats, "attention_spikes": spikes, "sentiment_vs_return_spearman": sent_vs_y}, f, indent=2)


if __name__ == "__main__":
    main()
