"""Step 1: download StockNet and build clean price + tweet tables (data/processed/*.pkl)."""
from trustband.config import load_config
from trustband.data import build, download_stocknet


def main():
    cfg = load_config()
    paths = cfg["paths"]
    zip_path = download_stocknet(paths["stocknet_url"], paths["raw_dir"] / "stocknet.zip")
    px, tw = build(zip_path)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    px.to_pickle(paths["processed_dir"] / "prices.pkl")
    tw.to_pickle(paths["processed_dir"] / "tweets.pkl")
    print(f"prices: {len(px):,} rows, {px.ticker.nunique()} tickers, {px.date.min():%Y-%m-%d}..{px.date.max():%Y-%m-%d}")
    print(f"tweets: {len(tw):,} rows, {tw.user_id.nunique():,} users, {tw.date.min():%Y-%m-%d}..{tw.date.max():%Y-%m-%d}")
    print(f"automated-client share: {tw.automated_client.mean():.3f}  near-duplicate share: {1 - tw.norm.nunique() / len(tw):.3f}")


if __name__ == "__main__":
    main()
