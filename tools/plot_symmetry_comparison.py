import argparse
import json
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_mean_opinions(topic_json: Dict, topic: str) -> np.ndarray:
    if "mean_opinions" in topic_json:
        return np.array(topic_json["mean_opinions"], dtype=float)
    if "summary_metrics" in topic_json and "mean_opinions" in topic_json["summary_metrics"]:
        return np.array(topic_json["summary_metrics"]["mean_opinions"], dtype=float)
    if "results" in topic_json and isinstance(topic_json["results"], dict):
        res = topic_json["results"]
        if topic in res and isinstance(res[topic], Dict):
            sm = res[topic].get("summary_metrics", {})
            if "mean_opinions" in sm:
                return np.array(sm["mean_opinions"], dtype=float)
        if len(res) == 1:
            only_val = next(iter(res.values()))
            if isinstance(only_val, dict):
                sm = only_val.get("summary_metrics", {})
                if "mean_opinions" in sm:
                    return np.array(sm["mean_opinions"], dtype=float)
    raise KeyError("mean_opinions not found")


TOPIC_MAP = {
    "activism": "corporate_activism",
    "cloning": "human_cloning",
    "democracy": "social_media_democracy",
    "economy": "environment_economy",
    "etiquette": "restaurant_etiquette",
    "paper": "toilet_paper",
    "safety": "gun_safety",
    "sandwich": "hot_dog_sandwich",
    "weddings": "child_free_weddings",
    "immigration": "immigration",
}


def load_topic_series(a_dir: str, b_dir: str, pure_path: str, topic: str) -> Dict[str, np.ndarray]:
    a_json = read_json(os.path.join(a_dir, f"{topic}.json"))
    b_name = TOPIC_MAP.get(topic, topic)
    b_json = read_json(os.path.join(b_dir, f"{b_name}.json"))
    pure_json = read_json(pure_path)

    a = extract_mean_opinions(a_json, topic)
    try:
        b = extract_mean_opinions(b_json, topic)
    except KeyError:
        b = extract_mean_opinions(b_json, b_name)
    pure = extract_mean_opinions(pure_json, "pure_math")

    n = min(len(a), len(b), len(pure))
    return {"a": a[:n], "b": b[:n], "pure": pure[:n]}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def plot_overlay(topic: str, series: Dict[str, np.ndarray], outdir: str) -> None:
    a = series["a"]
    b = series["b"]
    pure = series["pure"]
    n = min(len(a), len(b), len(pure))
    x = np.arange(n)

    r_a = rmse(a, pure)
    r_b = rmse(b, pure)
    # Order-effect not displayed per user request

    plt.figure(figsize=(8, 4))
    plt.plot(x, pure, label="pure_math", color="#4e79a7", linewidth=2)
    plt.plot(x, a, label="a_vs_b", color="#f28e2b")
    plt.plot(x, b, label="b_vs_a", color="#59a14f")
    plt.ylim(-1, 1)
    plt.xlabel("timestep")
    plt.ylabel("mean opinion")
    plt.title(f"{topic} — Overlay vs pure_math")
    plt.suptitle(f"RMSE a_vs_pure={r_a:.3f}, b_vs_pure={r_b:.3f}", y=0.98, fontsize=9)
    plt.legend()
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{topic}_overlay.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


# Delta plots removed per user request


# Summary plots removed per user request


# CSV output removed per user request


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot symmetry comparison: pure_math vs a_vs_b vs b_vs_a")
    parser.add_argument("--a", required=True, help="Path to a_vs_b nano dir")
    parser.add_argument("--b", required=True, help="Path to b_vs_a nano dir")
    parser.add_argument("--pure", required=True, help="Path to pure_math_smallworld.json")
    parser.add_argument("--outdir", default="results/symmetry_comparison", help="Output directory for plots")
    parser.add_argument("--topics", default="", help="Optional comma-separated subset of topics")
    args = parser.parse_args()

    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    else:
        topics = []
        for name in os.listdir(args.a):
            if name.endswith(".json") and name != "experiment_metadata.json":
                topics.append(name[:-5])
        topics.sort()

    for topic in topics:
        try:
            series = load_topic_series(args.a, args.b, args.pure, topic)
        except (FileNotFoundError, KeyError):
            continue

        plot_overlay(topic, series, args.outdir)

    print(f"Saved symmetry overlay plots to {args.outdir}")


if __name__ == "__main__":
    main()


