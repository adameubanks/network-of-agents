import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_mean_opinions(topic_json: Dict, topic: str) -> np.ndarray:
    if "mean_opinions" in topic_json:
        return np.array(topic_json["mean_opinions"], dtype=float)
    if "summary_metrics" in topic_json and "mean_opinions" in topic_json["summary_metrics"]:
        return np.array(topic_json["summary_metrics"]["mean_opinions"], dtype=float)
    if "results" in topic_json and topic in topic_json["results"]:
        sm = topic_json["results"][topic].get("summary_metrics", {})
        if "mean_opinions" in sm:
            return np.array(sm["mean_opinions"], dtype=float)
    # As a fallback for single-topic structures
    if "results" in topic_json and isinstance(topic_json["results"], dict) and len(topic_json["results"]) == 1:
        only_val = next(iter(topic_json["results"].values()))
        sm = only_val.get("summary_metrics", {})
        if "mean_opinions" in sm:
            return np.array(sm["mean_opinions"], dtype=float)
    raise KeyError("mean_opinions not found")


def load_pure_math_mean(pure_path: str) -> np.ndarray:
    data = read_json(pure_path)
    return extract_mean_opinions(data, "pure_math")


def load_nano_mean(nano_dir: str, topic: str) -> np.ndarray:
    data = read_json(os.path.join(nano_dir, f"{topic}.json"))
    return extract_mean_opinions(data, topic)


def load_mini_mean(mini_dir: str, topic: str) -> np.ndarray:
    mini_path = os.path.join(mini_dir, "degroot", TOPIC_MAP.get(topic, topic), f"{TOPIC_MAP.get(topic, topic)}.json")
    data = read_json(mini_path)
    return extract_mean_opinions(data, topic)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def plot_topic(pure: np.ndarray, nano: np.ndarray, mini: np.ndarray, topic: str, outdir: str) -> None:
    n = min(len(pure), len(nano), len(mini))
    x = np.arange(n)
    pure_n = pure[:n]
    nano_n = nano[:n]
    mini_n = mini[:n]

    nano_rmse = rmse(nano_n, pure_n)
    mini_rmse = rmse(mini_n, pure_n)

    plt.figure(figsize=(8, 4))
    plt.plot(x, pure_n, label="pure_math", color="#4e79a7", linewidth=2)
    plt.plot(x, nano_n, label="nano (a_vs_b)", color="#f28e2b")
    plt.plot(x, mini_n, label="mini (a_vs_b_mini)", color="#59a14f")
    plt.ylim(-1, 1)
    plt.xlabel("timestep")
    plt.ylabel("mean opinion")
    plt.title(f"{topic} — Convergence vs pure_math")
    plt.suptitle(f"RMSE nano={nano_rmse:.3f}, mini={mini_rmse:.3f}", y=0.98, fontsize=9)
    plt.legend()
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{topic}_convergence.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def discover_nano_topics(nano_dir: str) -> List[str]:
    topics = []
    for name in os.listdir(nano_dir):
        if name.endswith(".json") and name != "experiment_metadata.json":
            topics.append(name[:-5])
    topics.sort()
    return topics


def main():
    parser = argparse.ArgumentParser(description="Plot convergence vs pure_math for each topic")
    parser.add_argument("--nano", required=True, help="Path to a_vs_b dir")
    parser.add_argument("--mini", required=True, help="Path to a_vs_b_mini dir")
    parser.add_argument("--pure", required=True, help="Path to pure_math_smallworld.json")
    parser.add_argument("--outdir", default="results/pure_math_comparison", help="Output directory for plots")
    args = parser.parse_args()

    pure = load_pure_math_mean(args.pure)
    topics = discover_nano_topics(args.nano)

    for topic in topics:
        try:
            nano = load_nano_mean(args.nano, topic)
            mini = load_mini_mean(args.mini, topic)
            plot_topic(pure, nano, mini, topic, args.outdir)
        except FileNotFoundError:
            continue
        except KeyError:
            continue

    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()


