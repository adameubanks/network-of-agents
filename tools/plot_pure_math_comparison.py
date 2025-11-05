import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tools.plot_utils import to_long_topic_key, make_descriptive_title


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
    long_key = to_long_topic_key(topic)
    # Try both regular and streaming file names
    mini_path = os.path.join(mini_dir, "degroot", long_key, f"{long_key}.json")
    if not os.path.exists(mini_path):
        mini_path = os.path.join(mini_dir, "degroot", long_key, f"{long_key}_streaming.json")
    data = read_json(mini_path)
    return extract_mean_opinions(data, topic)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def plot_topic(pure: np.ndarray, nano_a: np.ndarray, nano_b: np.ndarray, mini: np.ndarray, 
               topic: str, outdir: str, mini_b: np.ndarray = None) -> None:
    # Build list of arrays that exist
    arrays = [("pure_math", pure, "#4e79a7", 2),
              ("nano (a_vs_b)", nano_a, "#f28e2b", 1),
              ("nano (b_vs_a)", nano_b, "#59a14f", 1),
              ("mini (a_vs_b_mini)", mini, "#e15759", 1)]
    
    if mini_b is not None:
        arrays.append(("mini (b_vs_a_mini)", mini_b, "#76b7b2", 1))
    
    # Find minimum length
    n = min(len(arr) for _, arr, _, _ in arrays)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    for label, arr, color, linewidth in arrays:
        ax.plot(x, arr[:n], label=label, color=color, linewidth=linewidth)
    
    ax.set_ylim(-1, 1)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean opinion")
    ax.set_title(make_descriptive_title(topic, multiline=True), pad=4)
    ax.legend()

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
    parser = argparse.ArgumentParser(description="Plot convergence vs pure_math per topic (supersedes symmetry overlays)")
    parser.add_argument("--nano", required=True, help="Path to a_vs_b dir")
    parser.add_argument("--nano_b", required=True, help="Path to b_vs_a dir")
    parser.add_argument("--mini", required=True, help="Path to a_vs_b_mini dir")
    parser.add_argument("--mini_b", help="Path to b_vs_a_mini dir (optional, for incomplete results)")
    parser.add_argument("--pure", required=True, help="Path to pure_math_smallworld.json")
    parser.add_argument("--outdir", default="results/pure_math_comparison", help="Output directory for plots")
    args = parser.parse_args()

    pure = load_pure_math_mean(args.pure)
    topics = discover_nano_topics(args.nano)

    for topic in topics:
        try:
            nano = load_nano_mean(args.nano, topic)
            long_key = to_long_topic_key(topic)
            b_json_path = os.path.join(args.nano_b, f"{long_key}.json")
            nano_b_data = read_json(b_json_path)
            try:
                nano_b = extract_mean_opinions(nano_b_data, topic)
            except KeyError:
                nano_b = extract_mean_opinions(nano_b_data, long_key)
            mini = load_mini_mean(args.mini, topic)
            
            # Try to load mini_b if provided
            mini_b = None
            if args.mini_b:
                try:
                    mini_b = load_mini_mean(args.mini_b, topic)
                except (FileNotFoundError, KeyError):
                    # Topic not available in mini_b yet (incomplete)
                    pass
            
            plot_topic(pure, nano, nano_b, mini, topic, args.outdir, mini_b)
        except FileNotFoundError:
            continue
        except KeyError:
            continue

    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()


