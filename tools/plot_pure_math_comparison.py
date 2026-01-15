import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from tools.plot_utils import to_long_topic_key, to_display_name

plt.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 36,
    'axes.labelsize': 34,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 32,
    'figure.titlesize': 38
})


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
    if "results" in topic_json and isinstance(topic_json["results"], dict) and len(topic_json["results"]) == 1:
        only_val = next(iter(topic_json["results"].values()))
        sm = only_val.get("summary_metrics", {})
        if "mean_opinions" in sm:
            return np.array(sm["mean_opinions"], dtype=float)
    raise KeyError("mean_opinions not found")


def extract_std_opinions(topic_json: Dict, topic: str) -> Optional[np.ndarray]:
    if "std_opinions" in topic_json:
        return np.array(topic_json["std_opinions"], dtype=float)
    if "summary_metrics" in topic_json and "std_opinions" in topic_json["summary_metrics"]:
        return np.array(topic_json["summary_metrics"]["std_opinions"], dtype=float)
    if "results" in topic_json and topic in topic_json["results"]:
        sm = topic_json["results"][topic].get("summary_metrics", {})
        if "std_opinions" in sm:
            return np.array(sm["std_opinions"], dtype=float)
    if "results" in topic_json and isinstance(topic_json["results"], dict) and len(topic_json["results"]) == 1:
        only_val = next(iter(topic_json["results"].values()))
        sm = only_val.get("summary_metrics", {})
        if "std_opinions" in sm:
            return np.array(sm["std_opinions"], dtype=float)
    if "opinion_history" in topic_json:
        history = topic_json["opinion_history"]
        stds = [np.std(np.array(step, dtype=float)) for step in history]
        return np.array(stds, dtype=float)
    if "results" in topic_json and topic in topic_json["results"]:
        topic_data = topic_json["results"][topic]
        if "opinion_history" in topic_data:
            history = topic_data["opinion_history"]
            stds = [np.std(np.array(step, dtype=float)) for step in history]
            return np.array(stds, dtype=float)
    return None


def load_pure_math_mean(pure_path: str) -> np.ndarray:
    data = read_json(pure_path)
    return extract_mean_opinions(data, "pure_math")


def load_nano_mean(nano_dir: str, topic: str) -> np.ndarray:
    data = read_json(os.path.join(nano_dir, f"{topic}.json"))
    return extract_mean_opinions(data, topic)


def load_nano_std(nano_dir: str, topic: str) -> Optional[np.ndarray]:
    data = read_json(os.path.join(nano_dir, f"{topic}.json"))
    return extract_std_opinions(data, topic)


def load_mini_mean(mini_dir: str, topic: str) -> np.ndarray:
    long_key = to_long_topic_key(topic)
    mini_path = os.path.join(mini_dir, "degroot", long_key, f"{long_key}.json")
    if not os.path.exists(mini_path):
        mini_path = os.path.join(mini_dir, "degroot", long_key, f"{long_key}_streaming.json")
    data = read_json(mini_path)
    return extract_mean_opinions(data, topic)


def load_mini_std(mini_dir: str, topic: str) -> Optional[np.ndarray]:
    long_key = to_long_topic_key(topic)
    mini_path = os.path.join(mini_dir, "degroot", long_key, f"{long_key}.json")
    if not os.path.exists(mini_path):
        mini_path = os.path.join(mini_dir, "degroot", long_key, f"{long_key}_streaming.json")
    data = read_json(mini_path)
    return extract_std_opinions(data, topic)


def plot_topic(pure: np.ndarray, nano_a: np.ndarray, nano_b: np.ndarray, mini: np.ndarray, 
               topic: str, outdir: str, mini_b: np.ndarray = None,
               nano_a_std: np.ndarray = None, nano_b_std: np.ndarray = None, 
               mini_std: np.ndarray = None, mini_b_std: np.ndarray = None) -> None:
    
    n = min(len(pure), len(nano_a), len(nano_b), len(mini))
    if mini_b is not None:
        n = min(n, len(mini_b))
    
    x = np.arange(n)
    pure_trimmed = pure[:n]
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    arrays = [
        ("nano (a_vs_b)", nano_a[:n] - pure_trimmed, nano_a_std, "#f28e2b", 2.5),
        ("nano (b_vs_a)", nano_b[:n] - pure_trimmed, nano_b_std, "#59a14f", 2.5),
        ("mini (a_vs_b_mini)", mini[:n] - pure_trimmed, mini_std, "#e15759", 2.5),
    ]
    
    if mini_b is not None:
        arrays.append(("mini (b_vs_a_mini)", mini_b[:n] - pure_trimmed, mini_b_std, "#76b7b2", 2.5))
    
    for label, diff_arr, std_arr, color, linewidth in arrays:
        ax.plot(x, diff_arr, label=label, color=color, linewidth=linewidth)
        
        if std_arr is not None and len(std_arr) >= n:
            std_vals = std_arr[:n]
            ax.fill_between(x, 
                          diff_arr - std_vals, 
                          diff_arr + std_vals, 
                          alpha=0.2, color=color)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel("Timestep", fontsize=36, fontweight='bold')
    ax.set_ylabel("Difference from Mathematical Convergence", fontsize=36, fontweight='bold')
    long_key = to_long_topic_key(topic)
    ax.set_title(to_display_name(long_key), pad=10, fontsize=40, fontweight='bold')
    
    n_legend_items = len(arrays)
    ncol = 2
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=True, 
              fontsize=34, framealpha=0.9, columnspacing=2.0, handlelength=2.5)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=34, width=2, length=8)
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{topic}_convergence.png")
    ax.set_position([0.08, 0.15, 0.90, 0.75])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


def discover_nano_topics(nano_dir: str) -> List[str]:
    topics = []
    for name in os.listdir(nano_dir):
        if name.endswith(".json") and name != "experiment_metadata.json":
            topics.append(name[:-5])
    topics.sort()
    return topics


def main():
    parser = argparse.ArgumentParser(description="Plot difference from mathematical convergence per topic")
    parser.add_argument("--nano", required=True, help="Path to a_vs_b dir")
    parser.add_argument("--nano_b", required=True, help="Path to b_vs_a dir")
    parser.add_argument("--mini", required=True, help="Path to a_vs_b_mini dir")
    parser.add_argument("--mini_b", help="Path to b_vs_a_mini dir (optional)")
    parser.add_argument("--pure", required=True, help="Path to pure_math_smallworld.json")
    parser.add_argument("--outdir", default="results/pure_math_comparison", help="Output directory for plots")
    args = parser.parse_args()

    pure = load_pure_math_mean(args.pure)
    topics = discover_nano_topics(args.nano)

    for topic in topics:
        try:
            nano_a = load_nano_mean(args.nano, topic)
            nano_a_std = load_nano_std(args.nano, topic)
            long_key = to_long_topic_key(topic)
            nano_b_data = read_json(os.path.join(args.nano_b, f"{long_key}.json"))
            try:
                nano_b = extract_mean_opinions(nano_b_data, topic)
                nano_b_std = extract_std_opinions(nano_b_data, topic)
            except KeyError:
                nano_b = extract_mean_opinions(nano_b_data, long_key)
                nano_b_std = extract_std_opinions(nano_b_data, long_key)
            mini = load_mini_mean(args.mini, topic)
            mini_std = load_mini_std(args.mini, topic)
            
            mini_b = None
            mini_b_std = None
            if args.mini_b:
                try:
                    mini_b = load_mini_mean(args.mini_b, topic)
                    mini_b_std = load_mini_std(args.mini_b, topic)
                except (FileNotFoundError, KeyError):
                    pass
            
            plot_topic(pure, nano_a, nano_b, mini, topic, args.outdir, mini_b,
                      nano_a_std, nano_b_std, mini_std, mini_b_std)
        except (FileNotFoundError, KeyError):
            continue

    print(f"Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()
