import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_nano_topic(nano_dir: str, topic: str) -> Dict:
    topic_file = os.path.join(nano_dir, f"{topic}.json")
    with open(topic_file, "r") as f:
        return json.load(f)


def load_mini_topic(mini_dir: str, topic: str) -> Dict:
    topic_file = os.path.join(mini_dir, "degroot", topic, f"{topic}.json")
    with open(topic_file, "r") as f:
        return json.load(f)


def _extract_arrays(topic_json: Dict, topic: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Case 1: Newer consolidated mini files: arrays at top-level
    if all(k in topic_json for k in ("mean_opinions", "std_opinions", "final_opinions")):
        return (
            np.array(topic_json["mean_opinions"], dtype=float),
            np.array(topic_json["std_opinions"], dtype=float),
            np.array(topic_json["final_opinions"], dtype=float),
        )

    # Case 2: summary_metrics at top-level
    if "summary_metrics" in topic_json:
        sm = topic_json["summary_metrics"]
        return (
            np.array(sm["mean_opinions"], dtype=float),
            np.array(sm["std_opinions"], dtype=float),
            np.array(sm["final_opinions"], dtype=float),
        )

    # Case 3: Nested under results[topic]
    if "results" in topic_json and isinstance(topic_json["results"], dict):
        res = topic_json["results"]
        if topic in res and "summary_metrics" in res[topic]:
            sm = res[topic]["summary_metrics"]
            return (
                np.array(sm["mean_opinions"], dtype=float),
                np.array(sm["std_opinions"], dtype=float),
                np.array(sm["final_opinions"], dtype=float),
            )
        if len(res) == 1:
            only_val = next(iter(res.values()))
            if isinstance(only_val, dict) and "summary_metrics" in only_val:
                sm = only_val["summary_metrics"]
                return (
                    np.array(sm["mean_opinions"], dtype=float),
                    np.array(sm["std_opinions"], dtype=float),
                    np.array(sm["final_opinions"], dtype=float),
                )
    raise KeyError("Could not locate mean/std/final opinions in JSON structure")


def extract_metrics_from_nano(topic_json: Dict, topic: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _extract_arrays(topic_json, topic)


def extract_metrics_from_mini(topic_json: Dict, topic: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _extract_arrays(topic_json, topic)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def same_sign_fraction(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
    return float(np.mean(np.sign(a) == np.sign(b)))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
    if a.size < 2:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_topic_metrics(nano_dir: str, mini_dir: str, topic: str) -> Dict[str, float]:
    nano_json = load_nano_topic(nano_dir, topic)
    mini_json = load_mini_topic(mini_dir, topic)

    n_mean, n_std, n_final = extract_metrics_from_nano(nano_json, topic)
    m_mean, m_std, m_final = extract_metrics_from_mini(mini_json, topic)

    traj_rmse = rmse(n_mean, m_mean)
    disp_rmse = rmse(n_std, m_std)
    final_rmse = rmse(n_final, m_final)

    # Normalize RMSEs by opinion range ~2.0 for [-1, 1]
    norm = 2.0
    traj_rmse_n = traj_rmse / norm
    disp_rmse_n = disp_rmse / norm
    final_rmse_n = final_rmse / norm

    final_corr = pearson_corr(n_final, m_final)
    sign_frac = same_sign_fraction(n_final, m_final)

    # Simple fidelity: higher is better
    # Combine as average of: (1 - normalized RMSEs) and the two agreement metrics
    components = [
        max(0.0, 1.0 - traj_rmse_n),
        max(0.0, 1.0 - disp_rmse_n),
        max(0.0, 1.0 - final_rmse_n),
        max(0.0, min(1.0, (final_corr + 1.0) / 2.0)),  # map [-1,1] -> [0,1]
        max(0.0, min(1.0, sign_frac)),
    ]
    fidelity = float(np.mean(components))

    return {
        "topic": topic,
        "trajectory_rmse": traj_rmse,
        "dispersion_rmse": disp_rmse,
        "final_rmse": final_rmse,
        "final_corr": final_corr,
        "same_sign_frac": sign_frac,
        "fidelity_score": fidelity,
    }


def discover_topics(nano_dir: str) -> List[str]:
    topics = []
    for name in os.listdir(nano_dir):
        if name.endswith(".json") and name != "experiment_metadata.json":
            topics.append(name.replace(".json", ""))
    topics.sort()
    return topics


def write_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    import csv

    fieldnames = [
        "topic",
        "trajectory_rmse",
        "dispersion_rmse",
        "final_rmse",
        "final_corr",
        "same_sign_frac",
        "fidelity_score",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_fidelity(rows: List[Dict[str, float]], out_path: str) -> None:
    topics = [r["topic"] for r in rows]
    scores = [r["fidelity_score"] for r in rows]
    overall = float(np.mean(scores)) if scores else 0.0

    plt.figure(figsize=(10, 4))
    plt.bar(topics, scores, color="#4e79a7")
    plt.axhline(overall, color="#f28e2b", linestyle="--", label=f"overall={overall:.3f}")
    plt.ylim(0, 1)
    plt.ylabel("Fidelity (0-1)")
    plt.title("Algorithmic Fidelity: mini vs nano (per topic)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare fidelity: mini vs nano")
    parser.add_argument("--nano", required=True, help="Path to nano run directory (a_vs_b)")
    parser.add_argument("--mini", required=True, help="Path to mini run directory (a_vs_b_mini)")
    parser.add_argument("--outdir", default="results", help="Output directory for reports")
    args = parser.parse_args()

    topics = discover_topics(args.nano)
    if not topics:
        raise SystemExit("No topic JSONs found in nano directory")

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for t in topics:
        try:
            row = compute_topic_metrics(args.nano, args.mini, t)
            rows.append(row)
        except FileNotFoundError:
            continue

    if not rows:
        raise SystemExit("No comparable topics found.")

    csv_path = os.path.join(args.outdir, "compare.csv")
    png_path = os.path.join(args.outdir, "fidelity_by_topic.png")
    write_csv(rows, csv_path)
    plot_fidelity(rows, png_path)

    overall = float(np.mean([r["fidelity_score"] for r in rows]))
    print(f"Wrote {csv_path} and {png_path}. Overall fidelity: {overall:.3f}")


if __name__ == "__main__":
    main()


