#!/usr/bin/env python3
import os
import json
import argparse
import logging
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run import create_visualizations  # reuse plotting


def slug(s: str) -> str:
    return str(s).lower().replace(" ", "_").replace("-", "_")


def find_json_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json") and fn != "experiment_metadata.json":
                yield os.path.join(dirpath, fn)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Repair and normalize an experiment directory")
    parser.add_argument("experiment_path", help="Path to experiment directory under results/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("repair")

    experiment_path = os.path.abspath(args.experiment_path)
    if not os.path.isdir(experiment_path):
        logger.error(f"Not a directory: {experiment_path}")
        return 1

    # Collect results by (model, topic)
    results = []
    for json_path in find_json_files(experiment_path):
        try:
            data = load_json(json_path)
        except Exception:
            continue

        meta = data.get("experiment_metadata", {})
        model = slug(meta.get("model", "degroot"))
        topics = meta.get("topics") or ["pure_math"]
        topic = slug(topics[0])
        if topic == "pure_math_model":
            topic = "pure_math"

        # Determine target location
        target_dir = os.path.join(experiment_path, model, topic)
        ensure_dir(target_dir)

        # Move file if not already there and if it looks like a topic JSON (not streaming)
        base = os.path.basename(json_path)
        is_streaming = base.endswith("_streaming.json")

        # Write normalized JSON under correct directory
        if topic == "pure_math":
            topology = slug(meta.get("topology", "unknown"))
            out_name = f"{topic}_{topology}.json"
        else:
            out_name = f"{topic}.json"
        out_path = os.path.join(target_dir, out_name)
        try:
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            continue

        # Optionally remove original if different
        if os.path.abspath(json_path) != os.path.abspath(out_path) and not is_streaming:
            try:
                os.remove(json_path)
            except Exception:
                pass

        results.append(data)

    # Remove stray _unknown dirs if empty
    for model_dir in [d for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]:
        unknown = os.path.join(experiment_path, model_dir, "_unknown")
        if os.path.isdir(unknown) and not os.listdir(unknown):
            try:
                os.rmdir(unknown)
            except Exception:
                pass

    # Recreate plots using normalized structure
    try:
        create_visualizations(results, experiment_path, {})
    except Exception:
        logger.warning("Plot regeneration failed; continue.")

    logger.info("Repair complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


