import argparse
import json
import os
from statistics import median
from typing import Any, Dict, List

from network_of_agents.simulation.controller import Controller


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


def run_seed_sweep(config_path: str, start_seed: int, end_seed: int, timesteps: int, tolerance: float) -> None:
    # Load configuration
    config = load_config(config_path)

    sim_cfg = config.get('simulation', {})

    n_agents = sim_cfg.get('n_agents', 10)
    epsilon = sim_cfg.get('epsilon', 0.001)
    theta = sim_cfg.get('theta', 7)
    initial_connection_probability = sim_cfg.get('initial_connection_probability', 0.2)

    topics = config.get('topics') or ['default topic']
    topic = topics[0] if isinstance(topics, list) and topics else 'default topic'

    per_seed_results: List[Dict[str, Any]] = []

    total = (end_seed - start_seed + 1)
    print(f"Running consensus sweep: seeds {start_seed}..{end_seed} (total {total}), timesteps={timesteps}, tolerance={tolerance}")

    for seed in range(start_seed, end_seed + 1):
        try:
            controller = Controller(
                n_agents=n_agents,
                epsilon=epsilon,
                theta=theta,
                num_timesteps=timesteps,
                initial_connection_probability=initial_connection_probability,
                llm_client=None,
                topics=[topic],
                random_seed=seed,
                llm_enabled=False,
                noise_enabled=False,
            )

            results = controller.run_simulation(progress_bar=False)

            final_opinions = results.get('final_opinions', [])
            if not final_opinions:
                raise RuntimeError('Empty final_opinions')

            fmin = min(final_opinions)
            fmax = max(final_opinions)
            consensus_gap = float(fmax - fmin)
            converged = consensus_gap <= tolerance

            per_seed_results.append({
                'seed': seed,
                'converged': converged,
                'consensus_gap': consensus_gap,
                'final_opinions': final_opinions,
            })

        except Exception as e:
            per_seed_results.append({
                'seed': seed,
                'converged': False,
                'consensus_gap': None,
                'error': str(e),
            })

    # Compute summary
    gaps = [r['consensus_gap'] for r in per_seed_results if isinstance(r.get('consensus_gap'), (int, float))]
    non_converged = [r['seed'] for r in per_seed_results if not r.get('converged', False)]

    min_gap = min(gaps) if gaps else None
    med_gap = median(gaps) if gaps else None
    max_gap = max(gaps) if gaps else None

    print("\nSweep Summary")
    print("-------------")
    print(f"Seeds tested: {len(per_seed_results)}")
    print(f"Non-converged seeds ({len(non_converged)}): {non_converged if non_converged else 'None'}")
    if gaps:
        print(f"consensus_gap stats: min={min_gap:.6e}, median={med_gap:.6e}, max={max_gap:.6e}")
    else:
        print("consensus_gap stats: no data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a seed sweep and report which seeds do not converge to a single opinion (stdout only)",
    )
    parser.add_argument('--start-seed', type=int, default=0, help='Starting seed (inclusive)')
    parser.add_argument('--end-seed', type=int, default=49, help='Ending seed (inclusive)')
    parser.add_argument('--timesteps', type=int, default=5000, help='Timesteps per seed run')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Consensus tolerance in agent domain [-1,1]')
    parser.add_argument('--config', type=str, default='/home/adam/Projects/IDeA/network-of-agents/config.json', help='Path to config.json')

    args = parser.parse_args()

    run_seed_sweep(
        config_path=args.config,
        start_seed=args.start_seed,
        end_seed=args.end_seed,
        timesteps=args.timesteps,
        tolerance=args.tolerance,
    )


if __name__ == '__main__':
    main()


