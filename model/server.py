"""
Flower server for FEMNIST federated learning.

Usage:
    python model/server.py --rounds 50 --min_clients 2
"""

import argparse
import csv
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    
    # Aggregate and return custom metric (weighted average)
    total_examples = sum(examples)
    return {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0,
        "loss": sum(losses) / total_examples if total_examples > 0 else 0,
        "num_clients": len(metrics),
        "total_examples": total_examples,
    }


def log_metrics_to_csv(csv_file: str, round_num: int, metrics: Dict, phase: str):
    """Log metrics to CSV file."""
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['timestamp', 'round', 'phase', 'loss', 'accuracy', 'num_clients', 'total_examples'])
        
        # Write metrics
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            round_num,
            phase,
            metrics.get('loss', 0),
            metrics.get('accuracy', 0),
            metrics.get('num_clients', 0),
            metrics.get('total_examples', 0)
        ])


def main():
    parser = argparse.ArgumentParser(description="FEMNIST Flower Server")
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)"
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=2,
        help="Minimum number of clients required per round (default: 2)"
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=1.0,
        help="Fraction of clients to sample for training per round (default: 1.0)"
    )
    parser.add_argument(
        "--fraction_evaluate",
        type=float,
        default=1.0,
        help="Fraction of clients to sample for evaluation per round (default: 1.0)"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address to bind to (default: 0.0.0.0:8080)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox"],
        help="Federated strategy to use: fedavg or fedprox (default: fedavg)"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="FedProx proximal term coefficient (only used if --strategy fedprox)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV file path
    csv_file = os.path.join(args.output_dir, f"{args.run_name}_metrics.csv")
    
    # Track current round for logging
    current_round = {'value': 0}

    # Config functions to send to clients
    def on_fit_config(rnd: int):
        cfg = {"round": rnd}
        if args.strategy == 'fedprox' and args.mu > 0:
            cfg['mu'] = float(args.mu)
        return cfg

    def on_evaluate_config(rnd: int):
        cfg = {"round": rnd}
        return cfg
    
    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate training metrics and log to CSV."""
        aggregated = weighted_average(metrics)
        current_round['value'] += 1
        log_metrics_to_csv(csv_file, current_round['value'], aggregated, 'train')
        print(f"Round {current_round['value']} - Train Loss: {aggregated['loss']:.4f}, Train Acc: {aggregated['accuracy']:.4f}")
        return aggregated
    
    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate evaluation metrics and log to CSV."""
        aggregated = weighted_average(metrics)
        log_metrics_to_csv(csv_file, current_round['value'], aggregated, 'evaluate')
        print(f"Round {current_round['value']} - Eval Loss: {aggregated['loss']:.4f}, Eval Acc: {aggregated['accuracy']:.4f}")
        return aggregated
    
    # Define strategy (FedAvg or FedProx)
    if args.strategy == 'fedprox':
        try:
            strategy = fl.server.strategy.FedProx(
                fraction_fit=args.fraction_fit,
                fraction_evaluate=args.fraction_evaluate,
                min_fit_clients=args.min_clients,
                min_evaluate_clients=args.min_clients,
                min_available_clients=args.min_clients,
                mu=float(args.mu),
                on_fit_config_fn=on_fit_config,
                on_evaluate_config_fn=on_evaluate_config,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            )
        except Exception:
            # Fall back to FedAvg if FedProx isn't available
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=args.fraction_fit,
                fraction_evaluate=args.fraction_evaluate,
                min_fit_clients=args.min_clients,
                min_evaluate_clients=args.min_clients,
                min_available_clients=args.min_clients,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_fit_clients=args.min_clients,
            min_evaluate_clients=args.min_clients,
            min_available_clients=args.min_clients,
            on_fit_config_fn=on_fit_config,
            on_evaluate_config_fn=on_evaluate_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
    
    print(f"Starting Flower server...")
    print(f"  Address: {args.server_address}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Min clients: {args.min_clients}")
    print(f"  Fraction fit: {args.fraction_fit}")
    print(f"  Fraction evaluate: {args.fraction_evaluate}")
    print(f"  Output CSV: {csv_file}")
    print(f"  Run name: {args.run_name}")
    
    # Start server with increased message size limit
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=100*1024*1024  # 100MB limit
    )


if __name__ == "__main__":
    main()