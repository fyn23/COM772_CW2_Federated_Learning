# FEMNIST Federated Learning with Flower

This project implements federated learning on the FEMNIST dataset using the Flower framework and TensorFlow/Keras. It supports both IID and non-IID data splits, configurable strategies (FedAvg, FedProx), and detailed per-client and global analysis.

## Project Structure

```
.
├── model/
│   ├── cnn.py         # CNN model and FlowerClient definition
│   ├── client.py      # Client entry point
│   └── server.py      # Server entry point
├── preprocess/
│   └── femnist/       # Scripts for FEMNIST data preprocessing
├── data_analysis/     # Analysis and plotting scripts
├── results/           # Output metrics and per-client results
├── run_federated.sh   # Main orchestration script
└── README.md
```

## Key Components

- **run_federated.sh**: Entry point. Launches the server and multiple clients, passing experiment parameters (clients, rounds, epochs, strategy, etc.).
- **model/server.py**: Starts the federated server, selects strategy (FedAvg/FedProx), manages rounds, aggregates updates, and logs results.
- **model/client.py**: Loads local FEMNIST data, builds the CNN, wraps it in a FlowerClient, and connects to the server for training and evaluation.
- **model/cnn.py**: Defines the CNN architecture for 28x28 grayscale FEMNIST images and the FlowerClient class, which handles local training, evaluation, and per-client metric saving.
- **preprocess/femnist/**: Scripts to download, process, group, filter, sample, and split FEMNIST data into IID and non-IID partitions.
- **data_analysis/**: Python scripts using pandas and matplotlib to visualize results (accuracy/loss vs. epochs, per-client metrics, strategy comparisons, etc.).

## Data Preprocessing

1. Download and extract raw FEMNIST data.
2. Convert to JSON, group by writer, filter users, sample, and split into train/test sets (IID and non-IID).
3. Output: Per-client data directories for federated simulation.
4. You will need to paste the outputted directories to data_niid or data_iid under preprocess/femnist/ or change the directory in line 42 of client.py to match the training/test splits
5. If this does not work please visit: https://leaf.cmu.edu/ and clone their data preprocessing scripts directly

## Federated Learning Flow

1. **Server** broadcasts the global model to all clients each round.
2. **Clients** train locally for several epochs on their own data, then send updated weights and metrics back.
3. **Server** aggregates updates (FedAvg or FedProx), updates the global model, and logs metrics.
4. Repeat for the specified number of rounds.

## Analysis

- Scripts in `data_analysis/` generate plots for:
  - Accuracy/loss vs. epochs
  - Per-client evaluation metrics
  - IID vs. non-IID comparison
  - FedAvg vs. FedProx performance

## How to Run

1. **Set up environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Preprocess data**  
   Run scripts in `preprocess/femnist/` as needed.

3. **Run federated experiment**  
   ```bash
   bash run_federated.sh
   ```
   Adjust variables in the script for your experiment (number of clients, rounds, epochs, strategy, etc.).

4. **Analyze results**  
   Use scripts in `data_analysis/` to generate plots and summaries.

## Technologies Used

- Python 3.11+
- TensorFlow/Keras
- Flower (flwr)
- pandas, matplotlib
- Bash

## Notes

- All client and global metrics are saved in the `results/` directory.
- The system is modular: you can easily swap models, strategies, or data splits.
- Designed for research and experimentation with federated learning on realistic, user-partitioned data.
