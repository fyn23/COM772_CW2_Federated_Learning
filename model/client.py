"""
Flower client for FEMNIST federated learning.
Loads one user's (writer's) data and trains locally.

Usage:
    python3 model/client.py --user_id f1083_27 --server_address 127.0.0.1:8080
    
Or let it pick a random user:
    python model/client.py
"""

import argparse
import json
import numpy as np
import os
import sys
import pathlib
import random

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.cnn import build_femnist_keras, FlowerClient
import flwr as fl


def load_user_data(user_id, train_file=None, test_file=None, data_type="niid"):
    """
    Load training and test data for a specific user.
    
    Args:
        user_id: Writer ID (e.g., 'f1083_27')
        train_file: Optional specific train JSON file path
        test_file: Optional specific test JSON file path
        data_type: "niid" or "iid" to select data directory
    
    Returns:
        (x_train, y_train, x_test, y_test) as numpy arrays
    """
    # Select data directory based on type
    data_dir = f'data_{data_type}' if data_type in ['niid', 'iid'] else 'data_niid'
    TRAIN_DIR = os.path.join('preprocess', 'femnist', data_dir, 'train')
    TEST_DIR = os.path.join('preprocess', 'femnist', data_dir, 'test')
    
    # If no specific files provided, find one that contains the user
    if train_file is None:
        train_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith('.json')])
        if not train_files:
            raise FileNotFoundError(f"No JSON files found in {TRAIN_DIR}")
        
        # Search for user in train files
        for tf in train_files:
            train_path = os.path.join(TRAIN_DIR, tf)
            with open(train_path, 'r') as f:
                data = json.load(f)
            if user_id in data['users']:
                train_file = tf
                break
        
        if train_file is None:
            raise ValueError(f"User {user_id} not found in any training file")
    
    # Load training data
    train_path = os.path.join(TRAIN_DIR, train_file)
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Find corresponding test file
    if test_file is None:
        test_file = train_file.replace('_train_', '_test_')
    
    test_path = os.path.join(TEST_DIR, test_file)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    # Extract user's data
    if user_id not in train_data['user_data']:
        raise ValueError(f"User {user_id} not found in {train_file}")
    if user_id not in test_data['user_data']:
        raise ValueError(f"User {user_id} not found in {test_file}")
    
    train_x = train_data['user_data'][user_id]['x']
    train_y = train_data['user_data'][user_id]['y']
    test_x = test_data['user_data'][user_id]['x']
    test_y = test_data['user_data'][user_id]['y']
    
    # Convert to numpy
    x_train = np.asarray(train_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int32)
    x_test = np.asarray(test_x, dtype=np.float32)
    y_test = np.asarray(test_y, dtype=np.int32)
    
    return x_train, y_train, x_test, y_test


def get_random_user(data_type="niid"):
    """Pick a random user from available training data."""
    data_dir = f'data_{data_type}' if data_type in ['niid', 'iid'] else 'data_niid'
    TRAIN_DIR = os.path.join('preprocess', 'femnist', data_dir, 'train')
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.json')]
    if not train_files:
        raise FileNotFoundError(f"No training files found in {TRAIN_DIR}")
    
    # Pick random file
    train_file = random.choice(train_files)
    train_path = os.path.join(TRAIN_DIR, train_file)
    
    with open(train_path, 'r') as f:
        data = json.load(f)
    
    # Pick random user from that file
    user_id = random.choice(data['users'])
    return user_id


def main():
    parser = argparse.ArgumentParser(description="FEMNIST Flower Client")
    parser.add_argument(
        "--user_id",
        type=str,
        default=None,
        help="Writer ID to use (e.g., 'f1083_27'). If not provided, picks random user."
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Flower server address (default: 127.0.0.1:8080)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of local epochs per round (default: 1)"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="niid",
        choices=["niid", "iid"],
        help="Data type to use: niid (non-IID) or iid (IID) (default: niid)"
    )
    
    args = parser.parse_args()
    
    # Get user ID
    if args.user_id is None:
        user_id = get_random_user(data_type=args.data_type)
        print(f"No user specified, randomly selected: {user_id}")
    else:
        user_id = args.user_id
        print(f"Using specified user: {user_id}")
    
    # Load data
    print(f"Loading {args.data_type.upper()} data for user {user_id}...")
    x_train, y_train, x_test, y_test = load_user_data(user_id, data_type=args.data_type)
    print(f"Loaded train: {x_train.shape}, test: {x_test.shape}")
    
    # Build model
    print("Building model...")
    model = build_femnist_keras(num_classes=62)
    
    # Create client (pass user_id so client can save per-client metrics)
    client = FlowerClient(model, x_train, y_train, x_test, y_test, epochs=args.epochs, user_id=user_id)
    
    # Start client with increased message size limit
    print(f"Starting Flower client, connecting to {args.server_address}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
        grpc_max_message_length=100*1024*1024  # 100MB limit change based on your PC specs
    )


if __name__ == "__main__":
    main()
