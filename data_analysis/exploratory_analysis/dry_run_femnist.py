import json
import numpy as np
import os
import sys
import pathlib

# Ensure project root is on sys.path so local packages (e.g., model) can be imported
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.cnn import build_femnist_keras

# Small dry-run loader and trainer for one user
TRAIN_DIR = os.path.join('preprocess', 'femnist', 'data_niid', 'train')
TEST_DIR = os.path.join('preprocess', 'femnist', 'data_niid', 'test')

# Find matching train and test JSON files
train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.json')]
if not train_files:
    raise SystemExit(f"No JSON files found in {TRAIN_DIR}; run preprocessing first")

# Pick first train file
train_file = train_files[0]
train_path = os.path.join(TRAIN_DIR, train_file)

# Find corresponding test file (same base name, but in test/ folder)
test_file = train_file.replace('_train_', '_test_')
test_path = os.path.join(TEST_DIR, test_file)

if not os.path.exists(test_path):
    raise SystemExit(f"Corresponding test file not found: {test_path}")

print(f"Loading TRAIN: {train_path}")
with open(train_path, 'r') as f:
    train_data = json.load(f)

print(f"Loading TEST:  {test_path}")
with open(test_path, 'r') as f:
    test_data = json.load(f)

# Pick first user from train file
users = train_data['users']
print(f"\nUsers in train file: {len(users)}; showing first 3: {users[:3]}")
user = users[0]
print(f"Using user: {user}")

# Load training data for this user
if user not in train_data['user_data']:
    raise SystemExit(f"User {user} not found in training data")
train_x = train_data['user_data'][user]['x']
train_y = train_data['user_data'][user]['y']

# Load test data for this user
if user not in test_data['user_data']:
    raise SystemExit(f"User {user} not found in test data")
test_x = test_data['user_data'][user]['x']
test_y = test_data['user_data'][user]['y']

# Convert to numpy arrays
x_train = np.asarray(train_x, dtype=np.float32)
y_train = np.asarray(train_y, dtype=np.int32)
x_test = np.asarray(test_x, dtype=np.float32)
y_test = np.asarray(test_y, dtype=np.int32)

print(f"\nLoaded user data shapes:")
print(f"  Train: x={x_train.shape}, y={y_train.shape}, x.dtype={x_train.dtype}, y.dtype={y_train.dtype}")
print(f"  Test:  x={x_test.shape}, y={y_test.shape}, x.dtype={x_test.dtype}, y.dtype={y_test.dtype}")

if x_train.ndim == 1:
    x_train = np.expand_dims(x_train, 0)
if x_test.ndim == 1:
    x_test = np.expand_dims(x_test, 0)

# Build model
model = build_femnist_keras(num_classes=62)

# Sanity train_on_batch to check shapes
batch = min(32, len(x_train))
print(f"Training 10 epochs, batch_size={batch}")
model.fit(x_train, y_train, epochs=10, batch_size=batch, verbose=1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Eval -> loss={loss:.6f}, accuracy={acc:.6f}")

# Show sample predictions
print("\n--- Sample Predictions (first 10 test samples) ---")
predictions = model.predict(x_test[:10], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Character mapping helper
def class_to_char(class_id):
    """Convert class ID (0-61) to character"""
    if class_id < 10:
        return str(class_id)  # digits 0-9
    elif class_id < 36:
        return chr(ord('A') + (class_id - 10))  # uppercase A-Z
    else:
        return chr(ord('a') + (class_id - 36))  # lowercase a-z

print(f"{'Sample':<8} {'True Label':<12} {'True Char':<12} {'Predicted':<12} {'Pred Char':<12} {'Confidence':<12} {'Correct?'}")
print("-" * 85)
for i in range(min(10, len(x_test))):
    true_label = y_test[i]
    pred_label = predicted_classes[i]
    confidence = predictions[i][pred_label] * 100
    correct = "✓" if true_label == pred_label else "✗"
    
    print(f"{i:<8} {true_label:<12} {class_to_char(true_label):<12} {pred_label:<12} {class_to_char(pred_label):<12} {confidence:>6.2f}%{'':<6} {correct}")

correct_count = np.sum(predicted_classes[:10] == y_test[:10])
print(f"\nAccuracy on these 10 samples: {correct_count}/10 = {correct_count*10}%")
