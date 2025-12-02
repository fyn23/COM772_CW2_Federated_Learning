import numpy as np
import tensorflow as tf
import flwr as fl

def build_femnist_keras(num_classes=62):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28,28,1), input_shape=(28*28,)),
        tf.keras.layers.Conv2D(32, 5, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, epochs: int = 1, user_id: str = None):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        # Default local epochs; can be overridden by server-sent config in fit()
        self.epochs = int(epochs)
        # Optional user id to tag client outputs
        self.user_id = user_id
        # Prepare results directory for client-level outputs
        from pathlib import Path
        self._client_results_dir = Path('results') / 'client'
        self._client_results_dir.mkdir(parents=True, exist_ok=True)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Prefer server-provided config if available (e.g., {'local_epochs': 2})
        epochs = self.epochs
        if config is not None:
            try:
                epochs = int(config.get('local_epochs', config.get('epochs', epochs)))
            except Exception:
                epochs = self.epochs

        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=32, verbose=0)
        # Return training metrics from the last epoch
        last_loss = float(history.history['loss'][-1]) if 'loss' in history.history else None
        last_acc = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None

        # Optionally save client-side train metrics
        try:
            if self.user_id is not None:
                import json
                record = {
                    'phase': 'train',
                    'round': int(config.get('round', -1)) if config is not None else -1,
                    'loss': last_loss,
                    'accuracy': last_acc
                }
                p = self._client_results_dir / f"{self.user_id}.json"
                # Append record to JSON list (create if missing)
                if p.exists():
                    with open(p, 'r') as fh:
                        try:
                            data = json.load(fh)
                        except Exception:
                            data = []
                else:
                    data = []
                data.append(record)
                with open(p, 'w') as fh:
                    json.dump(data, fh)
        except Exception:
            pass

        return self.model.get_weights(), len(self.x_train), {
            "loss": last_loss,
            "accuracy": last_acc
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        # Optionally save client-side evaluate metrics
        try:
            if self.user_id is not None:
                import json
                record = {
                    'phase': 'evaluate',
                    'round': int(config.get('round', -1)) if config is not None else -1,
                    'loss': float(loss),
                    'accuracy': float(acc)
                }
                p = self._client_results_dir / f"{self.user_id}.json"
                if p.exists():
                    with open(p, 'r') as fh:
                        try:
                            data = json.load(fh)
                        except Exception:
                            data = []
                else:
                    data = []
                data.append(record)
                with open(p, 'w') as fh:
                    json.dump(data, fh)
        except Exception:
            pass

        return float(loss), len(self.x_test), {"accuracy": float(acc), "loss": float(loss)}
