import json
import math
import os
import time

import tensorflow as tf

from src.training.losses import (
    combined_loss,
    mae_metric_masked,
    rmse_metric_masked,
    sam_acos_loss_masked,
)

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_progress = step / tf.maximum(warmup_steps, 1.0)
        warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * warmup_progress

        decay_steps = total_steps - warmup_steps
        cosine_progress = tf.minimum((step - warmup_steps) / tf.maximum(decay_steps, 1.0), 1.0)
        decay_lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + tf.cos(math.pi * cosine_progress))

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
        }

class ModelTrainer:
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4,
                 epochs=30, warmup_epochs=2, steps_per_epoch=2000,
                 clip_norm=1.0, lambda_l1=1.0, lambda_sam=0.5,
                 lambda_freq=0.0, lambda_wavelet=0.0,
                 band_weights=None, sam_mode='cosine',
                 checkpoint_dir='checkpoints', drive_path=None,
                 early_stopping_patience=None):
        self.model = model
        self.epochs = epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.checkpoint_dir = checkpoint_dir
        self.drive_path = drive_path
        self.clip_norm = clip_norm
        self.early_stopping_patience = early_stopping_patience

        self.lambda_l1 = lambda_l1
        self.lambda_sam = lambda_sam
        self.lambda_freq = lambda_freq
        self.lambda_wavelet = lambda_wavelet
        self.band_weights = band_weights
        self.sam_mode = sam_mode

        os.makedirs(checkpoint_dir, exist_ok=True)

        total_steps = self.steps_per_epoch * epochs
        warmup_steps = self.steps_per_epoch * warmup_epochs
        self.lr_schedule = WarmupCosineDecay(
            base_lr=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=weight_decay,
            clipnorm=clip_norm,
            epsilon=1e-7,
        )

        self.train_loss = tf.keras.metrics.Mean()
        self.val_l1 = tf.keras.metrics.Mean()
        self.val_rmse = tf.keras.metrics.Mean()
        self.val_sam = tf.keras.metrics.Mean()

        self.best_val = float('inf')
        self.epochs_no_improve = 0
        self.start_epoch = 0

    @tf.function
    def train_step(self, rgb, target, mask):
        with tf.GradientTape() as tape:
            pred = self.model(rgb, training=True)
            loss = combined_loss(
                target, pred, mask,
                lambda_l1=self.lambda_l1,
                lambda_sam=self.lambda_sam,
                lambda_freq=self.lambda_freq,
                lambda_wavelet=self.lambda_wavelet,
                band_weights=self.band_weights,
                sam_mode=self.sam_mode,
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [g if g is not None else tf.zeros_like(v)
                 for g, v in zip(grads, self.model.trainable_variables)]
        grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss.update_state(loss)
        return loss

    @tf.function
    def val_step(self, rgb, target, mask):
        pred = self.model(rgb, training=False)
        self.val_l1.update_state(mae_metric_masked(target, pred, mask))
        self.val_rmse.update_state(rmse_metric_masked(target, pred, mask))
        self.val_sam.update_state(sam_acos_loss_masked(target, pred, mask))

    def _best_path(self):
        return os.path.join(self.checkpoint_dir, 'best.weights.h5')

    def _state_path(self):
        return os.path.join(self.checkpoint_dir, 'trainer_state.json')

    def save_best(self):
        self.model.save_weights(self._best_path())
        with open(self._state_path(), 'w') as f:
            json.dump({'best_val': float(self.best_val),
                       'epochs_no_improve': int(self.epochs_no_improve)}, f)
        if self.drive_path:
            self._mirror_to_drive()

    def _mirror_to_drive(self):
        try:
            os.makedirs(self.drive_path, exist_ok=True)
            for fname in os.listdir(self.checkpoint_dir):
                src = os.path.join(self.checkpoint_dir, fname)
                dst = os.path.join(self.drive_path, fname)
                if os.path.isfile(src):
                    tf.io.gfile.copy(src, dst, overwrite=True)
        except Exception as exc:
            print(f"Warning: failed to mirror to {self.drive_path}: {exc}")

    def restore(self, weights_path=None):
        path = weights_path or self._best_path()
        if not os.path.exists(path):
            print(f"No checkpoint at {path}, starting from scratch.")
            return
        self.model.load_weights(path)
        print(f"Restored weights from {path}")
        if os.path.exists(self._state_path()):
            with open(self._state_path()) as f:
                state = json.load(f)
            self.best_val = float(state.get('best_val', float('inf')))
            self.epochs_no_improve = int(state.get('epochs_no_improve', 0))

    def fit(self, train_ds, val_ds, log_every=50):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_loss.reset_state()
            self.val_l1.reset_state()
            self.val_rmse.reset_state()
            self.val_sam.reset_state()

            t0 = time.time()
            for step, (rgb, target, mask) in enumerate(train_ds):
                loss = self.train_step(rgb, target, mask)
                if step % log_every == 0:
                    lr = float(self.lr_schedule(self.optimizer.iterations))
                    print(f"  epoch {epoch + 1} step {step:5d} | loss {float(loss):.4f} | lr {lr:.2e}")

            for rgb, target, mask in val_ds:
                self.val_step(rgb, target, mask)

            train_l = float(self.train_loss.result())
            val_l = float(self.val_l1.result())
            val_r = float(self.val_rmse.result())
            val_s = float(self.val_sam.result())
            elapsed = time.time() - t0

            print(f"Epoch {epoch + 1}/{self.epochs} | train {train_l:.4f} | "
                  f"val L1 {val_l:.4f} | val RMSE {val_r:.4f} | val SAM {val_s:.4f} | "
                  f"{elapsed:.1f}s")

            if val_l < self.best_val:
                self.best_val = val_l
                self.epochs_no_improve = 0
                self.save_best()
                print(f"  Best val L1 so far: {self.best_val:.4f} (saved)")
            else:
                self.epochs_no_improve += 1
                if (self.early_stopping_patience is not None
                        and self.epochs_no_improve >= self.early_stopping_patience):
                    print(f"Early stopping: no improvement for {self.epochs_no_improve} epochs.")
                    break

        return self.best_val
