"""TensorFlow/Keras version of the downscaling CNN (GPU training).

Same architecture and aggregation loss as models.FCNModel. Weights export to
the same JSON format, so a Keras-trained model can be served by the NumPy
backend and the TensorFlow.js front end.

    from hyperlocal_aq.models_keras import train_keras
    model_dict = train_keras(feats_z, coarse_days, factor=100, steps=3000)

Requires `pip install tensorflow` (optional dependency).
"""
from __future__ import annotations

import numpy as np

from .core import tile_windows
from .models import FCNModel


def build(in_ch=6):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(None, None, in_ch))
    x = inp
    for k, d, co, act in FCNModel.ARCH:
        x = tf.keras.layers.Conv2D(co, k, dilation_rate=d, padding="same",
                                   activation=None if act == "linear" else act)(x)
    return tf.keras.Model(inp, x)


def train_keras(feats_z, coarse_days, factor=100, steps=3000, window_cells=4, batch=8, lr=2e-3, seed=0):
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(coarse_days)
    y_mean, y_std = float(np.nanmean(coarse_days)), float(np.nanstd(coarse_days) + 1e-12)
    x_all = feats_z.transpose(1, 2, 0)
    model = build(feats_z.shape[0])
    pool = tf.keras.layers.AveragePooling2D(pool_size=factor)
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, steps, alpha=0.1)
    opt = tf.keras.optimizers.Adam(sched)
    ws = window_cells * factor

    @tf.function
    def step(xb, yb, mb):
        with tf.GradientTape() as tape:
            pooled = pool(model(xb, training=True))[..., 0]
            err = (pooled - yb) * mb
            loss = tf.reduce_sum(err ** 2) / tf.maximum(tf.reduce_sum(mb), 1.0)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    for it in range(steps):
        xb, yb, mb = [], [], []
        for _ in range(batch):
            d = int(rng.integers(0, coarse_days.shape[0]))
            (cy, cx), = tile_windows(valid[d], window_cells, rng, 1)
            xb.append(x_all[cy * factor:cy * factor + ws, cx * factor:cx * factor + ws])
            t = coarse_days[d, cy:cy + window_cells, cx:cx + window_cells]
            mb.append((~np.isnan(t)).astype(np.float32))
            yb.append(np.nan_to_num((t - y_mean) / y_std).astype(np.float32))
        loss = step(np.stack(xb), np.stack(yb), np.stack(mb))
        if it % 200 == 0:
            print(f"step {it} loss {float(loss):.4f}")

    # export in the FCNModel JSON layout: W reshaped (k*k*cin, cout), tap-major (i, j, c)
    layers = []
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    for (k, d, co, act), conv in zip(FCNModel.ARCH, convs):
        W, b = conv.get_weights()            # (k, k, cin, cout)
        layers.append({"k": k, "d": d, "act": act,
                       "W": W.reshape(-1, co).tolist(), "b": b.tolist()})
    return {"type": "fcn", "y_mean": y_mean, "y_std": y_std, "layers": layers}
