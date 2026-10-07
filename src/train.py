"""Train the MLP on raw pixels and log the run to Weights & Biases.

    python -m src.train --config configs/lab3_mlp.yaml

Given: pixel standardisation, the epoch loop with its logging, and the
checkpoint. Yours, each a few lines, each with its contract in the docstring:

    make_optimizer   TASK 3
    train_epoch      TASK 4   the minibatch loop
    predict          TASK 5   model outputs as numpy
    validate         TASK 6   RMSE on the scoreable validation rows

Read the given parts once. There is no framework underneath: this file is the
whole of supervised training.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src import config, tracking
from src.data.dataset import load_table, pixel_table
from src.data.splits import build_splits
from src.evaluate import rmse
from src.models.mlp import MLP
from src.seed import set_seed


class Standardiser:
    """Zero mean and unit variance over all pixels, fitted on the training rows only.

    One mean and one scale for the whole image, not one per pixel: with a
    thousand training rows a per-pixel estimate is mostly noise. Fitting on
    all rows would leak the validation distribution into training; a small
    leak, but the habit matters. The two numbers go into the checkpoint so
    that prediction uses exactly the same transform.
    """

    def fit(self, X: np.ndarray) -> Standardiser:
        self.mean = float(X.mean())
        self.scale = float(X.std()) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def to_tensor(a, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


# --------------------------------------------------------------- your part

def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """TASK 3. The optimiser, using cfg['train']['lr'] and cfg['train']['weight_decay']."""
    return torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator) -> float:
    """TASK 4. One pass over the training rows in shuffled minibatches.

    Draw the order with `torch.randperm(len(X), generator=generator)` so the
    run is reproducible, step the optimiser once per minibatch, and return the
    mean training loss over the epoch, weighted by minibatch size.
    """
    model.train()
    order = torch.randperm(len(X), generator=generator).to(X.device)
    total = 0.0
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(X)


def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """TASK 5. The model's outputs for every row of X, as a numpy array of shape (len(X),).

    Evaluation mode, no gradients, and it must give the same answer for a row
    whether that row is predicted alone or inside a batch.
    """
    model.eval()
    with torch.no_grad():
        out = [model(X[start:start + batch_size]) for start in range(0, len(X), batch_size)]
    return torch.cat(out).cpu().numpy()


def validate(model: torch.nn.Module, X_val: torch.Tensor, y_val: np.ndarray, ok: np.ndarray) -> float:
    """TASK 6. RMSE on the validation rows where `ok` is True (the scoreable ones)."""
    return rmse(y_val[ok], predict(model, X_val)[ok])


# ------------------------------------------------------------------ given

class EarlyStopping:
    """Remembers the weights of the epoch with the lowest RMSE on a stop set."""

    def __init__(self, X: torch.Tensor, y: np.ndarray, ok: np.ndarray):
        self.X, self.y, self.ok = X, y, ok
        self.best, self.epoch, self.state = float("inf"), 0, None

    def check(self, model: torch.nn.Module, epoch: int) -> float:
        score = validate(model, self.X, self.y, self.ok)
        if score < self.best:
            self.best, self.epoch = score, epoch
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return score

    def restore(self, model: torch.nn.Module) -> int:
        model.load_state_dict(self.state)
        return self.epoch


def set_aside(df: pd.DataFrame, train_idx: np.ndarray, es: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Split train_idx into (fit, stop) by whole units of es['group_column']; no stop rows without `es`."""
    if not es:
        return train_idx, np.array([], dtype=int)
    in_stop = np.isin(df[es["group_column"]].to_numpy()[train_idx], es["units"])
    assert 0 < in_stop.sum() < len(train_idx), "the early-stopping units must split the training rows"
    return train_idx[~in_stop], train_idx[in_stop]


def train_mlp(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              device: str | None = None, tags: list[str] | None = None, log=print) -> dict:
    """Train on the pixel rows train_idx, validate on val_idx after every epoch, log everything.

    `X` is the pixel table, one row per row of `df`. Returns the history, the
    validation predictions after the first and the last epoch, the trained
    model, the checkpoint path and the W&B run page.

    Early stopping, when cfg['train']['early_stopping'] is set: the training
    rows whose `group_column` value is in `units` are set aside as a stop set.
    The model trains on the rest, and the weights of the epoch with the lowest
    RMSE on the stop set are restored at the end. The validation rows never
    take part in that choice, so the returned `val_rmse` is untouched.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    tr = cfg["train"]

    es = tr.get("early_stopping")
    train_idx, stop_idx = set_aside(df, train_idx, es)

    scaler = Standardiser().fit(X[train_idx])
    X_tr = to_tensor(scaler.transform(X[train_idx]), device)
    X_va = to_tensor(scaler.transform(X[val_idx]), device)
    y_tr = to_tensor(df[config.TARGET].to_numpy()[train_idx], device)
    y_va = df[config.TARGET].to_numpy()[val_idx]
    ok = df["eval_ok"].to_numpy()[val_idx] == 1

    model = MLP(in_dim=X.shape[1], hidden=list(cfg["model"]["hidden"]), dropout=cfg["model"]["dropout"],
                prior=float(y_tr.mean())).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = torch.nn.MSELoss()           # the metric is RMSE; same minimiser
    generator = torch.Generator().manual_seed(cfg["seed"])

    stopper = None
    if es:
        stopper = EarlyStopping(to_tensor(scaler.transform(X[stop_idx]), device),
                                df[config.TARGET].to_numpy()[stop_idx], df["eval_ok"].to_numpy()[stop_idx] == 1)

    tracking.init_run(cfg, tags=tags)
    history, snapshots = [], {}
    log(f"{cfg['name']}: {len(train_idx)} train / {len(stop_idx)} stop / {len(val_idx)} val rows, "
        f"{X.shape[1]} inputs, {sum(p.numel() for p in model.parameters()):,} parameters, {device}")
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, optimizer, loss_fn, X_tr, y_tr, tr["batch_size"], generator)
        val = validate(model, X_va, y_va, ok)
        train_eval = rmse(y_tr.cpu().numpy(), predict(model, X_tr))   # dropout off, as in validation
        row = {"epoch": epoch, "train_loss": loss, "train_rmse": train_eval, "val_rmse": val}
        if stopper:
            row["stop_rmse"] = stopper.check(model, epoch)
        tracking.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        history.append(row | {"seconds": time.time() - t0})
        if epoch in (1, tr["epochs"]) or epoch % tr.get("log_every", 10) == 0:
            stop = f"  stop RMSE {row['stop_rmse']:.4f}" if stopper else ""
            log(f"epoch {epoch:3d}  train loss {loss:.4f}  train RMSE {train_eval:.4f}{stop}  val RMSE {val:.4f}")
        if epoch in (1, tr["epochs"]):
            snapshots[epoch] = predict(model, X_va)

    kept_epoch = tr["epochs"]
    if stopper:
        kept_epoch = stopper.restore(model)
        log(f"restored epoch {kept_epoch} (stop RMSE {stopper.best:.4f})")
    val_pred = predict(model, X_va)
    final_val = rmse(y_va[ok], val_pred[ok])

    ckpt = config.ROOT / "models" / f"{cfg['name']}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "in_dim": X.shape[1], "epoch": kept_epoch,
                "scaler_mean": scaler.mean, "scaler_scale": scaler.scale}, ckpt)
    tracking.log_checkpoint(ckpt, cfg["name"], {"val_rmse": final_val, "epoch": kept_epoch})
    tracking.log_metrics({"final_val_rmse": final_val, "kept_epoch": kept_epoch})
    url = tracking.run_url()
    tracking.finish()
    return {"history": pd.DataFrame(history), "val_pred": val_pred, "val_rmse": final_val, "kept_epoch": kept_epoch,
            "snapshots": snapshots, "model": model, "checkpoint": ckpt, "url": url, "val_index": val_idx, "ok": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MLP from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"], cfg["split"]["seed"])
    result = train_mlp(cfg, X, df, train_idx, val_idx)
    print(f"final val RMSE {result['val_rmse']:.4f} (epoch {result['kept_epoch']})  checkpoint {result['checkpoint']}  "
          f"run {result['url']}")


if __name__ == "__main__":
    main()
