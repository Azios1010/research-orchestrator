"""Training: the same recipe for every arm."""

from __future__ import annotations

import numpy as np
import torch

from .arms import arm_input, n_channels
from .model import UNet2D, loss_fn


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_arrays(data: dict, arm: str, sids: list[str], perms: dict):
    xs, ys, les = [], [], []
    for sid in sids:
        d, _ = data[sid]
        x = arm_input(arm, d, perms[sid])
        xs.append(x)
        ys.append(d["mask"][:, None])
        les.append(d["mask"].reshape(len(x), -1).any(1))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(les)


def train_one(cfg: dict, data: dict, arm: str, train_sids: list[str], perms: dict, seed: int, dev: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y, L = make_arrays(data, arm, train_sids, perms)
    pos, allidx = np.where(L)[0], np.arange(len(X))
    model = UNet2D(n_channels(arm, cfg["t_window"]), cfg["base_ch"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps_per_epoch = max(1, len(X) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    bs = cfg["batch_size"]
    nb_pos = int(cfg["batch_size"] * cfg["lesion_slice_frac"]) if len(pos) else 0
    model.train()
    for _ in range(cfg["epochs"]):
        for _ in range(steps_per_epoch):
            b = np.concatenate([np.random.choice(pos, nb_pos), np.random.choice(allidx, bs - nb_pos)])
            x = torch.from_numpy(X[b].astype(np.float32)).to(dev)
            y = torch.from_numpy(Y[b].astype(np.float32)).to(dev)
            if np.random.rand() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            with torch.autocast(device_type="cuda", enabled=(dev == "cuda")):
                logit = model(x)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model
