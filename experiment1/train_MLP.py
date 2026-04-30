from comet_ml import start
import torch
import pandas as pd
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float64)

from core.models import StandardMLP





experiment = start(
  api_key="R1QhUQFHMHW1IiYrde55Fu9eA",
  project_name="tefnn-benzene-august12",
  workspace="jerry2"
)


DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {DEVICE}")
BATCH_SIZE    = 128
LEARNING_RATE = 1e-4
EPOCHS        = 5000
LOSS_FN       = "SmoothL1Loss"

HIDDEN_DIM         = 256
NUM_HIDDEN_LAYERS  = 4

MODELNAME = f"StandardMLP_{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}"

model = StandardMLP(
    in_dim=12, out_dim=6,
    hidden_dim=HIDDEN_DIM,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    activation_fn=nn.Sigmoid
).to(DEVICE).double()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

experiment.log_parameters({
    "batch_size":    BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs":        EPOCHS,
    "loss_function": LOSS_FN,
    "device":        str(DEVICE),
    "num_parameters":num_params,
    "hidden_dim": HIDDEN_DIM,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
})

dataset = PSIDatasetMLP('data/train_2Benzene_10000_6.0_10.0_4.0_gamma1.cvs')
train_n = int(0.8 * len(dataset))
val_n   = len(dataset) - train_n

from torch.utils.data import Subset
train_ds = Subset(dataset, list(range(train_n)))
val_ds   = Subset(dataset, list(range(train_n, train_n + val_n)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

criterion = getattr(nn, LOSS_FN)()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    div_factor=4.0,
    final_div_factor=1e3
)

global_step = 0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    model.train()
    total_train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(X_b)
        loss = criterion(y_hat, y_b)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item() * X_b.size(0)

    avg_train_loss = total_train_loss / train_n
    global_step += 1
    experiment.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=global_step)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            y_hat = model(X_b)
            loss = criterion(y_hat, y_b)
            total_val_loss += loss.item() * X_b.size(0)
    avg_val_loss = total_val_loss / val_n

    epoch_time = time.time() - t0
    experiment.log_metric("epoch_time", epoch_time, step=epoch)
    experiment.log_metric("train_loss", avg_train_loss, epoch)
    experiment.log_metric("val_loss",   avg_val_loss,   epoch)

    print(f"Epoch {epoch:04d}/{EPOCHS}  train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}")

    if epoch % 200 == 0:
        ckpt_path = f"temp/{MODELNAME}_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        experiment.log_model(f"{MODELNAME}_{epoch}", ckpt_path)

MODEL_PATH = f"{MODELNAME}.pth"
torch.save(model.state_dict(), MODEL_PATH)
experiment.log_model(MODELNAME, MODEL_PATH)
