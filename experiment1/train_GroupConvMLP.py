from comet_ml import start
import torch
import pandas as pd
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from core.models import GroupConvMLP

class PSIDataset(Dataset):
    """
    Expect columns:
      R11..R13, R21..R23, R31..R33, x1,x2,x3, F1,F2,F3, M1,M2,M3
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)  # .cvs/.csv 都能读
        self.R = df[[f"R{i}{j}" for i in range(1,4) for j in range(1,4)]].to_numpy().reshape(-1,3,3)
        self.x = df[["x1","x2","x3"]].to_numpy().reshape(-1,3)
        self.y = df[["F1","F2","F3","M1","M2","M3"]].to_numpy().reshape(-1,6)

    def __len__(self): return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).double()        # (3,)
        R = torch.from_numpy(self.R[idx]).double()        # (3,3)
        y = torch.from_numpy(self.y[idx]).double()        # (6,)
        return (x, R), y


torch.set_default_dtype(torch.float64)

experiment = start(
  api_key="R1QhUQFHMHW1IiYrde55Fu9eA",
  project_name="tefnn-benzene-august12",
  workspace="jerry2"
)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {DEVICE}")
BATCH_SIZE    = 128
LEARNING_RATE = 1e-4
EPOCHS        = 3000
LOSS_FN       ="SmoothL1Loss"

HIDDEN_DIM=108
NUM_HIDDEN_LAYERS=3


class Sigmoid2Minus1(nn.Module):
    def forward(self, x):
        return 2 * torch.sigmoid(x) - 1

class Sigmoid3(nn.Module):
    def forward(self, x):
        return 3 * torch.sigmoid(x) - 3/2

MODELNAME     =f"GConvMLP{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}"
model  =  GroupConvMLP(hidden_dim=HIDDEN_DIM,num_hidden_layers=NUM_HIDDEN_LAYERS,act=nn.Sigmoid())

num_params = sum(p.numel() for p in model.parameters())
print(f"#params{num_params}")

# PRETRAINED_PATH = "StandardMLP_128_7.pth"
# model.load_state_dict(
#     torch.load(PRETRAINED_PATH, map_location=DEVICE),
#      strict=False
# )

experiment.log_parameters({
    "batch_size":    BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs":        EPOCHS,
    "loss_function": LOSS_FN,
    "device":        DEVICE,
    "num_parameters":num_params,
    "hidden_dim":HIDDEN_DIM,
    "num_hidden_layers":NUM_HIDDEN_LAYERS
})
experiment.set_model_graph(str(model))


dataset     = PSIDataset('data/train_2Benzene_10000_6.0_10.0_4.0_gamma1.cvs')
train_n     = int(0.8 * len(dataset))
val_n       = len(dataset) - train_n

from torch.utils.data import Subset
train_indices = list(range(train_n))
val_indices   = list(range(train_n, train_n + val_n))

train_ds = Subset(dataset, train_indices)
val_ds   = Subset(dataset, val_indices)


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
    final_div_factor=1e4
)


global_step = 0

for epoch in range(1, EPOCHS + 1):
    t0=time.time()

    model.train()
    total_train_loss = 0.0
    for (x_b, R_b), y_b in train_loader:
        x_b, R_b, y_b = x_b.to(DEVICE), R_b.to(DEVICE), y_b.to(DEVICE)  # x:(B,3) R:(B,3,3) y:(B,6)
        optimizer.zero_grad()

        y_hat = model(x_b, R_b)  # (B, 2, 3)
        y_hat = y_hat.reshape(y_hat.size(0), -1)  # -> (B, 6)  [F(3) + M(3)]
        loss = criterion(y_hat, y_b)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item() * x_b.size(0)

    avg_train_loss = total_train_loss / train_n

    global_step += 1
    current_lr = optimizer.param_groups[0]['lr']
    experiment.log_metric("learning_rate", current_lr, step=global_step)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for (x_b, R_b), y_b in val_loader:
            x_b, R_b, y_b = x_b.to(DEVICE), R_b.to(DEVICE), y_b.to(DEVICE)  # x:(B,3) R:(B,3,3) y:(B,6)
            y_hat = model(x_b, R_b)  # (B, 2, 3)
            y_hat = y_hat.reshape(y_hat.size(0), -1)  # -> (B, 6)  [F(3) + M(3)]
            loss = criterion(y_hat, y_b)
            total_val_loss += loss.item() * x_b.size(0)

    avg_val_loss = total_val_loss / val_n

    epoch_time = time.time() - t0

    experiment.log_metric("epoch_time", epoch_time, step=epoch)
    experiment.log_metric("train_loss", avg_train_loss, epoch)
    experiment.log_metric("val_loss",   avg_val_loss,   epoch)

    print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}")

    if epoch % 200 == 0:
        ckpt_path = f"temp/{MODELNAME}_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        experiment.log_model(f"{MODELNAME}_{epoch}", ckpt_path)


MODEL_PATH = f"{MODELNAME}.pth"
torch.save(model.state_dict(), MODEL_PATH)
experiment.log_model(MODELNAME, MODEL_PATH)