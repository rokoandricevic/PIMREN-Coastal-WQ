import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. GLOBAL CONFIGURATION & HYPERPARAMETERS
# ==========================================
NUM_MEMBERS = 5  
EPOCHS = 50
BATCH_SIZE = 32
OUTPUT_DIR = "Master_PIMREN_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Update these lambdas as needed for your Ablation Study (set to 0 for standard U-Net)
PINN_CONFIG = {
    0.6: {"name": "June",     "type": "exp", "a": 0.519, "b": 0.917, "lambda": 0.1},
    0.4: {"name": "April",    "type": "exp", "a": 0.077, "b": 2.145,  "lambda": 0.8},
    1.1: {"name": "November", "type": "lin", "m": -0.3747,   "c": 0.755,   "lambda": 0.1},
    0.3: {"name": "March",    "type": "lin", "m": 1.65,    "c": 0.41,    "lambda": 0.1}
}

# Geographics for Kaštela Bay
NX, NY = 1312, 212
EXTENT = [16.2507, 16.4873, 43.5132, 43.5524]
SPLIT_ASPECT = 1.0 / np.cos(np.radians(43.53))

# ==========================================
# 2. U-NET ARCHITECTURE (11-Channel Input)
# ==========================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(11, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        if u2.shape != e2.shape: u2 = torch.nn.functional.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        if u1.shape != e1.shape: u1 = torch.nn.functional.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.final(d1)

# ==========================================
# 3. DATA PREPARATION
# ==========================================
def prepare_training_data():
    print("Loading and normalizing training data...")
    X_raw = np.fromfile("inputs.bin", dtype=np.float32).reshape(4000, 57, 17, 10)
    Y_raw = np.fromfile("outputs.bin", dtype=np.float32).reshape(4000, 57, 17)
    
    Y_mean, Y_std = Y_raw.mean(), Y_raw.std()
    np.save(f"{OUTPUT_DIR}/Y_stats_train.npy", np.array([Y_mean, Y_std]))
    
    X_11 = np.zeros((4000, 57, 17, 11), dtype=np.float32)
    Y_phys_target = np.zeros_like(Y_raw)
    Y_phys_weight = np.zeros((4000, 1, 1, 1), dtype=np.float32)

    blocks = [(0, 1000, 0.6, "June"), (1000, 2000, 0.4, "April"), 
              (2000, 3000, 1.1, "November"), (3000, 4000, 0.3, "March")]

    for start, end, flag, name in blocks:
        cfg = PINN_CONFIG[flag]
        B2, B3, B4 = X_raw[start:end,:,:,0], X_raw[start:end,:,:,1], X_raw[start:end,:,:,2]
        
        # Seasonal Physics Rules[cite: 1]
        if cfg['type'] == 'exp':
            Y_p_raw = cfg['a'] * np.exp(cfg['b'] * (B2 / (B3 + 1e-8)))
        else:
            Y_p_raw = cfg['m'] * (B3 - B4) + cfg['c']
        
        Y_phys_target[start:end] = (Y_p_raw - Y_mean) / (Y_std + 1e-8)
        Y_phys_weight[start:end] = cfg['lambda']

        # Seasonal Normalization Keys[cite: 1, 2]
        X_block = X_raw[start:end]
        m_x, s_x = X_block.mean(axis=(0,1,2), keepdims=True), X_block.std(axis=(0,1,2), keepdims=True)
        np.save(f"{OUTPUT_DIR}/X_mean_{name}.npy", m_x)
        np.save(f"{OUTPUT_DIR}/X_std_{name}.npy", s_x)

        X_11[start:end, :, :, :10] = (X_block - m_x) / (s_x + 1e-8)
        X_11[start:end, :, :, 10] = flag
        
    return X_11, (Y_raw - Y_mean)/(Y_std + 1e-8), Y_phys_target, Y_phys_weight, Y_mean, Y_std

# ==========================================
# 4. TRAINING & INFERENCE EXECUTION
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X_11, Y_norm, YP_t, YW_t, Y_mean, Y_std = prepare_training_data()

# Tensor Conversion
X_t = torch.tensor(X_11).permute(0, 3, 1, 2)
Y_t = torch.tensor(Y_norm).unsqueeze(1)
YP_t = torch.tensor(YP_t).unsqueeze(1)
YW_t = torch.tensor(YW_t)

# FIX 1: Initialize the metrics list before the loop
ensemble_metrics = []

# Train Ensemble Loop
for m in range(NUM_MEMBERS):
    print(f"\n--- Training Member {m+1}/{NUM_MEMBERS} ---")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_t, Y_t, YP_t, YW_t), batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb, yp, yw in loader:
            xb, yb, yp, yw = xb.to(device), yb.to(device), yp.to(device), yw.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            # PINN Total Loss[cite: 1, 2]
            loss = nn.MSELoss()(pred, yb) + (yw[0,0,0,0] * nn.MSELoss()(pred, yp))
            loss.backward(); optimizer.step()
    
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/pimren_ensemble_member_{m}.pth")

    # FIX 2: Calculate validation metrics for the Ablation Study once per member
    model.eval()
    with torch.no_grad():
        preds = model(X_t.to(device))
        data_mse = nn.MSELoss()(preds, Y_t.to(device)).item()
        phys_mse = nn.MSELoss()(preds, YP_t.to(device)).item()
        
    ensemble_metrics.append([m, data_mse, phys_mse])
    print(f"Member {m} Finished -> Data MSE: {data_mse:.6f}, Physics MSE: {phys_mse:.6f}")

# FIX 3: Save the summary CSV once
np.savetxt(f"{OUTPUT_DIR}/ensemble_metrics_summary.csv", np.array(ensemble_metrics), 
           delimiter=",", header="member,data_mse,phys_mse", comments='')

# ==========================================
# 5. SEASONAL INFERENCE & MAPPING
# ==========================================
lons = np.linspace(EXTENT[0], EXTENT[1], NY)
lats = np.linspace(EXTENT[3], EXTENT[2], NX)
lon_grid, lat_grid = np.meshgrid(lons, lats)

for flag, name in [(0.6, "June"), (0.4, "April"), (1.1, "November"), (0.3, "March")]:
    print(f"\nGenerating Full Bay Maps for {name}...")
    m_x = np.load(f"{OUTPUT_DIR}/X_mean_{name}.npy")
    s_x = np.load(f"{OUTPUT_DIR}/X_std_{name}.npy")
    X_full = np.fromfile(f"full_bay_input_{name}.bin", dtype=np.float32).reshape(1, NX, NY, 10)
    
    X_norm = (X_full - m_x) / (s_x + 1e-8)
    
    # FIX 4: Explicitly use float32 for the flag channel to avoid MPS errors
    flag_sheet = np.ones((1, NX, NY, 1), dtype=np.float32) * flag
    X_in = np.concatenate([X_norm, flag_sheet], axis=-1)
    
    input_tensor = torch.tensor(X_in, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    all_preds = []
    # Ensemble Prediction[cite: 4]
    for m in range(NUM_MEMBERS):
        model.load_state_dict(torch.load(f"{OUTPUT_DIR}/pimren_ensemble_member_{m}.pth", map_location=device))
        model.eval()
        with torch.no_grad():
            p = model(input_tensor).squeeze().cpu().numpy()
            all_preds.append((p * Y_std) + Y_mean) # Un-normalize[cite: 3, 4]
    
    mean_img = np.mean(all_preds, axis=0)
    std_img = np.std(all_preds, axis=0)
    
    # Save CSV Map[cite: 4]
    export = np.stack([lon_grid.flatten(), lat_grid.flatten(), mean_img.flatten(), std_img.flatten()], axis=1)
    np.savetxt(f"{OUTPUT_DIR}/Kastela_Ensemble_{name}.csv", export, delimiter=",", header="lon,lat,mean,std")

    # FIX 5: Set specific VMIN/VMAX to handle land vegetation scaling issues
    plt.figure(figsize=(10, 5))
    plt.imshow(mean_img, extent=EXTENT, origin='upper', cmap='viridis', 
               aspect=SPLIT_ASPECT, vmin=0, vmax=2.5)
    plt.colorbar(label="Chl-a (mg/m³)")
    plt.title(f"PIMREN Ensemble Mean: {name} (Kaštela Bay)")
    plt.savefig(f"{OUTPUT_DIR}/Map_{name}_Mean.png")
    plt.close() # Close to save memory and allow the script to continue[cite: 3, 4]

print(f"\nPipeline complete! Results saved in '{OUTPUT_DIR}' folder.")