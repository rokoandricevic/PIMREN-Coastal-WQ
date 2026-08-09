import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = '/Users/rokoandricevic/Dropbox/Python_scripts/CNN/PIMREN_revised/PIMREN_Chl'
OUT_DIR = os.path.join(BASE_DIR, 'Master_Training_Results')
os.makedirs(OUT_DIR, exist_ok=True)

NUM_MEMBERS = 5  
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TOTAL_REALIZATIONS = 1000

PINN_CONFIG = {
    0.6: {"name": "June",     "a": 0.07,  "b": 3.0,  "lambda": 0.1},
    1.1: {"name": "November", "a": 2.7,   "b": -1.3, "lambda": 0.1},
    0.3: {"name": "March",    "a": 0.17,  "b": 1.1,  "lambda": 0.1},
    0.4: {"name": "April",    "a": 0.063, "b": 2.5,  "lambda": 0.1}
}

SEASONS = [('June', 0.6), ('November', 1.1), ('March', 0.3), ('April', 0.4)]

# ==========================================
# 2. U-NET ARCHITECTURE (11-Channel)
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
# 3. TRAINING PIPELINE & DIAGNOSTICS
# ==========================================
def execute_training():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 STAGE 1: Training Engine Initialized on {device}...")
    
    X_raw = np.fromfile(os.path.join(BASE_DIR, "X_PIMREN_Master.bin"), dtype=np.float32).reshape(-1, 20, 58, 11)
    Y_raw = np.fromfile(os.path.join(BASE_DIR, "Y_PIMREN_Master.bin"), dtype=np.float32).reshape(-1, 20, 58, 1)
    
    Y_MEAN, Y_STD = np.mean(Y_raw), np.std(Y_raw) + 1e-8
    Y_train = (Y_raw - Y_MEAN) / Y_STD
    np.save(os.path.join(OUT_DIR, "GLOBAL_Y_MEAN.npy"), Y_MEAN)
    np.save(os.path.join(OUT_DIR, "GLOBAL_Y_STD.npy"), Y_STD)
    
    Y_phys_target = np.zeros_like(Y_train)
    Y_phys_weight = np.zeros((len(X_raw), 1, 1, 1), dtype=np.float32)
    
    global_idx = 0
    for season_name, flag in SEASONS:
        cfg = PINN_CONFIG[flag]
        end_idx = global_idx + TOTAL_REALIZATIONS
        x_block = X_raw[global_idx:end_idx]
        
        # R1 is Channel 6 in the 11-channel setup
        r1_raw = x_block[..., 6]
        y_p_raw = cfg['a'] * np.exp(cfg['b'] * r1_raw)
        Y_phys_target[global_idx:end_idx, :, :, 0] = (y_p_raw - Y_MEAN) / Y_STD
        Y_phys_weight[global_idx:end_idx] = cfg['lambda']
        
        s_mean = x_block[..., :10].mean(axis=(0,1,2), keepdims=True)
        s_std = x_block[..., :10].std(axis=(0,1,2), keepdims=True) + 1e-8
        
        np.save(os.path.join(OUT_DIR, f"X_mean_{season_name}.npy"), s_mean)
        np.save(os.path.join(OUT_DIR, f"X_std_{season_name}.npy"), s_std)
        
        # 🌟 APISTEN LOGIC: Clip Z-Scores to prevent training collapse
        X_raw[global_idx:end_idx, ..., :10] = np.clip((x_block[..., :10] - s_mean) / s_std, -5.0, 5.0)
        global_idx = end_idx
        
    X_t = torch.tensor(X_raw).permute(0, 3, 1, 2)
    Y_t = torch.tensor(Y_train).permute(0, 3, 1, 2)
    YP_t = torch.tensor(Y_phys_target).permute(0, 3, 1, 2)
    YW_t = torch.tensor(Y_phys_weight)
    
    ensemble_metrics = []
    history_data_loss, history_phys_loss = [], []

    for m in range(NUM_MEMBERS):
        print(f"\n -> Training Member {m+1}/{NUM_MEMBERS}...")
        model = UNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loader = DataLoader(TensorDataset(X_t, Y_t, YP_t, YW_t), batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_d, epoch_p, batches = 0.0, 0.0, 0
            for xb, yb, yp, yw in loader:
                xb, yb, yp, yw = xb.to(device), yb.to(device), yp.to(device), yw.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                
                d_loss = nn.MSELoss()(pred, yb)
                p_loss = (nn.MSELoss(reduction='none')(pred, yp) * yw).mean()
                loss = d_loss + p_loss
                
                loss.backward(); optimizer.step()
                
                epoch_d += d_loss.item()
                epoch_p += p_loss.item()
                batches += 1
                
            if m == 0:
                history_data_loss.append(epoch_d / batches)
                history_phys_loss.append(epoch_p / batches)
                
        print(f"    Member {m+1} Final | Data MSE: {epoch_d/batches:.5f} | Phys MSE: {epoch_p/batches:.5f}")
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"pimren_member_{m}.pth"))
        
        model.eval()
        with torch.no_grad():
            preds = model(X_t.to(device))
            ensemble_metrics.append([m, nn.MSELoss()(preds, Y_t.to(device)).item(), nn.MSELoss()(preds, YP_t.to(device)).item()])
            
    np.savetxt(os.path.join(OUT_DIR, "ensemble_metrics.csv"), np.array(ensemble_metrics), delimiter=",", header="member,data_mse,phys_mse", comments='')
    print(" -> Exported ensemble_metrics.csv")

    # ==========================================
    # 4. DIAGNOSTIC PLOTTING
    # ==========================================
    print("\nGenerating Diagnostic Training Plots...")
    
    # Plot 1: Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), history_data_loss, label='Data MSE', color='blue', linewidth=2)
    plt.plot(range(1, EPOCHS+1), history_phys_loss, label='Physics Penalty', color='red', linewidth=2)
    plt.title('PIMREN Training Convergence (Member 1)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUT_DIR, "Training_Loss_Curve.png"), dpi=300)
    plt.close()
    
    # Plot 2: Physics Compliance Scatter
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PIMREN Physical Compliance (B2/B3 Ratio)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        final_preds = (model(X_t.to(device)).squeeze().cpu().numpy() * Y_STD) + Y_MEAN
        
    global_idx = 0
    # Re-load unnormalized X for the scatter plot x-axis
    raw_x_unnorm = np.fromfile(os.path.join(BASE_DIR, "X_PIMREN_Master.bin"), dtype=np.float32).reshape(-1, 20, 58, 11)
    
    for i, (season_name, flag) in enumerate(SEASONS):
        ax = axes[i]
        cfg = PINN_CONFIG[flag]
        end_idx = global_idx + TOTAL_REALIZATIONS
        
        s_preds = final_preds[global_idx:end_idx].flatten()
        s_r1_raw = raw_x_unnorm[global_idx:end_idx, ..., 6].flatten() # Channel 6 is B2/B3
        
        sub_idx = np.random.choice(len(s_preds), min(2000, len(s_preds)), replace=False)
        ax.scatter(s_r1_raw[sub_idx], s_preds[sub_idx], s=2, alpha=0.3, color='royalblue', label='U-Net Predictions')
        
        r1_sorted = np.sort(s_r1_raw[sub_idx])
        ax.plot(r1_sorted, cfg['a'] * np.exp(cfg['b'] * r1_sorted), 'k--', linewidth=2, label=f"Target: {cfg['a']}*exp({cfg['b']}*R1)")
        
        ax.set_title(f"{season_name} Compliance")
        ax.set_xlabel('Optical Ratio (B2/B3)')
        ax.set_ylabel('Chlorophyll-a [mg/m³]')
        ax.set_ylim(0, 3.0) 
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)
        global_idx = end_idx

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Physics_Compliance_Scatter.png"), dpi=300)
    plt.close()   

    print("\n✅ STAGE 1 COMPLETE: Models trained, metrics logged, and diagnostic plots exported.")

if __name__ == "__main__":
    execute_training()