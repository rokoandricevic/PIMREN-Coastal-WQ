import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import griddata

# ==========================================
# 1. EXPLICIT DIRECTORY CONFIGURATION
# ==========================================
# Root directory containing the Sentinel data folders
PROJECT_ROOT = '/Users/rokoandricevic/Dropbox/Python_scripts/CNN/PIMREN_revised'

# Sub-directory for Chlorophyll-specific outputs
CHL_DIR = os.path.join(PROJECT_ROOT, 'PIMREN_Chl')
TRAIN_DIR = os.path.join(CHL_DIR, 'Master_Training_Results')
OUT_DIR = os.path.join(CHL_DIR, 'Master_Inference_Maps')
os.makedirs(OUT_DIR, exist_ok=True)

EXTENT_FULL_BAY = [16.2507, 16.4750, 43.5130, 43.5524]
N_LON, N_LAT = 904, 219 
NUM_MEMBERS = 5

SEASONS = [
    ('June', 0.6, 'S2A_MSIL2A_20210626T095031_N0500_R079_T33TXJ_20230318T192106'),
    ('November', 1.1, 'S2B_MSIL2A_20211029T095029_N0500_R079_T33TXJ_20230114T070712'),
    ('March', 0.3, 'S2A_MSIL2A_20220303T095031_N0510_R079_T33TXJ_20240521T063733'),
    ('April', 0.4, 'S2A_MSIL2A_20220512T095031_N0510_R079_T33TXJ_20240618T004118')
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. U-NET ARCHITECTURE
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
# 3. INTERPOLATION (CONTINUOUS FIELDS)
# ==========================================
def extract_band(folder, band):
    # Explicitly search the PROJECT_ROOT for Sentinel data
    search_path_1 = os.path.join(PROJECT_ROOT, "Sentinel2_Data", folder, f"*_{band}_20m.csv")
    files = glob.glob(search_path_1)
    if not files: 
        search_path_2 = os.path.join(PROJECT_ROOT, folder, f"*_{band}_20m.csv")
        files = glob.glob(search_path_2)
        if not files: raise FileNotFoundError(f"Missing {band} in {folder}")
    return files[0]

def load_and_interpolate(filepath, target_shape, extent):
    df = pd.read_csv(filepath, comment='#')
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    lon_col = [c for c in df.columns if 16.0 <= df[c].min() <= 17.0][0]
    lat_col = [c for c in df.columns if 43.0 <= df[c].min() <= 44.0][0]
    val_col = [c for c in df.columns if c not in (lon_col, lat_col)][0]
            
    lon_arr = np.linspace(extent[0], extent[1], target_shape[0])
    lat_arr = np.linspace(extent[3], extent[2], target_shape[1])
    mesh_lon, mesh_lat = np.meshgrid(lon_arr, lat_arr)
    
    pts, vals = df[[lon_col, lat_col]].values, df[val_col].values
    
    # 🌟 APISTEN LOGIC: Linear interpolation with nearest-neighbor fallback for continuous arrays
    grid_data = griddata(pts, vals, (mesh_lon, mesh_lat), method='linear')
    nan_mask = np.isnan(grid_data)
    if np.any(nan_mask):
        grid_data[nan_mask] = griddata(pts, vals, (mesh_lon, mesh_lat), method='nearest')[nan_mask]
    return np.clip(grid_data, 1e-4, 1.0).astype(np.float32)

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
def execute_inference():
    print(f"\n🚀 STAGE 2: Inference Engine Initialized on {device}...")
    
    lon_arr = np.linspace(EXTENT_FULL_BAY[0], EXTENT_FULL_BAY[1], N_LON)
    lat_arr = np.linspace(EXTENT_FULL_BAY[3], EXTENT_FULL_BAY[2], N_LAT)
    mesh_lon, mesh_lat = np.meshgrid(lon_arr, lat_arr)
    np.save(os.path.join(OUT_DIR, "Grid_Longitude_FullBay.npy"), mesh_lon)
    np.save(os.path.join(OUT_DIR, "Grid_Latitude_FullBay.npy"), mesh_lat)

    Y_MEAN = np.load(os.path.join(TRAIN_DIR, "GLOBAL_Y_MEAN.npy"))
    Y_STD = np.load(os.path.join(TRAIN_DIR, "GLOBAL_Y_STD.npy"))

    ensemble = []
    for m in range(NUM_MEMBERS):
        model = UNet().to(device)
        model.load_state_dict(torch.load(os.path.join(TRAIN_DIR, f"pimren_member_{m}.pth"), map_location=device, weights_only=True))
        model.eval()
        ensemble.append(model)
    
    for season, flag, folder in SEASONS:
        print(f" -> Processing {season.upper()} Overpass...")
        
        b2 = load_and_interpolate(extract_band(folder, 'B02'), (N_LON, N_LAT), EXTENT_FULL_BAY)
        b3 = load_and_interpolate(extract_band(folder, 'B03'), (N_LON, N_LAT), EXTENT_FULL_BAY)
        b4 = load_and_interpolate(extract_band(folder, 'B04'), (N_LON, N_LAT), EXTENT_FULL_BAY)
        b5 = load_and_interpolate(extract_band(folder, 'B05'), (N_LON, N_LAT), EXTENT_FULL_BAY)
        b6 = load_and_interpolate(extract_band(folder, 'B06'), (N_LON, N_LAT), EXTENT_FULL_BAY)
        b7 = load_and_interpolate(extract_band(folder, 'B07'), (N_LON, N_LAT), EXTENT_FULL_BAY)
        
        r1 = b2 / (b3 + 1e-8)
        r2 = b3 / (b4 + 1e-8)
        r3 = b3 / (b2 + b4 + 1e-8)
        ndci = (b5 - b4) / (b5 + b4 + 1e-8)
        s_flag = np.full((N_LAT, N_LON), flag, dtype=np.float32)
        
        full_tensor = np.stack([b2, b3, b4, b5, b6, b7, r1, r2, r3, ndci, s_flag], axis=-1)
        
        X_MEAN = np.load(os.path.join(TRAIN_DIR, f"X_mean_{season}.npy")).reshape(1, 1, 10)
        X_STD = np.load(os.path.join(TRAIN_DIR, f"X_std_{season}.npy")).reshape(1, 1, 10)
        
        # 🌟 APISTEN LOGIC: Clip the Z-scores to [-5.0, 5.0] to stop coastal optical blowout
        full_tensor[..., :10] = np.clip((full_tensor[..., :10] - X_MEAN) / X_STD, -5.0, 5.0)
        
        in_t = torch.tensor(np.expand_dims(np.transpose(full_tensor, (2, 0, 1)), axis=0), dtype=torch.float32).to(device)
        
        all_preds = []
        with torch.no_grad():
            for model in ensemble:
                pred = (model(in_t).squeeze().cpu().numpy() * Y_STD) + Y_MEAN
                all_preds.append(pred)
                
        np.save(os.path.join(OUT_DIR, f"Predicted_Mean_Chl_{season}.npy"), np.mean(all_preds, axis=0))
        np.save(os.path.join(OUT_DIR, f"Predicted_STD_Chl_{season}.npy"), np.std(all_preds, axis=0))

    print("\n✅ STAGE 2 COMPLETE: Full Bay Inference Matrices Exported.")

if __name__ == "__main__":
    execute_inference()