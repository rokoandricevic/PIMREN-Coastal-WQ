import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import contextily as cx
import seaborn as sns
import warnings

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = '/Users/rokoandricevic/Dropbox/Python_scripts/CNN/PIMREN_revised/PIMREN_Chl'
RESULTS_DIR = os.path.join(BASE_DIR, "Master_Inference_Maps") 
MAPS_DIR = os.path.join(BASE_DIR, "Publication_Maps_Chl")
os.makedirs(MAPS_DIR, exist_ok=True)

EXTENT_FULL_BAY = [16.2507, 16.4750, 43.5130, 43.5524]
aspect_ratio = 1.0 / np.cos(np.radians(np.mean(EXTENT_FULL_BAY[2:])))

seasons = [
    ('June', 'June 2021', '2021-06-26'), 
    ('November', 'November 2021', '2021-10-29'), 
    ('March', 'March 2022', '2022-03-03'), 
    ('April', 'April 2022', '2022-05-12')
]

print(f"\n🚀 STAGE 3: Mapping Engine Initialized...")

# ==========================================
# 2. 🌟 APISTEN LOGIC: LATE-STAGE POLYGON MASKING
# ==========================================
poly_csv_path = os.path.join(BASE_DIR, 'KastelaBayPolygonLarge.csv')
if not os.path.exists(poly_csv_path):
    poly_csv_path = os.path.join(os.path.dirname(BASE_DIR), 'KastelaBayPolygonLarge.csv')

poly_df = pd.read_csv(poly_csv_path, comment='#')
lon_col = [c for c in poly_df.columns if 16.0 <= pd.to_numeric(poly_df[c], errors='coerce').min() <= 17.0][0]
lat_col = [c for c in poly_df.columns if 43.0 <= pd.to_numeric(poly_df[c], errors='coerce').min() <= 44.0][0]
bay_path = Path(poly_df[[lon_col, lat_col]].dropna().values)

lon_grid = np.load(os.path.join(RESULTS_DIR, "Grid_Longitude_FullBay.npy"))
lat_grid = np.load(os.path.join(RESULTS_DIR, "Grid_Latitude_FullBay.npy"))
mask_2d = bay_path.contains_points(np.vstack((lon_grid.flatten(), lat_grid.flatten())).T).reshape(lon_grid.shape)

def plot_satellite_map(data, ax, title, cmap='turbo', vmax=3.0, cbar_label="Chl-a (mg/m³)"):
    masked_data = np.where(mask_2d, data, np.nan)
    im = ax.imshow(masked_data, extent=EXTENT_FULL_BAY, origin='upper', cmap=cmap, vmin=0.0, vmax=vmax, aspect=aspect_ratio, zorder=2)
    cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, attribution=False, zorder=1)
    
    ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
    pad_x, pad_y = (EXTENT_FULL_BAY[1]-EXTENT_FULL_BAY[0])*0.15, (EXTENT_FULL_BAY[3]-EXTENT_FULL_BAY[2])*0.15
    ax.set_xlim(EXTENT_FULL_BAY[0]-pad_x, EXTENT_FULL_BAY[1]+pad_x)
    ax.set_ylim(EXTENT_FULL_BAY[2]-pad_y, EXTENT_FULL_BAY[3]+pad_y)
    ax.set_xticks([]); ax.set_yticks([])
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

# ==========================================
# 3. GENERATE MAPS & VIOLINS
# ==========================================
print("\nGenerating Individual Maps and Extracting Data Distributions...")
violin_plot_data = []

# Prepare the 2x2 Master Panel
fig_panel, axes = plt.subplots(2, 2, figsize=(18, 10))
fig_panel.suptitle("Seasonal Mean Chlorophyll Concentration", fontsize=20, fontweight='bold', y=0.98)
axes_flat = axes.flatten()

for i, (file_season, label_season, date_str) in enumerate(seasons):
    mean_img = np.load(os.path.join(RESULTS_DIR, f"Predicted_Mean_Chl_{file_season}.npy"))
    std_img = np.load(os.path.join(RESULTS_DIR, f"Predicted_STD_Chl_{file_season}.npy"))
    
    # Safely extract masked values, ignoring any structural NaNs
    valid_mean = mean_img[mask_2d]
    valid_mean = valid_mean[~np.isnan(valid_mean)]
    
    valid_std = std_img[mask_2d]
    valid_std = valid_std[~np.isnan(valid_std)]
    
    for val in valid_mean:
        violin_plot_data.append({'Date': date_str, 'Metric': 'Mean Chl-a', 'Value': val})
    for val in valid_std:
        violin_plot_data.append({'Date': date_str, 'Metric': 'σ Chl-a', 'Value': val})

    # 1. Generate standalone Mean map
    fig_mean, ax_mean = plt.subplots(figsize=(10, 6))
    plot_satellite_map(mean_img, ax_mean, f"Mean Chl map {label_season}, Kaštela Bay, Croatia", vmax=3.0)
    plt.tight_layout()
    plt.savefig(os.path.join(MAPS_DIR, f"Map_Chl_{file_season}_Mean.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_mean)
    
    # 2. Generate standalone STD map (plasma colormap, tighter scale)
    fig_std, ax_std = plt.subplots(figsize=(10, 6))
    plot_satellite_map(std_img, ax_std, f"STD Chl map {label_season}, Kaštela Bay, Croatia", cmap='plasma', vmax=0.8, cbar_label="Uncertainty (σ)")
    plt.tight_layout()
    plt.savefig(os.path.join(MAPS_DIR, f"Map_Chl_{file_season}_STD.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_std)

    # 3. Add to the 2x2 Master Panel
    plot_satellite_map(mean_img, axes_flat[i], f"Mean Chl map {label_season}, Kaštela Bay, Croatia", vmax=3.0)

# Save the stitched 2x2 Panel
plt.figure(fig_panel.number)
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.15) 
plt.savefig(os.path.join(MAPS_DIR, "Chl_Seasonal_Panel_Final_Sat.png"), dpi=400, bbox_inches='tight')
plt.close(fig_panel)

# 4. Violin Plot Generation
print("Generating Seasonal Distribution Violin Plot...")
df_dist = pd.DataFrame(violin_plot_data)
fig_violin, ax_violin = plt.subplots(figsize=(13, 7))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sns.violinplot(data=df_dist, x='Date', y='Value', hue='Metric', 
                   palette={'Mean Chl-a': 'seagreen', 'σ Chl-a': 'mediumpurple'}, 
                   split=False, inner='box', cut=0, ax=ax_violin)

ax_violin.set_ylim(-0.05, df_dist['Value'].max() * 1.1)
ax_violin.set_title("Kaštela Bay Seasonal Chlorophyll Domain Distributions", fontsize=16, fontweight='bold', pad=15)
ax_violin.set_xlabel("Satellite Acquisition Chronology", fontsize=14, fontweight='bold', labelpad=10)
ax_violin.set_ylabel("Concentration Magnitude (mg/m³)", fontsize=14, fontweight='bold', labelpad=10)
ax_violin.grid(axis='y', linestyle='--', alpha=0.5)

plt.xticks(rotation=30, fontweight='semibold')
plt.tight_layout()
plt.savefig(os.path.join(MAPS_DIR, "Chl_Seasonal_Domain_Violin_Distributions.png"), dpi=300, bbox_inches='tight')
plt.close(fig_violin)

print("\n✅ STAGE 3 COMPLETE: Final individual maps, publication panel, and violin plots generated successfully.")