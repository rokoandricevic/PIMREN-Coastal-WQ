PIMREN: Prior-Informed Multi-Regime Ensemble U-Net
This repository provides the official implementation of the PIMREN framework as described in:

"Bridging Scale Gaps and Seasonal Dynamics in Coastal Water Quality Monitoring: The Prior-Informed Multi-Regime Ensemble Network"

Project Overview
The PIMREN framework is designed to resolve the data-scarcity bottleneck in coastal water quality monitoring. It bridges the gap between centimeter-scale in situ observations and decameter-scale Sentinel-2 pixels using a stochastic scaling bridge.

Data Availability (Zenodo)
Due to file size constraints, the training and inference tensors are archived on Zenodo:
DOI: 10.5281/zenodo.20184062

The Zenodo archive contains:
- A folder Multi_Realizations_20m containg the 4000 realizations (1000 for each season)
- X_PIMREN_Master.bin (input tensor)
- Y_PIMREN_Master.bin (output tensor)
- 4 seasonal Sentinel-2 images 

Components
1. PIMREN training script in Python (PIMREN_Stage1_Training.py)
The core Python script for multi-seasonal ensemble inversion.
- Regime-Dependent Modulator: An 11th-channel input tensor that resolves seasonal non-stationarity by modulating shared convolutional kernels.
- Prior-Informed Loss: Enforces seasonal prior penalty with $\lambda$ to honor bio-optical relationships on the test polygon.
- Ensemble Uncertainty: A 5-member ensemble provides predictive mean and Coefficient of Variation (CV) maps.
2. PIMREN Inference script for entira Kaštela Bay (PIMREN_Stage2_Inference.py)
3. PIMREN_Stage3_Plotting.py (plotting script for generating final maps)
  
Getting Started
- Clone the repository:
git clone https://github.com/rokoandricevic/PIMREN-Coastal-WQ.git
- Download Data: Download the .bin tensors from the Zenodo DOI and place them in the project root.
- Run the scaling bridge: Open the Mathematica script to generate realizations (ensure measurementFile points to the In_situ samples folder)
- Execute Stage 1. Training, Stage 2. Inference, and Satge 3. plotting script. Results are automatically stored in the Publication_Maps_Chl directory.

Domain Visualization
- Refer to Considered polygons.png for a spatial overview of the training test field, in situ sample locations, and the four seasonal UAV hyperspectral polygons used to establish the centimetric spatial characteristics.
