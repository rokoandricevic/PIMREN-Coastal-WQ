PIMREN: Prior-Informed Multi-Regime Ensemble U-Net
This repository provides the official implementation of the PIMREN framework as described in:

"Bridging Scale Gaps and Seasonal Dynamics in Coastal Water Quality Monitoring: The Prior-Informed Multi-Regime Ensemble Network"

Project Overview
The persistent data-scarcity bottleneck in coastal monitoring is not merely a logistical challenge; it is a fundamental algorithmic limitation. The Prior-Informed Multi-Regime Ensemble U-Net (PIMREN) breaks this limitation. By employing a centimetric-to-decametric physical upscaling technique, the framework generated thousands of equally probable spatial realizations at the Sentinel-2 scale. Crucially, this massive expansion of the target space forced the convolutional architecture to actively stretch and search for microscopic signal gradients within the satellite input tensor. The network learned to map subtle optical variations to specific environmental realities, enabling the high-fidelity, independent spatial retrieval of Chlorophyll-$\alpha$, CDOM, and Turbidity across the case study area.

Data Availability (Zenodo)
Due to file size constraints, the training and inference tensors are archived on Zenodo:
https://doi.org/10.5281/zenodo.21863329

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
- Execute Stage 1. Training, Stage 2. Inference, and Satge 3. Plotting script. Results are automatically stored in the Publication_Maps_Chl directory.

Domain Visualization
- Refer to Considered polygons.png for a spatial overview of the training test field, in situ sample locations, and the four seasonal UAV hyperspectral polygons used to establish the centimetric spatial characteristics.
