PIMREN: Physics-Informed Multi-Regime Ensemble U-Net
This repository provides the official implementation of the PIMREN framework as described in:

"A Physics-Informed Multi-Regime Ensemble U-Net Framework for Satellite-Based Water Quality Inversion: Bridging Scale Gaps and Seasonal Non-Stationarity"

Project Overview
The PIMREN (Physics-Informed Multi-Regime Ensemble U-Net) framework is designed to resolve the data-scarcity bottleneck in coastal water quality monitoring. It bridges the gap between centimeter-scale in situ observations and decameter-scale Sentinel-2 pixels using a stochastic scaling bridge.

Data Availability (Zenodo)
Due to file size constraints, the training and inference tensors are archived on Zenodo:
DOI: 10.5281/zenodo.20184062

The Zenodo archive contains:
- & outputs.bin: The Master Dataset (4,000 realizations).
- PIMREN_4000_Realizations_Kastela_Bay.zip: Individual stacked realizations.
- Y_stats_train.npy: Pre-trained model weights ("The Brain").
- full_bay_input_season.bin: Inference tensors for Kaštela Bay.

Components
1. Stochastic Scaling Bridge (Batch_generation_June21_github.nb)
A Mathematica implementation for generating high-fidelity random field realizations.
- Function: Projects centimetric spatial anisotropy and covariance structures onto a decametric (20m) grid.
- Inputs: Utilizes provided In_situ_samples (June 2021 Chl-$\alpha$ example).
- Preview: See the PDF version for a quick overview of the code logic.

2. PIMREN Master Model (Master_pimren.py)
The core Python script for multi-seasonal ensemble inversion.
- Regime-Dependent Modulator: An 11th-channel input tensor that resolves seasonal non-stationarity by modulating shared convolutional kernels.
- Physics-Informed Loss: Enforces seasonal physical penalty ratios ($\lambda$) to honor empirically derived bio-optical relationships.
- Ensemble Uncertainty: A 5-member ensemble provides predictive mean and Coefficient of Variation (CV) maps.
- Epistemic Quality Constraint: CV maps identify trustworthy physical signals while masking sensor artifacts or meteorological noise.

Getting Started
- Clone the repository:
git clone https://github.com/rokoandricevic/PIMREN-Coastal-WQ.git
- Download Data: Download the .bin tensors from the Zenodo DOI and place them in the project root.
- Run the scaling bridge: Open the Mathematica script to generate realizations (ensure measurementFile points to the In_situ samples folder)
- Execute Training/Inference: Run Master_pimren.py to train the ensemble and perform inference for Kaštela Bay. Results are automatically stored in the Master_PIMREN_Results directory.

Domain Visualization
- Refer to Considered polygons.png for a spatial overview of the training test field, in situ sample locations, and the four seasonal UAV hyperspectral polygons used to establish the centimetric spatial characteristics.
