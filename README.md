**Master_pimren.py**
The provided Python script successfully implements the core mechanics of the PIMREN framework described in the manuscript: A Physics-Informed Multi-Regime Ensemble U-Net Framework for Satellite-Based Water Quality Inversion: Bridging Scale Gaps and Seasonal Non-Stationarity

11th-Channel Logic: The architecture is designed to accept an 11-channel input tensor, where the final channel serves as the regime-dependent modulator.

Physics-Informed Loss: The script defines the total loss function as the sum of empirical data loss and the seasonal physical penalty $\lambda(s)$. This ensures the model adheres to established bio-optical laws even when satellite signals are noisy.

Ensemble Integration: The code is structured to facilitate a 5-member ensemble, allowing you to calculate the predictive mean and the coefficient of variation (CV).

Epistemic Quality Constraint: By generating CV maps, the script provides the necessary tools to implement the automated quality filter or "blackout" for unreliable spectral states.

**Batch_generation_June21_github.nb**
This is a Mathematica code for genarating random field realizations using the "Stochastic Scaling Bridge" framework as an example for CHL for June 2021 in situ samples.

Before running the code make sure to change measurementFile directory (use provided folder In_situ samples) and ouput directory for storing random realizations.

Figure Considered polygons.png displays the random test field (for random generation) and in situ samples for each season with four UAV HS overpass polygons providing the sentimetric spatial correlative characteristics.
