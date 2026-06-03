Permafrost ERT Analysis

Automated algorithmic analysis of Electrical Resistivity Tomography (ERT) tomograms
for temporal monitoring of alpine permafrost degradation.

Bachelor thesis — University of Fribourg, Institute of Geography
Author : Mathilda Muster Labeau
Supervisor : Dr. Coline Mollaret, Department of Geoscience

Overview
Alpine permafrost is degrading under climate change, increasing the risk of gravitational instabilities. ERT (Electrical Resistivity Tomography) is a standard geophysical method to monitor this evolution — but interpreting tomograms manually over decades of data is slow and subjective.
This project proposes an automated Python pipeline to standardize the post-processing of inverted ERT tomograms from the IDGSP database, with a focus on temporal tracking of high-resistivity zones associated with potential permafrost.

Methodology
The pipeline combines two complementary approaches:
1. Dynamic segmentation (DBSCAN)
A density-based clustering algorithm adapted for ERT data. Unlike fixed rectangular ROIs, DBSCAN detects resistive bodies of arbitrary shape while penalizing unreliable deep measurements via coverage-weighted distances.
2. Fixed reference point tracking (Master ROI)


## Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Main processing script — single survey |
| `multi_run.py` | Batch processing over all surveys of a site |
| `functions.py` | Core functions: ROI masking, DBSCAN, metrics |
| `master_roi.py` | Fixed reference point time series analysis |
| `Plot.py` | Visualization module |
| `summary_results.csv` | DBSCAN results per survey |
| `summary_master.csv` | Master ROI results per survey |
| `requirements.txt` | Python dependencies |

Key Features

Automatic topographic masking adapted to each profile geometry
Coverage-weighted DBSCAN clustering in a custom feature space (x, z, resistivity, quality)
Vertical anisotropy factor to favor horizontally-dominant permafrost bodies
Multi-cluster fusion strategy for fragmented resistive zones
Temporal tracking with geometric indicators (area, length, depth, span)
Tested on 4 Alpine sites: CH_DOL, CH_ATT, IT_CER, CH_STT


Installation
bashgit clone https://github.com/MathiMuster/permafrost-ert-analysis.git
cd permafrost-ert-analysis
pip install -r requirements.txt

- pyGIMLi may require a separate installation via conda:
conda install -c gimli pygimli


Usage
Process a single survey:
bashpython main.py
Batch process all surveys of a site:
bashpython multi_run.py
Run fixed reference point analysis:
bashpython master_roi.py

Data
Input data comes from the IDGSP (International Database of Geoelectrical Surveys on Permafrost). Each survey consists of four files:
FileContenttopo.txtElectrode positions (GNSS)mesh_node.txtMesh node coordinatesmesh_cell.txtTriangular cell connectivitydata_inv.txtInverted resistivity + coverage index
Data is not included in this repository. Contact IDGSP for access.

Dependencies

pyGIMLi — geophysical mesh handling and inversion

scikit-learn — DBSCAN clustering

NumPy, Pandas — data processing

Matplotlib — visualization


Results
The dual-method approach was validated on the CH_ATT_MV1 profile (Les Attelas, Switzerland, 2007–2022):

Both DBSCAN and Master ROI show a consistent ~25% decrease in mean resistivity over 15 years
The DBSCAN approach captures interannual variability (cluster "breathing")
The Master ROI approach provides a smoother long-term trend

These results are compatible with warming trends documented by PERMOS and MeteoSwiss.

Limitations

Fixed parameter set: not yet auto-tuned per profile geometry
Deep resistive bodies may be under-detected due to coverage penalization
Resistivity is an indirect indicator; permafrost presence cannot be certified from ERT alone


Reference
Muster Labeau, M. (2026). Analyse algorithmique des tomogrammes : vers l'automatisation du suivi temporel du permafrost alpin. Bachelor thesis, University of Fribourg.
