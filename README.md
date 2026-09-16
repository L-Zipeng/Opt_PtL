# OptLCFdesigns PtL Code Package

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.enconman.2026.122177-blue)](https://doi.org/10.1016/j.enconman.2026.122177)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause/)

This code package contains the modelling workflow used for the article:

> Global economic and environmental trade-offs of power-to-liquid fuels

Zipeng Liu, Tom Terlouw, Christian Bauer, and Russell McKenna.
Energy Conversion and Management, 2026.
https://doi.org/10.1016/j.enconman.2026.122177

The framework evaluates global economic and environmental trade-offs for
power-to-liquid (PtL) fuel production, covering methanol, methanol-to-jet
synthetic kerosene, and Fischer-Tropsch synthetic kerosene.

## Relationship to Opt_Ammonia

This PtL framework is an extension of, and builds on, Tom Terlouw's
`Opt_Ammonia` framework for decentralized low-carbon ammonia production. The
general workflow structure, geospatial preprocessing logic, hourly energy-system
optimization approach, techno-economic assessment structure, and Brightway-based
life cycle assessment workflow were adapted from that earlier open-source work.

The present package extends the framework from ammonia to PtL fuels by adding:

- direct air capture inputs and location-dependent DAC energy requirements;
- PtL foreground inventories and process units;
- three PtL pathways: power-to-methanol, methanol-to-jet SAF, and
  Fischer-Tropsch SAF;
- PtL-specific optimization constraints, conversion factors, cost parameters,
  and LCA accounting;
- grid-connected, hybrid, and off-grid electricity-supply configurations;
- paper-specific case studies, sensitivity analyses, global maps, contribution
  figures, spider plots, carbon-price analyses, and SHAP-based driver analysis.

## Pathways

| Pathway key | Description | Main product |
| --- | --- | --- |
| `meoh` | Direct air capture + PEM electrolysis + methanol synthesis | Methanol |
| `meoh_to_saf` | Methanol production followed by methanol-to-jet conversion | Synthetic kerosene / SAF |
| `ftsaf` | Direct air capture + PEM electrolysis + Fischer-Tropsch synthesis | Synthetic kerosene / SAF |

## System Configurations

| Configuration | Electricity source | Grid exposure | Storage need |
| --- | --- | --- | --- |
| Grid-connected | Grid electricity only | High | Low |
| Hybrid | Grid electricity + dedicated solar/wind | Partial | Moderate |
| Off-grid | Dedicated solar/wind only | None | High |

The optimization can size and dispatch solar PV, onshore wind, battery storage,
grid connection, PEM electrolysis, DAC, hydrogen storage, CO2 storage, and the
corresponding PtL synthesis units.

## Software Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate bw_region_init
```

The full optimization workflow also requires a working Gurobi installation and
license. Full LCA database regeneration requires Brightway2, ecoinvent access,
and a valid premise license key.

## Credentials and Licensed Data

Licensed ecoinvent databases, ecoinvent credentials, premise license keys, and
Gurobi license files are not redistributed with this code package. Before
regenerating LCA databases, set local credentials:

```bash
export ECOINVENT_USER="your-ecoinvent-username"
export ECOINVENT_PASSWORD="your-ecoinvent-password"
export PREMISE_KEY="your-premise-license-key"
```

Do not commit real credentials or licensed database files to a public
repository.

## Repository Structure

| File | Purpose |
| --- | --- |
| `0_init_dbs_export_GHGs.ipynb` | Initializes Brightway databases and exports LCA/GHG factors. |
| `1_fetch_power_prices.py` | Fetches and processes country-level grid electricity price inputs. |
| `2_geo_plot_and_export.ipynb` | Prepares geospatial layers and exports the initial processed dataset. |
| `3_main_global_preprocess.py` | Adds hourly solar/wind profiles and DAC energy requirements to processed geospatial data. |
| `4_main_global_euler.py` | Main global PtL optimization script for hybrid and off-grid systems. |
| `5_main_global_grid_connected.py` | Grid-connected PtL optimization using representative country-level cases. |
| `6_case_studies_now_and_prospective.ipynb` | Detailed present-day and prospective case-study workflow. |
| `7_sensitivity_analysis_case_studies.py` | Case-study sensitivity analysis. |
| `opt_ptx_functions.py` | Core MILP optimization models for MeOH, MtJ-SAF, and FT-SAF. |
| `create_db_lca_functions.py` | Brightway/premise database and foreground-inventory helper functions. |
| `calculate_renewable_yield.py` | Solar PV and wind profile generation. |
| `energy_data_processor.py` | Grid prices, WACC, grid GHG factors, and regional capacity helper functions. |
| `config.py` | Main model settings, file paths, scenarios, pathways, and constants. |
| `mapping.py` | Plot labels, units, colors, and impact-category mappings. |
| `plot_ptx_*.py` | Scripts for paper figures and post-processing analyses. |
| `environment.yml` | Conda environment specification. |

## Workflow

The main workflow is:

1. Initialize or import LCA background databases and export component GHG/LCA
   factors with `0_init_dbs_export_GHGs.ipynb`.
2. Fetch or update country-level electricity prices:

```bash
python 1_fetch_power_prices.py
```

3. Prepare geospatial inputs with `2_geo_plot_and_export.ipynb`.
4. Add hourly renewable profiles and DAC energy requirements:

```bash
python 3_main_global_preprocess.py
```

5. Run global PtL optimization for hybrid and off-grid configurations:

```bash
python 4_main_global_euler.py
```

6. Run grid-connected cases:

```bash
python 5_main_global_grid_connected.py
```

7. Run case-study and sensitivity workflows:

```bash
jupyter lab 6_case_studies_now_and_prospective.ipynb
python 7_sensitivity_analysis_case_studies.py
```

8. Generate figures and analysis outputs with the plotting scripts, for example:

```bash
python plot_ptx_cost_figures.py --pathway meoh --db both
python plot_ptx_cost_figures.py --pathway meoh_to_saf --db both
python plot_ptx_cost_figures.py --pathway ftsaf --db both

python plot_ptx_contribution_figures.py --pathway meoh --db both
python plot_ptx_contribution_figures.py --pathway meoh_to_saf --db both
python plot_ptx_contribution_figures.py --pathway ftsaf --db both

python plot_ptx_sensitivity_figures.py --pathway meoh
python plot_ptx_sensitivity_figures.py --pathway meoh_to_saf
python plot_ptx_sensitivity_figures.py --pathway ftsaf

python plot_ptx_global_maps.py --pathway meoh
python plot_ptx_spider_tradeoffs.py
python plot_ptx_carbon_price.py
python plot_ptx_SHAP_global_driver_analysis.py
```

On systems where the default Matplotlib cache directory is not writable, use a
writable cache directory, for example:

```bash
MPLCONFIGDIR=/private/tmp/mpl python plot_ptx_cost_figures.py --pathway ftsaf --db both
```

## Outputs

The workflow produces:

- levelized production costs for methanol and synthetic kerosene;
- life cycle GHG emissions and, where full LCA is enabled, additional
  environmental impact categories;
- optimal technology capacities and hourly dispatch;
- grid-connected, hybrid, and off-grid results;
- current and prospective case-study results;
- sensitivity-analysis results;
- global maps, cost breakdowns, environmental contribution figures,
  carbon-price figures, spider plots, and driver-analysis figures.

## Notes on Reproducibility

The code package provides the workflow and foreground model logic. Full
reproduction from scratch depends on external licensed data and software,
including ecoinvent, premise, and Gurobi. When saved intermediate results are
available, the plotting scripts can be used without regenerating licensed LCA
databases.

## Citation

If you use this code, please cite:

```bibtex
@article{liu2026global_ptl_tradeoffs,
  title = {Global economic and environmental trade-offs of power-to-liquid fuels},
  author = {Liu, Zipeng and Terlouw, Tom and Bauer, Christian and McKenna, Russell},
  journal = {Energy Conversion and Management},
  year = {2026},
  doi = {10.1016/j.enconman.2026.122177},
  pages = {122177}
}
```

Please also acknowledge the original `Opt_Ammonia` framework by Tom Terlouw,
which provided the basis for this PtL extension.

## License

This code is intended for release under the BSD 3-Clause License. Include a
`LICENSE` file with the BSD-3-Clause text when publishing the package.
