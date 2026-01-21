# VLEO EO Coverage Analysis

A Python package and CLI tool for analyzing satellite coverage, downlink contacts, and data delivery timelines for VLEO (Very Low Earth Orbit) electro-optical (EO) constellations.

## Features

- **Orbit Propagation**: Uses Skyfield for high-fidelity orbit propagation
- **Coverage Analysis**: Calculate access windows and revisit statistics for target AOIs
- **Downlink Contacts**: Compute ground station contact windows with validation
- **Downlink Delay Metrics**: Payload downlink delay analysis for each collection event
- **Reporting**: Generate Excel and PowerPoint reports with plots and KPIs
- **Station-Keeping Analysis**: Hall effect thruster propulsion budgets

## Environment Setup

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies for Cartopy
brew install proj geos

# Clone the repository
git clone <repository-url>
cd satellite-coverage-analysis

# Create and activate virtual environment
python3 -m venv sat-cov-env
source sat-cov-env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Ubuntu / Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libproj-dev libgeos-dev

# Clone the repository
git clone <repository-url>
cd satellite-coverage-analysis

# Create and activate virtual environment
python3 -m venv sat-cov-env
source sat-cov-env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
# Ensure Python 3.9+ is installed from python.org or Microsoft Store

# Clone the repository
git clone <repository-url>
cd satellite-coverage-analysis

# Create and activate virtual environment
python -m venv sat-cov-env
.\sat-cov-env\Scripts\Activate.ps1

# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Note for Windows users**: If Cartopy installation fails, try installing via conda:
```powershell
# Alternative: Use conda for Cartopy
conda install -c conda-forge cartopy
pip install -r requirements.txt
```

## Quick Start

### Activate the Environment

**macOS / Linux:**
```bash
source sat-cov-env/bin/activate
```

**Windows PowerShell:**
```powershell
.\sat-cov-env\Scripts\Activate.ps1
```

### Run Analysis

Three baseline configurations are provided for different ground station networks:

```bash
# ViaSat RTE network only (10 stations)
python run_analysis.py --config configs/viasat_rte_baseline.yaml

# ViaSat RTE + ATLAS Space combined (22 stations)
python run_analysis.py --config configs/viasat_rte_atlas_baseline.yaml

# KSAT network only (15 stations)
python run_analysis.py --config configs/ksat_baseline.yaml
```

### Additional Options

```bash
# Skip PowerPoint generation (faster)
python run_analysis.py --config configs/ksat_baseline.yaml --skip-ppt

# Skip plot generation
python run_analysis.py --config configs/ksat_baseline.yaml --skip-plots

# Custom output directory
python run_analysis.py --config configs/ksat_baseline.yaml --output-dir results/my_run

# Verbose output
python run_analysis.py --config configs/ksat_baseline.yaml -v
```

## Baseline Configurations

All baseline configs share these parameters:
- **Satellite**: 1 satellite, 400 km SSO, 10:30 LTDN
- **Analysis Period**: 30 days starting 2027-02-15
- **Targets**: APAC target deck (`targets/target_aoi_deck_apac.geojson`)
- **Image Size**: 3 GB
- **Downlink Rate**: 1000 Mbps

| Config | Ground Stations | Description |
|--------|-----------------|-------------|
| `viasat_rte_baseline.yaml` | 10 | ViaSat Real-Time Earth network |
| `viasat_rte_atlas_baseline.yaml` | 22 | ViaSat RTE + ATLAS Space combined |
| `ksat_baseline.yaml` | 15 | KSAT (Kongsberg Satellite Services) |

## Configuration

The analysis is configured via a YAML file. Key sections include:

```yaml
# Analysis time window
start_date: "2027-02-15"
duration_days: 30
time_step_s: 10.0

# Orbit configuration
orbit:
  ltdn_hours: 10.5      # Local Time of Descending Node
  auto_sso: true        # Auto-calculate SSO parameters

# Constellation definition
satellites:
  - sat_id: 1
    inclination_deg: 97.0
    altitude_km: 400
    raan_deg: 0

# Downlink parameters
image_size_gb: 3.0
downlink_rate_mbps: 1000

# Elevation masks per band
min_elevation_sx_deg: 5.0   # S-band and X-band (TT&C)
min_elevation_ka_deg: 10.0  # Ka-band (payload downlink)

# Ground stations
ground_stations:
  - name: KSAT-Svalbard-SvalSat
    lat: 78.23
    lon: 15.41
    s_band: true
    x_band: true
    ka_band: true
    provider: KSAT

# Imaging modes
imaging_modes:
  - name: "Panchromatic"
    fov_half_angle_deg: 2.5
    collect_dwell_time_s: 5.0
    collection_rate_gbps: 2.0
    off_nadir_max_deg: 60.0

# Target AOIs
targets:
  geojson_path: "targets/target_aoi_deck_apac.geojson"
  use_grid: false

# Output settings
output_dir: "results/ksat_baseline"
excel_filename: "ksat_baseline.xlsx"
ppt_filename: "ksat_baseline.pptx"
```

## Output Files

After running the analysis, you'll find:

```
results/<config_name>/
├── <config_name>.xlsx        # Excel report with all data tables
├── <config_name>.pptx        # PowerPoint slides with plots and KPIs
└── plots/
    ├── ground_tracks.png     # Satellite ground tracks
    ├── coverage_map.png      # Access coverage visualization
    ├── comm_cones.png        # Ground station communication cones
    ├── contact_validity.png  # Valid/invalid contact breakdown
    ├── downlink_delay_distribution.png # Payload downlink delay statistics
    ├── slide_*.png           # Additional presentation slides
    └── ...
```

### Excel Report Sheets

| Sheet | Description |
|-------|-------------|
| Config | Analysis parameters |
| Satellites | Constellation definition |
| Ground_Stations | GS locations and capabilities |
| Imaging_Modes | Mode parameters |
| Coverage_KPIs | Access statistics |
| Access_Windows | All computed access windows |
| Contacts | Contact windows per mode |
| Downlink_KPIs | Downlink statistics by GS |
| Downlink_Delay | Payload downlink delay per collection |
| Downlink_Delay_Summary | Downlink delay statistics by mode |
| Propulsion_Analysis | Station-keeping propellant budget |

## Package Structure

```
satellite-coverage-analysis/
├── run_analysis.py           # CLI entrypoint
├── configs/
│   ├── viasat_rte_baseline.yaml
│   ├── viasat_rte_atlas_baseline.yaml
│   └── ksat_baseline.yaml
├── targets/
│   └── target_aoi_deck_apac.geojson
├── scripts/
│   ├── generate_full_presentation.py
│   └── validate_outputs.py
├── src/
│   └── vleo_eo/
│       ├── __init__.py
│       ├── config.py         # Configuration dataclasses
│       ├── orbits.py         # TLE generation and propagation
│       ├── coverage.py       # Access window calculation
│       ├── contacts.py       # Ground station contacts
│       ├── data_model.py     # Data volume modeling
│       ├── plots.py          # Visualization functions
│       ├── reports.py        # Excel and PPT generation
│       ├── optimization.py   # Ground station optimization
│       ├── propulsion.py     # Station-keeping analysis
│       └── ground_station_db.py
└── results/                  # Output directory
```

## API Usage

You can also use the package programmatically:

```python
from src.vleo_eo.config import load_config
from src.vleo_eo.orbits import create_tle_data, propagate_orbits
from src.vleo_eo.coverage import load_targets, calculate_access_windows
from src.vleo_eo.contacts import calculate_contact_windows, calculate_downlink_delay

# Load configuration
config = load_config('configs/ksat_baseline.yaml')

# Create TLE data
tle_data = {}
for sat in config.satellites:
    tle = create_tle_data(
        sat_id=sat.sat_id,
        inclination_deg=sat.inclination_deg,
        altitude_km=sat.altitude_km,
        raan_deg=sat.raan_deg,
        epoch=config.start_datetime,
    )
    tle_data[sat.sat_id] = tle

# Propagate orbits
orbit_df = propagate_orbits(
    tle_data,
    config.start_datetime,
    config.end_datetime,
    time_step_s=config.time_step_s,
)

# Calculate access windows
targets_gdf = load_targets(geojson_path=config.targets.geojson_path)
access_df = calculate_access_windows(orbit_df, targets_gdf)

# Calculate contacts
mode_dfs = calculate_contact_windows(
    access_df, tle_data,
    config.ground_stations,
    config.imaging_modes,
)

# Calculate payload downlink delay
downlink_delay_df = calculate_downlink_delay(mode_dfs, config.ground_stations)
```

## Validation

Run output validation to verify analysis results:

```bash
python scripts/validate_outputs.py results/ksat_baseline
```

### Automated Validation Checks

Each phase includes automated sanity checks:

- **Phase 1 (Orbits)**: Altitude range, lat/lon bounds, finite values
- **Phase 2 (Coverage)**: Non-zero access count, plausible durations
- **Phase 3 (Contacts)**: Contact existence, non-negative downlink delay

## Troubleshooting

### Cartopy Installation Issues

**macOS:**
```bash
brew install proj geos
pip install cartopy --no-binary cartopy
```

**Ubuntu:**
```bash
sudo apt-get install libproj-dev libgeos-dev
pip install cartopy
```

**Windows:**
If pip installation fails, use conda:
```powershell
conda install -c conda-forge cartopy
```

### Virtual Environment Not Found

Ensure you've created and activated the virtual environment:
```bash
# Create if it doesn't exist
python3 -m venv sat-cov-env

# Activate (macOS/Linux)
source sat-cov-env/bin/activate

# Activate (Windows PowerShell)
.\sat-cov-env\Scripts\Activate.ps1
```

### Missing Target Files

Ensure target GeoJSON files exist in the `targets/` directory. The baseline configs expect:
```
targets/target_aoi_deck_apac.geojson
```

## License

MIT License
