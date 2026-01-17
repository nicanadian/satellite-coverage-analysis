# VLEO EO Coverage Analysis

A Python package and CLI tool for analyzing satellite coverage, downlink contacts, and data delivery timelines for VLEO (Very Low Earth Orbit) electro-optical (EO) constellations.

## Features

- **Orbit Propagation**: Uses Skyfield for high-fidelity orbit propagation
- **Coverage Analysis**: Calculate access windows and revisit statistics for target AOIs
- **Downlink Contacts**: Compute ground station contact windows with validation
- **TTNC Metrics**: Time To Next Ka Contact analysis for each collection event
- **Data Backlog**: Simulate data generation and downlink to compute backlog over time
- **Reporting**: Generate Excel and PowerPoint reports with plots and KPIs

## Installation

### Quick Install

```bash
cd satellite-coverage-analysis
pip install -r requirements.txt
```

### Manual Installation

```bash
# Core dependencies
pip install numpy pandas geopandas
pip install matplotlib cartopy
pip install skyfield
pip install shapely pyproj
pip install pyyaml openpyxl
pip install python-pptx  # Optional, for PowerPoint generation
```

**Note**: Cartopy may require additional system dependencies. On macOS:
```bash
brew install proj geos
```

On Ubuntu/Debian:
```bash
sudo apt-get install libproj-dev libgeos-dev
```

## Quick Start

```bash
# Run analysis with default configuration
python run_analysis.py --config configs/vleo_eo_default.yaml

# Run with custom output directory
python run_analysis.py --config configs/vleo_eo_default.yaml --output-dir results/my_run

# Skip PowerPoint generation (if python-pptx not installed)
python run_analysis.py --config configs/vleo_eo_default.yaml --skip-ppt

# Verbose output
python run_analysis.py --config configs/vleo_eo_default.yaml -v
```

## Configuration

The analysis is configured via a YAML file. See `configs/vleo_eo_default.yaml` for a complete example.

### Key Configuration Sections

```yaml
# Analysis time window
start_date: "2025-11-01"
duration_days: 7
time_step_s: 10.0

# Constellation definition
satellites:
  - sat_id: 1
    inclination_deg: 97.4
    altitude_km: 350
    raan_deg: 0

# Ground stations
ground_stations:
  - name: Svalbard
    lat: 78.2
    lon: 15.4
    min_elevation_deg: 5.0
    ka_capable: true
    downlink_rate_mbps: 1200

# Imaging modes
imaging_modes:
  - name: "Panchromatic"
    collect_dwell_time_s: 5.0
    collection_rate_gbps: 2.0
    off_nadir_max_deg: 30.0

# Target AOIs
targets:
  target_points:
    - [35.7, 139.7]   # lat, lon
```

## Output Files

After running the analysis, you'll find:

```
results/
├── results_summary.xlsx      # Excel report with all data tables
├── results_summary.pptx      # PowerPoint slides with plots and KPIs
└── plots/
    ├── ground_tracks.png     # Satellite ground tracks
    ├── coverage_map.png      # Access coverage visualization
    ├── comm_cones.png        # Ground station communication cones
    ├── contact_validity.png  # Valid/invalid contact breakdown
    ├── backlog_timeseries.png # Data backlog over time
    └── ttnc_distribution.png # TTNC Ka statistics
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
| Backlog_TimeSeries | Data backlog over time |
| TTNC_Ka | Time to next Ka contact per collection |
| TTNC_Summary | TTNC statistics by mode |

## Package Structure

```
satellite-coverage-analysis/
├── run_analysis.py           # CLI entrypoint
├── configs/
│   └── vleo_eo_default.yaml  # Default configuration
├── src/
│   └── vleo_eo/
│       ├── __init__.py
│       ├── config.py         # Configuration dataclasses
│       ├── orbits.py         # TLE generation and propagation
│       ├── coverage.py       # Access window calculation
│       ├── contacts.py       # Ground station contacts
│       ├── data_model.py     # Data volume and backlog
│       ├── plots.py          # Visualization functions
│       └── reports.py        # Excel and PPT generation
└── results/                  # Output directory
```

## API Usage

You can also use the package programmatically:

```python
from src.vleo_eo.config import load_config
from src.vleo_eo.orbits import create_tle_data, propagate_orbits
from src.vleo_eo.coverage import load_targets, calculate_access_windows
from src.vleo_eo.contacts import calculate_contact_windows, calculate_ttnc_ka

# Load configuration
config = load_config('configs/vleo_eo_default.yaml')

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
targets_gdf = load_targets(target_points=config.targets.target_points)
access_df = calculate_access_windows(orbit_df, targets_gdf)

# Calculate contacts
mode_dfs = calculate_contact_windows(
    access_df, tle_data,
    config.ground_stations,
    config.imaging_modes,
)

# Calculate TTNC
ttnc_df = calculate_ttnc_ka(mode_dfs, config.ground_stations)
```

## Plot Styling

The plotting functions match the style of the original notebook:

- **Map projection**: PlateCarree (Cartopy)
- **Land color**: `#fafaf9`
- **Ocean color**: `#d4dbdc`
- **Border color**: `#ebd6d9`
- **Figure size**: 15x10 inches
- **DPI**: 300

## Validation Checks

Each phase includes automated sanity checks:

- **Phase 1 (Orbits)**: Altitude range, lat/lon bounds, finite values
- **Phase 2 (Coverage)**: Non-zero access count, plausible durations
- **Phase 3 (Contacts)**: Contact existence, non-negative TTNC
- **Phase 4 (Backlog)**: Non-negative values, consistent totals

## License

MIT License
