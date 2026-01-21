"""
Configuration dataclasses and loading utilities.

All configuration is managed through typed dataclasses for validation
and IDE support.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class SatelliteConfig:
    """Configuration for a single satellite."""
    sat_id: int
    inclination_deg: float
    altitude_km: float
    raan_deg: float = 0.0
    # Optional: provide TLE lines directly instead of generating
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None


@dataclass
class GroundStationConfig:
    """Configuration for a ground station."""
    name: str
    lat: float
    lon: float
    elevation_m: float = 0.0
    min_elevation_deg: float = 5.0
    # Band capabilities (separate S/X for TT&C, Ka for payload)
    s_band: bool = True       # S-band uplink (command)
    x_band: bool = True       # X-band downlink (telemetry)
    ka_band: bool = True      # Ka-band (payload data)
    # Provider name for optimization grouping
    provider: str = ""
    # Downlink rate in Mbps (Ka-band payload)
    downlink_rate_mbps: float = 800.0

    @property
    def ka_capable(self) -> bool:
        """Backward compatibility: ka_capable maps to ka_band."""
        return self.ka_band

    @property
    def ttc_capable(self) -> bool:
        """Station can support TT&C operations (needs S or X band)."""
        return self.s_band or self.x_band


@dataclass
class ImagingModeConfig:
    """Configuration for an EO imaging mode."""
    name: str
    # Field of view half-angle in degrees
    fov_half_angle_deg: float = 5.0
    # Dwell/integration time in seconds
    collect_dwell_time_s: float = 10.0
    # Data rate during collection in Gbps
    collection_rate_gbps: float = 1.0
    # Ground processing time in minutes
    processing_time_min: float = 45.0
    # Off-nadir pointing limits
    off_nadir_min_deg: float = 0.0
    off_nadir_max_deg: float = 45.0


@dataclass
class OptimizationConfig:
    """Configuration for RGT-Min ground station optimization."""
    # Enable optimization
    enabled: bool = False
    # Optimization approach: 'exhaustive' or 'greedy'
    approach: str = 'greedy'
    # TT&C SLA: minimum contacts per orbital revolution per satellite
    ttc_contacts_per_rev: int = 1
    # Ka Payload SLA: percentage of collects delivered within time limit
    ka_delivery_percent: float = 95.0
    # Ka Payload SLA: delivery time limit in hours
    ka_delivery_hours: float = 4.0
    # Providers to optimize (empty = all available)
    providers: List[str] = field(default_factory=list)
    # Use database stations (if True, ignores manually specified stations)
    use_database_stations: bool = True
    # Max stations to consider in exhaustive search (for performance)
    max_exhaustive_stations: int = 15


@dataclass
class TargetConfig:
    """Configuration for target AOIs."""
    # Path to GeoJSON file with target points/polygons
    geojson_path: Optional[str] = None
    # Or specify targets directly as list of (lat, lon) tuples
    target_points: List[tuple] = field(default_factory=list)
    # Grid-based targets
    use_grid: bool = False
    grid_lat_min: float = -60.0
    grid_lat_max: float = 60.0
    grid_lon_min: float = -180.0
    grid_lon_max: float = 180.0
    grid_spacing_deg: float = 5.0


@dataclass
class AnalysisConfig:
    """Main configuration for the analysis."""
    # Analysis time window
    start_date: str  # ISO format: YYYY-MM-DD
    duration_days: int = 30
    time_step_s: float = 10.0

    # Constellation
    satellites: List[SatelliteConfig] = field(default_factory=list)

    # Ground stations
    ground_stations: List[GroundStationConfig] = field(default_factory=list)

    # Imaging modes
    imaging_modes: List[ImagingModeConfig] = field(default_factory=list)

    # Targets
    targets: TargetConfig = field(default_factory=TargetConfig)

    # Optimization settings
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Slew parameters
    slew_rate_deg_per_s: float = 0.6
    mode_switch_time_s: float = 60.0
    contact_buffer_s: float = 30.0

    # Access filtering
    min_access_duration_s: float = 45.0
    access_merge_buffer_s: float = 240.0  # 4 minutes

    # Payload downlink delay search window (max time to search for next Ka contact)
    downlink_search_hours: float = 3.0

    # Elevation masks per band
    min_elevation_sx_deg: float = 5.0   # S-band and X-band (TT&C)
    min_elevation_ka_deg: float = 10.0  # Ka-band (payload downlink)

    # Output paths
    output_dir: str = "results"
    excel_filename: str = "results_summary.xlsx"
    ppt_filename: str = "results_summary.pptx"
    plots_subdir: str = "plots"

    # Plot settings (matching notebook style)
    plot_dpi: int = 300
    plot_figsize: tuple = (15, 10)

    # Map colors (from notebook)
    land_color: str = "#fafaf9"
    ocean_color: str = "#d4dbdc"
    border_color: str = "#ebd6d9"

    # Satellite colors
    satellite_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

    # Raw config for extended parameters (propulsion, orbit, etc.)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def start_datetime(self) -> datetime:
        """Parse start date to datetime with UTC timezone."""
        return datetime.strptime(self.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    @property
    def end_datetime(self) -> datetime:
        """Calculate end datetime."""
        from datetime import timedelta
        return self.start_datetime + timedelta(days=self.duration_days)

    @property
    def output_path(self) -> Path:
        """Get output directory path."""
        return Path(self.output_dir)

    @property
    def plots_path(self) -> Path:
        """Get plots subdirectory path."""
        return self.output_path / self.plots_subdir


def _parse_satellite(data: Dict[str, Any]) -> SatelliteConfig:
    """Parse satellite config from dict."""
    return SatelliteConfig(
        sat_id=data['sat_id'],
        inclination_deg=data['inclination_deg'],
        altitude_km=data['altitude_km'],
        raan_deg=data.get('raan_deg', 0.0),
        tle_line1=data.get('tle_line1'),
        tle_line2=data.get('tle_line2'),
    )


def _parse_ground_station(data: Dict[str, Any]) -> GroundStationConfig:
    """Parse ground station config from dict."""
    # Handle both old (ka_capable) and new (s_band/x_band/ka_band) formats
    ka_capable = data.get('ka_capable', True)
    return GroundStationConfig(
        name=data['name'],
        lat=data['lat'],
        lon=data['lon'],
        elevation_m=data.get('elevation_m', 0.0),
        min_elevation_deg=data.get('min_elevation_deg', 5.0),
        s_band=data.get('s_band', True),
        x_band=data.get('x_band', True),
        ka_band=data.get('ka_band', ka_capable),  # Fallback to ka_capable for compatibility
        provider=data.get('provider', ''),
        downlink_rate_mbps=data.get('downlink_rate_mbps', 800.0),
    )


def _parse_optimization(data: Dict[str, Any]) -> OptimizationConfig:
    """Parse optimization config from dict."""
    return OptimizationConfig(
        enabled=data.get('enabled', False),
        approach=data.get('approach', 'greedy'),
        ttc_contacts_per_rev=data.get('ttc_contacts_per_rev', 1),
        ka_delivery_percent=data.get('ka_delivery_percent', 95.0),
        ka_delivery_hours=data.get('ka_delivery_hours', 4.0),
        providers=data.get('providers', []),
        use_database_stations=data.get('use_database_stations', True),
        max_exhaustive_stations=data.get('max_exhaustive_stations', 15),
    )


def _parse_imaging_mode(data: Dict[str, Any]) -> ImagingModeConfig:
    """Parse imaging mode config from dict."""
    return ImagingModeConfig(
        name=data['name'],
        fov_half_angle_deg=data.get('fov_half_angle_deg', 5.0),
        collect_dwell_time_s=data.get('collect_dwell_time_s', 10.0),
        collection_rate_gbps=data.get('collection_rate_gbps', 1.0),
        processing_time_min=data.get('processing_time_min', 45.0),
        off_nadir_min_deg=data.get('off_nadir_min_deg', 0.0),
        off_nadir_max_deg=data.get('off_nadir_max_deg', 45.0),
    )


def _parse_targets(data: Dict[str, Any]) -> TargetConfig:
    """Parse target config from dict."""
    target_points = []
    if 'target_points' in data:
        target_points = [tuple(p) for p in data['target_points']]

    return TargetConfig(
        geojson_path=data.get('geojson_path'),
        target_points=target_points,
        use_grid=data.get('use_grid', False),
        grid_lat_min=data.get('grid_lat_min', -60.0),
        grid_lat_max=data.get('grid_lat_max', 60.0),
        grid_lon_min=data.get('grid_lon_min', -180.0),
        grid_lon_max=data.get('grid_lon_max', 180.0),
        grid_spacing_deg=data.get('grid_spacing_deg', 5.0),
    )


def load_config(config_path: str) -> AnalysisConfig:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    AnalysisConfig
        Parsed and validated configuration.
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Parse nested configs
    satellites = [_parse_satellite(s) for s in data.get('satellites', [])]
    ground_stations = [_parse_ground_station(gs) for gs in data.get('ground_stations', [])]
    imaging_modes = [_parse_imaging_mode(m) for m in data.get('imaging_modes', [])]
    targets = _parse_targets(data.get('targets', {}))
    optimization = _parse_optimization(data.get('optimization', {}))

    # Build main config
    config = AnalysisConfig(
        start_date=data['start_date'],
        duration_days=data.get('duration_days', 30),
        time_step_s=data.get('time_step_s', 10.0),
        satellites=satellites,
        ground_stations=ground_stations,
        imaging_modes=imaging_modes,
        targets=targets,
        optimization=optimization,
        slew_rate_deg_per_s=data.get('slew_rate_deg_per_s', 0.6),
        mode_switch_time_s=data.get('mode_switch_time_s', 60.0),
        contact_buffer_s=data.get('contact_buffer_s', 30.0),
        min_access_duration_s=data.get('min_access_duration_s', 45.0),
        access_merge_buffer_s=data.get('access_merge_buffer_s', 240.0),
        downlink_search_hours=data.get('downlink_search_hours', data.get('ttnc_max_search_hours', 3.0)),
        min_elevation_sx_deg=data.get('min_elevation_sx_deg', 5.0),
        min_elevation_ka_deg=data.get('min_elevation_ka_deg', 10.0),
        output_dir=data.get('output_dir', 'results'),
        excel_filename=data.get('excel_filename', 'results_summary.xlsx'),
        ppt_filename=data.get('ppt_filename', 'results_summary.pptx'),
        plots_subdir=data.get('plots_subdir', 'plots'),
        plot_dpi=data.get('plot_dpi', 300),
        plot_figsize=tuple(data.get('plot_figsize', [15, 10])),
        land_color=data.get('land_color', '#fafaf9'),
        ocean_color=data.get('ocean_color', '#d4dbdc'),
        border_color=data.get('border_color', '#ebd6d9'),
        satellite_colors=data.get('satellite_colors', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]),
        raw_config=data,  # Store raw config for extended parameters
    )

    return config


def save_config(config: AnalysisConfig, config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Parameters
    ----------
    config : AnalysisConfig
        Configuration to save.
    config_path : str
        Path to write the YAML file.
    """
    data = {
        'start_date': config.start_date,
        'duration_days': config.duration_days,
        'time_step_s': config.time_step_s,
        'satellites': [
            {
                'sat_id': s.sat_id,
                'inclination_deg': s.inclination_deg,
                'altitude_km': s.altitude_km,
                'raan_deg': s.raan_deg,
            }
            for s in config.satellites
        ],
        'ground_stations': [
            {
                'name': gs.name,
                'lat': gs.lat,
                'lon': gs.lon,
                'elevation_m': gs.elevation_m,
                'min_elevation_deg': gs.min_elevation_deg,
                's_band': gs.s_band,
                'x_band': gs.x_band,
                'ka_band': gs.ka_band,
                'provider': gs.provider,
                'downlink_rate_mbps': gs.downlink_rate_mbps,
            }
            for gs in config.ground_stations
        ],
        'imaging_modes': [
            {
                'name': m.name,
                'fov_half_angle_deg': m.fov_half_angle_deg,
                'collect_dwell_time_s': m.collect_dwell_time_s,
                'collection_rate_gbps': m.collection_rate_gbps,
                'processing_time_min': m.processing_time_min,
                'off_nadir_min_deg': m.off_nadir_min_deg,
                'off_nadir_max_deg': m.off_nadir_max_deg,
            }
            for m in config.imaging_modes
        ],
        'targets': {
            'geojson_path': config.targets.geojson_path,
            'target_points': config.targets.target_points,
            'use_grid': config.targets.use_grid,
            'grid_lat_min': config.targets.grid_lat_min,
            'grid_lat_max': config.targets.grid_lat_max,
            'grid_lon_min': config.targets.grid_lon_min,
            'grid_lon_max': config.targets.grid_lon_max,
            'grid_spacing_deg': config.targets.grid_spacing_deg,
        },
        'optimization': {
            'enabled': config.optimization.enabled,
            'approach': config.optimization.approach,
            'ttc_contacts_per_rev': config.optimization.ttc_contacts_per_rev,
            'ka_delivery_percent': config.optimization.ka_delivery_percent,
            'ka_delivery_hours': config.optimization.ka_delivery_hours,
            'providers': config.optimization.providers,
            'use_database_stations': config.optimization.use_database_stations,
            'max_exhaustive_stations': config.optimization.max_exhaustive_stations,
        },
        'slew_rate_deg_per_s': config.slew_rate_deg_per_s,
        'mode_switch_time_s': config.mode_switch_time_s,
        'contact_buffer_s': config.contact_buffer_s,
        'min_access_duration_s': config.min_access_duration_s,
        'access_merge_buffer_s': config.access_merge_buffer_s,
        'downlink_search_hours': config.downlink_search_hours,
        'min_elevation_sx_deg': config.min_elevation_sx_deg,
        'min_elevation_ka_deg': config.min_elevation_ka_deg,
        'output_dir': config.output_dir,
        'excel_filename': config.excel_filename,
        'ppt_filename': config.ppt_filename,
        'plots_subdir': config.plots_subdir,
        'plot_dpi': config.plot_dpi,
        'plot_figsize': list(config.plot_figsize),
        'land_color': config.land_color,
        'ocean_color': config.ocean_color,
        'border_color': config.border_color,
        'satellite_colors': config.satellite_colors,
    }

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
