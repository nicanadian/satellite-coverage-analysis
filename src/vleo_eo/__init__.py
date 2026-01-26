"""
VLEO EO Coverage Analysis Package

A Python package for analyzing satellite coverage, downlink contacts,
and data delivery timelines for VLEO electro-optical (EO) constellations.
"""

__version__ = "0.1.0"

# Core utilities and constants
from .constants import (
    EARTH_RADIUS_KM,
    EARTH_MU_KM3_S2,
    DEFAULT_MIN_ELEVATION_SX_DEG,
    DEFAULT_MIN_ELEVATION_KA_DEG,
    SATELLITE_COLORS,
    PROVIDER_COLORS,
)
from .utils import (
    haversine_distance_km,
    get_orbital_period_minutes,
    calculate_coverage_radius_km,
    find_contact_boundaries,
)
from .config import (
    AnalysisConfig,
    SatelliteConfig,
    GroundStationConfig,
    ImagingModeConfig,
    OptimizationConfig,
    load_config,
)
from .orbits import (
    TLEData,
    create_tle_data,
    propagate_orbits,
    calculate_ground_track,
    ltdn_to_raan,
    calculate_sso_inclination,
)
from .coverage import (
    load_targets,
    calculate_access_windows,
    filter_access_by_swath,
    validate_coverage,
)
from .contacts import (
    create_communication_cone,
    calculate_contact_windows,
    calculate_downlink_delay,
    calculate_raw_contact_windows,
)
from .data_model import (
    calculate_downlink_kpis,
)
from .plots import (
    plot_ground_tracks,
    plot_coverage_map,
    plot_comm_cones,
    plot_access_statistics,
    plot_contact_validity,
    plot_downlink_delay_distribution,
)
from .reports import (
    generate_excel_report,
)
from .optimization import (
    run_optimization,
    OptimizationResult,
    optimization_results_to_dataframe,
)
from .comparison import (
    ComparisonResult,
    load_comparison_results,
    load_comparison_result,
    load_raw_delay_data,
    comparison_to_dataframe,
    generate_comparison_excel,
    get_provider_color,
)
