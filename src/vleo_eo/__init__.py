"""
VLEO EO Coverage Analysis Package

A Python package for analyzing satellite coverage, downlink contacts,
and data delivery timelines for VLEO electro-optical (EO) constellations.
"""

__version__ = "0.1.0"

from .config import (
    AnalysisConfig,
    SatelliteConfig,
    GroundStationConfig,
    ImagingModeConfig,
    OptimizationConfig,
    load_config,
)
from .orbits import (
    generate_tle,
    propagate_orbits,
    calculate_ground_track,
)
from .coverage import (
    calculate_look_angle,
    calculate_access_windows,
    filter_access_by_swath,
)
from .contacts import (
    create_communication_cone,
    find_next_contact,
    calculate_contact_windows,
    calculate_downlink_delay,
    calculate_ttnc_ka,  # Backward compatibility alias
    calculate_raw_contact_windows,
)
from .data_model import (
    calculate_data_volume,
)
from .plots import (
    plot_ground_tracks,
    plot_coverage_map,
    plot_comm_cones,
    plot_access_statistics,
    plot_contact_validity,
    plot_delivery_timeline,
    plot_downlink_delay_distribution,
    plot_ttnc_distribution,  # Backward compatibility alias
)
from .reports import (
    generate_excel_report,
    generate_ppt_report,
)
from .optimization import (
    run_optimization,
    OptimizationResult,
    optimization_results_to_dataframe,
    identify_orbital_revolutions,
    calculate_ttc_contacts_per_revolution,
    calculate_delivery_times,
)
from .ground_station_db import (
    GROUND_STATION_DATABASE,
    get_stations_by_provider,
    get_ttc_capable_stations,
    get_ka_capable_stations,
    get_all_providers,
)
from .comparison import (
    ComparisonResult,
    load_comparison_results,
    load_comparison_result,
    load_raw_delay_data,
    comparison_to_dataframe,
    generate_comparison_excel,
    PROVIDER_COLORS,
    get_provider_color,
)
