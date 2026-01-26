"""
Presentation generation package.

Provides modular slide generation with parallel execution support.
"""

from .core import (
    setup_map_axes,
    create_comm_cone_geodesic,
    calculate_coverage_radius,
    get_elevation_for_band,
    PlotConfig,
)
from .parallel import (
    generate_plots_parallel,
    PlotTask,
)

__all__ = [
    'setup_map_axes',
    'create_comm_cone_geodesic',
    'calculate_coverage_radius',
    'get_elevation_for_band',
    'PlotConfig',
    'generate_plots_parallel',
    'PlotTask',
]
