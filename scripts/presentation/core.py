"""
Core utilities for presentation generation.

Shared functions for map setup, comm cones, and styling.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from shapely.geometry import Polygon

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vleo_eo.constants import (
    EARTH_RADIUS_KM,
    DEFAULT_LAND_COLOR,
    DEFAULT_OCEAN_COLOR,
    DEFAULT_BORDER_COLOR,
    DEFAULT_MIN_ELEVATION_SX_DEG,
    DEFAULT_MIN_ELEVATION_KA_DEG,
)
from src.vleo_eo.cache import get_cached_comm_cone


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    land_color: str = DEFAULT_LAND_COLOR
    ocean_color: str = DEFAULT_OCEAN_COLOR
    border_color: str = DEFAULT_BORDER_COLOR
    dpi: int = 150
    figsize: Tuple[float, float] = (16, 9)


# Cache for map features (loaded once)
_LAND_FEATURE = None
_OCEAN_FEATURE = None


def _get_map_features(land_color: str, ocean_color: str):
    """Get or create cached map features."""
    global _LAND_FEATURE, _OCEAN_FEATURE

    # Only cache if using default colors
    if land_color == DEFAULT_LAND_COLOR and ocean_color == DEFAULT_OCEAN_COLOR:
        if _LAND_FEATURE is None:
            _LAND_FEATURE = cfeature.NaturalEarthFeature(
                'physical', 'land', '50m', facecolor=land_color
            )
            _OCEAN_FEATURE = cfeature.NaturalEarthFeature(
                'physical', 'ocean', '50m', facecolor=ocean_color
            )
        return _LAND_FEATURE, _OCEAN_FEATURE

    # Create new features for custom colors
    return (
        cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=land_color),
        cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=ocean_color),
    )


def setup_map_axes(
    ax,
    land_color: str = DEFAULT_LAND_COLOR,
    ocean_color: str = DEFAULT_OCEAN_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
):
    """
    Setup map with standard styling.

    Parameters
    ----------
    ax : matplotlib axes
        Axes with cartopy projection.
    land_color : str
        Land fill color.
    ocean_color : str
        Ocean fill color.
    border_color : str
        Country border color.
    """
    land, ocean = _get_map_features(land_color, ocean_color)
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=border_color, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray', zorder=2)
    ax.gridlines(draw_labels=True, alpha=0.3)


def calculate_coverage_radius(sat_altitude_km: float, min_elevation_deg: float) -> float:
    """
    Calculate ground coverage radius in km.

    Parameters
    ----------
    sat_altitude_km : float
        Satellite altitude in km.
    min_elevation_deg : float
        Minimum elevation angle in degrees.

    Returns
    -------
    float
        Coverage radius in km.
    """
    elevation_rad = np.radians(min_elevation_deg)
    central_angle = np.arccos(
        EARTH_RADIUS_KM * np.cos(elevation_rad) / (EARTH_RADIUS_KM + sat_altitude_km)
    ) - elevation_rad
    return EARTH_RADIUS_KM * central_angle


def create_comm_cone_geodesic(
    lon: float,
    lat: float,
    min_elevation_deg: float,
    sat_altitude_km: float,
) -> Optional[Polygon]:
    """
    Create comm cone using geodesic geometry.

    Handles antimeridian crossing for stations near ±180° longitude.

    Parameters
    ----------
    lon, lat : float
        Ground station coordinates.
    min_elevation_deg : float
        Minimum elevation angle.
    sat_altitude_km : float
        Satellite altitude in km.

    Returns
    -------
    Optional[Polygon]
        Communication cone polygon, or None if creation fails.
    """
    try:
        elevation_rad = np.radians(min_elevation_deg)
        central_angle = np.arccos(
            EARTH_RADIUS_KM * np.cos(elevation_rad) / (EARTH_RADIUS_KM + sat_altitude_km)
        ) - elevation_rad
        max_ground_range_m = EARTH_RADIUS_KM * central_angle * 1000

        geod = Geod(ellps='WGS84')
        circle_lons = []
        circle_lats = []

        for azimuth in np.linspace(0, 360, 73):  # 5-degree increments
            dest_lon, dest_lat, _ = geod.fwd(lon, lat, azimuth, max_ground_range_m)
            circle_lons.append(dest_lon)
            circle_lats.append(dest_lat)

        # Normalize longitudes to avoid antimeridian discontinuity
        normalized_lons = []
        for clon in circle_lons:
            offset = clon - lon
            while offset > 180:
                offset -= 360
            while offset < -180:
                offset += 360
            normalized_lons.append(lon + offset)

        coords = list(zip(normalized_lons, circle_lats))
        poly = Polygon(coords)

        if not poly.is_valid:
            poly = poly.buffer(0)

        return poly

    except Exception:
        return None


def get_elevation_for_band(config, is_ka_band: bool) -> float:
    """
    Get the appropriate elevation mask from config based on band.

    Parameters
    ----------
    config : AnalysisConfig
        Configuration object with raw_config dict.
    is_ka_band : bool
        True for Ka-band (payload downlink), False for S/X-band (TT&C).

    Returns
    -------
    float
        Elevation mask in degrees.
    """
    if is_ka_band:
        return config.raw_config.get('min_elevation_ka_deg', DEFAULT_MIN_ELEVATION_KA_DEG)
    else:
        return config.raw_config.get('min_elevation_sx_deg', DEFAULT_MIN_ELEVATION_SX_DEG)


def plot_ground_stations_and_cones(
    ax,
    ground_stations: List[Dict],
    sat_alt_km: float,
    config=None,
):
    """
    Plot ground stations with communication cones.

    Parameters
    ----------
    ax : matplotlib axes
        Map axes.
    ground_stations : List[Dict]
        Ground station configurations.
    sat_alt_km : float
        Satellite altitude in km.
    config : AnalysisConfig, optional
        Configuration for elevation masks.
    """
    for gs in ground_stations:
        is_ka = gs.get('ka_band', False)
        marker_color = 'green' if is_ka else 'black'
        cone_color = 'green' if is_ka else 'black'

        # Use config elevation if available
        if config:
            cone_elev = get_elevation_for_band(config, is_ka)
        else:
            cone_elev = DEFAULT_MIN_ELEVATION_KA_DEG if is_ka else DEFAULT_MIN_ELEVATION_SX_DEG

        ax.plot(
            gs['lon'], gs['lat'], '^',
            color=marker_color, markersize=6,
            transform=ccrs.PlateCarree(), zorder=8
        )

        # Use cached comm cone
        comm_cone = get_cached_comm_cone(
            gs.get('name', f"{gs['lat']},{gs['lon']}"),
            gs['lon'], gs['lat'],
            cone_elev, sat_alt_km,
            create_fn=lambda lon, lat, elev, alt_m: create_comm_cone_geodesic(
                lon, lat, elev, alt_m / 1000
            ),
        )

        if comm_cone is not None and comm_cone.is_valid:
            ax.add_geometries(
                [comm_cone], crs=ccrs.PlateCarree(),
                facecolor='none', edgecolor=cone_color, alpha=0.5,
                linewidth=0.8, zorder=2
            )
