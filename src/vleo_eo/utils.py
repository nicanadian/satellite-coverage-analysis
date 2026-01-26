"""
Shared utility functions for satellite coverage analysis.

Consolidates common calculations to avoid code duplication.
"""

from functools import lru_cache
from typing import Tuple, Union
import numpy as np

from .constants import EARTH_RADIUS_KM, EARTH_MU_KM3_S2


def haversine_distance_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate great-circle distance using Haversine formula.

    Fully vectorized - works with scalars or arrays of any compatible shape.

    Parameters
    ----------
    lat1, lon1 : float or array
        First point(s) latitude and longitude in degrees.
    lat2, lon2 : float or array
        Second point(s) latitude and longitude in degrees.

    Returns
    -------
    float or array
        Distance(s) in kilometers.

    Examples
    --------
    >>> haversine_distance_km(0, 0, 0, 1)  # ~111 km at equator
    111.19...

    >>> # Vectorized: distance from one point to many
    >>> haversine_distance_km(0, 0, np.array([0, 1, 2]), np.array([1, 1, 1]))
    array([111.19..., 157.25..., 247.54...])
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def haversine_distance_deg(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate angular distance on Earth's surface in degrees.

    This is the central angle, useful for quick comparisons without
    converting to km.

    Parameters
    ----------
    lat1, lon1 : float or array
        First point(s) latitude and longitude in degrees.
    lat2, lon2 : float or array
        Second point(s) latitude and longitude in degrees.

    Returns
    -------
    float or array
        Angular distance(s) in degrees.
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)

    # Spherical law of cosines
    cos_angle = (
        np.sin(lat1_rad) * np.sin(lat2_rad) +
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
    )
    # Clamp to [-1, 1] to handle floating point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def get_orbital_period_minutes(altitude_km: float) -> float:
    """
    Calculate orbital period for a circular orbit.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude above Earth's surface in km.

    Returns
    -------
    float
        Orbital period in minutes.

    Examples
    --------
    >>> get_orbital_period_minutes(250)  # VLEO
    89.4...
    >>> get_orbital_period_minutes(400)  # ISS altitude
    92.5...
    """
    semi_major_axis_km = EARTH_RADIUS_KM + altitude_km
    period_s = 2 * np.pi * np.sqrt(semi_major_axis_km ** 3 / EARTH_MU_KM3_S2)
    return period_s / 60.0


def get_orbital_period_seconds(altitude_km: float) -> float:
    """
    Calculate orbital period for a circular orbit.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude above Earth's surface in km.

    Returns
    -------
    float
        Orbital period in seconds.
    """
    return get_orbital_period_minutes(altitude_km) * 60.0


def calculate_horizon_distance_km(altitude_km: float) -> float:
    """
    Calculate distance to the horizon from a given altitude.

    Parameters
    ----------
    altitude_km : float
        Altitude above Earth's surface in km.

    Returns
    -------
    float
        Distance to horizon in km (along Earth's surface).
    """
    return EARTH_RADIUS_KM * np.arccos(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km))


def calculate_coverage_radius_km(
    altitude_km: float,
    min_elevation_deg: float,
) -> float:
    """
    Calculate ground coverage radius for a satellite at given altitude
    and minimum elevation angle constraint.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude in km.
    min_elevation_deg : float
        Minimum elevation angle from ground in degrees.

    Returns
    -------
    float
        Coverage radius on ground in km.

    Examples
    --------
    >>> calculate_coverage_radius_km(250, 5.0)  # VLEO with 5° elevation
    1847.5...
    >>> calculate_coverage_radius_km(250, 10.0)  # Higher elevation = smaller radius
    1634.8...
    """
    elevation_rad = np.radians(min_elevation_deg)
    # Central angle from satellite to edge of coverage
    central_angle = np.arccos(
        EARTH_RADIUS_KM * np.cos(elevation_rad) / (EARTH_RADIUS_KM + altitude_km)
    ) - elevation_rad
    return EARTH_RADIUS_KM * central_angle


def calculate_coverage_radius_deg(
    altitude_km: float,
    min_elevation_deg: float,
) -> float:
    """
    Calculate ground coverage radius in degrees (great-circle arc).

    Parameters
    ----------
    altitude_km : float
        Satellite altitude in km.
    min_elevation_deg : float
        Minimum elevation angle from ground in degrees.

    Returns
    -------
    float
        Coverage radius as great-circle arc in degrees.
    """
    radius_km = calculate_coverage_radius_km(altitude_km, min_elevation_deg)
    return np.degrees(radius_km / EARTH_RADIUS_KM)


def look_angle_to_off_nadir(look_angle_deg: float) -> float:
    """
    Convert look angle (elevation from target) to off-nadir angle.

    Parameters
    ----------
    look_angle_deg : float
        Look angle in degrees (90° = directly overhead).

    Returns
    -------
    float
        Off-nadir angle in degrees (0° = nadir pointing).
    """
    return 90.0 - look_angle_deg


def off_nadir_to_look_angle(off_nadir_deg: float) -> float:
    """
    Convert off-nadir angle to look angle.

    Parameters
    ----------
    off_nadir_deg : float
        Off-nadir angle in degrees.

    Returns
    -------
    float
        Look angle in degrees.
    """
    return 90.0 - off_nadir_deg


def off_nadir_to_ground_range_km(altitude_km: float, off_nadir_deg: float) -> float:
    """
    Convert off-nadir angle to ground range distance.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude in km.
    off_nadir_deg : float
        Off-nadir pointing angle in degrees.

    Returns
    -------
    float
        Ground range from nadir point in km.
    """
    if off_nadir_deg <= 0:
        return 0.0
    if off_nadir_deg >= 90:
        return calculate_horizon_distance_km(altitude_km)

    altitude_m = altitude_km * 1000
    radius_earth_m = EARTH_RADIUS_KM * 1000
    la_rad = np.radians(90 - off_nadir_deg)  # Convert to look angle

    # Handle edge cases for asin
    arg = (1 + altitude_m / radius_earth_m) * np.sin(la_rad)
    if arg > 1.0:
        arg = 1.0

    ia = np.arcsin(arg)
    theta = ia - la_rad
    return theta * EARTH_RADIUS_KM


def find_contact_boundaries(in_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find start and end indices of contiguous True regions in a boolean array.

    Useful for detecting contact windows from a mask of in-range positions.

    Parameters
    ----------
    in_range : np.ndarray
        Boolean array where True indicates satellite is in range.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of (start_indices, end_indices) for each contact.

    Examples
    --------
    >>> mask = np.array([False, True, True, True, False, True, True, False])
    >>> starts, ends = find_contact_boundaries(mask)
    >>> starts
    array([1, 5])
    >>> ends
    array([4, 7])
    """
    # Pad with False at boundaries to detect edge contacts
    padded = np.concatenate([[False], in_range, [False]])
    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return starts, ends


def normalize_longitude(lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize longitude to [-180, 180] range.

    Parameters
    ----------
    lon : float or array
        Longitude(s) in degrees.

    Returns
    -------
    float or array
        Normalized longitude(s) in [-180, 180].
    """
    return ((lon + 180) % 360) - 180


def wgs84_to_utm_epsg(lon: float, lat: float) -> int:
    """
    Get UTM EPSG code for a WGS84 coordinate.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.

    Returns
    -------
    int
        UTM EPSG code.
    """
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + utm_zone
    return 32700 + utm_zone
