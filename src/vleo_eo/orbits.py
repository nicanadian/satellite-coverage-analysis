"""
Orbit propagation and TLE generation utilities.

Uses Skyfield for high-fidelity orbit propagation, matching the approach
from the original notebook.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite, wgs84


@dataclass
class TLEData:
    """Container for TLE data and derived orbital parameters."""
    sat_id: int
    line1: str
    line2: str
    inclination_deg: float
    altitude_km: float
    raan_deg: float
    mean_motion_rev_per_day: float


def calculate_checksum(line: str) -> int:
    """
    Calculate the checksum for a TLE line.

    Parameters
    ----------
    line : str
        TLE line (without checksum digit).

    Returns
    -------
    int
        Checksum digit (0-9).
    """
    checksum = 0
    for i in range(min(68, len(line))):
        char = line[i]
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10


def generate_tle(
    epoch: datetime,
    inclination_deg: float,
    altitude_km: float,
    sat_num: int,
    raan_deg: float = 0.0,
    no_drag: bool = False,
) -> Tuple[str, str]:
    """
    Generate a TLE for a satellite given basic orbital parameters.

    Assumes a circular orbit.

    Parameters
    ----------
    epoch : datetime
        Epoch time for the TLE (should be timezone-aware UTC).
    inclination_deg : float
        Orbital inclination in degrees.
    altitude_km : float
        Mean orbital altitude in kilometers.
    sat_num : int
        Satellite catalog number.
    raan_deg : float, optional
        Right Ascension of Ascending Node in degrees (default 0).
    no_drag : bool, optional
        If True, set BSTAR drag term to 0 for idealized propagation
        without atmospheric drag decay (default False).

    Returns
    -------
    tuple of (str, str)
        The two lines of the TLE.
    """
    # Constants
    earth_radius_km = 6378.137
    earth_mu = 398600.4418  # km^3/s^2

    # Calculate semi-major axis and mean motion
    semi_major_axis = earth_radius_km + altitude_km
    mean_motion = np.sqrt(earth_mu / semi_major_axis**3) * 86400 / (2 * np.pi)  # revs per day

    # Convert epoch to TLE format
    year = epoch.year % 100
    day_of_year = epoch.timetuple().tm_yday
    # Add fractional day
    fraction = (epoch.hour * 3600 + epoch.minute * 60 + epoch.second) / 86400.0
    epoch_str = f"{year:02d}{day_of_year + fraction:012.8f}"

    # Format satellite number
    sat_num_str = f"{sat_num:05d}"

    # BSTAR drag term: 0 for no drag, otherwise default value for VLEO
    bstar = "00000-0" if no_drag else "50000-4"

    # Line 1
    line1 = f"1 {sat_num_str}U 20001A   {epoch_str} .00000000  00000-0  {bstar} 0  999"
    line1 = line1 + str(calculate_checksum(line1))

    # Line 2
    line2 = f"2 {sat_num_str} {inclination_deg:8.4f} {raan_deg:8.4f} 0001000   0.0000   0.0000 {mean_motion:11.8f}    1"
    line2 = line2 + str(calculate_checksum(line2))

    return line1, line2


def create_tle_data(
    sat_id: int,
    inclination_deg: float,
    altitude_km: float,
    raan_deg: float,
    epoch: datetime,
    tle_line1: Optional[str] = None,
    tle_line2: Optional[str] = None,
    no_drag: bool = False,
) -> TLEData:
    """
    Create TLE data for a satellite.

    If TLE lines are provided, uses those directly. Otherwise generates
    TLE from orbital parameters.

    Parameters
    ----------
    sat_id : int
        Satellite ID.
    inclination_deg : float
        Inclination in degrees.
    altitude_km : float
        Altitude in km.
    raan_deg : float
        RAAN in degrees.
    epoch : datetime
        TLE epoch.
    tle_line1 : str, optional
        Pre-existing TLE line 1.
    tle_line2 : str, optional
        Pre-existing TLE line 2.
    no_drag : bool, optional
        If True, generate TLE with BSTAR=0 for no atmospheric drag (default False).

    Returns
    -------
    TLEData
        TLE data container.
    """
    if tle_line1 and tle_line2:
        line1, line2 = tle_line1, tle_line2
    else:
        line1, line2 = generate_tle(epoch, inclination_deg, altitude_km, sat_id, raan_deg, no_drag=no_drag)

    # Calculate mean motion from orbital parameters
    earth_radius_km = 6378.137
    earth_mu = 398600.4418
    semi_major_axis = earth_radius_km + altitude_km
    mean_motion = np.sqrt(earth_mu / semi_major_axis**3) * 86400 / (2 * np.pi)

    return TLEData(
        sat_id=sat_id,
        line1=line1,
        line2=line2,
        inclination_deg=inclination_deg,
        altitude_km=altitude_km,
        raan_deg=raan_deg,
        mean_motion_rev_per_day=mean_motion,
    )


def propagate_orbits(
    tle_data: Dict[int, TLEData],
    start_time: datetime,
    end_time: datetime,
    time_step_s: float = 10.0,
    satellite_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Propagate satellite orbits over a time window.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        Dictionary mapping satellite IDs to TLE data.
    start_time : datetime
        Start time (UTC).
    end_time : datetime
        End time (UTC).
    time_step_s : float
        Time step in seconds.
    satellite_ids : List[int], optional
        Specific satellite IDs to propagate. If None, propagates all.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Satellite, Epoch, Latitude, Longitude, Altitude_km
    """
    ts = load.timescale()

    # Convert times to Skyfield format
    t0 = ts.from_datetime(start_time)
    tf = ts.from_datetime(end_time)
    num_steps = int((end_time - start_time).total_seconds() / time_step_s)
    times = ts.linspace(t0, tf, num_steps)

    # Select satellites
    if satellite_ids is None:
        satellite_ids = list(tle_data.keys())

    orbit_points = []

    for sat_id in satellite_ids:
        if sat_id not in tle_data:
            continue

        tle = tle_data[sat_id]
        satellite = EarthSatellite(tle.line1, tle.line2, str(sat_id), ts)

        # Propagate (vectorized)
        geocentric = satellite.at(times)
        subpoint = geocentric.subpoint()

        lats = subpoint.latitude.degrees
        lons = subpoint.longitude.degrees
        distances = geocentric.distance().km
        altitudes = distances - 6378.135  # Approximate altitude

        # Build records
        for t, lat, lon, alt in zip(times, lats, lons, altitudes):
            orbit_points.append({
                'Satellite': sat_id,
                'Epoch': t.utc_datetime(),
                'Latitude': lat,
                'Longitude': lon,
                'Altitude_km': alt,
            })

    df = pd.DataFrame(orbit_points)
    if not df.empty:
        df['Epoch'] = pd.to_datetime(df['Epoch'], utc=True)

    return df


def calculate_ground_track(
    tle_data: Dict[int, TLEData],
    sat_id: int,
    start_time: datetime,
    duration: timedelta = timedelta(minutes=10),
    step_s: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Calculate satellite ground track for a given time period.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        Dictionary mapping satellite IDs to TLE data.
    sat_id : int
        Satellite ID.
    start_time : datetime
        Start time.
    duration : timedelta
        Duration to calculate ground track.
    step_s : float
        Time step in seconds.

    Returns
    -------
    List[Tuple[float, float]]
        List of (longitude, latitude) tuples representing ground track.
    """
    if sat_id not in tle_data:
        raise KeyError(f"Satellite ID {sat_id} not found in TLE data")

    # Ensure timezone-aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    tle = tle_data[sat_id]
    ts = load.timescale()

    # Generate time array
    end_time = start_time + duration
    times = pd.date_range(start=start_time, end=end_time, freq=f'{step_s}s')

    # Convert to Skyfield times
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = ts.from_datetimes([pd.Timestamp(t).to_pydatetime() for t in times])

    # Get satellite positions
    satellite = EarthSatellite(tle.line1, tle.line2, str(sat_id), ts)
    positions = satellite.at(t)

    # Convert to geographic coordinates
    subpoints = wgs84.geographic_position_of(positions)
    lats = subpoints.latitude.degrees
    lons = subpoints.longitude.degrees

    return list(zip(lons, lats))


def get_satellite_position(
    tle_data: Dict[int, TLEData],
    sat_id: int,
    time: datetime,
) -> Tuple[float, float, float]:
    """
    Get satellite position at a specific time.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        Dictionary mapping satellite IDs to TLE data.
    sat_id : int
        Satellite ID.
    time : datetime
        Time (timezone-aware UTC).

    Returns
    -------
    Tuple[float, float, float]
        (latitude, longitude, altitude_km) of satellite.
    """
    if sat_id not in tle_data:
        raise KeyError(f"Satellite ID {sat_id} not found in TLE data")

    # Ensure timezone-aware
    if isinstance(time, pd.Timestamp):
        time = time.to_pydatetime()
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)

    tle = tle_data[sat_id]
    ts = load.timescale()

    t = ts.from_datetime(time)
    satellite = EarthSatellite(tle.line1, tle.line2, str(sat_id), ts)

    geocentric = satellite.at(t)
    subpoint = geocentric.subpoint()

    lat = subpoint.latitude.degrees
    lon = subpoint.longitude.degrees
    alt = subpoint.elevation.km

    return lat, lon, alt


def get_orbital_period_minutes(altitude_km: float) -> float:
    """
    Calculate orbital period for a circular orbit.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude in kilometers.

    Returns
    -------
    float
        Orbital period in minutes.
    """
    earth_radius_km = 6378.137
    earth_mu = 398600.4418  # km^3/s^2

    semi_major_axis = earth_radius_km + altitude_km
    period_s = 2 * np.pi * np.sqrt(semi_major_axis**3 / earth_mu)
    return period_s / 60.0


def validate_propagation(
    orbit_df: pd.DataFrame,
    expected_alt_range: Tuple[float, float] = (200, 600),
    expected_inc_range: Tuple[float, float] = (0, 100),
) -> Dict[str, Any]:
    """
    Run sanity checks on propagated orbit data.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Propagated orbit DataFrame.
    expected_alt_range : Tuple[float, float]
        Expected altitude range (min, max) in km for VLEO.
    expected_inc_range : Tuple[float, float]
        Expected latitude range (proxy for inclination check).

    Returns
    -------
    Dict[str, Any]
        Validation results with 'passed', 'errors', and 'warnings'.
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    if orbit_df.empty:
        results['passed'] = False
        results['errors'].append("Orbit DataFrame is empty")
        return results

    # Check for finite values
    if orbit_df[['Latitude', 'Longitude', 'Altitude_km']].isnull().any().any():
        results['passed'] = False
        results['errors'].append("NaN values found in position data")

    if not np.isfinite(orbit_df[['Latitude', 'Longitude', 'Altitude_km']].values).all():
        results['passed'] = False
        results['errors'].append("Non-finite values found in position data")

    # Check altitude range
    alt_min = orbit_df['Altitude_km'].min()
    alt_max = orbit_df['Altitude_km'].max()
    results['stats']['altitude_min_km'] = alt_min
    results['stats']['altitude_max_km'] = alt_max

    if alt_min < expected_alt_range[0] or alt_max > expected_alt_range[1]:
        results['warnings'].append(
            f"Altitude range [{alt_min:.1f}, {alt_max:.1f}] km outside expected "
            f"VLEO range [{expected_alt_range[0]}, {expected_alt_range[1]}] km"
        )

    # Check latitude range (sanity check)
    lat_min = orbit_df['Latitude'].min()
    lat_max = orbit_df['Latitude'].max()
    results['stats']['latitude_min'] = lat_min
    results['stats']['latitude_max'] = lat_max

    if lat_min < -90 or lat_max > 90:
        results['passed'] = False
        results['errors'].append(f"Invalid latitude range: [{lat_min}, {lat_max}]")

    # Check longitude range
    lon_min = orbit_df['Longitude'].min()
    lon_max = orbit_df['Longitude'].max()
    results['stats']['longitude_min'] = lon_min
    results['stats']['longitude_max'] = lon_max

    if lon_min < -180 or lon_max > 180:
        results['passed'] = False
        results['errors'].append(f"Invalid longitude range: [{lon_min}, {lon_max}]")

    # Check number of satellites
    n_sats = orbit_df['Satellite'].nunique()
    results['stats']['num_satellites'] = n_sats

    # Check time span
    time_span = (orbit_df['Epoch'].max() - orbit_df['Epoch'].min()).total_seconds() / 3600
    results['stats']['time_span_hours'] = time_span

    print(f"Orbit validation: {n_sats} satellites, {time_span:.1f} hours")
    print(f"  Altitude: {alt_min:.1f} - {alt_max:.1f} km")
    print(f"  Latitude: {lat_min:.1f} - {lat_max:.1f} deg")

    if results['errors']:
        print(f"  ERRORS: {results['errors']}")
    if results['warnings']:
        print(f"  Warnings: {results['warnings']}")

    return results


def ltdn_to_raan(ltdn_hours: float, epoch: datetime) -> float:
    """
    Convert Local Time of Descending Node to RAAN.

    For a sun-synchronous orbit, RAAN is related to the local solar time
    at which the satellite crosses the equator going southward (descending node).

    Parameters
    ----------
    ltdn_hours : float
        Local Time of Descending Node in hours (e.g., 10.5 for 10:30 AM).
    epoch : datetime
        Mission start date/time.

    Returns
    -------
    float
        RAAN in degrees.
    """
    # Days since J2000.0 (Jan 1, 2000, 12:00 TT)
    if epoch.tzinfo is not None:
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    else:
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
    days_since_j2000 = (epoch - j2000).total_seconds() / 86400

    # Mean longitude of the Sun (simplified)
    L_sun = 280.46 + 0.9856474 * days_since_j2000
    L_sun = L_sun % 360

    # Mean anomaly of the Sun
    g_sun = 357.528 + 0.9856003 * days_since_j2000
    g_sun_rad = np.radians(g_sun)

    # Ecliptic longitude of the Sun
    lambda_sun = L_sun + 1.915 * np.sin(g_sun_rad) + 0.020 * np.sin(2 * g_sun_rad)

    # Right ascension of the Sun (simplified)
    ra_sun = lambda_sun % 360

    # RAAN from LTDN: RAAN = RA_sun + (LTDN - 12) * 15 degrees
    raan = ra_sun + (ltdn_hours - 12) * 15
    raan = raan % 360

    return raan


def calculate_sso_inclination(altitude_km: float) -> float:
    """
    Calculate sun-synchronous inclination for given altitude.

    For sun-synchronous orbit, the nodal precession rate must equal
    the Earth's mean motion around the Sun (~0.9856 deg/day).

    Parameters
    ----------
    altitude_km : float
        Orbital altitude in kilometers.

    Returns
    -------
    float
        Required inclination in degrees.
    """
    J2 = 1.08263e-3
    Re = 6378.137  # km
    mu = 398600.4418  # km^3/s^2

    a = Re + altitude_km  # Semi-major axis (km)
    n = np.sqrt(mu / a**3) * 86400  # Mean motion (rad/day)

    # Required precession rate for SSO: 0.9856 deg/day
    omega_dot_required = np.radians(0.9856)  # rad/day

    # Solve for cos(i)
    cos_i = -omega_dot_required / (1.5 * n * J2 * (Re/a)**2)
    cos_i = np.clip(cos_i, -1, 1)

    inclination = np.degrees(np.arccos(cos_i))

    return inclination
