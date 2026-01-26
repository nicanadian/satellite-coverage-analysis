"""
RGT-Min Ground Station Optimization Module

Finds minimum ground station configurations that satisfy:
- TT&C SLA: ≥N contacts per orbital revolution per satellite (S/X band)
- Ka Payload SLA: ≥P% of collects delivered within T hours

Supports both exhaustive and greedy optimization approaches.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisConfig, GroundStationConfig
from .constants import EARTH_RADIUS_KM, EARTH_MU_KM3_S2

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of RGT-Min optimization."""
    provider: str
    ttc_stations: List[GroundStationConfig]
    ka_stations: List[GroundStationConfig]
    ttc_coverage_percent: float  # % of revolutions with ≥N contacts
    ka_delivery_percent: float   # % of collects delivered within time limit
    ttc_satisfied: bool
    ka_satisfied: bool
    optimization_approach: str
    iterations: int


def calculate_orbital_period(altitude_km: float) -> float:
    """
    Calculate orbital period in minutes for a circular orbit.

    Parameters
    ----------
    altitude_km : float
        Orbital altitude in kilometers.

    Returns
    -------
    float
        Orbital period in minutes.
    """
    semi_major_axis_km = EARTH_RADIUS_KM + altitude_km
    period_s = 2 * np.pi * np.sqrt(semi_major_axis_km**3 / EARTH_MU_KM3_S2)
    return period_s / 60.0


def identify_orbital_revolutions(
    orbit_df: pd.DataFrame,
    sat_id: int,
) -> pd.DataFrame:
    """
    Identify orbital revolutions by detecting ascending node crossings.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit propagation data with columns: Satellite, Epoch, Latitude, Longitude.
    sat_id : int
        Satellite ID to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with revolution boundaries: rev_num, start_time, end_time.
    """
    sat_data = orbit_df[orbit_df['Satellite'] == sat_id].copy()
    sat_data = sat_data.sort_values('Epoch').reset_index(drop=True)

    # Find ascending node crossings (latitude crosses 0 going north)
    lat = sat_data['Latitude'].values
    crossing_idx = []

    for i in range(1, len(lat)):
        if lat[i-1] < 0 and lat[i] >= 0:
            crossing_idx.append(i)

    # Build revolution boundaries
    revolutions = []
    for i in range(len(crossing_idx) - 1):
        start_idx = crossing_idx[i]
        end_idx = crossing_idx[i + 1]
        revolutions.append({
            'sat_id': sat_id,
            'rev_num': i + 1,
            'start_time': sat_data.iloc[start_idx]['Epoch'],
            'end_time': sat_data.iloc[end_idx]['Epoch'],
        })

    return pd.DataFrame(revolutions)


def calculate_ttc_contacts_per_revolution(
    contact_df: pd.DataFrame,
    revolutions_df: pd.DataFrame,
    ground_stations: List[GroundStationConfig],
    sat_id: int,
) -> pd.DataFrame:
    """
    Calculate TT&C contacts per orbital revolution.

    Parameters
    ----------
    contact_df : pd.DataFrame
        Contact windows with columns: sat_id, gs_name, contact_start, contact_end.
    revolutions_df : pd.DataFrame
        Revolution boundaries from identify_orbital_revolutions.
    ground_stations : List[GroundStationConfig]
        Ground stations to consider (should be TT&C capable).
    sat_id : int
        Satellite ID to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sat_id, rev_num, ttc_contacts, stations_contacted.
    """
    # Filter to TT&C-capable stations only
    ttc_gs_names = {gs.name for gs in ground_stations if gs.ttc_capable}

    sat_contacts = contact_df[
        (contact_df['sat_id'] == sat_id) &
        (contact_df['gs_name'].isin(ttc_gs_names))
    ].copy()

    sat_revs = revolutions_df[revolutions_df['sat_id'] == sat_id]

    results = []
    for _, rev in sat_revs.iterrows():
        # Find contacts within this revolution
        mask = (
            (sat_contacts['contact_start'] >= rev['start_time']) &
            (sat_contacts['contact_start'] < rev['end_time'])
        )
        rev_contacts = sat_contacts[mask]

        results.append({
            'sat_id': sat_id,
            'rev_num': rev['rev_num'],
            'ttc_contacts': len(rev_contacts),
            'stations_contacted': list(rev_contacts['gs_name'].unique()),
        })

    return pd.DataFrame(results)


def calculate_delivery_times(
    collect_df: pd.DataFrame,
    contact_df: pd.DataFrame,
    ground_stations: List[GroundStationConfig],
    downlink_rate_mbps: float = 1200.0,
) -> pd.DataFrame:
    """
    Calculate delivery time for each collection event.

    Delivery time = time from collect_end to when all data has been downlinked.

    Parameters
    ----------
    collect_df : pd.DataFrame
        Collection events with: sat_id, collect_start, collect_end, data_volume_gb.
    contact_df : pd.DataFrame
        Ka-band contact windows.
    ground_stations : List[GroundStationConfig]
        Ground stations (should be Ka-capable).
    downlink_rate_mbps : float
        Downlink data rate.

    Returns
    -------
    pd.DataFrame
        DataFrame with delivery_time_hours for each collection.
    """
    # Filter to Ka-capable stations
    ka_gs_names = {gs.name for gs in ground_stations if gs.ka_band}

    ka_contacts = contact_df[contact_df['gs_name'].isin(ka_gs_names)].copy()
    ka_contacts = ka_contacts.sort_values('contact_start')

    results = []

    for _, collect in collect_df.iterrows():
        collect_end = collect['collect_end']
        data_volume_gb = collect.get('data_volume_gb', 1.0)  # Default 1GB if not specified

        # Find next available Ka contacts after collection
        future_contacts = ka_contacts[
            (ka_contacts['sat_id'] == collect['sat_id']) &
            (ka_contacts['contact_start'] >= collect_end)
        ]

        if len(future_contacts) == 0:
            # No contacts found - delivery not possible
            results.append({
                **collect.to_dict(),
                'delivery_time_hours': np.nan,
                'delivered': False,
            })
            continue

        # Calculate time to downlink all data
        remaining_data_gb = data_volume_gb
        delivery_complete = None

        for _, contact in future_contacts.iterrows():
            contact_duration_s = (contact['contact_end'] - contact['contact_start']).total_seconds()
            downlink_capacity_gb = (downlink_rate_mbps / 8 / 1000) * contact_duration_s  # Convert to GB

            if remaining_data_gb <= downlink_capacity_gb:
                # Can complete delivery in this contact
                time_needed_s = remaining_data_gb / (downlink_rate_mbps / 8 / 1000)
                delivery_complete = contact['contact_start'] + timedelta(seconds=time_needed_s)
                break
            else:
                remaining_data_gb -= downlink_capacity_gb

        if delivery_complete is not None:
            delivery_time_hours = (delivery_complete - collect_end).total_seconds() / 3600
            results.append({
                **collect.to_dict(),
                'delivery_time_hours': delivery_time_hours,
                'delivered': True,
            })
        else:
            results.append({
                **collect.to_dict(),
                'delivery_time_hours': np.nan,
                'delivered': False,
            })

    return pd.DataFrame(results)


def evaluate_ttc_coverage(
    orbit_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    ground_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    required_contacts_per_rev: int = 1,
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate TT&C coverage for a set of ground stations.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit propagation data.
    tle_data : Dict[int, Tuple[str, str]]
        TLE data for satellites.
    ground_stations : List[GroundStationConfig]
        TT&C-capable ground stations to evaluate.
    config : AnalysisConfig
        Analysis configuration.
    required_contacts_per_rev : int
        Required TT&C contacts per revolution.

    Returns
    -------
    Tuple[float, pd.DataFrame]
        Coverage percentage and detailed per-revolution data.
    """
    from .contacts import calculate_raw_contact_windows

    # Calculate contact windows for these stations
    ttc_stations = [gs for gs in ground_stations if gs.ttc_capable]
    if not ttc_stations:
        return 0.0, pd.DataFrame()

    contact_df = calculate_raw_contact_windows(tle_data, ttc_stations, config)

    all_rev_data = []
    total_revs = 0
    satisfied_revs = 0

    for sat_id in tle_data.keys():
        revolutions = identify_orbital_revolutions(orbit_df, sat_id)
        ttc_per_rev = calculate_ttc_contacts_per_revolution(
            contact_df, revolutions, ttc_stations, sat_id
        )

        total_revs += len(ttc_per_rev)
        satisfied_revs += (ttc_per_rev['ttc_contacts'] >= required_contacts_per_rev).sum()
        all_rev_data.append(ttc_per_rev)

    coverage_percent = (satisfied_revs / total_revs * 100) if total_revs > 0 else 0.0
    combined_df = pd.concat(all_rev_data, ignore_index=True) if all_rev_data else pd.DataFrame()

    return coverage_percent, combined_df


def evaluate_ka_delivery(
    collect_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    ground_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    delivery_hours: float = 4.0,
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate Ka-band delivery performance for a set of ground stations.

    Parameters
    ----------
    collect_df : pd.DataFrame
        Collection events.
    tle_data : Dict[int, Tuple[str, str]]
        TLE data for satellites.
    ground_stations : List[GroundStationConfig]
        Ka-capable ground stations to evaluate.
    config : AnalysisConfig
        Analysis configuration.
    delivery_hours : float
        Delivery time limit in hours.

    Returns
    -------
    Tuple[float, pd.DataFrame]
        Delivery percentage within time limit and detailed delivery data.
    """
    from .contacts import calculate_raw_contact_windows

    # Calculate contact windows for Ka-capable stations
    ka_stations = [gs for gs in ground_stations if gs.ka_band]
    if not ka_stations:
        return 0.0, pd.DataFrame()

    contact_df = calculate_raw_contact_windows(tle_data, ka_stations, config)

    # Calculate delivery times
    delivery_df = calculate_delivery_times(
        collect_df, contact_df, ka_stations,
        downlink_rate_mbps=ka_stations[0].downlink_rate_mbps if ka_stations else 1200.0
    )

    # Calculate percentage within time limit
    delivered = delivery_df[delivery_df['delivered']]
    if len(delivered) == 0:
        return 0.0, delivery_df

    within_limit = (delivered['delivery_time_hours'] <= delivery_hours).sum()
    delivery_percent = within_limit / len(collect_df) * 100

    return delivery_percent, delivery_df


def greedy_ttc_optimization(
    orbit_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    candidate_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    required_contacts_per_rev: int = 1,
    target_coverage: float = 100.0,
) -> Tuple[List[GroundStationConfig], float, int]:
    """
    Greedy optimization for TT&C coverage.

    Iteratively adds the station that provides the greatest improvement
    until the coverage target is met.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit propagation data.
    tle_data : Dict[int, Tuple[str, str]]
        TLE data for satellites.
    candidate_stations : List[GroundStationConfig]
        Candidate stations to consider.
    config : AnalysisConfig
        Analysis configuration.
    required_contacts_per_rev : int
        Required contacts per revolution.
    target_coverage : float
        Target coverage percentage (0-100).

    Returns
    -------
    Tuple[List[GroundStationConfig], float, int]
        Selected stations, achieved coverage, and number of iterations.
    """
    selected_stations = []
    remaining_stations = list(candidate_stations)
    current_coverage = 0.0
    iterations = 0

    while current_coverage < target_coverage and remaining_stations:
        best_station = None
        best_improvement = -1
        best_coverage = current_coverage

        for station in remaining_stations:
            test_stations = selected_stations + [station]
            coverage, _ = evaluate_ttc_coverage(
                orbit_df, tle_data, test_stations, config, required_contacts_per_rev
            )
            improvement = coverage - current_coverage

            if improvement > best_improvement:
                best_improvement = improvement
                best_station = station
                best_coverage = coverage

        if best_station is None or best_improvement <= 0:
            break

        selected_stations.append(best_station)
        remaining_stations.remove(best_station)
        current_coverage = best_coverage
        iterations += 1

        logger.info(f"TT&C Greedy iteration {iterations}: Added {best_station.name}, "
                   f"coverage now {current_coverage:.1f}%")

    return selected_stations, current_coverage, iterations


def greedy_ka_optimization(
    collect_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    candidate_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    delivery_hours: float = 4.0,
    target_percent: float = 95.0,
) -> Tuple[List[GroundStationConfig], float, int]:
    """
    Greedy optimization for Ka-band delivery.

    Parameters
    ----------
    collect_df : pd.DataFrame
        Collection events.
    tle_data : Dict[int, Tuple[str, str]]
        TLE data for satellites.
    candidate_stations : List[GroundStationConfig]
        Candidate Ka-capable stations.
    config : AnalysisConfig
        Analysis configuration.
    delivery_hours : float
        Delivery time limit.
    target_percent : float
        Target delivery percentage.

    Returns
    -------
    Tuple[List[GroundStationConfig], float, int]
        Selected stations, achieved percentage, and iterations.
    """
    selected_stations = []
    remaining_stations = list(candidate_stations)
    current_percent = 0.0
    iterations = 0

    while current_percent < target_percent and remaining_stations:
        best_station = None
        best_improvement = -1
        best_percent = current_percent

        for station in remaining_stations:
            test_stations = selected_stations + [station]
            percent, _ = evaluate_ka_delivery(
                collect_df, tle_data, test_stations, config, delivery_hours
            )
            improvement = percent - current_percent

            if improvement > best_improvement:
                best_improvement = improvement
                best_station = station
                best_percent = percent

        if best_station is None or best_improvement <= 0:
            break

        selected_stations.append(best_station)
        remaining_stations.remove(best_station)
        current_percent = best_percent
        iterations += 1

        logger.info(f"Ka Greedy iteration {iterations}: Added {best_station.name}, "
                   f"delivery now {current_percent:.1f}%")

    return selected_stations, current_percent, iterations


def exhaustive_ttc_optimization(
    orbit_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    candidate_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    required_contacts_per_rev: int = 1,
    target_coverage: float = 100.0,
    max_stations: int = 15,
) -> Tuple[List[GroundStationConfig], float, int]:
    """
    Exhaustive search for minimum TT&C stations.

    Tries all combinations starting from size 1 until target is met.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit propagation data.
    tle_data : Dict[int, Tuple[str, str]]
        TLE data for satellites.
    candidate_stations : List[GroundStationConfig]
        Candidate stations.
    config : AnalysisConfig
        Analysis configuration.
    required_contacts_per_rev : int
        Required contacts per revolution.
    target_coverage : float
        Target coverage percentage.
    max_stations : int
        Maximum stations to consider (limits combinatorial explosion).

    Returns
    -------
    Tuple[List[GroundStationConfig], float, int]
        Selected stations, achieved coverage, and combinations tried.
    """
    # Limit candidates if too many
    if len(candidate_stations) > max_stations:
        logger.warning(f"Limiting exhaustive search to {max_stations} stations")
        candidate_stations = candidate_stations[:max_stations]

    best_stations = []
    best_coverage = 0.0
    total_iterations = 0

    for size in range(1, len(candidate_stations) + 1):
        found_solution = False

        for combo in combinations(candidate_stations, size):
            total_iterations += 1
            coverage, _ = evaluate_ttc_coverage(
                orbit_df, tle_data, list(combo), config, required_contacts_per_rev
            )

            if coverage >= target_coverage:
                logger.info(f"TT&C Exhaustive: Found solution with {size} stations, "
                           f"coverage {coverage:.1f}%")
                return list(combo), coverage, total_iterations

            if coverage > best_coverage:
                best_coverage = coverage
                best_stations = list(combo)

        if total_iterations > 10000:
            logger.warning("Exhaustive search exceeded 10000 iterations, stopping")
            break

    return best_stations, best_coverage, total_iterations


def exhaustive_ka_optimization(
    collect_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    candidate_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    delivery_hours: float = 4.0,
    target_percent: float = 95.0,
    max_stations: int = 15,
) -> Tuple[List[GroundStationConfig], float, int]:
    """
    Exhaustive search for minimum Ka stations.
    """
    if len(candidate_stations) > max_stations:
        logger.warning(f"Limiting exhaustive search to {max_stations} stations")
        candidate_stations = candidate_stations[:max_stations]

    best_stations = []
    best_percent = 0.0
    total_iterations = 0

    for size in range(1, len(candidate_stations) + 1):
        for combo in combinations(candidate_stations, size):
            total_iterations += 1
            percent, _ = evaluate_ka_delivery(
                collect_df, tle_data, list(combo), config, delivery_hours
            )

            if percent >= target_percent:
                logger.info(f"Ka Exhaustive: Found solution with {size} stations, "
                           f"delivery {percent:.1f}%")
                return list(combo), percent, total_iterations

            if percent > best_percent:
                best_percent = percent
                best_stations = list(combo)

        if total_iterations > 10000:
            logger.warning("Exhaustive search exceeded 10000 iterations, stopping")
            break

    return best_stations, best_percent, total_iterations


def run_optimization(
    orbit_df: pd.DataFrame,
    collect_df: pd.DataFrame,
    tle_data: Dict[int, Tuple[str, str]],
    config: AnalysisConfig,
) -> Dict[str, OptimizationResult]:
    """
    Run RGT-Min optimization for all configured providers.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit propagation data.
    collect_df : pd.DataFrame
        Collection events.
    tle_data : Dict[int, Tuple[str, str]]
        TLE data for satellites.
    config : AnalysisConfig
        Analysis configuration with optimization settings.

    Returns
    -------
    Dict[str, OptimizationResult]
        Optimization results keyed by provider name.
    """
    opt_config = config.optimization

    if not opt_config.enabled:
        logger.info("Optimization disabled in config")
        return {}

    # Get candidate stations from config
    all_stations = config.ground_stations

    # Filter to TT&C-capable and Ka-capable stations
    ttc_candidates = [gs for gs in all_stations if gs.ttc_capable]
    ka_candidates = [gs for gs in all_stations if gs.ka_band]

    if not ttc_candidates:
        logger.warning("No TT&C-capable stations found in config")
        return {}

    # Determine provider name for results (use config provider or 'config')
    provider_names = list(set(gs.provider for gs in all_stations if gs.provider))
    provider_label = provider_names[0] if len(provider_names) == 1 else 'config'

    logger.info(f"Running optimization with {len(ttc_candidates)} TT&C and {len(ka_candidates)} Ka stations from config")

    # Run TT&C optimization
    if opt_config.approach == 'exhaustive':
        ttc_stations, ttc_coverage, ttc_iters = exhaustive_ttc_optimization(
            orbit_df, tle_data, ttc_candidates, config,
            required_contacts_per_rev=opt_config.ttc_contacts_per_rev,
            max_stations=opt_config.max_exhaustive_stations,
        )
    else:
        ttc_stations, ttc_coverage, ttc_iters = greedy_ttc_optimization(
            orbit_df, tle_data, ttc_candidates, config,
            required_contacts_per_rev=opt_config.ttc_contacts_per_rev,
        )

    # Run Ka optimization (only if we have Ka candidates and collections)
    ka_stations = []
    ka_percent = 0.0
    ka_iters = 0

    if ka_candidates and len(collect_df) > 0:
        if opt_config.approach == 'exhaustive':
            ka_stations, ka_percent, ka_iters = exhaustive_ka_optimization(
                collect_df, tle_data, ka_candidates, config,
                delivery_hours=opt_config.ka_delivery_hours,
                target_percent=opt_config.ka_delivery_percent,
                max_stations=opt_config.max_exhaustive_stations,
            )
        else:
            ka_stations, ka_percent, ka_iters = greedy_ka_optimization(
                collect_df, tle_data, ka_candidates, config,
                delivery_hours=opt_config.ka_delivery_hours,
                target_percent=opt_config.ka_delivery_percent,
            )

    result = OptimizationResult(
        provider=provider_label,
        ttc_stations=ttc_stations,
        ka_stations=ka_stations,
        ttc_coverage_percent=ttc_coverage,
        ka_delivery_percent=ka_percent,
        ttc_satisfied=ttc_coverage >= 100.0,  # All revs have required contacts
        ka_satisfied=ka_percent >= opt_config.ka_delivery_percent,
        optimization_approach=opt_config.approach,
        iterations=ttc_iters + ka_iters,
    )

    logger.info(f"Optimization Results:")
    logger.info(f"  TT&C: {len(ttc_stations)} stations, {ttc_coverage:.1f}% coverage")
    logger.info(f"  Ka: {len(ka_stations)} stations, {ka_percent:.1f}% delivery")

    return {provider_label: result}


def optimization_results_to_dataframe(results: Dict[str, OptimizationResult]) -> pd.DataFrame:
    """Convert optimization results to a summary DataFrame."""
    rows = []
    for provider, result in results.items():
        rows.append({
            'Provider': provider,
            'Approach': result.optimization_approach,
            'TT&C Stations': len(result.ttc_stations),
            'TT&C Station Names': ', '.join(s.name for s in result.ttc_stations),
            'TT&C Coverage %': result.ttc_coverage_percent,
            'TT&C Satisfied': result.ttc_satisfied,
            'Ka Stations': len(result.ka_stations),
            'Ka Station Names': ', '.join(s.name for s in result.ka_stations),
            'Ka Delivery %': result.ka_delivery_percent,
            'Ka Satisfied': result.ka_satisfied,
            'Total Iterations': result.iterations,
        })
    return pd.DataFrame(rows)
