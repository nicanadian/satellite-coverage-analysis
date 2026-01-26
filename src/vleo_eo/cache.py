"""
Caching utilities for expensive computations.

Provides in-memory and optional disk caching for:
- Communication cone geometries
- Orbit propagation results
- Ground track calculations
"""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon


# =============================================================================
# In-Memory Caches
# =============================================================================

# Communication cone cache: keyed by (station_name, altitude_km, elevation_deg)
_COMM_CONE_CACHE: Dict[Tuple[str, float, float], Polygon] = {}


def get_cached_comm_cone(
    station_name: str,
    lon: float,
    lat: float,
    min_elevation_deg: float,
    sat_altitude_km: float,
    create_fn: Callable,
) -> Optional[Polygon]:
    """
    Get or create a cached communication cone.

    Parameters
    ----------
    station_name : str
        Ground station identifier.
    lon, lat : float
        Station coordinates.
    min_elevation_deg : float
        Minimum elevation angle.
    sat_altitude_km : float
        Satellite altitude in km.
    create_fn : Callable
        Function to create the cone if not cached.
        Should accept (lon, lat, min_elevation_deg, sat_altitude_m).

    Returns
    -------
    Optional[Polygon]
        Communication cone polygon, or None if creation fails.
    """
    # Round altitude to avoid cache misses from floating point differences
    alt_key = round(sat_altitude_km, 1)
    elev_key = round(min_elevation_deg, 1)
    cache_key = (station_name, alt_key, elev_key)

    if cache_key not in _COMM_CONE_CACHE:
        try:
            cone = create_fn(lon, lat, min_elevation_deg, sat_altitude_km * 1000)
            _COMM_CONE_CACHE[cache_key] = cone
        except Exception:
            return None

    return _COMM_CONE_CACHE.get(cache_key)


def clear_comm_cone_cache():
    """Clear the communication cone cache."""
    global _COMM_CONE_CACHE
    _COMM_CONE_CACHE.clear()


def get_comm_cone_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        'comm_cone_entries': len(_COMM_CONE_CACHE),
    }


# =============================================================================
# Coverage Radius Cache (using lru_cache for simplicity)
# =============================================================================

@lru_cache(maxsize=256)
def cached_coverage_radius_km(altitude_km: float, min_elevation_deg: float) -> float:
    """
    Cached version of coverage radius calculation.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude in km.
    min_elevation_deg : float
        Minimum elevation angle in degrees.

    Returns
    -------
    float
        Coverage radius in km.
    """
    from .utils import calculate_coverage_radius_km
    return calculate_coverage_radius_km(altitude_km, min_elevation_deg)


@lru_cache(maxsize=256)
def cached_coverage_radius_deg(altitude_km: float, min_elevation_deg: float) -> float:
    """
    Cached version of coverage radius in degrees.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude in km.
    min_elevation_deg : float
        Minimum elevation angle in degrees.

    Returns
    -------
    float
        Coverage radius as great-circle arc in degrees.
    """
    from .utils import calculate_coverage_radius_deg
    return calculate_coverage_radius_deg(altitude_km, min_elevation_deg)


# =============================================================================
# Disk Cache (Optional, for very expensive computations)
# =============================================================================

class DiskCache:
    """
    Simple disk-based cache for large computation results.

    Stores pickled results in a .cache directory.
    """

    def __init__(self, cache_dir: Path = None):
        """
        Initialize disk cache.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory for cache files. Defaults to .cache in current directory.
        """
        self.cache_dir = cache_dir or Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)

    def _make_key(self, *args, **kwargs) -> str:
        """Create a hash key from arguments."""
        key_data = str((args, sorted(kwargs.items())))
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Optional[Any]
            Cached value, or None if not found.
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set cached value.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache (must be picklable).
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass  # Silently fail on cache write errors

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Get cached value or compute and cache it.

        Parameters
        ----------
        key : str
            Cache key.
        compute_fn : Callable
            Function to compute the value if not cached.
        *args, **kwargs
            Arguments to pass to compute_fn.

        Returns
        -------
        Any
            Cached or computed value.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        result = compute_fn(*args, **kwargs)
        self.set(key, result)
        return result

    def clear(self) -> int:
        """
        Clear all cache files.

        Returns
        -------
        int
            Number of files cleared.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count


# Global disk cache instance
_disk_cache: Optional[DiskCache] = None


def get_disk_cache(cache_dir: Path = None) -> DiskCache:
    """
    Get or create the global disk cache instance.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for cache files.

    Returns
    -------
    DiskCache
        Global disk cache instance.
    """
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = DiskCache(cache_dir)
    return _disk_cache
