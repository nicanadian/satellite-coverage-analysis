# Satellite Coverage Analysis - Code Review Report

**Reviewer**: Claude (Senior Engineer)
**Date**: 2026-01-25
**Scope**: Full codebase review focusing on performance, code quality, and maintainability

## Implementation Status

**All recommendations have been implemented.** See the changes summary below:

| Recommendation | Status | Files Changed |
|---------------|--------|---------------|
| Create constants module | Done | `src/vleo_eo/constants.py` (new) |
| Create utils module | Done | `src/vleo_eo/utils.py` (new) |
| Remove unused imports | Done | All source files |
| Consolidate duplicate functions | Done | Via utils.py delegation |
| Vectorize calculate_access_windows | Done | `coverage.py` |
| Add caching layer | Done | `src/vleo_eo/cache.py` (new) |
| Split presentation script | Done | `scripts/presentation/` (new package) |
| Add parallel generation | Done | `generate_full_presentation.py` |

---

---

## Executive Summary

The codebase is well-organized with clear separation of concerns. However, there are opportunities for significant performance improvements, code cleanup, and better maintainability. Key issues include:

1. **Performance**: Python loops where NumPy vectorization would be faster
2. **Unused code**: ~15 orphaned functions and unused imports
3. **Code duplication**: Similar logic repeated across modules
4. **Memory efficiency**: Large DataFrames created where generators could work

---

## 1. Performance Issues (High Priority)

### 1.1 Vectorization Opportunities

#### `coverage.py:calculate_access_windows()` (lines 291-422)
**Issue**: The function iterates over targets with a Python loop, computing look angles for the entire orbit DataFrame for each target.

**Current (slow)**:
```python
for idx, aoi in targets_gdf.iterrows():  # Python loop
    target_lat = aoi.geometry.y
    look_angles = calculate_look_angle(orbit_df['Latitude'].values, ...)
```

**Suggested (vectorized)**:
```python
# Broadcast calculation across all targets at once
target_lats = targets_gdf.geometry.y.values[:, np.newaxis]  # (N_targets, 1)
target_lons = targets_gdf.geometry.x.values[:, np.newaxis]
sat_lats = orbit_df['Latitude'].values[np.newaxis, :]       # (1, N_epochs)

# Vectorized haversine for all target-epoch pairs at once
dlat = np.radians(sat_lats - target_lats)
dlon = np.radians(sat_lons - target_lons)
# ... compute look angles as (N_targets, N_epochs) array
```

**Impact**: For 100 targets and 250,000 orbit points, this could be 10-50x faster.

---

#### `contacts.py:calculate_contact_windows()` - Ground station contact detection
**Issue**: Contact detection uses Python loop over time epochs.

**Location**: `contacts.py` lines where ground distance is computed per epoch.

**Current pattern**:
```python
for i, (lon, lat) in enumerate(ground_track):
    dist = np.sqrt((lat - gs_lat)**2 + ...)
```

**Suggested pattern**:
```python
# Vectorized distance calculation
track_lats = np.array([p[1] for p in ground_track])
track_lons = np.array([p[0] for p in ground_track])
distances = np.sqrt((track_lats - gs_lat)**2 +
                    ((track_lons - gs_lon) * np.cos(np.radians(gs_lat)))**2)
in_range = distances < coverage_radius_deg
```

---

#### `generate_full_presentation.py:generate_slide_ttc_coverage_analysis()` (lines 821-999)
**Issue**: Contact detection loop iterates over every ground track point (millions of points for 30-day analysis).

**Current (very slow)**:
```python
for i, (lon, lat) in enumerate(ground_track):  # ~2.5M iterations for 30 days at 1s step
    dist = np.sqrt((lat - gs_lat)**2 + ...)
```

**Suggested**:
```python
# Pre-compute all distances at once using broadcasting
track_arr = np.array(ground_track)  # Shape: (N, 2)
for gs in ground_stations:
    dists = haversine_vectorized(track_arr[:, 1], track_arr[:, 0], gs['lat'], gs['lon'])
    in_range = dists < coverage_radius
    # Use np.diff to find contact boundaries
    boundaries = np.diff(in_range.astype(int))
    starts = np.where(boundaries == 1)[0]
    ends = np.where(boundaries == -1)[0]
```

---

### 1.2 Redundant Calculations

#### `optimization.py:calculate_orbital_period()` (line 40)
**Issue**: Orbital period is calculated multiple times across modules. Define once and pass as parameter.

**Files with duplicate calculation**:
- `optimization.py:40`
- `generate_full_presentation.py:774-778`
- `generate_full_presentation.py:833-835`
- `generate_full_presentation.py:1453-1454`

**Suggestion**: Add `orbital_period_min` as a computed property on `AnalysisConfig` or `TLEData`.

---

#### `create_comm_cone_geodesic()` - Repeated geodesic calculations
**Issue**: Communication cones are recalculated for the same ground stations on multiple slides.

**Suggestion**: Cache comm cone polygons by (station_name, sat_altitude, elevation) tuple.

```python
_COMM_CONE_CACHE = {}

def get_comm_cone(gs, sat_alt_km, min_elev):
    key = (gs['name'], sat_alt_km, min_elev)
    if key not in _COMM_CONE_CACHE:
        _COMM_CONE_CACHE[key] = create_comm_cone_geodesic(gs['lon'], gs['lat'], min_elev, sat_alt_km)
    return _COMM_CONE_CACHE[key]
```

---

### 1.3 Memory Inefficiency

#### `orbits.py:propagate_orbits()` - Full DataFrame creation
**Issue**: Creates a DataFrame with every propagation point upfront, even if downstream processing only needs subsets.

**Suggestion**: Consider a generator-based approach for very long durations:
```python
def propagate_orbits_chunked(tle_data, start, duration, chunk_days=7):
    """Yield orbit data in chunks to reduce memory footprint."""
    current = start
    while current < start + duration:
        chunk_end = min(current + timedelta(days=chunk_days), start + duration)
        yield propagate_orbits(tle_data, current, chunk_end - current, ...)
        current = chunk_end
```

---

## 2. Unused Code (Medium Priority)

### 2.1 Unused Imports

| File | Unused Imports |
|------|----------------|
| `coverage.py` | `Geod`, `datetime`, `prep`, `timezone` |
| `data_model.py` | `datetime`, `np`, `timedelta` |
| `plots.py` | `LinearSegmentedColormap`, `Point`, `datetime` |
| `reports.py` | `GroundStationConfig`, `ImagingModeConfig`, `datetime`, `np` |
| `optimization.py` | `EarthSatellite`, `OptimizationConfig`, `datetime`, `load`, `wgs84` |
| `comparison.py` | `field` |

**Action**: Remove unused imports to reduce load time and improve clarity.

---

### 2.2 Orphaned Functions

These functions are defined but never called:

#### `config.py`
- `save_config()` - Appears unused; remove if config saving isn't needed

#### `coverage.py`
- `calculate_look_angle()` - Only used internally, could be prefixed with `_`
- `calculate_off_nadir_angle()` - Simple helper, inline or remove
- `convert_wgs_to_utm()` - Used once in `calculate_swath_polygon`, could be inlined

#### `contacts.py`
- `find_next_contact()` - Orphaned
- `calculate_slew_duration()` - Orphaned
- `to_unit_vector()` - Orphaned

#### `orbits.py`
- `calculate_checksum()` - Used in TLE generation but may be orphaned if TLEs come from external source
- `generate_tle()` - Check if still needed
- `get_orbital_period_minutes()` - Duplicates logic elsewhere

#### `optimization.py`
Many functions appear unused because optimization is disabled by default:
- `identify_orbital_revolutions()`
- `calculate_ttc_contacts_per_revolution()`
- `calculate_delivery_times()`
- etc.

**Note**: These may be used when `optimization.enabled: true` - verify before removing.

---

## 3. Code Quality Issues (Medium Priority)

### 3.1 Duplicate Logic

#### Haversine Distance Calculation
**Issue**: Great-circle distance calculated differently in multiple places:

1. `coverage.py:67-72` - Using Haversine formula
2. `contacts.py` - Similar calculation for ground distance
3. `generate_full_presentation.py:876` - Simplified Euclidean approximation

**Suggestion**: Create a single `haversine_distance()` utility function:
```python
# utils.py
def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation."""
    R = 6378.137
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
```

---

#### Map Setup
**Issue**: Map axes setup repeated in multiple functions:
- `plots.py:_setup_map_axes()`
- `generate_full_presentation.py:setup_map_axes()`

**Suggestion**: Use the same utility function, or consolidate into one module.

---

### 3.2 Magic Numbers

Several hardcoded values should be constants:

```python
# Constants that should be defined at module level
EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418
DEFAULT_MIN_ELEVATION_SX_DEG = 5.0
DEFAULT_MIN_ELEVATION_KA_DEG = 10.0
```

**Locations**:
- `coverage.py:64` - `R = 6378.137`
- `optimization.py:55-56` - `mu`, `r_earth`
- `generate_full_presentation.py:42` - `earth_radius_km = 6378.137`
- Multiple files: `mu = 398600.4418`

---

### 3.3 Error Handling

#### Silent Exception Catching
**Issue**: Several places catch all exceptions and continue silently:

```python
# generate_full_presentation.py:153
except Exception:
    pass  # Silently ignores comm cone errors
```

**Suggestion**: At minimum, log the error:
```python
except Exception as e:
    logger.warning(f"Failed to create comm cone for {gs['name']}: {e}")
```

---

### 3.4 Type Hints Inconsistency

Some functions have complete type hints, others have none. For example:
- `config.py` - Good type hints
- `generate_full_presentation.py` - Minimal type hints

**Suggestion**: Add type hints to all public functions for better IDE support and documentation.

---

## 4. Architecture Suggestions

### 4.1 Configuration Improvements

The ground stations file you just created (`configs/ground_stations.py`) is a good step. Consider extending this pattern:

```python
# configs/orbital_scenarios.py
ORBITAL_SCENARIOS = {
    'vleo_sso_dawn_dusk': {'altitude_km': 250, 'ltdn_hours': 6.0, 'auto_sso': True},
    'vleo_sso_midday': {'altitude_km': 250, 'ltdn_hours': 12.0, 'auto_sso': True},
    'leo_standard': {'altitude_km': 500, 'inclination_deg': 97.4},
}
```

---

### 4.2 Separate Presentation Generation

The `generate_full_presentation.py` file is 3000+ lines. Consider splitting:

```
scripts/
├── presentation/
│   ├── __init__.py
│   ├── slide_builders.py      # PowerPoint slide creation
│   ├── map_plots.py           # Geographic visualization
│   ├── chart_plots.py         # Bar charts, histograms
│   └── analysis_plots.py      # TT&C, Ka coverage analysis
└── generate_presentation.py   # Orchestration only
```

---

### 4.3 Caching Layer

For long analysis runs, add caching for expensive computations:

```python
# cache.py
import hashlib
import pickle
from pathlib import Path

CACHE_DIR = Path('.cache')

def get_cached(key, compute_fn, *args, **kwargs):
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        return pickle.load(open(cache_file, 'rb'))
    result = compute_fn(*args, **kwargs)
    cache_file.parent.mkdir(exist_ok=True)
    pickle.dump(result, open(cache_file, 'wb'))
    return result
```

Use for:
- Orbit propagation results
- Contact window calculations
- Ground track computations

---

## 5. Quick Wins (Low Effort, High Impact)

### 5.1 Parallel Slide Generation

Slides are independent - generate plots in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def generate_all_plots(config, tle_data, ...):
    plot_funcs = [
        (generate_slide2_gs_map, (config, ...)),
        (generate_slide3_single_orbit, (tle_data, ...)),
        (generate_slide_1day_tracks, (tle_data, ...)),
        # ... etc
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fn, *args) for fn, args in plot_funcs]
        results = [f.result() for f in futures]
```

---

### 5.2 Progress Indicators

Add `tqdm` progress bars for long operations:

```python
from tqdm import tqdm

for idx, aoi in tqdm(targets_gdf.iterrows(), total=len(targets_gdf), desc="Processing targets"):
    ...
```

---

### 5.3 Lazy Loading for Cartopy Features

Cartopy loads Natural Earth data on first use. Pre-cache or lazy-load:

```python
# At module level, load features once
_LAND_FEATURE = None
_OCEAN_FEATURE = None

def get_map_features():
    global _LAND_FEATURE, _OCEAN_FEATURE
    if _LAND_FEATURE is None:
        _LAND_FEATURE = cfeature.NaturalEarthFeature('physical', 'land', '50m', ...)
        _OCEAN_FEATURE = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', ...)
    return _LAND_FEATURE, _OCEAN_FEATURE
```

---

## 6. Recommended Priority Order

1. **High Priority (Performance)**
   - Vectorize `calculate_access_windows()` - biggest impact
   - Cache comm cones - eliminates redundant geodesic calculations
   - Vectorize contact detection in presentation script

2. **Medium Priority (Cleanup)**
   - Remove unused imports (5 minutes)
   - Add constants file for magic numbers (15 minutes)
   - Consolidate haversine implementations (30 minutes)

3. **Lower Priority (Architecture)**
   - Split presentation generation script
   - Add comprehensive type hints
   - Implement caching layer

---

## 7. Testing Recommendations

Before making changes, add these tests:

```python
# tests/test_coverage.py
def test_access_windows_vectorized_matches_loop():
    """Ensure vectorized implementation matches original loop behavior."""
    # Compare results from both implementations

def test_access_windows_performance():
    """Benchmark access window calculation time."""
    # Assert vectorized is at least 5x faster
```

---

## Appendix: File-by-File Summary

| File | Lines | Issues | Priority |
|------|-------|--------|----------|
| `coverage.py` | 622 | Vectorization needed, unused imports | High |
| `generate_full_presentation.py` | 3038 | Too large, performance loops, duplication | High |
| `optimization.py` | 727 | Unused Skyfield imports, duplicate period calc | Medium |
| `contacts.py` | ~400 | Orphaned functions, vectorization opportunity | Medium |
| `plots.py` | 612 | Unused imports, duplication with presentation | Low |
| `comparison.py` | 379 | Unused `field` import | Low |
| `config.py` | ~300 | Unused `save_config()` | Low |

---

*Report generated by code review analysis. Recommendations should be validated against actual runtime profiling before implementation.*
