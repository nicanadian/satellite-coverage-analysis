"""
Physical and orbital constants used throughout the analysis.

Centralizes magic numbers to ensure consistency and easy updates.
"""

# =============================================================================
# Earth Parameters (WGS84)
# =============================================================================
EARTH_RADIUS_KM = 6378.137  # Equatorial radius in km
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000  # Equatorial radius in meters
EARTH_MU_KM3_S2 = 398600.4418  # Gravitational parameter (km^3/s^2)
EARTH_MU_M3_S2 = EARTH_MU_KM3_S2 * 1e9  # Gravitational parameter (m^3/s^2)
EARTH_J2 = 1.08263e-3  # J2 perturbation coefficient
EARTH_ROTATION_DEG_PER_DAY = 360.9856  # Sidereal rotation rate

# =============================================================================
# Default Elevation Masks
# =============================================================================
DEFAULT_MIN_ELEVATION_SX_DEG = 5.0  # S/X-band (TT&C) minimum elevation
DEFAULT_MIN_ELEVATION_KA_DEG = 10.0  # Ka-band (payload) minimum elevation

# =============================================================================
# Default Downlink Parameters
# =============================================================================
DEFAULT_DOWNLINK_RATE_MBPS = 800.0
DEFAULT_IMAGE_SIZE_GB = 6.0
DEFAULT_PROCESSING_BUFFER_MIN = 5.0

# =============================================================================
# Plot Styling Defaults
# =============================================================================
DEFAULT_LAND_COLOR = '#fafaf9'
DEFAULT_OCEAN_COLOR = '#d4dbdc'
DEFAULT_BORDER_COLOR = '#ebd6d9'

# Satellite track colors (cycling palette)
SATELLITE_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]

# Provider colors for comparison charts
PROVIDER_COLORS = {
    'ViaSat RTE + ATLAS': '#5DADE2',  # Light blue
    'ViaSat RTE': '#F5B041',  # Light orange
    'ViaSat': '#F5B041',  # Light orange
    'KSAT': '#BB8FCE',  # Light purple
    'ATLAS': '#58D68D',  # Light green
}

# =============================================================================
# Time Constants
# =============================================================================
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
MINUTES_PER_DAY = 1440

# =============================================================================
# Analysis Defaults
# =============================================================================
DEFAULT_MIN_ACCESS_DURATION_S = 30.0
DEFAULT_MIN_CONTACT_DURATION_S = 30.0
DEFAULT_TIME_STEP_S = 10.0
DEFAULT_CONTACT_BUFFER_S = 30.0
