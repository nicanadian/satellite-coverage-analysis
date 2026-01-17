"""
Propulsion and Station-Keeping Module for VLEO Satellites

Models Hall Effect Thruster (HET) station-keeping to counteract atmospheric drag
and maintain orbital altitude in Very Low Earth Orbit (VLEO).

Key features:
- Exponential atmospheric density model with solar activity correction
- Drag force and deceleration calculation
- Hall Effect Thruster performance modeling
- Delta-V budgeting for drag compensation
- Propellant mass consumption tracking
- LTDN (Local Time of Descending Node) to RAAN conversion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta
import pandas as pd


# Physical constants
G0 = 9.80665  # Standard gravity (m/s²)
RE = 6371.0  # Earth radius (km)
MU = 398600.4418  # Earth gravitational parameter (km³/s²)


@dataclass
class AtmosphericModel:
    """
    Exponential atmospheric density model with solar activity correction.

    Based on simplified NRLMSISE-00 approximation for VLEO altitudes (150-400 km).
    """
    # Reference density at 200 km (kg/m³) for moderate solar activity (F10.7 = 150)
    rho_ref: float = 2.5e-10
    h_ref: float = 200.0  # Reference altitude (km)

    # Scale heights for different altitude bands (km)
    scale_heights: Dict[str, float] = field(default_factory=lambda: {
        '150-200': 27.0,
        '200-250': 32.0,
        '250-300': 38.0,
        '300-400': 46.0,
        '400-500': 55.0,
    })

    # Solar activity index (F10.7 in SFU - Solar Flux Units)
    # Low: 70, Moderate: 150, High: 250
    f107: float = 150.0

    def get_scale_height(self, altitude_km: float) -> float:
        """Get appropriate scale height for altitude."""
        if altitude_km < 200:
            return self.scale_heights['150-200']
        elif altitude_km < 250:
            return self.scale_heights['200-250']
        elif altitude_km < 300:
            return self.scale_heights['250-300']
        elif altitude_km < 400:
            return self.scale_heights['300-400']
        else:
            return self.scale_heights['400-500']

    def get_solar_correction(self) -> float:
        """
        Calculate solar activity correction factor.
        Density scales roughly linearly with F10.7.
        """
        f107_ref = 150.0  # Reference F10.7 for rho_ref
        return self.f107 / f107_ref

    def density(self, altitude_km: float) -> float:
        """
        Calculate atmospheric density at given altitude.

        Args:
            altitude_km: Altitude above Earth surface (km)

        Returns:
            Atmospheric density (kg/m³)
        """
        H = self.get_scale_height(altitude_km)
        solar_factor = self.get_solar_correction()

        # Exponential atmosphere model
        rho = self.rho_ref * solar_factor * np.exp(-(altitude_km - self.h_ref) / H)

        return rho

    def density_at_points(self, altitudes_km: np.ndarray) -> np.ndarray:
        """Calculate density at multiple altitude points."""
        return np.array([self.density(h) for h in altitudes_km])


@dataclass
class SpacecraftConfig:
    """Spacecraft physical properties for drag calculation."""
    mass_kg: float = 500.0  # Spacecraft dry mass
    cross_section_m2: float = 2.0  # Cross-sectional area perpendicular to velocity
    drag_coefficient: float = 2.2  # Typical for flat plate normal to flow

    @property
    def ballistic_coefficient(self) -> float:
        """Ballistic coefficient B = m / (Cd * A) in kg/m²"""
        return self.mass_kg / (self.drag_coefficient * self.cross_section_m2)


@dataclass
class HallThrusterConfig:
    """
    Hall Effect Thruster (HET) configuration.

    Typical values for small-to-medium HETs (e.g., BHT-200, SPT-50, PPS-1350):
    - Thrust: 10-100 mN
    - Isp: 1000-2000 s
    - Power: 200W - 2kW
    - Propellant: Xenon (most common) or Krypton
    """
    name: str = "Generic HET"
    thrust_mN: float = 50.0  # Nominal thrust (milliNewtons)
    isp_s: float = 1500.0  # Specific impulse (seconds)
    power_W: float = 1000.0  # Input power (Watts)
    propellant: str = "Xenon"  # Propellant type
    efficiency: float = 0.55  # Total thruster efficiency

    # Operational constraints
    max_duty_cycle: float = 0.5  # Maximum fraction of orbit with thruster firing
    min_firing_duration_s: float = 60.0  # Minimum firing duration per maneuver

    @property
    def thrust_N(self) -> float:
        """Thrust in Newtons."""
        return self.thrust_mN / 1000.0

    @property
    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity (m/s)."""
        return self.isp_s * G0

    @property
    def mass_flow_rate(self) -> float:
        """Propellant mass flow rate (kg/s)."""
        return self.thrust_N / self.exhaust_velocity

    @property
    def specific_power(self) -> float:
        """Specific power (W/mN)."""
        return self.power_W / self.thrust_mN


@dataclass
class StationKeepingResult:
    """Results from station-keeping analysis."""
    # Mission parameters
    altitude_km: float
    duration_days: float

    # Drag analysis
    mean_density_kg_m3: float
    mean_drag_N: float
    mean_drag_acceleration_m_s2: float

    # Delta-V budget
    total_delta_v_m_s: float
    delta_v_per_day_m_s: float
    delta_v_per_orbit_m_s: float

    # Propellant consumption
    propellant_mass_kg: float
    propellant_per_day_kg: float
    propellant_per_year_kg: float

    # Thruster operations
    total_firing_time_hours: float
    firing_time_per_day_hours: float
    duty_cycle_percent: float
    total_impulse_Ns: float

    # Energy
    total_energy_kWh: float
    power_per_orbit_Wh: float

    # Orbit info
    orbital_period_min: float
    orbits_per_day: float
    total_orbits: float

    # Margin analysis
    margin_factor: float = 1.5  # Typical margin for drag uncertainty

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            'Altitude (km)': self.altitude_km,
            'Duration (days)': self.duration_days,
            'Mean Atmospheric Density (kg/m³)': f'{self.mean_density_kg_m3:.2e}',
            'Mean Drag Force (mN)': f'{self.mean_drag_N * 1000:.3f}',
            'Mean Drag Acceleration (m/s²)': f'{self.mean_drag_acceleration_m_s2:.2e}',
            'Total Delta-V (m/s)': f'{self.total_delta_v_m_s:.2f}',
            'Delta-V per Day (m/s)': f'{self.delta_v_per_day_m_s:.3f}',
            'Delta-V per Orbit (m/s)': f'{self.delta_v_per_orbit_m_s:.4f}',
            'Propellant Mass (kg)': f'{self.propellant_mass_kg:.2f}',
            'Propellant per Day (g)': f'{self.propellant_per_day_kg * 1000:.1f}',
            'Propellant per Year (kg)': f'{self.propellant_per_year_kg:.1f}',
            'Total Firing Time (hours)': f'{self.total_firing_time_hours:.1f}',
            'Firing Time per Day (hours)': f'{self.firing_time_per_day_hours:.2f}',
            'Duty Cycle (%)': f'{self.duty_cycle_percent:.1f}',
            'Total Impulse (N·s)': f'{self.total_impulse_Ns:.1f}',
            'Total Energy (kWh)': f'{self.total_energy_kWh:.1f}',
            'Power per Orbit (Wh)': f'{self.power_per_orbit_Wh:.1f}',
            'Orbital Period (min)': f'{self.orbital_period_min:.2f}',
            'Orbits per Day': f'{self.orbits_per_day:.2f}',
            'Total Orbits': f'{self.total_orbits:.0f}',
            'Margin Factor': self.margin_factor,
            'With Margin - Delta-V (m/s)': f'{self.total_delta_v_m_s * self.margin_factor:.2f}',
            'With Margin - Propellant (kg)': f'{self.propellant_mass_kg * self.margin_factor:.2f}',
        }


def calculate_orbital_velocity(altitude_km: float) -> float:
    """
    Calculate circular orbital velocity.

    Args:
        altitude_km: Altitude above Earth surface (km)

    Returns:
        Orbital velocity (m/s)
    """
    r = (RE + altitude_km) * 1000  # Convert to meters
    v = np.sqrt(MU * 1e9 / r)  # MU in m³/s²
    return v


def calculate_orbital_period(altitude_km: float) -> float:
    """
    Calculate orbital period.

    Args:
        altitude_km: Altitude above Earth surface (km)

    Returns:
        Orbital period (seconds)
    """
    a = (RE + altitude_km) * 1000  # Semi-major axis in meters
    T = 2 * np.pi * np.sqrt(a**3 / (MU * 1e9))
    return T


def calculate_drag_force(
    altitude_km: float,
    spacecraft: SpacecraftConfig,
    atmosphere: AtmosphericModel
) -> float:
    """
    Calculate atmospheric drag force.

    Args:
        altitude_km: Altitude above Earth surface (km)
        spacecraft: Spacecraft configuration
        atmosphere: Atmospheric model

    Returns:
        Drag force (N)
    """
    rho = atmosphere.density(altitude_km)
    v = calculate_orbital_velocity(altitude_km)

    # Drag equation: F = 0.5 * rho * v² * Cd * A
    F_drag = 0.5 * rho * v**2 * spacecraft.drag_coefficient * spacecraft.cross_section_m2

    return F_drag


def calculate_drag_acceleration(
    altitude_km: float,
    spacecraft: SpacecraftConfig,
    atmosphere: AtmosphericModel
) -> float:
    """
    Calculate drag deceleration.

    Args:
        altitude_km: Altitude above Earth surface (km)
        spacecraft: Spacecraft configuration
        atmosphere: Atmospheric model

    Returns:
        Drag deceleration (m/s²)
    """
    F_drag = calculate_drag_force(altitude_km, spacecraft, atmosphere)
    a_drag = F_drag / spacecraft.mass_kg
    return a_drag


def calculate_delta_v_drag(
    altitude_km: float,
    duration_days: float,
    spacecraft: SpacecraftConfig,
    atmosphere: AtmosphericModel,
    time_steps: int = 1000
) -> Tuple[float, float, float]:
    """
    Calculate total delta-V needed to counteract drag over mission duration.

    Uses numerical integration accounting for potential altitude variations.
    For station-keeping, we assume altitude is maintained constant.

    Args:
        altitude_km: Target altitude (km)
        duration_days: Mission duration (days)
        spacecraft: Spacecraft configuration
        atmosphere: Atmospheric model
        time_steps: Number of integration steps

    Returns:
        Tuple of (total_delta_v_m_s, mean_drag_N, mean_acceleration_m_s2)
    """
    duration_s = duration_days * 86400
    dt = duration_s / time_steps

    # For constant altitude station-keeping, drag is approximately constant
    # (ignoring diurnal density variations)
    drag_force = calculate_drag_force(altitude_km, spacecraft, atmosphere)
    drag_accel = drag_force / spacecraft.mass_kg

    # Delta-V = integral of drag acceleration over time
    # For constant drag: delta_v = a_drag * t
    total_delta_v = drag_accel * duration_s

    return total_delta_v, drag_force, drag_accel


def calculate_propellant_mass(
    delta_v_m_s: float,
    spacecraft_mass_kg: float,
    isp_s: float
) -> float:
    """
    Calculate propellant mass using Tsiolkovsky rocket equation.

    Args:
        delta_v_m_s: Required delta-V (m/s)
        spacecraft_mass_kg: Spacecraft dry mass (kg)
        isp_s: Specific impulse (seconds)

    Returns:
        Propellant mass (kg)
    """
    v_exhaust = isp_s * G0

    # Tsiolkovsky: delta_v = v_e * ln(m0/m1)
    # m0 = m1 + m_prop (initial mass = final mass + propellant)
    # m_prop = m1 * (exp(delta_v/v_e) - 1)

    mass_ratio = np.exp(delta_v_m_s / v_exhaust)
    propellant_mass = spacecraft_mass_kg * (mass_ratio - 1)

    return propellant_mass


def calculate_firing_time(
    delta_v_m_s: float,
    spacecraft_mass_kg: float,
    thruster: HallThrusterConfig
) -> float:
    """
    Calculate total thruster firing time.

    Args:
        delta_v_m_s: Required delta-V (m/s)
        spacecraft_mass_kg: Spacecraft mass (kg)
        thruster: Thruster configuration

    Returns:
        Total firing time (seconds)
    """
    # F = m * a, so t = m * delta_v / F
    firing_time = spacecraft_mass_kg * delta_v_m_s / thruster.thrust_N
    return firing_time


def ltdn_to_raan(ltdn_hours: float, epoch: datetime) -> float:
    """
    Convert Local Time of Descending Node to RAAN.

    For a sun-synchronous orbit, RAAN is related to the local solar time
    at which the satellite crosses the equator going southward (descending node).

    Args:
        ltdn_hours: Local Time of Descending Node in hours (e.g., 10.5 for 10:30 AM)
        epoch: Mission start date/time

    Returns:
        RAAN in degrees
    """
    # Calculate the Sun's right ascension at epoch
    # Simplified calculation using day of year
    doy = epoch.timetuple().tm_yday
    year = epoch.year

    # Days since J2000.0 (Jan 1, 2000, 12:00 TT)
    # Make j2000 timezone-aware if epoch is timezone-aware
    from datetime import timezone as tz
    if epoch.tzinfo is not None:
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=tz.utc)
    else:
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
    days_since_j2000 = (epoch - j2000).total_seconds() / 86400

    # Mean longitude of the Sun (simplified)
    # L = 280.46 + 0.9856474 * d (degrees)
    L_sun = 280.46 + 0.9856474 * days_since_j2000
    L_sun = L_sun % 360

    # Mean anomaly of the Sun
    # g = 357.528 + 0.9856003 * d (degrees)
    g_sun = 357.528 + 0.9856003 * days_since_j2000
    g_sun_rad = np.radians(g_sun)

    # Ecliptic longitude of the Sun
    # lambda = L + 1.915 * sin(g) + 0.020 * sin(2g)
    lambda_sun = L_sun + 1.915 * np.sin(g_sun_rad) + 0.020 * np.sin(2 * g_sun_rad)

    # Right ascension of the Sun (simplified, ignoring obliquity for approximation)
    # For more accuracy, should account for ecliptic obliquity
    ra_sun = lambda_sun % 360

    # LTDN relates to the hour angle at the descending node
    # For LTDN at local time T hours:
    # RAAN = RA_sun + (LTDN - 12) * 15 degrees
    # The -12 is because local noon corresponds to the Sun's RA

    raan = ra_sun + (ltdn_hours - 12) * 15
    raan = raan % 360

    return raan


def calculate_sso_inclination(altitude_km: float) -> float:
    """
    Calculate sun-synchronous inclination for given altitude.

    For sun-synchronous orbit, the nodal precession rate must equal
    the Earth's mean motion around the Sun (~0.9856°/day).

    Args:
        altitude_km: Orbital altitude (km)

    Returns:
        Required inclination (degrees)
    """
    # Nodal precession rate: Omega_dot = -1.5 * n * J2 * (Re/a)^2 * cos(i)
    # where n = mean motion, J2 = 1.08263e-3, Re = 6378.137 km

    J2 = 1.08263e-3
    Re = 6378.137  # km

    a = Re + altitude_km  # Semi-major axis (km)
    n = np.sqrt(MU / a**3) * 86400  # Mean motion (rad/day)

    # Required precession rate for SSO: 360/365.25 deg/day = 0.9856 deg/day
    omega_dot_required = np.radians(0.9856)  # rad/day

    # Solve for cos(i):
    # cos(i) = -omega_dot / (1.5 * n * J2 * (Re/a)^2)
    cos_i = -omega_dot_required / (1.5 * n * J2 * (Re/a)**2)

    # Clamp to valid range
    cos_i = np.clip(cos_i, -1, 1)

    inclination = np.degrees(np.arccos(cos_i))

    return inclination


def analyze_station_keeping(
    altitude_km: float,
    duration_days: float,
    spacecraft: SpacecraftConfig,
    thruster: HallThrusterConfig,
    atmosphere: Optional[AtmosphericModel] = None,
    margin_factor: float = 1.5
) -> StationKeepingResult:
    """
    Perform comprehensive station-keeping analysis for VLEO orbit.

    Args:
        altitude_km: Target orbital altitude (km)
        duration_days: Mission duration (days)
        spacecraft: Spacecraft configuration
        thruster: Hall thruster configuration
        atmosphere: Atmospheric model (default: moderate solar activity)
        margin_factor: Safety margin for delta-V and propellant

    Returns:
        StationKeepingResult with all computed metrics
    """
    if atmosphere is None:
        atmosphere = AtmosphericModel()

    # Orbital parameters
    orbital_period_s = calculate_orbital_period(altitude_km)
    orbital_period_min = orbital_period_s / 60
    orbits_per_day = 86400 / orbital_period_s
    total_orbits = orbits_per_day * duration_days

    # Drag analysis
    total_delta_v, mean_drag, mean_accel = calculate_delta_v_drag(
        altitude_km, duration_days, spacecraft, atmosphere
    )

    mean_density = atmosphere.density(altitude_km)

    # Propellant calculation
    propellant_mass = calculate_propellant_mass(
        total_delta_v, spacecraft.mass_kg, thruster.isp_s
    )

    # Thruster operations
    firing_time_s = calculate_firing_time(
        total_delta_v, spacecraft.mass_kg, thruster
    )
    firing_time_hours = firing_time_s / 3600

    # Per-period calculations
    delta_v_per_day = total_delta_v / duration_days
    delta_v_per_orbit = total_delta_v / total_orbits
    propellant_per_day = propellant_mass / duration_days
    firing_per_day_hours = firing_time_hours / duration_days

    # Duty cycle
    duty_cycle = firing_time_s / (duration_days * 86400) * 100

    # Total impulse
    total_impulse = thruster.thrust_N * firing_time_s

    # Energy
    total_energy_Wh = thruster.power_W * firing_time_hours
    total_energy_kWh = total_energy_Wh / 1000
    power_per_orbit = total_energy_Wh / total_orbits

    # Annual propellant (for reference)
    propellant_per_year = propellant_per_day * 365.25

    return StationKeepingResult(
        altitude_km=altitude_km,
        duration_days=duration_days,
        mean_density_kg_m3=mean_density,
        mean_drag_N=mean_drag,
        mean_drag_acceleration_m_s2=mean_accel,
        total_delta_v_m_s=total_delta_v,
        delta_v_per_day_m_s=delta_v_per_day,
        delta_v_per_orbit_m_s=delta_v_per_orbit,
        propellant_mass_kg=propellant_mass,
        propellant_per_day_kg=propellant_per_day,
        propellant_per_year_kg=propellant_per_year,
        total_firing_time_hours=firing_time_hours,
        firing_time_per_day_hours=firing_per_day_hours,
        duty_cycle_percent=duty_cycle,
        total_impulse_Ns=total_impulse,
        total_energy_kWh=total_energy_kWh,
        power_per_orbit_Wh=power_per_orbit,
        orbital_period_min=orbital_period_min,
        orbits_per_day=orbits_per_day,
        total_orbits=total_orbits,
        margin_factor=margin_factor,
    )


def generate_propulsion_report(
    result: StationKeepingResult,
    spacecraft: SpacecraftConfig,
    thruster: HallThrusterConfig,
    atmosphere: AtmosphericModel
) -> pd.DataFrame:
    """
    Generate a detailed propulsion report as a DataFrame.

    Args:
        result: Station-keeping analysis result
        spacecraft: Spacecraft configuration
        thruster: Thruster configuration
        atmosphere: Atmospheric model

    Returns:
        DataFrame with all propulsion parameters and results
    """
    data = []

    # Mission Parameters
    data.append(('Mission', 'Target Altitude', f'{result.altitude_km:.0f}', 'km'))
    data.append(('Mission', 'Duration', f'{result.duration_days:.0f}', 'days'))
    data.append(('Mission', 'Total Orbits', f'{result.total_orbits:.0f}', ''))
    data.append(('Mission', 'Orbital Period', f'{result.orbital_period_min:.2f}', 'min'))

    # Spacecraft
    data.append(('Spacecraft', 'Dry Mass', f'{spacecraft.mass_kg:.0f}', 'kg'))
    data.append(('Spacecraft', 'Cross-Section Area', f'{spacecraft.cross_section_m2:.2f}', 'm²'))
    data.append(('Spacecraft', 'Drag Coefficient', f'{spacecraft.drag_coefficient:.2f}', ''))
    data.append(('Spacecraft', 'Ballistic Coefficient', f'{spacecraft.ballistic_coefficient:.1f}', 'kg/m²'))

    # Atmosphere
    data.append(('Atmosphere', 'Solar Activity (F10.7)', f'{atmosphere.f107:.0f}', 'SFU'))
    data.append(('Atmosphere', 'Mean Density', f'{result.mean_density_kg_m3:.2e}', 'kg/m³'))

    # Drag
    data.append(('Drag', 'Mean Drag Force', f'{result.mean_drag_N * 1000:.3f}', 'mN'))
    data.append(('Drag', 'Mean Drag Acceleration', f'{result.mean_drag_acceleration_m_s2:.2e}', 'm/s²'))

    # Thruster
    data.append(('Thruster', 'Type', thruster.name, ''))
    data.append(('Thruster', 'Thrust', f'{thruster.thrust_mN:.0f}', 'mN'))
    data.append(('Thruster', 'Specific Impulse', f'{thruster.isp_s:.0f}', 's'))
    data.append(('Thruster', 'Power', f'{thruster.power_W:.0f}', 'W'))
    data.append(('Thruster', 'Propellant', thruster.propellant, ''))
    data.append(('Thruster', 'Efficiency', f'{thruster.efficiency * 100:.0f}', '%'))

    # Delta-V Budget
    data.append(('Delta-V', 'Total Required', f'{result.total_delta_v_m_s:.2f}', 'm/s'))
    data.append(('Delta-V', 'Per Day', f'{result.delta_v_per_day_m_s:.3f}', 'm/s'))
    data.append(('Delta-V', 'Per Orbit', f'{result.delta_v_per_orbit_m_s:.4f}', 'm/s'))
    data.append(('Delta-V', f'With {result.margin_factor}x Margin',
                 f'{result.total_delta_v_m_s * result.margin_factor:.2f}', 'm/s'))

    # Propellant
    data.append(('Propellant', 'Mission Total', f'{result.propellant_mass_kg:.2f}', 'kg'))
    data.append(('Propellant', 'Per Day', f'{result.propellant_per_day_kg * 1000:.1f}', 'g'))
    data.append(('Propellant', 'Per Year (annualized)', f'{result.propellant_per_year_kg:.1f}', 'kg'))
    data.append(('Propellant', f'With {result.margin_factor}x Margin',
                 f'{result.propellant_mass_kg * result.margin_factor:.2f}', 'kg'))

    # Operations
    data.append(('Operations', 'Total Firing Time', f'{result.total_firing_time_hours:.1f}', 'hours'))
    data.append(('Operations', 'Firing Time per Day', f'{result.firing_time_per_day_hours:.2f}', 'hours'))
    data.append(('Operations', 'Duty Cycle', f'{result.duty_cycle_percent:.1f}', '%'))
    data.append(('Operations', 'Total Impulse', f'{result.total_impulse_Ns:.1f}', 'N·s'))

    # Power/Energy
    data.append(('Energy', 'Total Energy', f'{result.total_energy_kWh:.1f}', 'kWh'))
    data.append(('Energy', 'Energy per Orbit', f'{result.power_per_orbit_Wh:.1f}', 'Wh'))

    df = pd.DataFrame(data, columns=['Category', 'Parameter', 'Value', 'Unit'])
    return df


# Preset thruster configurations
THRUSTER_PRESETS = {
    'BHT-200': HallThrusterConfig(
        name='Busek BHT-200',
        thrust_mN=13.0,
        isp_s=1390,
        power_W=200,
        propellant='Xenon',
        efficiency=0.43,
    ),
    'SPT-50': HallThrusterConfig(
        name='Fakel SPT-50',
        thrust_mN=20.0,
        isp_s=1100,
        power_W=350,
        propellant='Xenon',
        efficiency=0.35,
    ),
    'BHT-600': HallThrusterConfig(
        name='Busek BHT-600',
        thrust_mN=39.0,
        isp_s=1500,
        power_W=600,
        propellant='Xenon',
        efficiency=0.52,
    ),
    'PPS-1350': HallThrusterConfig(
        name='Safran PPS-1350',
        thrust_mN=90.0,
        isp_s=1660,
        power_W=1500,
        propellant='Xenon',
        efficiency=0.55,
    ),
    'SPT-100': HallThrusterConfig(
        name='Fakel SPT-100',
        thrust_mN=83.0,
        isp_s=1600,
        power_W=1350,
        propellant='Xenon',
        efficiency=0.50,
    ),
    'AEPS': HallThrusterConfig(
        name='Aerojet AEPS',
        thrust_mN=240.0,
        isp_s=2800,
        power_W=12000,
        propellant='Xenon',
        efficiency=0.58,
    ),
}


def print_station_keeping_summary(
    result: StationKeepingResult,
    thruster: HallThrusterConfig
) -> None:
    """Print a formatted station-keeping analysis summary."""
    print("\n" + "=" * 60)
    print("HALL EFFECT THRUSTER STATION-KEEPING ANALYSIS")
    print("=" * 60)

    print(f"\nMission Profile:")
    print(f"  Target Altitude: {result.altitude_km:.0f} km")
    print(f"  Duration: {result.duration_days:.0f} days")
    print(f"  Total Orbits: {result.total_orbits:.0f}")
    print(f"  Orbital Period: {result.orbital_period_min:.2f} min")

    print(f"\nAtmospheric Drag:")
    print(f"  Mean Density: {result.mean_density_kg_m3:.2e} kg/m³")
    print(f"  Mean Drag Force: {result.mean_drag_N * 1000:.3f} mN")
    print(f"  Mean Deceleration: {result.mean_drag_acceleration_m_s2:.2e} m/s²")

    print(f"\nThruster: {thruster.name}")
    print(f"  Thrust: {thruster.thrust_mN:.0f} mN")
    print(f"  Isp: {thruster.isp_s:.0f} s")
    print(f"  Power: {thruster.power_W:.0f} W")

    print(f"\nDelta-V Budget:")
    print(f"  Total Required: {result.total_delta_v_m_s:.2f} m/s")
    print(f"  Per Day: {result.delta_v_per_day_m_s:.3f} m/s")
    print(f"  Per Orbit: {result.delta_v_per_orbit_m_s:.4f} m/s")
    print(f"  With {result.margin_factor}x Margin: {result.total_delta_v_m_s * result.margin_factor:.2f} m/s")

    print(f"\nPropellant ({thruster.propellant}):")
    print(f"  Mission Total: {result.propellant_mass_kg:.2f} kg")
    print(f"  Per Day: {result.propellant_per_day_kg * 1000:.1f} g")
    print(f"  Annualized: {result.propellant_per_year_kg:.1f} kg/year")
    print(f"  With {result.margin_factor}x Margin: {result.propellant_mass_kg * result.margin_factor:.2f} kg")

    print(f"\nThruster Operations:")
    print(f"  Total Firing Time: {result.total_firing_time_hours:.1f} hours")
    print(f"  Firing per Day: {result.firing_time_per_day_hours:.2f} hours")
    print(f"  Duty Cycle: {result.duty_cycle_percent:.1f}%")
    print(f"  Total Impulse: {result.total_impulse_Ns:.1f} N·s")

    print(f"\nPower/Energy:")
    print(f"  Total Energy: {result.total_energy_kWh:.1f} kWh")
    print(f"  Energy per Orbit: {result.power_per_orbit_Wh:.1f} Wh")

    print("=" * 60)
