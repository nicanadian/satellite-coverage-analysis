"""
Comparison Module for VLEO EO Coverage Analysis

Load and compare results from multiple baseline configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class ComparisonResult:
    """Comparison metrics for a single baseline configuration."""
    label: str
    provider: str
    config_path: str
    results_dir: Path

    # Ground station info
    num_stations: int = 0
    num_ka_stations: int = 0

    # Coverage metrics
    total_accesses: int = 0
    accesses_per_day: float = 0.0

    # Contact metrics
    total_contacts: int = 0
    total_valid_contacts: int = 0
    valid_contact_pct: float = 0.0

    # Downlink metrics
    total_data_collected_gb: float = 0.0
    total_data_downlinked_gb: float = 0.0
    downlink_efficiency_pct: float = 0.0

    # Delay metrics
    delay_median_min: float = 0.0
    delay_p90_min: float = 0.0
    delay_p95_min: float = 0.0
    delay_max_min: float = 0.0

    # Raw data paths
    excel_path: Optional[Path] = None


def find_excel_file(results_dir: Path) -> Optional[Path]:
    """Find the Excel file in a results directory."""
    # Look for common patterns
    patterns = ['*.xlsx', 'results_summary.xlsx']
    for pattern in patterns:
        matches = list(results_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_comparison_result(
    label: str,
    provider: str,
    config_path: str,
    results_dir: Path,
) -> ComparisonResult:
    """
    Load comparison metrics from a single results directory.

    Args:
        label: Display label for this configuration
        provider: Ground station provider name
        config_path: Path to the config file used
        results_dir: Path to the results directory

    Returns:
        ComparisonResult with populated metrics
    """
    result = ComparisonResult(
        label=label,
        provider=provider,
        config_path=config_path,
        results_dir=results_dir,
    )

    excel_path = find_excel_file(results_dir)
    if excel_path is None or not excel_path.exists():
        print(f"  Warning: No Excel file found in {results_dir}")
        return result

    result.excel_path = excel_path

    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # Load ground station info
        if 'Ground_Stations' in sheet_names:
            gs_df = pd.read_excel(excel_path, sheet_name='Ground_Stations')
            result.num_stations = len(gs_df)
            # Try different possible column names for Ka capability
            for col in ['Ka Capable', 'ka_band', 'ka_capable', 'Ka_Capable']:
                if col in gs_df.columns:
                    result.num_ka_stations = int(gs_df[col].sum())
                    break

        # Load coverage KPIs
        if 'Coverage_KPIs' in sheet_names:
            kpis_df = pd.read_excel(excel_path, sheet_name='Coverage_KPIs')
            if not kpis_df.empty:
                # Handle both row-based (KPI, Value) and column-based formats
                if 'KPI' in kpis_df.columns and 'Value' in kpis_df.columns:
                    kpis = kpis_df.set_index('KPI')['Value'].to_dict()
                else:
                    kpis = kpis_df.set_index(kpis_df.columns[0])[kpis_df.columns[1]].to_dict()

                result.total_accesses = int(kpis.get('Total Accesses', kpis.get('total_accesses', 0)))
                result.accesses_per_day = float(kpis.get('Accesses per Day', kpis.get('accesses_per_day', 0)))

                # Also load data metrics from Coverage_KPIs if available
                if 'Total Data Collected' in kpis:
                    result.total_data_collected_gb = float(kpis.get('Total Data Collected', 0))
                if 'Total Data Downlinked' in kpis:
                    result.total_data_downlinked_gb = float(kpis.get('Total Data Downlinked', 0))
                if 'Downlink Efficiency' in kpis:
                    result.downlink_efficiency_pct = float(kpis.get('Downlink Efficiency', 0))
                if 'Total Valid Contacts' in kpis:
                    result.total_valid_contacts = int(kpis.get('Total Valid Contacts', 0))
                if 'Valid Contact Percentage' in kpis:
                    result.valid_contact_pct = float(kpis.get('Valid Contact Percentage', 0))

        # Load contact data
        if 'Contacts' in sheet_names:
            contacts_df = pd.read_excel(excel_path, sheet_name='Contacts')
            if not contacts_df.empty:
                result.total_contacts = len(contacts_df)
                # Try different possible column names for valid contacts
                for col in ['Total_Valid_Contact', 'Valid', 'valid_contact', 'Valid_Contact']:
                    if col in contacts_df.columns:
                        result.total_valid_contacts = int(contacts_df[col].sum())
                        result.valid_contact_pct = (result.total_valid_contacts / result.total_contacts * 100) if result.total_contacts > 0 else 0
                        break

        # Load downlink KPIs (only if data metrics weren't already loaded from Coverage_KPIs)
        if 'Downlink_KPIs' in sheet_names and result.total_data_collected_gb == 0:
            dl_df = pd.read_excel(excel_path, sheet_name='Downlink_KPIs')
            if not dl_df.empty:
                # Check if this is a summary sheet (has specific keys) vs per-station data
                if 'Ground Station' not in dl_df.columns:
                    dl_kpis = dl_df.set_index(dl_df.columns[0])[dl_df.columns[1]].to_dict()
                    if 'Total Data Collected (GB)' in dl_kpis:
                        result.total_data_collected_gb = float(dl_kpis.get('Total Data Collected (GB)', 0))
                    if 'Total Data Downlinked (GB)' in dl_kpis:
                        result.total_data_downlinked_gb = float(dl_kpis.get('Total Data Downlinked (GB)', 0))

                    if result.total_data_collected_gb > 0:
                        result.downlink_efficiency_pct = (result.total_data_downlinked_gb / result.total_data_collected_gb) * 100

        # Load downlink delay metrics
        if 'Downlink_Delay' in sheet_names:
            delay_df = pd.read_excel(excel_path, sheet_name='Downlink_Delay')
            if not delay_df.empty and 'Downlink_Delay_minutes' in delay_df.columns:
                valid_delay = delay_df['Downlink_Delay_minutes'].dropna()
                if len(valid_delay) > 0:
                    result.delay_median_min = valid_delay.median()
                    result.delay_p90_min = valid_delay.quantile(0.90)
                    result.delay_p95_min = valid_delay.quantile(0.95)
                    result.delay_max_min = valid_delay.max()

    except Exception as e:
        print(f"  Error loading results from {excel_path}: {e}")

    return result


def load_comparison_results(
    results_dirs: Dict[str, Dict[str, Any]],
) -> List[ComparisonResult]:
    """
    Load comparison results from multiple baseline configurations.

    Args:
        results_dirs: Dictionary mapping labels to config info:
            {
                'ViaSat RTE': {
                    'provider': 'ViaSat RTE',
                    'config_path': 'configs/viasat_rte_baseline.yaml',
                    'results_dir': Path('results/viasat_rte_baseline'),
                },
                ...
            }

    Returns:
        List of ComparisonResult objects
    """
    results = []

    for label, info in results_dirs.items():
        print(f"Loading results for: {label}")
        result = load_comparison_result(
            label=label,
            provider=info.get('provider', label),
            config_path=info.get('config_path', ''),
            results_dir=Path(info.get('results_dir', '')),
        )
        results.append(result)

    return results


def load_raw_delay_data(
    results_dirs: Dict[str, Path],
) -> Dict[str, pd.DataFrame]:
    """
    Load raw downlink delay data from multiple results directories.

    Args:
        results_dirs: Dictionary mapping labels to results directory paths

    Returns:
        Dictionary mapping labels to delay DataFrames
    """
    delay_data = {}

    for label, results_dir in results_dirs.items():
        excel_path = find_excel_file(results_dir)
        if excel_path is None:
            continue

        try:
            df = pd.read_excel(excel_path, sheet_name='Downlink_Delay')
            if not df.empty and 'Downlink_Delay_minutes' in df.columns:
                delay_data[label] = df
        except Exception as e:
            print(f"  Warning: Could not load delay data for {label}: {e}")

    return delay_data


def load_contacts_data(
    results_dirs: Dict[str, Path],
) -> Dict[str, pd.DataFrame]:
    """
    Load contacts data from multiple results directories.

    Args:
        results_dirs: Dictionary mapping labels to results directory paths

    Returns:
        Dictionary mapping labels to contacts DataFrames
    """
    contacts_data = {}

    for label, results_dir in results_dirs.items():
        excel_path = find_excel_file(results_dir)
        if excel_path is None:
            continue

        try:
            df = pd.read_excel(excel_path, sheet_name='Contacts')
            if not df.empty:
                contacts_data[label] = df
        except Exception as e:
            print(f"  Warning: Could not load contacts data for {label}: {e}")

    return contacts_data


def load_ground_stations_data(
    results_dirs: Dict[str, Path],
) -> Dict[str, pd.DataFrame]:
    """
    Load ground station data from multiple results directories.

    Args:
        results_dirs: Dictionary mapping labels to results directory paths

    Returns:
        Dictionary mapping labels to ground stations DataFrames
    """
    gs_data = {}

    for label, results_dir in results_dirs.items():
        excel_path = find_excel_file(results_dir)
        if excel_path is None:
            continue

        try:
            df = pd.read_excel(excel_path, sheet_name='Ground_Stations')
            if not df.empty:
                gs_data[label] = df
        except Exception as e:
            print(f"  Warning: Could not load ground stations data for {label}: {e}")

    return gs_data


def comparison_to_dataframe(results: List[ComparisonResult]) -> pd.DataFrame:
    """
    Convert list of ComparisonResult objects to a comparison DataFrame.

    Args:
        results: List of ComparisonResult objects

    Returns:
        DataFrame with one row per configuration
    """
    data = []
    for r in results:
        data.append({
            'Configuration': r.label,
            'Provider': r.provider,
            'Ground Stations': r.num_stations,
            'Ka-Band Stations': r.num_ka_stations,
            'Total Accesses': r.total_accesses,
            'Accesses/Day': r.accesses_per_day,
            'Total Contacts': r.total_contacts,
            'Valid Contacts': r.total_valid_contacts,
            'Valid Contact %': r.valid_contact_pct,
            'Data Collected (GB)': r.total_data_collected_gb,
            'Data Downlinked (GB)': r.total_data_downlinked_gb,
            'Downlink Efficiency %': r.downlink_efficiency_pct,
            'Delay Median (min)': r.delay_median_min,
            'Delay P90 (min)': r.delay_p90_min,
            'Delay P95 (min)': r.delay_p95_min,
            'Delay Max (min)': r.delay_max_min,
        })

    return pd.DataFrame(data)


def generate_comparison_excel(
    results: List[ComparisonResult],
    delay_data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """
    Generate comparison Excel report.

    Args:
        results: List of ComparisonResult objects
        delay_data: Dictionary of delay DataFrames
        output_path: Output path for Excel file

    Returns:
        Path to generated Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary comparison
        summary_df = comparison_to_dataframe(results)
        summary_df.to_excel(writer, sheet_name='Comparison_Summary', index=False)

        # Individual delay data
        for label, df in delay_data.items():
            sheet_name = f"Delay_{label[:20]}"  # Truncate for Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  Comparison Excel saved to: {output_path}")
    return output_path


# Provider color scheme for consistent visualizations
# Order matters: more specific keys must come first for matching
PROVIDER_COLORS = {
    'ViaSat RTE + ATLAS': '#5DADE2',   # Light blue
    'ViaSat RTE': '#F5B041',            # Light orange
    'KSAT': '#BB8FCE',                  # Light purple
}


def get_provider_color(label: str) -> str:
    """Get color for a provider label."""
    # Check more specific matches first (longer keys)
    for key, color in PROVIDER_COLORS.items():
        if key.lower() in label.lower():
            return color
    # Default colors for unknown providers
    default_colors = ['#5DADE2', '#F5B041', '#BB8FCE', '#d62728', '#9467bd']
    return default_colors[hash(label) % len(default_colors)]
