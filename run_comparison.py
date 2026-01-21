#!/usr/bin/env python3
"""
VLEO EO Coverage Analysis - Ground Station Provider Comparison

Run all baseline configurations and generate a comparison report.

Usage:
    # Default: run all 3 baselines
    python run_comparison.py

    # Custom configs with labels
    python run_comparison.py --configs config1.yaml config2.yaml --labels "A" "B"

    # Skip re-running, just regenerate comparison from existing results
    python run_comparison.py --skip-runs

    # Specify output directory
    python run_comparison.py --output-dir results/my_comparison
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default baseline configurations (APAC targets)
DEFAULT_BASELINES = [
    {
        'config': 'configs/viasat_rte_apac.yaml',
        'label': 'ViaSat RTE',
        'provider': 'ViaSat RTE',
    },
    {
        'config': 'configs/viasat_rte_atlas_apac.yaml',
        'label': 'ViaSat RTE + ATLAS',
        'provider': 'ViaSat RTE + ATLAS',
    },
    {
        'config': 'configs/ksat_apac.yaml',
        'label': 'KSAT',
        'provider': 'KSAT',
    },
]


def run_sequential_analyses(
    baselines: List[Dict],
    skip_ppt: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict]:
    """
    Run analysis for each baseline configuration sequentially.

    Args:
        baselines: List of baseline config dictionaries
        skip_ppt: Skip individual PowerPoint generation
        verbose: Enable verbose output

    Returns:
        Dictionary mapping labels to results info
    """
    from run_analysis import run_single_analysis
    from src.vleo_eo.config import load_config

    results_info = {}

    for i, baseline in enumerate(baselines, 1):
        config_path = baseline['config']
        label = baseline['label']
        provider = baseline.get('provider', label)

        print("\n" + "=" * 70)
        print(f"Running Baseline {i}/{len(baselines)}: {label}")
        print(f"Config: {config_path}")
        print("=" * 70)

        # Check config exists
        if not Path(config_path).exists():
            print(f"ERROR: Config file not found: {config_path}")
            continue

        try:
            # Run the analysis
            output_path = run_single_analysis(
                config_path=config_path,
                skip_ppt=skip_ppt,  # Skip individual PPTs for faster comparison
                verbose=verbose,
            )

            results_info[label] = {
                'provider': provider,
                'config_path': config_path,
                'results_dir': output_path,
            }

            print(f"\nCompleted: {label} -> {output_path}")

        except Exception as e:
            print(f"ERROR running {label}: {e}")
            import traceback
            traceback.print_exc()

    return results_info


def collect_existing_results(baselines: List[Dict]) -> Dict[str, Dict]:
    """
    Collect results from existing analysis runs without re-running.

    Args:
        baselines: List of baseline config dictionaries

    Returns:
        Dictionary mapping labels to results info
    """
    from src.vleo_eo.config import load_config

    results_info = {}

    for baseline in baselines:
        config_path = baseline['config']
        label = baseline['label']
        provider = baseline.get('provider', label)

        if not Path(config_path).exists():
            print(f"  Warning: Config not found: {config_path}")
            continue

        # Load config to find output directory
        config = load_config(config_path)
        output_path = Path(config.output_dir)

        if output_path.exists():
            # Check for Excel file
            excel_files = list(output_path.glob('*.xlsx'))
            if excel_files:
                results_info[label] = {
                    'provider': provider,
                    'config_path': config_path,
                    'results_dir': output_path,
                }
                print(f"  Found existing results: {label} -> {output_path}")
            else:
                print(f"  Warning: No Excel file in {output_path}")
        else:
            print(f"  Warning: Results directory not found: {output_path}")

    return results_info


def generate_comparison_report(
    results_info: Dict[str, Dict],
    output_dir: Path,
) -> Path:
    """
    Generate comparison report from collected results.

    Args:
        results_info: Dictionary mapping labels to results info
        output_dir: Output directory for comparison report

    Returns:
        Path to comparison output directory
    """
    from src.vleo_eo.comparison import (
        load_comparison_results, load_raw_delay_data,
        generate_comparison_excel,
    )

    print("\n" + "=" * 70)
    print("Generating Comparison Report")
    print("=" * 70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Load comparison results
    print("\nLoading comparison data...")
    results = load_comparison_results(results_info)

    # Build results_dirs mapping for raw data loading
    results_dirs = {label: info['results_dir'] for label, info in results_info.items()}

    # Load raw data for detailed comparisons
    delay_data = load_raw_delay_data(results_dirs)

    # Generate comparison Excel
    print("\nGenerating comparison Excel...")
    excel_path = generate_comparison_excel(
        results, delay_data,
        output_dir / 'comparison_summary.xlsx',
    )

    # Generate comparison visualizations
    print("\nGenerating comparison visualizations...")
    try:
        from scripts.generate_comparison_report import (
            generate_all_comparison_plots, build_comparison_presentation,
        )

        generate_all_comparison_plots(
            results, delay_data, results_info,
            plots_dir,
        )

        # Build comparison PowerPoint
        print("\nBuilding comparison PowerPoint...")
        ppt_path = build_comparison_presentation(
            plots_dir, results, results_info,
            output_dir / 'comparison_report.pptx',
        )

    except ImportError as e:
        print(f"  Warning: Could not import comparison report generator: {e}")
        print("  Skipping visualization generation.")

    print(f"\nComparison outputs saved to: {output_dir}")
    return output_dir


def main():
    """CLI entry point for comparison analysis."""
    parser = argparse.ArgumentParser(
        description='Run all baseline configurations and generate comparison report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all 3 default baselines
    python run_comparison.py

    # Custom configs
    python run_comparison.py --configs a.yaml b.yaml --labels "Config A" "Config B"

    # Skip re-running, use existing results
    python run_comparison.py --skip-runs

    # Custom output directory
    python run_comparison.py --output-dir results/my_comparison
        """
    )

    parser.add_argument(
        '--configs',
        nargs='+',
        default=None,
        help='Config files to compare (default: 3 baseline configs)'
    )

    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Labels for each config (must match number of configs)'
    )

    parser.add_argument(
        '--skip-runs',
        action='store_true',
        help='Skip running analyses, just regenerate comparison from existing results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for comparison report'
    )

    parser.add_argument(
        '--skip-individual-ppts',
        action='store_true',
        help='Skip generating individual PowerPoints for each baseline (faster)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("VLEO EO Coverage Analysis - Ground Station Provider Comparison")
    print("=" * 70)

    # Build baselines list
    if args.configs:
        if args.labels and len(args.labels) != len(args.configs):
            print("ERROR: Number of labels must match number of configs")
            sys.exit(1)

        baselines = []
        for i, config in enumerate(args.configs):
            label = args.labels[i] if args.labels else Path(config).stem
            baselines.append({
                'config': config,
                'label': label,
                'provider': label,
            })
    else:
        baselines = DEFAULT_BASELINES

    print(f"\nConfigurations to compare:")
    for baseline in baselines:
        print(f"  - {baseline['label']}: {baseline['config']}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'results/comparison_{timestamp}')

    print(f"\nComparison output: {output_dir}")

    # Run or collect results
    if args.skip_runs:
        print("\nSkipping analysis runs, collecting existing results...")
        results_info = collect_existing_results(baselines)

        if not results_info:
            print("ERROR: No existing results found. Run without --skip-runs first.")
            sys.exit(1)
    else:
        print("\nRunning baseline analyses...")
        results_info = run_sequential_analyses(
            baselines,
            skip_ppt=args.skip_individual_ppts,
            verbose=args.verbose,
        )

        if not results_info:
            print("ERROR: No analyses completed successfully.")
            sys.exit(1)

    # Generate comparison report
    generate_comparison_report(results_info, output_dir)

    print("\n" + "=" * 70)
    print("Comparison Complete")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Excel:      {output_dir / 'comparison_summary.xlsx'}")
    print(f"  PowerPoint: {output_dir / 'comparison_report.pptx'}")
    print(f"  Plots:      {output_dir / 'plots/'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
