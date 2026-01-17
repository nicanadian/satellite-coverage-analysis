#!/usr/bin/env python3
"""
Output Validation Script for VLEO EO Coverage Analysis

Validates that all expected outputs are generated and contain valid data.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd


def check_file_exists(path: Path, min_size_kb: float = 0) -> Tuple[bool, str]:
    """Check if file exists and meets minimum size requirement."""
    if not path.exists():
        return False, f"File not found: {path}"

    size_kb = path.stat().st_size / 1024
    if size_kb < min_size_kb:
        return False, f"File too small ({size_kb:.1f} KB < {min_size_kb} KB): {path}"

    return True, f"OK ({size_kb:.1f} KB)"


def validate_excel_report(excel_path: Path) -> Dict[str, Any]:
    """Validate Excel report contents."""
    results = {
        "passed": True,
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    if not excel_path.exists():
        results["passed"] = False
        results["errors"].append(f"Excel file not found: {excel_path}")
        return results

    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # Required sheets
        required_sheets = [
            "Config", "Satellites", "Ground_Stations", "Imaging_Modes",
            "Coverage_KPIs", "Access_Windows", "Contacts",
            "Backlog_TimeSeries", "TTNC_Ka"
        ]

        for sheet in required_sheets:
            if sheet in sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet)
                row_count = len(df)
                results["checks"].append(f"Sheet '{sheet}': {row_count} rows")

                if row_count == 0 and sheet not in ["Coverage_KPIs", "TTNC_Summary"]:
                    results["warnings"].append(f"Sheet '{sheet}' is empty")
            else:
                results["errors"].append(f"Missing required sheet: {sheet}")
                results["passed"] = False

        # Validate specific data integrity
        if "Access_Windows" in sheet_names:
            df = pd.read_excel(excel_path, sheet_name="Access_Windows")
            if len(df) > 0:
                # Check for negative durations
                if "Access_Duration" in df.columns:
                    neg_durations = (df["Access_Duration"] < 0).sum()
                    if neg_durations > 0:
                        results["errors"].append(f"Found {neg_durations} negative access durations")
                        results["passed"] = False
                    else:
                        results["checks"].append("Access durations: all positive")

        if "Backlog_TimeSeries" in sheet_names:
            df = pd.read_excel(excel_path, sheet_name="Backlog_TimeSeries")
            if len(df) > 0 and "backlog_gb" in df.columns:
                neg_backlog = (df["backlog_gb"] < 0).sum()
                if neg_backlog > 0:
                    results["errors"].append(f"Found {neg_backlog} negative backlog values")
                    results["passed"] = False
                else:
                    results["checks"].append("Backlog values: all non-negative")

        if "TTNC_Ka" in sheet_names:
            df = pd.read_excel(excel_path, sheet_name="TTNC_Ka")
            if len(df) > 0 and "TTNC_Ka_minutes" in df.columns:
                valid_ttnc = df["TTNC_Ka_minutes"].dropna()
                if len(valid_ttnc) > 0:
                    neg_ttnc = (valid_ttnc < 0).sum()
                    if neg_ttnc > 0:
                        results["errors"].append(f"Found {neg_ttnc} negative TTNC values")
                        results["passed"] = False
                    else:
                        results["checks"].append("TTNC values: all non-negative")

                    # Check ordering: Median <= P90 <= P95
                    median = valid_ttnc.median()
                    p90 = valid_ttnc.quantile(0.90)
                    p95 = valid_ttnc.quantile(0.95)

                    if median <= p90 <= p95:
                        results["checks"].append(f"TTNC ordering valid: Median({median:.1f}) <= P90({p90:.1f}) <= P95({p95:.1f})")
                    else:
                        results["warnings"].append(f"TTNC ordering issue: Median={median:.1f}, P90={p90:.1f}, P95={p95:.1f}")

    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Error reading Excel: {str(e)}")

    return results


def validate_plots(plots_dir: Path) -> Dict[str, Any]:
    """Validate that all expected plots exist and are non-empty."""
    results = {
        "passed": True,
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    expected_plots = [
        ("ground_tracks.png", 10),
        ("coverage_map.png", 10),
        ("comm_cones.png", 10),
        ("access_statistics.png", 10),
        ("contact_validity.png", 10),
        ("backlog_timeseries.png", 10),
        ("ttnc_distribution.png", 10),
    ]

    for plot_name, min_size_kb in expected_plots:
        plot_path = plots_dir / plot_name
        ok, msg = check_file_exists(plot_path, min_size_kb)

        if ok:
            results["checks"].append(f"{plot_name}: {msg}")
        else:
            results["errors"].append(msg)
            results["passed"] = False

    return results


def validate_output_directory(output_dir: str) -> Dict[str, Any]:
    """Validate all outputs in a directory."""
    output_path = Path(output_dir)

    results = {
        "directory": output_dir,
        "passed": True,
        "file_checks": {},
        "excel_validation": {},
        "plot_validation": {},
        "summary": {},
    }

    # Check main files exist
    excel_path = output_path / "results_summary.xlsx"
    ppt_path = output_path / "results_summary.pptx"
    plots_path = output_path / "plots"

    # Excel check
    ok, msg = check_file_exists(excel_path, 10)
    results["file_checks"]["excel"] = {"exists": ok, "message": msg}
    if not ok:
        results["passed"] = False

    # PPT check (optional)
    ok, msg = check_file_exists(ppt_path, 10)
    results["file_checks"]["ppt"] = {"exists": ok, "message": msg}
    # PPT is optional, don't fail if missing

    # Plots directory check
    if plots_path.exists():
        results["plot_validation"] = validate_plots(plots_path)
        if not results["plot_validation"]["passed"]:
            results["passed"] = False
    else:
        results["plot_validation"] = {"passed": False, "errors": ["Plots directory not found"]}
        results["passed"] = False

    # Excel content validation
    if excel_path.exists():
        results["excel_validation"] = validate_excel_report(excel_path)
        if not results["excel_validation"]["passed"]:
            results["passed"] = False

    return results


def print_validation_results(results: Dict[str, Any]) -> None:
    """Print validation results in a readable format."""
    print(f"\n{'=' * 70}")
    print(f"Validation Results: {results['directory']}")
    print("=" * 70)

    status = "PASSED" if results["passed"] else "FAILED"
    print(f"Overall Status: {status}")

    # File checks
    print(f"\n{'─' * 40}")
    print("File Checks:")
    for file_type, check in results["file_checks"].items():
        status = "✓" if check["exists"] else "✗"
        print(f"  {status} {file_type}: {check['message']}")

    # Plot validation
    if results["plot_validation"]:
        print(f"\n{'─' * 40}")
        print("Plot Validation:")
        pv = results["plot_validation"]
        for check in pv.get("checks", []):
            print(f"  ✓ {check}")
        for error in pv.get("errors", []):
            print(f"  ✗ {error}")
        for warning in pv.get("warnings", []):
            print(f"  ⚠ {warning}")

    # Excel validation
    if results["excel_validation"]:
        print(f"\n{'─' * 40}")
        print("Excel Validation:")
        ev = results["excel_validation"]
        for check in ev.get("checks", []):
            print(f"  ✓ {check}")
        for error in ev.get("errors", []):
            print(f"  ✗ {error}")
        for warning in ev.get("warnings", []):
            print(f"  ⚠ {warning}")

    print()


def main():
    """Run validation on specified directories or default test outputs."""

    print("=" * 70)
    print("VLEO EO Coverage Analysis - Output Validation")
    print("=" * 70)

    # Directories to validate
    if len(sys.argv) > 1:
        output_dirs = sys.argv[1:]
    else:
        # Default test output directories
        output_dirs = [
            "results/test_single_sat",
            "results/test_europe",
            "results/test_middleeast",
            "results/test_apac",
        ]

    all_results = []

    for output_dir in output_dirs:
        if not Path(output_dir).exists():
            print(f"\n[SKIP] Directory not found: {output_dir}")
            continue

        results = validate_output_directory(output_dir)
        all_results.append(results)
        print_validation_results(results)

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)

    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}: {r['directory']}")

    print("-" * 70)
    print(f"Total: {passed}/{total} passed")
    print("=" * 70)

    # Exit with error if any validation failed
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
