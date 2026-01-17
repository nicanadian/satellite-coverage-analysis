#!/usr/bin/env python3
"""
Performance Benchmarking Script for VLEO EO Coverage Analysis

Measures execution time for various parameter configurations.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_analysis(config_path: str, output_dir: str, skip_ppt: bool = True) -> dict:
    """
    Run the analysis and measure execution time.

    Returns dict with timing and status information.
    """
    cmd = [
        sys.executable, "run_analysis.py",
        "--config", config_path,
        "--output-dir", output_dir,
    ]
    if skip_ppt:
        cmd.append("--skip-ppt")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        return {
            "success": success,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "elapsed_seconds": 600,
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout after 600 seconds",
        }
    except Exception as e:
        return {
            "success": False,
            "elapsed_seconds": time.time() - start_time,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


def run_benchmark_suite():
    """Run the full benchmark suite."""

    print("=" * 70)
    print("VLEO EO Coverage Analysis - Performance Benchmark")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Test configurations to benchmark
    test_configs = [
        ("Single Satellite", "configs/test_single_sat.yaml", "results/bench_single_sat"),
        ("Europe AOI (3 sats, 10 targets)", "configs/test_europe_aoi.yaml", "results/bench_europe"),
        ("Middle East AOI (3 sats, 8 targets)", "configs/test_middleeast_aoi.yaml", "results/bench_middleeast"),
        ("APAC AOI (3 sats, 10 targets)", "configs/test_apac_aoi.yaml", "results/bench_apac"),
        ("Default (5 sats, 5 targets, 7 days)", "configs/vleo_eo_default.yaml", "results/bench_default"),
    ]

    results = []

    for name, config_path, output_dir in test_configs:
        if not Path(config_path).exists():
            print(f"\n[SKIP] {name}: Config not found at {config_path}")
            continue

        print(f"\n{'─' * 70}")
        print(f"Running: {name}")
        print(f"Config: {config_path}")
        print(f"Output: {output_dir}")
        print("─" * 70)

        result = run_analysis(config_path, output_dir)
        result["name"] = name
        result["config"] = config_path
        results.append(result)

        if result["success"]:
            print(f"✓ Completed in {result['elapsed_seconds']:.1f} seconds")
        else:
            print(f"✗ Failed (code {result['returncode']})")
            if result["stderr"]:
                print(f"  Error: {result['stderr'][:200]}")

    # Print summary
    print("\n")
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Test Name':<45} {'Status':<10} {'Time (s)':<10}")
    print("-" * 70)

    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"{r['name']:<45} {status:<10} {r['elapsed_seconds']:>8.1f}")

    print("-" * 70)

    # Calculate totals
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    total_time = sum(r["elapsed_seconds"] for r in results)

    print(f"{'TOTAL':<45} {passed}/{total:<8} {total_time:>8.1f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_benchmark_suite()

    # Exit with error if any test failed
    if not all(r["success"] for r in results):
        sys.exit(1)
