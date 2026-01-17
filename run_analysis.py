#!/usr/bin/env python3
"""
VLEO EO Coverage Analysis CLI

Run satellite coverage, downlink, and delivery timeline analysis.

Usage:
    python run_analysis.py --config configs/vleo_eo_default.yaml
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='VLEO EO Coverage and Downlink Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py --config configs/vleo_eo_default.yaml
    python run_analysis.py --config configs/vleo_eo_default.yaml --skip-ppt
    python run_analysis.py --config configs/vleo_eo_default.yaml --output-dir results/run1
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--skip-excel',
        action='store_true',
        help='Skip Excel report generation'
    )

    parser.add_argument(
        '--skip-ppt',
        action='store_true',
        help='Skip PowerPoint report generation'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip RGT-Min optimization even if enabled in config'
    )

    parser.add_argument(
        '--optimization-only',
        action='store_true',
        help='Run only the optimization phase (requires existing analysis data)'
    )

    args = parser.parse_args()

    # Import after argument parsing to speed up --help
    from src.vleo_eo.config import load_config, AnalysisConfig
    from src.vleo_eo.orbits import (
        create_tle_data, propagate_orbits, validate_propagation, TLEData
    )
    from src.vleo_eo.coverage import (
        load_targets, calculate_access_windows, filter_access_by_swath,
        calculate_mid_point, validate_coverage
    )
    from src.vleo_eo.contacts import (
        calculate_contact_windows, calculate_ttnc_ka, validate_contacts
    )
    from src.vleo_eo.data_model import (
        simulate_backlog, calculate_downlink_kpis, validate_backlog
    )
    from src.vleo_eo.plots import (
        plot_ground_tracks, plot_coverage_map, plot_comm_cones,
        plot_access_statistics, plot_contact_validity,
        plot_backlog_timeseries, plot_ttnc_distribution
    )
    from src.vleo_eo.reports import generate_excel_report, generate_ppt_report
    from src.vleo_eo.optimization import run_optimization, optimization_results_to_dataframe
    from src.vleo_eo.propulsion import (
        AtmosphericModel, SpacecraftConfig, HallThrusterConfig,
        analyze_station_keeping, generate_propulsion_report,
        print_station_keeping_summary, ltdn_to_raan, calculate_sso_inclination,
        THRUSTER_PRESETS
    )

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for CLI
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("VLEO EO Coverage Analysis")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Create output directories
    output_path = Path(config.output_dir)
    plots_path = output_path / config.plots_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_path}")
    print(f"Analysis period: {config.start_date} to {config.end_datetime.strftime('%Y-%m-%d')}")
    print(f"Satellites: {len(config.satellites)}")
    print(f"Ground stations: {len(config.ground_stations)}")
    print(f"Imaging modes: {len(config.imaging_modes)}")

    # =========================================================================
    # Parse orbit configuration (LTDN, auto SSO)
    # =========================================================================
    orbit_config = config.raw_config.get('orbit', {})
    ltdn_hours = orbit_config.get('ltdn_hours')
    auto_sso = orbit_config.get('auto_sso', False)

    # Calculate RAAN from LTDN and SSO inclination if enabled
    satellite_overrides = {}
    if auto_sso and ltdn_hours is not None:
        print(f"\nOrbit Configuration:")
        print(f"  LTDN: {ltdn_hours:.2f} hours ({int(ltdn_hours)}:{int((ltdn_hours % 1) * 60):02d} local time)")
        print(f"  Auto SSO: enabled")

        for sat in config.satellites:
            # Calculate SSO inclination for this altitude
            sso_inc = calculate_sso_inclination(sat.altitude_km)
            # Calculate RAAN from LTDN
            sso_raan = ltdn_to_raan(ltdn_hours, config.start_datetime)

            satellite_overrides[sat.sat_id] = {
                'inclination_deg': sso_inc,
                'raan_deg': sso_raan,
            }
            print(f"  Satellite {sat.sat_id}: SSO inclination={sso_inc:.2f}°, RAAN={sso_raan:.2f}°")

    # =========================================================================
    # Parse propulsion configuration
    # =========================================================================
    propulsion_config = config.raw_config.get('propulsion', {})
    propulsion_enabled = propulsion_config.get('enabled', False)
    propulsion_result = None
    spacecraft_config = None
    thruster_config = None
    atmosphere_config = None

    if propulsion_enabled:
        # Spacecraft config
        sc_cfg = propulsion_config.get('spacecraft', {})
        spacecraft_config = SpacecraftConfig(
            mass_kg=sc_cfg.get('dry_mass_kg', 500),
            cross_section_m2=sc_cfg.get('cross_section_m2', 2.0),
            drag_coefficient=sc_cfg.get('drag_coefficient', 2.2),
        )

        # Thruster config
        thruster_preset = propulsion_config.get('thruster_preset', 'BHT-600')
        if thruster_preset.lower() == 'custom':
            custom = propulsion_config.get('custom_thruster', {})
            thruster_config = HallThrusterConfig(
                name=custom.get('name', 'Custom HET'),
                thrust_mN=custom.get('thrust_mN', 50.0),
                isp_s=custom.get('isp_s', 1500),
                power_W=custom.get('power_W', 1000),
                propellant=custom.get('propellant', 'Xenon'),
                efficiency=custom.get('efficiency', 0.55),
            )
        else:
            thruster_config = THRUSTER_PRESETS.get(thruster_preset, THRUSTER_PRESETS['BHT-600'])

        # Atmosphere config
        atm_cfg = propulsion_config.get('atmosphere', {})
        atmosphere_config = AtmosphericModel(
            f107=atm_cfg.get('solar_activity_f107', 150),
        )

    # =========================================================================
    # Phase 1: Orbit Propagation
    # =========================================================================
    print("\n" + "-" * 60)
    print("Phase 1: Orbit Propagation")
    print("-" * 60)

    # Create TLE data
    tle_data = {}
    for sat in config.satellites:
        # Apply SSO overrides if available
        inc_deg = sat.inclination_deg
        raan_deg = sat.raan_deg
        if sat.sat_id in satellite_overrides:
            inc_deg = satellite_overrides[sat.sat_id]['inclination_deg']
            raan_deg = satellite_overrides[sat.sat_id]['raan_deg']

        tle = create_tle_data(
            sat_id=sat.sat_id,
            inclination_deg=inc_deg,
            altitude_km=sat.altitude_km,
            raan_deg=raan_deg,
            epoch=config.start_datetime,
            tle_line1=sat.tle_line1,
            tle_line2=sat.tle_line2,
        )
        tle_data[sat.sat_id] = tle
        if args.verbose:
            print(f"  Satellite {sat.sat_id}: {sat.altitude_km} km, {inc_deg:.2f} deg inc, {raan_deg:.2f} deg RAAN")

    # Propagate orbits
    print("\nPropagating orbits...")
    orbit_df = propagate_orbits(
        tle_data,
        config.start_datetime,
        config.end_datetime,
        time_step_s=config.time_step_s,
    )

    # Validate propagation
    prop_validation = validate_propagation(orbit_df)
    if not prop_validation['passed']:
        print("ERROR: Orbit propagation validation failed!")
        for error in prop_validation['errors']:
            print(f"  - {error}")
        sys.exit(1)

    # =========================================================================
    # Phase 1.5: Hall Effect Thruster Station-Keeping Analysis
    # =========================================================================
    propulsion_df = None

    if propulsion_enabled and spacecraft_config and thruster_config:
        print("\n" + "-" * 60)
        print("Phase 1.5: Hall Effect Thruster Station-Keeping Analysis")
        print("-" * 60)

        # Use average altitude from satellites
        avg_altitude = sum(s.altitude_km for s in config.satellites) / len(config.satellites)
        margin_factor = propulsion_config.get('margin_factor', 1.5)

        print(f"\nSpacecraft: {spacecraft_config.mass_kg} kg, {spacecraft_config.cross_section_m2} m² area")
        print(f"Thruster: {thruster_config.name}")
        print(f"Target Altitude: {avg_altitude:.0f} km")
        print(f"Solar Activity (F10.7): {atmosphere_config.f107} SFU")

        propulsion_result = analyze_station_keeping(
            altitude_km=avg_altitude,
            duration_days=config.duration_days,
            spacecraft=spacecraft_config,
            thruster=thruster_config,
            atmosphere=atmosphere_config,
            margin_factor=margin_factor,
        )

        # Generate propulsion report DataFrame
        propulsion_df = generate_propulsion_report(
            propulsion_result, spacecraft_config, thruster_config, atmosphere_config
        )

        # Print summary
        print_station_keeping_summary(propulsion_result, thruster_config)

    # =========================================================================
    # Phase 2: Coverage/Access Calculation
    # =========================================================================
    print("\n" + "-" * 60)
    print("Phase 2: Coverage/Access Calculation")
    print("-" * 60)

    # Load targets
    print("\nLoading targets...")
    targets_gdf = load_targets(
        geojson_path=config.targets.geojson_path,
        target_points=config.targets.target_points,
        use_grid=config.targets.use_grid,
        grid_params={
            'lat_min': config.targets.grid_lat_min,
            'lat_max': config.targets.grid_lat_max,
            'lon_min': config.targets.grid_lon_min,
            'lon_max': config.targets.grid_lon_max,
            'spacing_deg': config.targets.grid_spacing_deg,
        } if config.targets.use_grid else None,
    )
    print(f"  Loaded {len(targets_gdf)} targets")

    # Calculate access windows
    print("\nCalculating access windows...")
    # Use first imaging mode's off-nadir limits
    if config.imaging_modes:
        off_nadir_min = config.imaging_modes[0].off_nadir_min_deg
        off_nadir_max = config.imaging_modes[0].off_nadir_max_deg
    else:
        off_nadir_min = 0.0
        off_nadir_max = 45.0

    # Determine pass type filter based on LTDN
    # For LTDN-defined SSO orbits, imaging occurs on descending passes (daytime)
    if ltdn_hours is not None:
        pass_type = "descending"
        print(f"  Filtering for descending passes only (LTDN: {ltdn_hours:.1f}h)")
    else:
        pass_type = "both"

    access_df = calculate_access_windows(
        orbit_df,
        targets_gdf,
        off_nadir_min_deg=off_nadir_min,
        off_nadir_max_deg=off_nadir_max,
        min_access_duration_s=config.min_access_duration_s,
        pass_type=pass_type,
    )

    if access_df.empty:
        print("WARNING: No access windows found!")
    else:
        print(f"  Found {len(access_df)} access windows")

        # Filter by swath
        print("\nFiltering by sensor swath...")
        access_df, swaths_gdf = filter_access_by_swath(
            access_df, tle_data,
            off_nadir_min_deg=off_nadir_min,
            off_nadir_max_deg=off_nadir_max,
        )
        print(f"  {len(access_df)} accesses after swath filtering")

        # Add midpoints
        access_df = calculate_mid_point(access_df, tle_data)

    # Validate coverage
    coverage_validation = validate_coverage(access_df, config.duration_days)

    # =========================================================================
    # Phase 3: Downlink/Contacts + TTNC
    # =========================================================================
    print("\n" + "-" * 60)
    print("Phase 3: Downlink Contacts and TTNC")
    print("-" * 60)

    if access_df.empty:
        print("Skipping contact calculation (no access windows)")
        mode_dfs = {}
        ttnc_df = __import__('pandas').DataFrame()
    else:
        # Calculate contact windows
        mode_dfs = calculate_contact_windows(
            access_df,
            tle_data,
            config.ground_stations,
            config.imaging_modes,
            slew_rate_deg_per_s=config.slew_rate_deg_per_s,
            mode_switch_time_s=config.mode_switch_time_s,
            contact_buffer_s=config.contact_buffer_s,
            ttnc_max_search_hours=config.ttnc_max_search_hours,
        )

        # Calculate TTNC
        print("\nCalculating TTNC Ka...")
        ttnc_df = calculate_ttnc_ka(mode_dfs, config.ground_stations)

    # Validate contacts
    contact_validation = validate_contacts(mode_dfs, config.ground_stations)

    # =========================================================================
    # Phase 4: Data Generation + Backlog
    # =========================================================================
    print("\n" + "-" * 60)
    print("Phase 4: Data Generation and Backlog")
    print("-" * 60)

    print("\nSimulating data backlog...")
    backlog_df = simulate_backlog(
        mode_dfs,
        config.ground_stations,
        config.start_datetime,
        config.end_datetime,
        time_step_minutes=15.0,
    )

    # Calculate downlink KPIs
    downlink_kpis = calculate_downlink_kpis(mode_dfs, config.ground_stations, backlog_df)

    # Validate backlog
    backlog_validation = validate_backlog(backlog_df, mode_dfs)

    # =========================================================================
    # Phase 4.5: RGT-Min Optimization (if enabled)
    # =========================================================================
    optimization_results = None

    if config.optimization.enabled and not args.skip_optimization:
        print("\n" + "-" * 60)
        print("Phase 4.5: RGT-Min Ground Station Optimization")
        print("-" * 60)

        print(f"\nOptimization Settings:")
        print(f"  Approach: {config.optimization.approach}")
        print(f"  TT&C Requirement: {config.optimization.ttc_contacts_per_rev} contact(s) per revolution")
        print(f"  Ka Delivery SLA: {config.optimization.ka_delivery_percent}% within {config.optimization.ka_delivery_hours} hours")
        print(f"  Providers: {config.optimization.providers if config.optimization.providers else 'All'}")

        # Build collection events DataFrame from mode_dfs
        collect_events = []
        for mode_name, df in mode_dfs.items():
            for _, row in df.iterrows():
                if 'Imaging_Stop' in row and 'Raw_Data_Size_GB' in row:
                    collect_events.append({
                        'sat_id': row['Satellite'],
                        'collect_start': row.get('Squint_Start_Time', row['Start_Time']),
                        'collect_end': row['Imaging_Stop'],
                        'data_volume_gb': row['Raw_Data_Size_GB'],
                        'mode': mode_name,
                    })

        import pandas as pd
        collect_df = pd.DataFrame(collect_events)

        print(f"\nRunning optimization for {len(collect_df)} collection events...")

        optimization_results = run_optimization(
            orbit_df,
            collect_df,
            tle_data,
            config,
        )

        if optimization_results:
            print("\n" + "=" * 40)
            print("Optimization Results Summary")
            print("=" * 40)
            for provider, result in optimization_results.items():
                print(f"\n{provider}:")
                print(f"  TT&C Stations: {len(result.ttc_stations)} ({result.ttc_coverage_percent:.1f}% coverage)")
                for s in result.ttc_stations:
                    print(f"    - {s.name}")
                if result.ka_stations:
                    print(f"  Ka Stations: {len(result.ka_stations)} ({result.ka_delivery_percent:.1f}% delivery)")
                    for s in result.ka_stations:
                        print(f"    - {s.name}")
                print(f"  TT&C SLA: {'MET' if result.ttc_satisfied else 'NOT MET'}")
                print(f"  Ka SLA: {'MET' if result.ka_satisfied else 'NOT MET'}")
        else:
            print("\nNo optimization results generated.")

    # =========================================================================
    # Phase 5: Reporting
    # =========================================================================
    print("\n" + "-" * 60)
    print("Phase 5: Generating Reports")
    print("-" * 60)

    plot_paths = {}

    if not args.skip_plots:
        print("\nGenerating plots...")

        # Ground tracks
        print("  - Ground tracks...")
        fig = plot_ground_tracks(tle_data, config, duration_minutes=100,
                                 output_path=plots_path / 'ground_tracks.png')
        plt.close(fig)
        plot_paths['ground_tracks'] = plots_path / 'ground_tracks.png'

        # Coverage map
        if not access_df.empty:
            print("  - Coverage map...")
            fig = plot_coverage_map(access_df, targets_gdf, config,
                                    output_path=plots_path / 'coverage_map.png')
            plt.close(fig)
            plot_paths['coverage_map'] = plots_path / 'coverage_map.png'

            # Access statistics
            print("  - Access statistics...")
            fig = plot_access_statistics(access_df, config,
                                         output_path=plots_path / 'access_statistics.png')
            plt.close(fig)
            plot_paths['access_statistics'] = plots_path / 'access_statistics.png'

        # Communication cones
        if config.ground_stations:
            print("  - Communication cones...")
            fig = plot_comm_cones(config.ground_stations, tle_data, targets_gdf, config,
                                  output_path=plots_path / 'comm_cones.png')
            plt.close(fig)
            plot_paths['comm_cones'] = plots_path / 'comm_cones.png'

        # Contact validity
        if mode_dfs:
            print("  - Contact validity...")
            fig = plot_contact_validity(mode_dfs, config.ground_stations, config,
                                        output_path=plots_path / 'contact_validity.png')
            plt.close(fig)
            plot_paths['contact_validity'] = plots_path / 'contact_validity.png'

        # Backlog
        if not backlog_df.empty:
            print("  - Backlog timeseries...")
            fig = plot_backlog_timeseries(backlog_df, config,
                                          output_path=plots_path / 'backlog_timeseries.png')
            plt.close(fig)
            plot_paths['backlog_timeseries'] = plots_path / 'backlog_timeseries.png'

        # TTNC distribution
        if not ttnc_df.empty:
            print("  - TTNC distribution...")
            fig = plot_ttnc_distribution(ttnc_df, config,
                                         output_path=plots_path / 'ttnc_distribution.png')
            plt.close(fig)
            plot_paths['ttnc_distribution'] = plots_path / 'ttnc_distribution.png'

    # Generate Excel report
    if not args.skip_excel:
        print("\nGenerating Excel report...")
        excel_path = generate_excel_report(
            config,
            access_df,
            mode_dfs,
            ttnc_df,
            backlog_df,
            coverage_validation,
            contact_validation,
            downlink_kpis,
            output_path=output_path / config.excel_filename,
            optimization_results=optimization_results,
            propulsion_df=propulsion_df,
        )

    # Generate PowerPoint report
    if not args.skip_ppt:
        print("\nGenerating PowerPoint report...")
        ppt_path = generate_ppt_report(
            config,
            coverage_validation,
            contact_validation,
            downlink_kpis,
            ttnc_df,
            plot_paths,
            output_path=output_path / config.ppt_filename,
            optimization_results=optimization_results,
        )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    print(f"\nOutput files:")
    print(f"  Excel: {output_path / config.excel_filename}")
    if not args.skip_ppt:
        print(f"  PowerPoint: {output_path / config.ppt_filename}")
    print(f"  Plots: {plots_path}/")

    # Print key metrics
    print(f"\nKey Metrics:")
    if coverage_validation.get('stats'):
        stats = coverage_validation['stats']
        print(f"  Total Accesses: {stats.get('total_accesses', 0)}")
        print(f"  Accesses/Day: {stats.get('accesses_per_day', 0):.1f}")

    if contact_validation.get('stats'):
        stats = contact_validation['stats']
        print(f"  Valid Contacts: {stats.get('total_valid_contacts', 0)}")
        print(f"  Contact Success: {stats.get('valid_percentage', 0):.1f}%")

    if downlink_kpis:
        print(f"  Data Collected: {downlink_kpis.get('total_data_collected_gb', 0):.1f} GB")
        print(f"  Data Downlinked: {downlink_kpis.get('total_data_downlinked_gb', 0):.1f} GB")
        print(f"  Peak Backlog: {downlink_kpis.get('peak_backlog_gb', 0):.2f} GB")

    if not ttnc_df.empty:
        valid_ttnc = ttnc_df['TTNC_Ka_minutes'].dropna()
        if not valid_ttnc.empty:
            print(f"  TTNC Ka Median: {valid_ttnc.median():.1f} min")
            print(f"  TTNC Ka P95: {valid_ttnc.quantile(0.95):.1f} min")

    if optimization_results:
        print(f"\nOptimization Results:")
        for provider, result in optimization_results.items():
            status = "MET" if (result.ttc_satisfied and result.ka_satisfied) else "NOT MET"
            print(f"  {provider}: {len(result.ttc_stations)} TT&C + {len(result.ka_stations)} Ka stations (SLA: {status})")

    if propulsion_result:
        print(f"\nStation-Keeping (HET):")
        print(f"  Thruster: {thruster_config.name}")
        print(f"  Delta-V Required: {propulsion_result.total_delta_v_m_s:.2f} m/s")
        print(f"  Propellant ({thruster_config.propellant}): {propulsion_result.propellant_mass_kg:.2f} kg")
        print(f"  Propellant (with margin): {propulsion_result.propellant_mass_kg * propulsion_result.margin_factor:.2f} kg")
        print(f"  Firing Time: {propulsion_result.total_firing_time_hours:.1f} hours ({propulsion_result.duty_cycle_percent:.1f}% duty)")
        print(f"  Annualized Propellant: {propulsion_result.propellant_per_year_kg:.1f} kg/year")

    print("\nDone!")


if __name__ == '__main__':
    main()
