"""Run with python -m seaice_diagnostics --help."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="2023–2025 ver2 sea-ice forecast diagnostics"
    )
    parser.add_argument(
        "--config", help="Path configuration JSON (or set SEAICE_CONFIG)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-data", help="Verify bundled numerical data hashes")
    commands.add_parser("audit", help="Verify all 1128 frozen external archives")
    p = commands.add_parser(
        "figures", help="Redraw manuscript figures from bundled numerical inputs"
    )
    p.add_argument(
        "--only", type=int, nargs="+", choices=range(1, 8), default=list(range(1, 8))
    )
    p = commands.add_parser(
        "infer", help="Run checkpoints; creates new archives, not the frozen release"
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["cnn", "unet", "gnn"],
        default=["cnn", "unet", "gnn"],
    )
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument(
        "--init-date", action="append", help="Limit to listed initialization date(s)"
    )
    p = commands.add_parser(
        "evaluate", help="Recompute the published frozen experiment"
    )
    p.add_argument("stage", choices=["ocean", "states", "conditions", "all"])
    commands.add_parser(
        "prepare-cache", help="Build cache from already regridded NetCDF files"
    )
    args = parser.parse_args()
    if args.config:
        os.environ["SEAICE_CONFIG"] = args.config
    from . import config as c
    from . import pipeline as p

    if args.command == "check-data":
        p.check_data()
    elif args.command == "audit":
        p.audit()
    elif args.command == "figures":
        p.check_data()
        p.plot(args.only)
    elif args.command == "infer":
        from .inference import run_inference
        from .protocol import paired_initializations

        dates = (
            args.init_date
            or paired_initializations(
                [2023, 2024, 2025], (7, 8, 9, 10)
            ).init_date.tolist()
        )
        run_inference(
            models=args.models,
            years=[2023, 2024, 2025],
            output_root=c.ARCHIVES,
            init_dates=dates,
            device=args.device,
        )
    elif args.command == "evaluate":
        p.initialize()
        if args.stage in ["all", "ocean"]:
            p.evaluate_ocean()
        if args.stage in ["all", "states"]:
            from .evaluate_states import main as compute
            from .summarize_states import main as summarize

            compute()
            summarize()
        if args.stage in ["all", "conditions"]:
            from .evaluate_conditions import main as compute
            from .summarize_conditions import main as summarize

            compute()
            summarize()
    elif args.command == "prepare-cache":
        import glob
        import pandas as pd
        from .prepare_cache import create_or_load_grid_daily_cache
        from .inference import VARIABLES

        paths = [
            sorted(glob.glob(c.raw_path(k)))
            for k in ["era5_glob", "piomas_sic_glob", "piomas_sit_glob"]
        ]
        if not all(paths):
            raise FileNotFoundError(
                "Missing ERA5 or regridded PIOMAS files; see docs/data.md"
            )
        create_or_load_grid_daily_cache(
            era5_file_list=paths[0],
            piomas_sic_file_list=paths[1],
            piomas_sit_file_list=paths[2],
            variables=VARIABLES,
            valid_dates=pd.date_range("1979-01-01", "2025-12-31"),
            cache_dir=c.CACHE,
        )


if __name__ == "__main__":
    main()
