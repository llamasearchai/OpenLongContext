import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(
        prog="openlongcontext",
        description="OpenLongContext: Production-ready FastAPI service and research platform for long-context document QA and retrieval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run Experiment
    run_exp = subparsers.add_parser("run-experiment", help="Run a single experiment with a config.")
    run_exp.add_argument("--config-name", required=True, help="Name of the experiment config YAML.")

    # Sweep
    sweep = subparsers.add_parser("sweep", help="Run a hyperparameter sweep.")
    sweep.add_argument("--config-name", required=True, help="Name of the sweep config YAML.")
    sweep.add_argument("--multirun", action="store_true", help="Enable multirun mode.")

    # Ablate
    ablate = subparsers.add_parser("ablate", help="Run an ablation study.")
    ablate.add_argument("--config-name", required=True, help="Name of the ablation config YAML.")

    # Analyze
    analyze = subparsers.add_parser("analyze", help="Analyze experiment results.")
    analyze.add_argument("--results-path", required=True, help="Path to results file or directory.")

    # Plot Scaling
    plot = subparsers.add_parser("plot-scaling", help="Plot scaling law results.")
    plot.add_argument("--results-path", required=True, help="Path to scaling law results.")

    # API Server
    api = subparsers.add_parser("api-server", help="Run the FastAPI document QA server.")
    api.add_argument("--host", default="0.0.0.0", help="Host to bind the API server.")
    api.add_argument("--port", type=int, default=8000, help="Port for the API server.")

    args = parser.parse_args()

    if args.command == "run-experiment":
        from .run_experiment import main as run_experiment_main
        run_experiment_main(args)
    elif args.command == "sweep":
        from .sweep import main as sweep_main
        sweep_main(args)
    elif args.command == "ablate":
        from .ablate import main as ablate_main
        ablate_main(args)
    elif args.command == "analyze":
        from .analyze_results import main as analyze_main
        analyze_main(args)
    elif args.command == "plot-scaling":
        from .plot_scaling import main as plot_main
        plot_main(args)
    elif args.command == "api-server":
        import uvicorn
        uvicorn.run("openlongcontext.api:app", host=args.host, port=args.port, reload=True)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()