def main(args):
    from openlongcontext.core.engine import run_experiment
    print(f"Running experiment with config: {args.config_name}")
    run_experiment(args.config_name)