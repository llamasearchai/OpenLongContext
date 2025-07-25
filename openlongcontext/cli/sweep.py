def main(args):
    from openlongcontext.ablation.experiment_registry import run_sweep
    print(f"Running sweep with config: {args.config_name}, multirun: {args.multirun}")
    run_sweep(args.config_name, multirun=args.multirun)
