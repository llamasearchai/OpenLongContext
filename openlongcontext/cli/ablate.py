def main(args):
    from openlongcontext.ablation.experiment_registry import run_ablation
    print(f"Running ablation with config: {args.config_name}")
    run_ablation(args.config_name)
