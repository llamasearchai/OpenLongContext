def main(args):
    from openlongcontext.evaluation.ablation import analyze_results
    print(f"Analyzing results at: {args.results_path}")
    analyze_results(args.results_path)