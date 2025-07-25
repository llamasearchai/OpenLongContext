def main(args):
    from openlongcontext.theory.capacity import plot_scaling_law
    print(f"Plotting scaling law results from: {args.results_path}")
    plot_scaling_law(args.results_path)