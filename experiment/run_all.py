"""Run all attention output variants sequentially."""

import argparse
import sys
from .config import baseline_config, rezero_config, rezero_pvh_config
from .train import train


VARIANT_CONFIGS = {
    "baseline": baseline_config,
    "rezero": rezero_config,
    "rezero_pvh": rezero_pvh_config,
}


def main():
    parser = argparse.ArgumentParser(description="Run QK-Norm attention variant experiments")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANT_CONFIGS.keys()),
        choices=list(VARIANT_CONFIGS.keys()),
        help="Which variants to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", default="runs")
    args = parser.parse_args()

    results = {}

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"  Running variant: {variant}")
        print(f"{'='*60}\n")

        overrides = {"seed": args.seed, "log_dir": args.log_dir}
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps

        config = VARIANT_CONFIGS[variant](**overrides)
        val_loss, val_ppl = train(config)
        results[variant] = {"val_loss": val_loss, "val_ppl": val_ppl}

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Val Loss':>10} {'Val PPL':>10}")
    print("-" * 42)
    for variant, r in results.items():
        print(f"{variant:<20} {r['val_loss']:>10.4f} {r['val_ppl']:>10.2f}")


if __name__ == "__main__":
    main()
