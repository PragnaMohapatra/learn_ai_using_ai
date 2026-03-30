"""
CLI Entry Point — Decode AI Using AI, Episode 1
Commands:
  clean   — Clean a CSV file
  run-nb  — Execute a Jupyter notebook
"""

import argparse
import json
import sys
from src.cleaner import clean_csv
from src.notebook_runner import run_notebook


def cmd_clean(args):
    """Handle the 'clean' subcommand."""
    print(f"🧹 Cleaning: {args.input}")
    print(f"   Strategy: {args.missing}")
    print(f"   Output:   {args.output}")
    print()

    report = clean_csv(args.input, args.output, missing_strategy=args.missing)

    print("✅ Cleaning complete!")
    print(f"   Original rows:      {report['original_rows']}")
    print(f"   Cleaned rows:       {report['cleaned_rows']}")
    print(f"   Duplicates dropped: {report['duplicates_dropped']}")
    print(f"   Missing handled:    {report['missing_values_handled']}")
    print(f"   Saved to:           {report['output_file']}")

    if args.json:
        print("\n📊 JSON Report:")
        print(json.dumps(report, indent=2))

    return report

def cmd_run_nb(args):
    """Handle the 'run-nb' subcommand."""
    print(f"📓 Running notebook: {args.notebook}")
    print()

    report = run_notebook(
        notebook_path=args.notebook,
        output_path=args.output,
        timeout=args.timeout,
    )

    status = "✅ Success" if report["success"] else "❌ Failed"
    print(f"{status}")
    print(f"   Code cells:      {report['code_cells']}")
    print(f"   Execution time:  {report['execution_time_seconds']}s")
    print(f"   Output saved to: {report['output']}")

    if report["errors"]:
        print(f"   Errors: {report['errors']}")

    if args.json:
        print("\n📊 JSON Report:")
        print(json.dumps(report, indent=2))

    return report

def main():
    parser = argparse.ArgumentParser(
        prog="ai-workflow",
        description="🤖 Decode AI Using AI — Episode 1: Data Cleaner + Notebook Runner",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- clean command ---
    clean_parser = subparsers.add_parser("clean", help="Clean a messy CSV file")
    clean_parser.add_argument("input", help="Path to input CSV file")
    clean_parser.add_argument("output", help="Path to save cleaned CSV")
    clean_parser.add_argument(
        "--missing",
        choices=["drop", "fill"],
        default="drop",
        help="Strategy for missing values: drop rows or fill (default: drop)",
    )
    clean_parser.add_argument(
        "--json", action="store_true", help="Print JSON report",
    )

    # --- run-nb command ---
    nb_parser = subparsers.add_parser("run-nb", help="Execute a Jupyter notebook")
    nb_parser.add_argument("notebook", help="Path to .ipynb file")
    nb_parser.add_argument(
        "--output", default=None, help="Path to save executed notebook",
    )
    nb_parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout per cell in seconds (default: 300)",
    )
    nb_parser.add_argument(
        "--json", action="store_true", help="Print JSON report",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "clean":
        cmd_clean(args)
    elif args.command == "run-nb":
        cmd_run_nb(args)


if __name__ == "__main__":
    main()

