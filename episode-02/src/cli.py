"""CLI entry point for Episode 2: Data Analysis for AI."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eda import generate_basic_profile

def cmd_analyze(args):
    print(f"Analyzing: {args.input}")
    print(f"Output:    {args.output}")
    print()

    report = generate_basic_profile(
        args.input,
        args.output,
        with_charts=args.charts,
        with_ai_summary=args.ai_summary,
    )

    print("Analysis complete!")
    print(f"   Rows:            {report['rows']}")
    print(f"   Columns:         {report['columns_count']}")
    print(f"   Numeric columns: {len(report['numeric_columns'])}")
    print(f"   Date columns:    {len(report['datetime_columns'])}")
    print(f"   Report file:     {report['report_file']}")
    print(f"   HTML report:     {report['html_report_file']}")
    print(f"   Charts:          {'enabled' if args.charts else 'disabled'}")

    if args.ai_summary:
        ai_narrative = report["ai_narrative"] or {}
        print(f"   AI summary:      {ai_narrative.get('status', 'unknown')}")
        if ai_narrative.get("reason"):
            print(f"   AI detail:       {ai_narrative['reason']}")

    if args.json:
        print("   JSON artifact:   written")

def main():
    parser = argparse.ArgumentParser(
        prog="ai-workflow-ep2",
        description="Decode AI Using AI — Episode 2: Data Analysis for AI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Profile a cleaned CSV file")
    analyze_parser.add_argument("input", help="Path to the cleaned CSV")
    analyze_parser.add_argument("output", help="Directory where reports should be written")
    analyze_parser.add_argument("--charts", action="store_true", help="Embed chart images in the HTML report")
    analyze_parser.add_argument(
        "--ai-summary",
        action="store_true",
        help="Request an AI-written narrative summary when OPENAI_API_KEY is configured",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Confirm JSON artifact generation in summary output",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

