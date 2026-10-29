"""Command line: `dnd2json <command>` (or `python -m dnd2json.cli <command>`).

    dnd2json train                      train the block classifier on data/gold
    dnd2json convert --all              convert every PDF in media/eng/pdf
    dnd2json convert "path/to/x.pdf"    convert specific PDFs
    dnd2json check                      compare outputs with data/expected
    dnd2json extract                    only stage 1 (blocks CSVs), e.g. to pre-run OCR
    dnd2json evaluate [models...]       compare classifier models (step 3/4 harness)

Typical first run:  dnd2json train  ->  dnd2json convert --all  ->  dnd2json check
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dnd2json import config
from dnd2json.errors import DnD2JsonError


def cmd_train(args) -> int:
    from dnd2json.classifier import train
    meta = train()
    print(f"Trained on {meta['rows']} blocks {meta['label_counts']}")
    print(f"Saved to {config.MODEL_PATH}")
    return 0


def cmd_convert(args) -> int:
    from dnd2json.pipeline import convert_many
    pdfs = _pdf_list(args)
    if not pdfs:
        print("Nothing to convert: give PDF paths or --all.")
        return 2
    report = convert_many(pdfs, Path(args.out), force_extract=args.force_extract,
                          debug=args.debug)
    counts = {status: sum(r["status"] == status for r in report)
              for status in ("ok", "warnings", "failed", "error")}
    print(f"\n{len(report)} PDFs: {counts['ok']} ok, {counts['warnings']} with warnings, "
          f"{counts['failed']} failed, {counts['error']} crashed (bugs)")
    print(f"JSON in {args.out}, run report in {Path(args.out) / '_report.json'}")
    return 1 if counts["failed"] or counts["error"] else 0


def cmd_check(args) -> int:
    from dnd2json.check import run_check
    run_check(Path(args.out), Path(args.expected))
    return 0


def cmd_extract(args) -> int:
    """Stage 1 only. Text-layer PDFs here; scanned ones through the OCR environment."""
    from dnd2json import extract
    common = ["--in", str(args.input), "--out", str(config.BLOCKS_DIR)]
    if args.force:
        common.append("--force")
    status = extract.main(common + ["--only", "text"])
    if args.text_only:
        return status
    if not config.OCR_PYTHON.exists():
        print(f"\nOCR environment not found ({config.OCR_PYTHON}): scanned PDFs skipped. "
              "See README, 'OCR setup'.")
        return status or 1
    ocr = subprocess.run([str(config.OCR_PYTHON), str(config.EXTRACT_SCRIPT), *common,
                          "--only", "ocr"])
    return status or ocr.returncode


def cmd_evaluate(args) -> int:
    from dnd2json import evaluate
    evaluate.main(args.models)
    return 0


def _pdf_list(args) -> list[Path]:
    if args.all:
        return sorted(Path(args.input).glob("*.pdf"))
    missing = [p for p in args.pdfs if not Path(p).exists()]
    for path in missing:
        print(f"Not found: {path}")
    return [Path(p) for p in args.pdfs if Path(p).exists()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnd2json", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("train", help="train the block classifier").set_defaults(func=cmd_train)

    convert = commands.add_parser("convert", help="PDF -> JSON")
    convert.add_argument("pdfs", nargs="*", help="PDF files to convert")
    convert.add_argument("--all", action="store_true", help="every PDF in --in")
    convert.add_argument("--in", dest="input", default=str(config.PDF_DIR))
    convert.add_argument("--out", default=str(config.OUTPUT_DIR))
    convert.add_argument("--force-extract", action="store_true",
                         help="redo stage 1 even if blocks CSVs are up to date")
    convert.add_argument("--debug", action="store_true",
                         help="also write <out>/debug/<name>.debug.csv: every block's role")
    convert.set_defaults(func=cmd_convert)

    check = commands.add_parser("check", help="compare outputs with expected JSON")
    check.add_argument("--out", default=str(config.OUTPUT_DIR))
    check.add_argument("--expected", default=str(config.EXPECTED_DIR))
    check.set_defaults(func=cmd_check)

    extract = commands.add_parser("extract", help="stage 1 only: PDFs -> blocks CSVs")
    extract.add_argument("--in", dest="input", default=str(config.PDF_DIR))
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--text-only", action="store_true", help="skip OCR PDFs")
    extract.set_defaults(func=cmd_extract)

    evaluate = commands.add_parser("evaluate", help="compare classifier models")
    evaluate.add_argument("models", nargs="*")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DnD2JsonError as e:  # expected problems: show the message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
