"""Command line interface for the PoC pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from privacy_vlm_poc.analyzer import analyze_video
from privacy_vlm_poc.evaluation import evaluate_labels
from privacy_vlm_poc.schemas import MaskMethod, ROI, SamplingMethod, VLMBackend

console = Console()


def _parse_roi(value: str | None) -> ROI | None:
    if value is None or value.strip() == "":
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        msg = "--roi must be formatted as x1,y1,x2,y2"
        raise argparse.ArgumentTypeError(msg)
    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        msg = "--roi values must be integers"
        raise argparse.ArgumentTypeError(msg) from exc
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sampling", choices=[item.value for item in SamplingMethod], default=SamplingMethod.HYBRID.value)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--mask", choices=[item.value for item in MaskMethod], default=MaskMethod.NONE.value)
    parser.add_argument("--roi", type=_parse_roi, default=None, help="Optional ROI formatted as x1,y1,x2,y2")
    parser.add_argument("--vlm-backend", choices=[item.value for item in VLMBackend], default=VLMBackend.MOCK.value)
    parser.add_argument("--resize-width", type=int, default=None)


def analyze_command(args: argparse.Namespace) -> int:
    result = analyze_video(
        video_path=args.video,
        sampling_method=args.sampling,
        num_frames=args.num_frames,
        mask_method=args.mask,
        roi=args.roi,
        vlm_backend=args.vlm_backend,
        resize_width=args.resize_width,
    )
    table = Table(title="Analysis Complete")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("run_dir", str(result.run_dir))
    table.add_row("grid", str(result.grid_path))
    table.add_row("result", str(result.result_path))
    table.add_row("report", str(result.report_path))
    table.add_row("selected_frames", ", ".join(str(frame.frame_index) for frame in result.selected_frames))
    console.print(table)
    console.print_json(json.dumps(result.vlm_response.model_dump(mode="json"), ensure_ascii=False))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    metrics = evaluate_labels(
        labels_csv=args.labels,
        sampling_method=args.sampling,
        num_frames=args.num_frames,
        mask_method=args.mask,
        roi=args.roi,
        vlm_backend=args.vlm_backend,
        resize_width=args.resize_width,
    )
    console.print_json(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="privacy-vlm-poc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one video")
    analyze_parser.add_argument("--video", required=True, type=Path)
    _add_common_options(analyze_parser)
    analyze_parser.set_defaults(func=analyze_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a labels.csv file")
    evaluate_parser.add_argument("--labels", required=True, type=Path)
    _add_common_options(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
