"""Command line interface for the PoC pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from privacy_vlm_poc.analyzer import analyze_video
from privacy_vlm_poc.evaluation import evaluate_labels
from privacy_vlm_poc.model_selection import UI_OLLAMA_MODELS, ensure_ollama_models, model_candidates, ollama_doctor
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
    parser.add_argument("--vlm-model", default=None, help="Optional per-run model override for the selected backend")
    parser.add_argument("--resize-width", type=int, default=None)


def analyze_command(args: argparse.Namespace) -> int:
    result = analyze_video(
        video_path=args.video,
        sampling_method=args.sampling,
        num_frames=args.num_frames,
        mask_method=args.mask,
        roi=args.roi,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
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
        vlm_model=args.vlm_model,
        resize_width=args.resize_width,
    )
    console.print_json(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
    return 0


def doctor_command(_args: argparse.Namespace) -> int:
    table = Table(title="Research VLM Model Selection")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Command")
    for candidate in model_candidates():
        table.add_row(candidate.role, candidate.name, candidate.expected_download_size, candidate.command)
    console.print(table)

    result = ollama_doctor()
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
    if result.host_reachable and result.configured_model_present and result.ollama_command_available:
        console.print("[green]Ollama VLM backend is ready.[/green]")
        return 0
    console.print("[yellow]Ollama VLM backend is not fully ready. See notes above.[/yellow]")
    return 1


def bootstrap_command(args: argparse.Namespace) -> int:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    result = ensure_ollama_models(models)
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
    if result.host_reachable and result.sample_data_ready:
        console.print("[green]Runtime assets are ready.[/green]")
        return 0
    console.print("[yellow]Runtime assets are not fully ready. See notes above.[/yellow]")
    return 1


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

    doctor_parser = subparsers.add_parser("doctor", help="Check local VLM research readiness")
    doctor_parser.set_defaults(func=doctor_command)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Create local defaults, sample data, and pull UI Ollama models",
    )
    bootstrap_parser.add_argument(
        "--models",
        default=",".join(UI_OLLAMA_MODELS),
        help="Comma-separated Ollama models to ensure locally",
    )
    bootstrap_parser.set_defaults(func=bootstrap_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
