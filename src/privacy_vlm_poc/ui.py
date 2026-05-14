"""Gradio web UI for local PoC experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from privacy_vlm_poc.analyzer import analyze_video
from privacy_vlm_poc.schemas import MaskMethod, ROI, SamplingMethod, VLMBackend


def _video_path_from_gradio(value: Any) -> Path:
    if value is None:
        raise gr.Error("動画ファイルを指定してください。")
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        if "name" in value:
            return Path(value["name"])
        if "path" in value:
            return Path(value["path"])
    name = getattr(value, "name", None)
    if name:
        return Path(name)
    raise gr.Error("アップロードされた動画パスを解決できませんでした。")


def _roi_from_values(x1: float | None, y1: float | None, x2: float | None, y2: float | None) -> ROI | None:
    values = [x1, y1, x2, y2]
    if all(value is None for value in values):
        return None
    if all(value == 0 for value in values):
        return None
    if any(value is None for value in values):
        raise gr.Error("ROIは x1, y1, x2, y2 をすべて指定するか、すべて空にしてください。")
    return ROI(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))


def run_analysis(
    video_file: Any,
    sampling_method: str,
    num_frames: int,
    mask_method: str,
    roi_x1: float | None,
    roi_y1: float | None,
    roi_x2: float | None,
    roi_y2: float | None,
    vlm_backend: str,
) -> tuple[list[str], list[str], str, dict, str]:
    roi = _roi_from_values(roi_x1, roi_y1, roi_x2, roi_y2)
    result = analyze_video(
        video_path=_video_path_from_gradio(video_file),
        sampling_method=sampling_method,
        num_frames=int(num_frames),
        mask_method=mask_method,
        roi=roi,
        vlm_backend=vlm_backend,
    )
    selected = [str(frame.path) for frame in result.selected_frames if frame.path is not None]
    masked = [str(path) for path in result.masked_frame_paths]
    report = result.report_path.read_text(encoding="utf-8")
    return selected, masked, str(result.grid_path), result.vlm_response.model_dump(mode="json"), report


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Privacy VLM PoC") as demo:
        gr.Markdown("# Privacy VLM PoC")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="動画アップロード")
                sampling = gr.Dropdown(
                    label="sampling_method",
                    choices=[item.value for item in SamplingMethod],
                    value=SamplingMethod.HYBRID.value,
                )
                num_frames = gr.Slider(label="num_frames", minimum=1, maximum=24, step=1, value=8)
                mask = gr.Dropdown(
                    label="mask_method",
                    choices=[item.value for item in MaskMethod],
                    value=MaskMethod.NONE.value,
                )
                with gr.Row():
                    roi_x1 = gr.Number(label="x1", precision=0, value=None)
                    roi_y1 = gr.Number(label="y1", precision=0, value=None)
                    roi_x2 = gr.Number(label="x2", precision=0, value=None)
                    roi_y2 = gr.Number(label="y2", precision=0, value=None)
                backend = gr.Dropdown(
                    label="vlm_backend",
                    choices=[item.value for item in VLMBackend],
                    value=VLMBackend.MOCK.value,
                )
                run_button = gr.Button("実行", variant="primary")
            with gr.Column(scale=2):
                selected_gallery = gr.Gallery(label="選択フレーム一覧", columns=4, height=260)
                masked_gallery = gr.Gallery(label="マスク後フレーム一覧", columns=4, height=260)
                grid = gr.Image(label="グリッド画像", type="filepath")
                result_json = gr.JSON(label="VLM出力JSON")
                report = gr.Markdown(label="report.md")

        run_button.click(
            fn=run_analysis,
            inputs=[video, sampling, num_frames, mask, roi_x1, roi_y1, roi_x2, roi_y2, backend],
            outputs=[selected_gallery, masked_gallery, grid, result_json, report],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
