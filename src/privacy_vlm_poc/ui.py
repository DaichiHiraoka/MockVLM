"""Streamlit UI for local PoC experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from privacy_vlm_poc.analyzer import analyze_video
from privacy_vlm_poc.config import get_settings
from privacy_vlm_poc.schemas import AnalyzeResult, MaskMethod, ROI, SamplingMethod, VLMBackend

UPLOAD_DIR = Path("outputs/uploads")
SAMPLE_DIR = Path("data/sample")
OLLAMA_MODEL_OPTIONS = ["gemma3:4b", "gemma3:12b"]


def _sample_video_options() -> dict[str, Path | None]:
    options: dict[str, Path | None] = {"アップロード動画を使う": None}
    for path in sorted(SAMPLE_DIR.glob("*.mp4")):
        options[path.name] = path
    return options


def _save_uploaded_video(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi"}:
        raise ValueError("対応形式は mp4, mov, avi です。")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = UPLOAD_DIR / f"{stamp}_{Path(uploaded_file.name).name}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def _roi_from_sidebar(enabled: bool, x1: int, y1: int, x2: int, y2: int) -> ROI | None:
    if not enabled:
        return None
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2)


def _render_gallery(paths: list[Path], columns: int = 4) -> None:
    if not paths:
        st.info("表示する画像がありません。")
        return
    cols = st.columns(min(columns, len(paths)))
    for index, path in enumerate(paths):
        with cols[index % len(cols)]:
            st.image(str(path), caption=path.name, use_container_width=True)


def _render_result(result: AnalyzeResult) -> None:
    run_config = json.loads(result.config_path.read_text(encoding="utf-8")).get("config", {})
    st.subheader("Run")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("判定", str(result.vlm_response.unauthorized_object_interaction_suspected))
    c2.metric("confidence", f"{result.vlm_response.confidence:.2f}")
    c3.metric("selected frames", str(len(result.selected_frames)))
    c4.metric("processing sec", f"{result.processing_time_sec:.2f}")
    if run_config.get("vlm_backend") == VLMBackend.OLLAMA.value:
        st.caption(f"model: {run_config.get('vlm_model') or get_settings().ollama_model}")
    st.code(str(result.run_dir), language="text")

    tab_grid, tab_frames, tab_json, tab_report, tab_files = st.tabs(
        ["グリッド", "フレーム", "JSON", "Report", "Files"]
    )

    with tab_grid:
        st.image(str(result.grid_path), caption="grid.jpg", use_container_width=True)

    with tab_frames:
        st.markdown("#### Selected")
        _render_gallery([frame.path for frame in result.selected_frames if frame.path is not None])
        st.markdown("#### Masked")
        _render_gallery(result.masked_frame_paths)

    with tab_json:
        payload = result.vlm_response.model_dump(mode="json")
        st.json(payload)
        st.download_button(
            "result.json",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name="result.json",
            mime="application/json",
        )

    with tab_report:
        report_text = result.report_path.read_text(encoding="utf-8")
        st.markdown(report_text)
        st.download_button("report.md", data=report_text, file_name="report.md", mime="text/markdown")

    with tab_files:
        st.write(
            {
                "grid": str(result.grid_path),
                "result": str(result.result_path),
                "report": str(result.report_path),
                "config": str(result.config_path),
            }
        )


def run_app() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Privacy VLM PoC", layout="wide")
    st.title("Privacy VLM PoC")

    with st.sidebar:
        uploaded_file = st.file_uploader("動画アップロード", type=["mp4", "mov", "avi"])
        sample_options = _sample_video_options()
        sample_label = st.selectbox("サンプル動画", list(sample_options.keys()))

        sampling_method = st.selectbox(
            "sampling_method",
            [item.value for item in SamplingMethod],
            index=[item.value for item in SamplingMethod].index(SamplingMethod.HYBRID.value),
        )
        num_frames = st.slider("num_frames", min_value=1, max_value=24, value=8, step=1)
        mask_method = st.selectbox(
            "mask_method",
            [item.value for item in MaskMethod],
            index=[item.value for item in MaskMethod].index(MaskMethod.NONE.value),
        )
        vlm_backend = st.selectbox(
            "vlm_backend",
            [item.value for item in VLMBackend],
            index=[item.value for item in VLMBackend].index(VLMBackend.MOCK.value),
        )
        ollama_model = None
        if vlm_backend == VLMBackend.OLLAMA.value:
            default_model = settings.ollama_model if settings.ollama_model in OLLAMA_MODEL_OPTIONS else "gemma3:4b"
            ollama_model = st.selectbox(
                "ollama_model",
                OLLAMA_MODEL_OPTIONS,
                index=OLLAMA_MODEL_OPTIONS.index(default_model),
                help="UI から切り替えられる Gemma 3 モデルです。",
            )

        use_roi = st.checkbox("ROIを指定する", value=False)
        roi_cols = st.columns(2)
        with roi_cols[0]:
            roi_x1 = st.number_input("x1", min_value=0, value=0, step=1, disabled=not use_roi)
            roi_x2 = st.number_input("x2", min_value=0, value=320, step=1, disabled=not use_roi)
        with roi_cols[1]:
            roi_y1 = st.number_input("y1", min_value=0, value=0, step=1, disabled=not use_roi)
            roi_y2 = st.number_input("y2", min_value=0, value=240, step=1, disabled=not use_roi)

        resize_width = st.number_input("resize_width", min_value=160, max_value=1920, value=640, step=32)
        run_button = st.button("実行", type="primary", use_container_width=True)

    if run_button:
        try:
            sample_path = sample_options[sample_label]
            if uploaded_file is not None:
                video_path = _save_uploaded_video(uploaded_file)
            elif sample_path is not None:
                video_path = sample_path
            else:
                st.error("動画をアップロードするか、サンプル動画を選択してください。")
                return

            roi = _roi_from_sidebar(use_roi, int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2))
            with st.spinner("解析中..."):
                st.session_state["analysis_result"] = analyze_video(
                    video_path=video_path,
                    sampling_method=sampling_method,
                    num_frames=int(num_frames),
                    mask_method=mask_method,
                    roi=roi,
                    vlm_backend=vlm_backend,
                    vlm_model=ollama_model,
                    resize_width=int(resize_width),
                )
        except Exception as exc:  # noqa: BLE001 - UI boundary should report recoverable errors.
            st.exception(exc)

    result = st.session_state.get("analysis_result")
    if isinstance(result, AnalyzeResult):
        _render_result(result)
    else:
        st.info("左の設定を選び、実行してください。")


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
