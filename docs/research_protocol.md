# Research Protocol

This protocol turns the repository from a pipeline PoC into a repeatable experiment harness.

## 1. Setup

```bash
uv sync
uv run python scripts/generate_synthetic_video.py
```

For pipeline-only verification:

```bash
uv run pytest
uv run python -m privacy_vlm_poc.cli analyze --video data/sample/sample_suspicious.mp4 --sampling hybrid --num-frames 8 --mask background_blur_with_roi --vlm-backend mock
```

For real VLM verification, install Ollama and pull the selected local model:

```bash
ollama pull gemma3:4b
```

Set `.env`:

```env
OLLAMA_ENABLED=true
OLLAMA_MODEL=gemma3:4b
```

Then verify readiness:

```bash
uv run python -m privacy_vlm_poc.cli doctor
```

## 2. Single-Video Real VLM Run

```bash
uv run python -m privacy_vlm_poc.cli analyze \
  --video data/sample/sample_suspicious.mp4 \
  --sampling event_window \
  --num-frames 8 \
  --mask background_blur_with_roi \
  --vlm-backend ollama
```

Inspect:

- `outputs/runs/<timestamp>/grid.jpg`
- `outputs/runs/<timestamp>/result.json`
- `outputs/runs/<timestamp>/report.md`
- `outputs/runs/<timestamp>/config.json`

## 3. Sampling x Masking Matrix

Quick smoke matrix:

```bash
uv run python scripts/run_research_matrix.py --quick --vlm-backend mock
```

Real VLM quick matrix:

```bash
uv run python scripts/run_research_matrix.py --quick --vlm-backend ollama
```

Full matrix:

```bash
uv run python scripts/run_research_matrix.py --vlm-backend ollama
```

The matrix writes:

- `summary.csv`: per-run prediction, confidence, selected frames, explanation, limitations
- `by_condition.csv`: average confidence, selected-frame recall, processing time by sampling/mask condition
- `summary.md`: readable summary
- `config.json`: experiment configuration

## 4. What To Compare

Primary comparisons:

- Sampling method vs. `selected_frame_recall`
- Sampling method vs. VLM explanation changes in `reason` and `limitations`
- Mask method vs. confidence and false/low-confidence outcomes
- Mask method vs. privacy-sensitive output flag

Do not report the result as theft detection accuracy. The permitted label is `unauthorized_object_interaction_suspected`.

## 5. Minimum Evidence For Graduation Research Demo

The repository is ready for an initial research demo when all of the following are true:

- `uv run pytest` passes
- `uv run python -m privacy_vlm_poc.cli doctor` reports the selected Ollama model present
- one suspicious and one normal synthetic video run succeed with `--vlm-backend ollama`
- `scripts/run_research_matrix.py --quick --vlm-backend ollama` produces `summary.csv`
- report discussion focuses on uncertainty and limited visual information, not crime determination
