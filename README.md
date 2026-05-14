# Privacy VLM PoC

スマートホーム内の「許可外物品操作疑い」を、限定された視覚情報で検知・説明するための研究用PoCです。

このリポジトリは高精度な盗難検知器ではありません。目的は、フレームサンプリング方式、入力制限、軽量VLMまたはMockモデルの違いによって、判定と説明がどう変わるかを比較できる最小実装を提供することです。

出力ラベルは `theft` ではなく `unauthorized_object_interaction_suspected` です。犯罪や盗難の断定には使えません。

## Setup

Windows PowerShellでそのまま実行できるように、コマンドはすべて1行で記載しています。

```powershell
uv sync
uv run python scripts/generate_synthetic_video.py
uv run pytest
```

## Selected Research VLM

実証用の最小推奨VLMは Ollama の `gemma3:4b` です。ローカル実行でき、Text+Image入力に対応し、約3.3GBの構成で反復比較に使いやすいためです。

```powershell
ollama pull gemma3:4b
uv run python -m privacy_vlm_poc.cli doctor
```

`mock` backend は分類精度の検証ではなく、パイプライン接続確認用です。モデル選定根拠は `docs/model_selection.md`、実験手順は `docs/research_protocol.md` を参照してください。

## Streamlit UI

```powershell
uv run streamlit run src/privacy_vlm_poc/ui.py --server.address 127.0.0.1 --server.port 8501
```

UIでは動画アップロードまたは合成サンプルを選択し、`sampling_method`、`num_frames`、`mask_method`、ROI、`vlm_backend` を切り替えて解析できます。結果は `outputs/runs/YYYYMMDD_HHMMSS/` に保存されます。

## CLI

```powershell
uv run python -m privacy_vlm_poc.cli analyze --video data/sample/sample_suspicious.mp4 --sampling hybrid --num-frames 8 --mask background_blur_with_roi --vlm-backend mock
uv run python -m privacy_vlm_poc.cli evaluate --labels data/sample/labels.csv --sampling event_window --num-frames 8 --mask lower_body_only --vlm-backend mock
```

## Research Matrix

```powershell
uv run python scripts/run_research_matrix.py --quick --vlm-backend mock
uv run python scripts/run_research_matrix.py --quick --vlm-backend ollama
```

結果は `outputs/runs/research_matrix_*/summary.csv`, `by_condition.csv`, `summary.md` に保存されます。

## Sampling Methods

- `uniform`: 動画全体から等間隔にN枚選び、全体文脈を残します。
- `motion`: 隣接フレーム差分が大きいフレームを優先します。
- `hybrid`: `uniform` と `motion` を混ぜ、全体文脈と変化の大きい瞬間を両方残します。
- `event_window`: 高モーションフレームの前後も含め、対象物が「ある -> 触る/動く -> なくなる」前後関係を見やすくします。

## Mask Methods

- `none`: 元画像をそのまま使います。
- `face_like_top_mask`: 顔検出ではなく、画像上部中央を矩形で黒塗りするPoC用匿名化です。
- `background_blur_with_roi`: ROI以外をぼかします。
- `lower_body_only`: 上半分を黒塗りし、下半分だけ残します。
- `object_area_only`: ROI以外を黒塗りします。

## VLM Backends

- `mock`: 外部モデルなしで動く決定的なMockです。
- `ollama`: 実証用の主backendです。`OLLAMA_ENABLED=true`, `OLLAMA_MODEL=gemma3:4b` を設定します。
- `openai_compatible`: 任意のOpenAI互換API用です。

外部APIへ raw video は送りません。送る設計になっているのは、選択・マスク済みフレームから生成した `grid.jpg` のみです。

## Limits

- 実際の盗難検知精度は保証しません。
- 物体検出や手検出は簡易実装です。
- マスク処理はPoC用であり、匿名化を保証しません。
- VLMの出力は誤る可能性があります。
- 犯罪判断には使えません。
- 顔認識、個人識別、人物属性推定は実装していません。
