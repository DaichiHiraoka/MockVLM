# Privacy VLM PoC

スマートホーム内の「許可外物品操作疑い」を、限定された視覚情報で検知・説明するための研究用PoCです。

このリポジトリの目的は、高精度な盗難検知器を完成させることではありません。次の比較実験を最小実装で回せるようにすることです。

- フレームサンプリング方法の違いによる判定・説明の変化
- 入力情報を制限した場合の判定・説明の変化
- 軽量モデルまたはMockモデルでシステム全体が動くことの確認

出力ラベルは `theft` ではなく `unauthorized_object_interaction_suspected` です。犯罪や盗難を断定する用途には使えません。

## Setup

Python 3.12 と uv を使います。

```bash
uv sync
```

任意の外部VLM設定は `.env.example` を参照してください。未設定でも `mock` backend は動きます。

## Synthetic Video

実動画がなくても動作確認できるように、単純な図形の動画を生成できます。

```bash
uv run python scripts/generate_synthetic_video.py
```

生成物:

- `data/sample/sample_suspicious.mp4`
- `data/sample/sample_normal.mp4`
- `data/sample/labels.csv`

## CLI

1本の動画を解析します。

```bash
uv run python -m privacy_vlm_poc.cli analyze \
  --video data/sample/sample_suspicious.mp4 \
  --sampling hybrid \
  --num-frames 8 \
  --mask background_blur_with_roi \
  --vlm-backend mock
```

labels.csv を使って簡易評価します。

```bash
uv run python -m privacy_vlm_poc.cli evaluate \
  --labels data/sample/labels.csv \
  --sampling event_window \
  --num-frames 8 \
  --mask lower_body_only \
  --vlm-backend mock
```

出力は `outputs/runs/YYYYMMDD_HHMMSS/` に保存されます。

- `selected_frames/`
- `masked_frames/`
- `grid.jpg`
- `result.json`
- `report.md`
- `config.json`

## UI

```bash
uv run python -m privacy_vlm_poc.ui
```

ブラウザで表示されるGradio UIから、動画アップロード、サンプリング方式、マスク方式、ROI、VLM backendを切り替えて実行できます。

## Sampling Methods

- `uniform`: 動画全体から等間隔にN枚選び、全体文脈を残します。
- `motion`: 隣接フレーム差分が大きいフレームを優先します。厳密な行動認識ではありません。
- `hybrid`: `uniform` と `motion` を混ぜ、全体文脈と変化の大きい瞬間を両方残します。
- `event_window`: 高モーションフレームの前後も含め、対象物が「ある -> 触る/動く -> なくなる」前後関係を見やすくします。

## Mask Methods

- `none`: 元画像をそのまま使います。
- `face_like_top_mask`: 顔検出ではなく、画像上部中央を矩形で黒塗りするPoC用匿名化です。
- `background_blur_with_roi`: ROI以外をぼかし、手元・机上・バッグ付近だけ残す想定です。
- `lower_body_only`: 上半分を黒塗りし、下半分だけ残します。
- `object_area_only`: ROI以外を黒塗りし、対象物・手元領域だけで判定可能か確認します。

ROIは `x1,y1,x2,y2` のピクセル座標です。未指定の場合は中央下部の仮ROIを使います。

## VLM Backends

- `mock`: 外部モデルなしで必ず動く決定的なMockです。画像内容の理解はしません。
- `ollama`: 任意です。`OLLAMA_ENABLED=true`, `OLLAMA_HOST`, `OLLAMA_MODEL` を設定します。
- `openai_compatible`: 任意です。`OPENAI_COMPATIBLE_ENABLED=true`, `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_MODEL` を設定します。

外部APIへ raw video は送りません。送る設計になっているのは、選択・マスク済みフレームから生成した `grid.jpg` のみです。

## Evaluation

`labels.csv` の形式:

```csv
video_path,label,evidence_frames
data/sample/sample_suspicious.mp4,1,"24;36;58"
data/sample/sample_normal.mp4,0,""
```

指標:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `selected_frame_recall`: 正解 evidence frame の近傍、デフォルト ±2 フレームが選択された割合
- `average_num_selected_frames`
- `average_processing_time_sec`

`MockVLMClient` の分類指標は実精度を意味しません。まずはパイプライン評価とサンプリング評価を重視してください。

## Tests

```bash
uv run pytest
```

## Privacy and Safety Limits

- 実際の盗難検知精度は保証しません。
- 物体検出や手検出は簡易実装です。
- マスク処理はPoC用であり、匿名化を保証しません。
- VLMの出力は誤る可能性があります。
- 犯罪判断には使えません。
- 顔認識、個人識別、人物属性推定は実装していません。
