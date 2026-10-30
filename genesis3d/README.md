# Genesis 3D版（Franka Panda + 上方RGB-Dカメラ）

2D版と同じ「AIで形を見分けて仕分ける」タスクを、Genesis の3D物理シミュレーションで実際に把持して行います。

## Demo (YouTube)
- https://youtu.be/iVbuWvqwlQo
- Local copy: `out/PAI_final_demo_repro_02_genesis.mp4`

## 環境
- Python 3.10〜3.12、NVIDIA GPU推奨（CPUでも動くが遅い）
- `pip install torch genesis-world opencv-python scikit-learn joblib`
- 画面のないLinuxサーバでは `export PYOPENGL_PLATFORM=egl` を設定

## 実行
```bash
python shape_ai.py        # 形状AIを学習（CPUで約20秒、テスト精度 約99.9%）
python test_offline.py    # Genesisなしで「認識→3D位置」の部分を確認（任意）
python genesis_sort.py    # 本番（--cpu / --vis / --episodes N）
ffmpeg -i out/genesis_raw.mp4 -c:v libx264 -pix_fmt yuv420p out/genesis_demo.mp4
```

## 処理の流れ
上方RGB-Dカメラ → 深度から「机からの高さ」を計算 → 物体ごとのシルエット → MLPで circle / square / triangle に分類
→ 画素と深度から3D位置、minAreaRectから把持の向き → IK → 関節PD制御 → 指を力制御して把持
→ 持ち上げた後の指の開き幅で把持の成否を判定（失敗したら再試行） → 色別のビンに置く

## うまく動かないときに調整する所（genesis_sort.py の先頭）
| 症状 | 調整する値 |
|---|---|
| 物体が滑って落ちる | `GRIP_FORCE` を -3.0 くらいまで強める |
| 指が物体に届かない／机に当たる | `HAND_DOWN_OFFSET`（0.10〜0.13） |
| 物体が検出されない | `perceive()` の高さしきい値（0.015〜0.08） |
| 起動時に表示される `[calib]` 誤差が大きい | `CAM_POS` / `CAM_FOV` と、`add_camera` の `up` の設定 |
| `cam.render` の戻り値でエラー | `render()` 関数を、使っているGenesisのバージョンに合わせて修正 |
