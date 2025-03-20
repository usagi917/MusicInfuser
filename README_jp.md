# MusicInfuser
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://susunghong.github.io/MusicInfuser/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2503.14505)

MusicInfuserは音楽に合わせて動画を生成するテキスト・トゥ・ビデオのディフュージョンモデルを応用したもので、音楽とテキストプロンプトに基づいてダンス動画を生成します。

## 必要要件
Python 3.10、`torch>=2.4.1+cu118`、`torchaudio>=2.4.1+cu118`、および`torchvision>=0.19.1+cu118`でテスト済みです。このリポジトリではトレーニングと推論に単一のA100 GPUが必要です。

## インストール
```bash
# リポジトリをクローン
git clone https://github.com/SusungHong/MusicInfuser
cd MusicInfuser
# condaの環境を作成して有効化
conda create -n musicinfuser python=3.10
conda activate musicinfuser
# 依存パッケージをインストール
pip install -r requirements.txt
pip install -e ./mochi --no-build-isolation
# モデルウェイトをダウンロード
python ./music_infuser/download_weights.py weights/
```

## 推論
音楽入力から動画を生成するには：
```bash
python inference.py --input-file {音声を抽出するMP3またはMP4ファイル} \
                    --prompt {プロンプト} \
                    --num-frames {フレーム数}
```

引数の説明：
- `--input-file`：音声を抽出するための入力ファイル（MP3またはMP4）
- `--prompt`：ダンサー生成のためのプロンプト。より具体的なプロンプトほど良い結果が得られますが、具体性が高まると音声の効果は減少します。デフォルト：`"a professional female dancer dancing K-pop in an advanced dance setting in a studio with a white background, captured from a front view"`
- `--num-frames`：生成するフレーム数。元々は73フレームでトレーニングされていますが、MusicInfuserはより長いシーケンスへの外挿が可能です。デフォルト：`145`

その他の考慮事項：
- `--seed`：生成のためのランダムシード。生成されるダンスはランダムシードにも依存するため、自由に変更してください。デフォルト：`None`
- `--cfg-scale`：テキストプロンプトのためのClassifier-Free Guidance（CFG）スケール。デフォルト：`6.0`

## データセット
AISTデータセットについては、[AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)で利用規約を確認し、ダウンロードしてください。

## トレーニング
自分のデータセットでモデルをトレーニングするには：
1. データを前処理します：
```bash
bash music_infuser/preprocess.bash -v {データセットのパス} -o {処理済み動画の出力ディレクトリ} -w {事前学習済みmochiへのパス} --num_frames {フレーム数}
```
2. トレーニングを実行します：
```bash
bash music_infuser/run.bash -c music_infuser/configs/music_infuser.yaml -n 1
```
**注意:** 現在の実装はシングルGPUトレーニングのみをサポートしており、73フレームのシーケンスでトレーニングするには約80GBのVRAMが必要です。

## VLM評価
ビジュアル言語モデルを使用してモデルを評価するには：
1. `vlm_eval/README.md`の指示に従ってVideoLLaMA2評価フレームワークをセットアップします
2. 評価にはMusicInfuserとは別の環境を使用することをお勧めします

## 引用
```bibtex
@article{hong2025musicinfuser,
  title={MusicInfuser: Making Video Diffusion Listen and Dance},
  author={Hong, Susung and Kemelmacher-Shlizerman, Ira and Curless, Brian and Seitz, Steven M},
  journal={arXiv preprint arXiv:2503.14505},
  year={2025}
}
```

## 謝辞
このコードは以下の素晴らしいリポジトリを基にしています：
- [Mochi](https://github.com/genmoai/mochi)
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [VideoChat2](https://github.com/OpenGVLab/Ask-Anything)

著者の方々がコードとモデルをオープンソース化してくださったことに感謝します。これによりこの研究が可能になりました。