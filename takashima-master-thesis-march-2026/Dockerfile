FROM python:3.10.17-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:${PATH}"

ARG TORCH_DEVICE=cu121
ENV TORCH_DEVICE=${TORCH_DEVICE}

# 基本ツールとビルドに必要なパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorchをインストール（CPU or CUDA）
# TORCH_DEVICE=cpu | cu121 を指定可能（既定: cpu）
RUN python -m pip install --upgrade pip && \
  if [ "$TORCH_DEVICE" = "cpu" ]; then \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 ; \
  else \
    pip install --index-url https://download.pytorch.org/whl/${TORCH_DEVICE} \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 ; \
  fi

# PyTorch Geometric と関連拡張をインストール（対応するwheelを選択）
RUN if [ "$TORCH_DEVICE" = "cpu" ]; then \
      pip install \
        torch-scatter \
        torch-sparse \
        pyg-lib \
        torch-geometric==2.5.3 \
        -f https://data.pyg.org/whl/torch-2.4.1+cpu.html ; \
    else \
      pip install \
        torch-scatter \
        torch-sparse \
        pyg-lib \
        torch-geometric==2.5.3 \
        -f https://data.pyg.org/whl/torch-2.4.1+${TORCH_DEVICE}.html ; \
    fi

# 依存関係解決用にメタデータを先にコピー（ビルドキャッシュ最適化）
COPY pyproject.toml README.md ./
COPY requirements.txt ./requirements.txt

# プロジェクト依存関係（dev含む）をインストール
# transformers/lightning等は pyproject.toml の dependencies で解決
RUN pip install ".[dev]" || true

# tests補助（重複していても問題なし）
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# ソースをコピーして開発用途でインストール（編集内容反映が必要ならホスト側ボリュームを使用）
COPY . .

# デフォルトは対話シェル。docker-compose 側でコマンド上書きを推奨
CMD ["/bin/bash"]


