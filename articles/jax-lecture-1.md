---
title: "ゼロから始めるJAX Part1 〜インストールと基本操作〜"
emoji: "✨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "JAX", "NumPy", "科学技術計算", "機械学習"]
published: false
---
# はじめに
この記事では、Googleが開発した数値計算ライブラリであるJAXについて、基礎から応用までを解説していきます。JAXは、NumPyのような使いやすさと、GPUやTPUでの高速な計算を両立しており、機械学習や科学計算の分野で注目されています。

![JAX Logo](https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png)

# インストール方法
基本的には他のライブラリと同様に`pip`を使ってインストールできます。しかし、JAXは使用するハードウェア（CPU、GPU、TPU）に応じて異なるバージョンが提供されているため、適切なコマンドを選択する必要があります。特に、GPU上でも動作させたい場合は、CUDAのバージョンに注意してください。

```bash
# CPU版のインストール
pip install -U jax

# GPU版のインストール (CUDA 12.1の場合)
pip install -U jax[cuda12]

# TPU版のインストール
pip install -U jax[tpu]
```

また、JAXが利用可能な環境はOSやCPU/GPUアーキテクチャによって異なる場合があるので、以下の公式ドキュメントを参照してください。

https://docs.jax.dev/en/latest/installation.html

インストールが完了したら、以下のコードを実行して動作するか確認しましょう。

```python
import jax
print(jax.devices())
```

# 基本的な使い方
JAXはNumPyに似たAPIを提供しており、基本的な操作はNumPyとほぼ同じです。以下に、JAXの基本的な使い方を示します。
## 配列の作成と演算
```python
import jax.numpy as jnp

# 1次元配列の作成
x = jnp.array([1, 2, 3])

# 2次元配列の作成
y = jnp.array([[1, 2, 3], [4, 5, 6]])

# 3次元配列の作成
z = jnp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
```
