#This is a test readme for dnd2json project.

## Stage 1: extract blocks from PDFs

`src/dnd2json/extract.py` turns each PDF into `data/blocks/<name>.csv`: one row per text
block, in reading order, with font size/boldness (text-layer PDFs) or layout labels and OCR
confidence (scanned PDFs).

```
# PDFs with a usable text layer (any Python, fast)
python src/dnd2json/extract.py --only text

# scanned or garbled PDFs (OCR env; a few seconds per page on GPU)
.venv-ocr/Scripts/python src/dnd2json/extract.py --only ocr
```

Outputs are skipped when they're up to date (same extractor version and settings, stored
in `<name>.meta.json`). Use `--force` to redo them.

## OCR setup

PaddlePaddle has no wheels for Python 3.14, so OCR runs in its own Python 3.12 env:

```
py -3.12 -m venv .venv-ocr
# NVIDIA GPU (CUDA 12.9 build; ~2.7 GB with the CUDA libraries):
.venv-ocr/Scripts/python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
# or CPU only (much slower):
#   .venv-ocr/Scripts/python -m pip install paddlepaddle
.venv-ocr/Scripts/python -m pip install "paddleocr[doc-parser]>=3.3" pymupdf
```

Plain `pip install paddlepaddle` is the CPU-only build: it silently ignores the GPU.
`extract.py` picks the GPU automatically and warns when it falls back to CPU.

Models download on first use (a few hundred MB). On CPU, oneDNN is turned off in
`extract.py` because it crashes on some layout models in Paddle 3.3.
