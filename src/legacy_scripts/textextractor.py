import argparse
import functools
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


@functools.lru_cache(maxsize=None)
def get_pipeline(device: str):
    # imported lazily so digital PDFs don't pay the model-load cost
    from paddleocr import PPStructureV3
    return PPStructureV3(
        device=device,
        text_recognition_model_name="en_PP-OCRv4_mobile_rec",
        use_doc_orientation_classify=False,  # PDF renders are already upright and flat
        use_doc_unwarping=False, 
        text_det_limit_type="max",
        text_det_limit_side_len=1024,
        )


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import paddle
        return "gpu" if paddle.device.cuda.device_count() > 0 else "cpu"
    except Exception as e:
        print(f"[INFO] GPU check failed ({e}), using CPU")
        return "cpu"


def is_bad_text(text: str) -> bool:
    t = text.strip()
    if len(t) < 10:
        return True
    garbage = sum(1 for c in t if c == "\ufffd" or (ord(c) < 32 and c not in "\n\t\r"))
    return garbage / len(t) > 0.05


def ocr_page(page, device, tmp_dir: Path, img_dir: Path) -> str:
    png = tmp_dir / "page.png"
    page.get_pixmap(dpi=200).save(png)
    pipeline = get_pipeline(device)
    md_list = []
    for res in pipeline.predict(input=str(png)):
        md = res.markdown
        md_list.append(md)
        for rel_path, img in md.get("markdown_images", {}).items():
            dest = img_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest)
    return pipeline.concatenate_markdown_pages(md_list)


def process_pdf(pdf_path: Path, device: str, out_dir: Path):
    out_file = out_dir / f"{pdf_path.stem}.md"
    if out_file.exists():
        print(f"[SKIP] {pdf_path.name} already done")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    with fitz.open(pdf_path) as doc, tempfile.TemporaryDirectory() as tmp:
        for n, page in enumerate(doc, start=1):
            text = page.get_text()  # default order; sort=True would interleave columns
            if is_bad_text(text):
                print(f"[OCR] {pdf_path.name} page {n}")
                parts.append(ocr_page(page, device, Path(tmp), out_dir / "imgs"))
            else:
                print(f"[TEXT] {pdf_path.name} page {n}")
                parts.append(text)
    out_file.write_text("\n\n<!-- page break -->\n\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    ap.add_argument("--in", dest="in_dir", default="./eng/pdf")
    ap.add_argument("--out", dest="out_dir", default="./output")
    args = ap.parse_args()

    device = pick_device(args.device)
    for pdf in sorted(Path(args.in_dir).glob("*.pdf")):
        process_pdf(pdf, device, Path(args.out_dir))