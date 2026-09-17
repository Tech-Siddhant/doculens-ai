"""Generates the 10 evaluation corpus PDFs for DocuLens AI."""

from pathlib import Path
import pymupdf

from scripts.benchmark_data import DOCUMENTS_SPEC


def build_corpus_pdfs(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for spec in DOCUMENTS_SPEC:
        filename = spec["filename"]
        pages_data = spec["pages"]
        doc = pymupdf.open()

        for p_info in pages_data:
            page = doc.new_page(width=595, height=842)
            title = p_info.get("title", "")
            page_num = p_info.get("page_num", 1)

            # Header banner
            page.draw_rect(pymupdf.Rect(40, 30, 555, 65), color=(0.15, 0.25, 0.45), fill=(0.93, 0.95, 0.98))
            page.insert_text((50, 52), title, fontsize=13, fontname="helv", color=(0.1, 0.2, 0.4))
            page.insert_text((490, 52), f"Page {page_num}", fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4))

            y = 85.0
            for b in p_info.get("blocks", []):
                if "heading" in b:
                    page.insert_text((50, y), b["heading"], fontsize=11, fontname="helv", color=(0.15, 0.25, 0.45))
                    y += 16.0
                h = b.get("height", 60)
                b_type = b.get("type", "paragraph")
                if b_type == "table":
                    page.draw_rect(pymupdf.Rect(50, y, 545, y + h), color=(0.7, 0.7, 0.7), fill=(0.98, 0.98, 0.99))
                    page.insert_textbox(pymupdf.Rect(55, y + 5, 540, y + h - 5), b["content"], fontsize=9.0, fontname="cour", color=(0.05, 0.05, 0.05))
                elif b_type == "figure_box":
                    page.draw_rect(pymupdf.Rect(50, y, 545, y + h), color=(0.3, 0.5, 0.7), fill=(0.92, 0.96, 1.0))
                    page.insert_text((60, y + 16), b.get("fig_title", "Figure"), fontsize=10, fontname="helv", color=(0.1, 0.3, 0.6))
                    page.insert_textbox(pymupdf.Rect(60, y + 22, 535, y + h - 5), b["content"], fontsize=9.0, fontname="helv", color=(0.15, 0.15, 0.15))
                else:
                    page.insert_textbox(pymupdf.Rect(50, y, 545, y + h), b["content"], fontsize=9.5, fontname="helv", color=(0.1, 0.1, 0.1))
                y += h + 12.0

            page.draw_line(pymupdf.Point(40, 800), pymupdf.Point(555, 800), color=(0.8, 0.8, 0.8))
            page.insert_text((50, 815), f"DocuLens AI Benchmark Corpus — {filename}", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

        out_path = output_dir / filename
        doc.save(str(out_path))
        doc.close()
        created.append(out_path)

    return created
