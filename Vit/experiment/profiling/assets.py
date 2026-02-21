import base64, re
from pathlib import Path

# torch.profilerが生成したHTMLファイルの中から埋め込まれているPNGデータを抽出して別のPNGファイルにする

def extract_png_from_html(html: Path, *, save_to: Path | None = None) -> Path:
    """Extract embedded PNG from *torch.profiler* HTML file."""
    with html.open("r", encoding="utf-8") as f:
        html_txt = f.read()

    m = re.search(r"data:image/png;base64,([^']+)", html_txt)
    if not m:
        raise ValueError("PNG <img> tag not found in HTML")

    png_bytes = base64.b64decode(m.group(1))
    dst = save_to or html.with_suffix(".png")
    dst.write_bytes(png_bytes)
    return dst