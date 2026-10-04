"""Render REPORT.md (local source) to PAPER.pdf with headless Chrome. Image paths are relative to the repo root."""

import pathlib
import re
import subprocess

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "REPORT.md", ROOT / "PAPER.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.85in 0.9in 0.8in 0.9in; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; color: #1f1f1f; font-size: 14.5px; line-height: 1.6;
       font-family: Charter, "Bitstream Charter", Cambria, Georgia, serif; }
h1, h2, h3, table, figcaption, h1 + p {
       font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: 25px; line-height: 1.25; margin: 0 0 8px; letter-spacing: -0.01em; }
h1 + p { font-size: 13px; color: #666; margin: 0 0 26px; }
h2 { font-size: 19px; line-height: 1.3; margin: 28px 0 8px; break-after: avoid; }
h3 { font-size: 15.5px; margin: 20px 0 6px; break-after: avoid; }
p { margin: 0 0 13px; orphans: 3; widows: 3; }
code { font-family: "SF Mono", Menlo, Monaco, monospace; font-size: 12px;
       background: #f2f2f2; border-radius: 3px; padding: 1px 4px; }
table { border-collapse: collapse; margin: 4px 0 14px; font-size: 11.5px; line-height: 1.35; width: 100%; }
tr { break-inside: avoid; }
th, td { border: 1px solid #ddd; padding: 4px 6px; text-align: left; }
th { background: #f7f7f7; font-weight: 600; }
ul, ol { margin: 0 0 14px; padding-left: 22px; }
li { margin: 0 0 6px; }
figure { margin: 4px 0 20px; break-inside: avoid; }
figure .frame { border: 1px solid #e3e3e3; border-radius: 4px; padding: 4px; }
figure img { display: block; width: 100%; }
figcaption { font-size: 12px; line-height: 1.5; color: #666; font-style: italic;
             text-align: center; margin: 6px 10px 0; }
"""


def main():
    body = markdown.markdown(SRC.read_text(), extensions=["tables", "fenced_code"])
    # an image and the italic caption under it become one unbreakable figure
    body = re.sub(r"<p>(<img [^>]*>)</p>\s*<p><em>(Figure \d+\..*?)</em></p>",
                  r'<figure><div class="frame">\1</div><figcaption>\2</figcaption></figure>', body, flags=re.S)
    page = ROOT / "REPORT.html"
    page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", f"--print-to-pdf={OUT}", page.as_uri()],
                   check=True, capture_output=True)
    print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
