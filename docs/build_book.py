#!/usr/bin/env python3
"""Convert all lesson markdown files into a single styled PDF and EPUB book."""

import subprocess
import sys
from pathlib import Path

import re
import uuid

import markdown
from ebooklib import epub
from pygments.formatters import HtmlFormatter

DOCS_DIR = Path(__file__).parent
LESSONS_DIR = DOCS_DIR / "lessons"
OUTPUT_PDF = DOCS_DIR / "ai-bootcamp-book.pdf"
OUTPUT_EPUB = DOCS_DIR / "ai-bootcamp-book.epub"
TMP_HTML = DOCS_DIR / "_book.html"

PYGMENTS_CSS = HtmlFormatter(style="monokai").get_style_defs(".codehilite")

PAGE_CSS = f"""
@page {{
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center {{
        content: counter(page);
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-size: 9pt;
        color: #888;
    }}
}}

body {{
    font-family: "Georgia", "Palatino Linotype", "Book Antiqua", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
}}

h1 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 22pt;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 0.8em;
    color: #111;
    border-bottom: 2px solid #333;
    padding-bottom: 0.3em;
}}

h2 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 16pt;
    font-weight: 600;
    margin-top: 1.5em;
    color: #222;
}}

h3 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13pt;
    font-weight: 600;
    margin-top: 1.2em;
    color: #333;
}}

h4 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    font-weight: 600;
    color: #444;
}}

p {{
    margin: 0.6em 0;
    text-align: justify;
    hyphens: auto;
}}

a {{
    color: #2563eb;
    text-decoration: none;
}}

code {{
    font-family: "Menlo", "Monaco", "Courier New", monospace;
    font-size: 9.5pt;
    background: #f3f4f6;
    padding: 0.15em 0.35em;
    border-radius: 3px;
}}

.codehilite {{
    background: #272822;
    color: #f8f8f2;
    padding: 0.8em 1em;
    border-radius: 6px;
    overflow-x: auto;
    margin: 0.8em 0;
    font-size: 9pt;
    line-height: 1.45;
}}

.codehilite code {{
    background: none;
    padding: 0;
    color: inherit;
    font-size: inherit;
}}

pre {{
    font-family: "Menlo", "Monaco", "Courier New", monospace;
    font-size: 9pt;
    background: #272822;
    color: #f8f8f2;
    padding: 0.8em 1em;
    border-radius: 6px;
    overflow-x: auto;
    line-height: 1.45;
    white-space: pre-wrap;
    word-wrap: break-word;
}}

pre code {{
    background: none;
    padding: 0;
    color: inherit;
    font-size: inherit;
}}

blockquote {{
    border-left: 3px solid #6b7280;
    margin: 1em 0;
    padding: 0.4em 1em;
    color: #4b5563;
    background: #f9fafb;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
}}

th, td {{
    border: 1px solid #d1d5db;
    padding: 0.5em 0.8em;
    text-align: left;
}}

th {{
    background: #f3f4f6;
    font-weight: 600;
}}

tr:nth-child(even) {{
    background: #f9fafb;
}}

ul, ol {{
    margin: 0.5em 0;
    padding-left: 1.5em;
}}

li {{
    margin: 0.3em 0;
}}

hr {{
    border: none;
    border-top: 1px solid #d1d5db;
    margin: 1.5em 0;
}}

.lesson-break {{
    page-break-before: always;
}}

.toc-page {{
    page-break-after: always;
}}

.toc-page h1 {{
    text-align: center;
    border-bottom: none;
    margin-bottom: 1.2em;
}}

.toc-page ol {{
    list-style: none;
    padding-left: 0;
    margin: 0;
}}

.toc-page li {{
    margin: 0.45em 0;
    font-size: 11.5pt;
    line-height: 1.5;
    border-bottom: 1px dotted #d1d5db;
    padding-bottom: 0.35em;
}}

.toc-page li a {{
    color: #1a1a1a;
    text-decoration: none;
}}

.toc-page li a:hover {{
    color: #2563eb;
}}

.toc-page .toc-number {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-weight: 600;
    color: #2563eb;
    margin-right: 0.4em;
}}

.toc-page .toc-desc {{
    display: block;
    font-size: 8.5pt;
    color: #6b7280;
    margin-top: 0.1em;
    line-height: 1.35;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-style: italic;
}}

{PYGMENTS_CSS}
"""

MD_EXTENSIONS = [
    "fenced_code",
    "codehilite",
    "tables",
    "toc",
    "smarty",
    "attr_list",
]

MD_EXTENSION_CONFIGS = {
    "codehilite": {
        "css_class": "codehilite",
        "guess_lang": True,
        "noclasses": False,
    },
    "toc": {
        "permalink": False,
    },
}


def collect_lessons() -> list[Path]:
    files = sorted(LESSONS_DIR.glob("topic_*.md"))
    if not files:
        print(f"No lesson files found in {LESSONS_DIR}", file=sys.stderr)
        sys.exit(1)
    return files


def md_to_html(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=MD_EXTENSIONS,
        extension_configs=MD_EXTENSION_CONFIGS,
    )


def extract_title(md_text: str) -> str:
    match = re.match(r"^#\s+(.+)", md_text.strip())
    return match.group(1).strip() if match else "Без названия"


def extract_subtopics(md_text: str) -> str:
    headings = re.findall(r"^###\s+\d+\.\s+(.+)", md_text, re.MULTILINE)
    if not headings:
        return ""
    return " · ".join(headings)


def build_toc(titles: list[str], descriptions: list[str]) -> str:
    items: list[str] = []
    for i, (title, desc) in enumerate(zip(titles, descriptions)):
        anchor = f"lesson-{i}"
        num = i + 1
        desc_html = f'\n<span class="toc-desc">{desc}</span>' if desc else ""
        items.append(
            f'<li><a href="#{anchor}">'
            f'<span class="toc-number">{num}.</span>{title}</a>'
            f"{desc_html}</li>"
        )
    return (
        '<section class="toc-page">\n'
        "<h1>Содержание</h1>\n"
        "<ol>\n" + "\n".join(items) + "\n</ol>\n"
        "</section>"
    )


def build_html(lessons: list[Path]) -> str:
    titles: list[str] = []
    descriptions: list[str] = []
    parts: list[str] = []
    for i, path in enumerate(lessons):
        md_text = path.read_text(encoding="utf-8")
        titles.append(extract_title(md_text))
        descriptions.append(extract_subtopics(md_text))
        html = md_to_html(md_text)
        anchor = f"lesson-{i}"
        css_class = ' class="lesson-break"' if i > 0 else ""
        parts.append(f'<article id="{anchor}"{css_class}>\n{html}\n</article>')

    toc = build_toc(titles, descriptions)
    body = toc + "\n" + "\n".join(parts)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="ru">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<style>{PAGE_CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>"
    )


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    result = subprocess.run(
        ["weasyprint", str(html_path), str(pdf_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"weasyprint failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


EPUB_CSS = """
body {
    font-family: Georgia, "Palatino Linotype", serif;
    font-size: 1em;
    line-height: 1.6;
    color: #1a1a1a;
    margin: 1em 1.5em;
}
h1 { font-size: 1.8em; font-weight: 700; border-bottom: 1px solid #333; padding-bottom: 0.3em; }
h2 { font-size: 1.4em; font-weight: 600; margin-top: 1.4em; }
h3 { font-size: 1.2em; font-weight: 600; margin-top: 1.1em; }
h4 { font-size: 1em; font-weight: 600; }
p  { margin: 0.6em 0; }
a  { color: #2563eb; text-decoration: none; }
code {
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 0.88em;
    background: #f3f4f6;
    padding: 0.1em 0.3em;
    border-radius: 3px;
}
pre {
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 0.82em;
    background: #272822;
    color: #f8f8f2;
    padding: 0.8em 1em;
    border-radius: 4px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}
pre code { background: none; padding: 0; color: inherit; }
blockquote {
    border-left: 3px solid #6b7280;
    margin: 1em 0;
    padding: 0.4em 1em;
    color: #4b5563;
    background: #f9fafb;
}
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th, td { border: 1px solid #d1d5db; padding: 0.4em 0.7em; text-align: left; }
th { background: #f3f4f6; font-weight: 600; }
tr:nth-child(even) { background: #f9fafb; }
ul, ol { margin: 0.5em 0; padding-left: 1.5em; }
li { margin: 0.3em 0; }
hr { border: none; border-top: 1px solid #d1d5db; margin: 1.5em 0; }
"""


def build_epub(lessons: list[Path]) -> None:
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title("AI Engineering Bootcamp")
    book.set_language("ru")

    style = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS,
    )
    book.add_item(style)

    chapters: list[epub.EpubHtml] = []
    toc_entries: list[epub.Link] = []

    for i, path in enumerate(lessons):
        md_text = path.read_text(encoding="utf-8")
        title = extract_title(md_text)
        html_body = md_to_html(md_text)

        chapter = epub.EpubHtml(
            title=title,
            file_name=f"lesson_{i:02d}.xhtml",
            lang="ru",
        )
        chapter.content = (
            f'<html xmlns="http://www.w3.org/1999/xhtml">'
            f"<head><title>{title}</title>"
            f'<link rel="stylesheet" href="../style/main.css" type="text/css"/>'
            f"</head><body>{html_body}</body></html>"
        )
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)
        toc_entries.append(epub.Link(f"lesson_{i:02d}.xhtml", title, f"lesson_{i}"))

    book.toc = toc_entries
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]

    epub.write_epub(str(OUTPUT_EPUB), book)


def main() -> None:
    lessons = collect_lessons()
    print(f"Found {len(lessons)} lessons")

    full_html = build_html(lessons)
    TMP_HTML.write_text(full_html, encoding="utf-8")
    print("HTML assembled, converting to PDF...")

    html_to_pdf(TMP_HTML, OUTPUT_PDF)
    TMP_HTML.unlink(missing_ok=True)

    size_mb = OUTPUT_PDF.stat().st_size / (1024 * 1024)
    print(f"Done PDF: {OUTPUT_PDF} ({size_mb:.1f} MB)")

    print("Building EPUB...")
    build_epub(lessons)
    epub_mb = OUTPUT_EPUB.stat().st_size / (1024 * 1024)
    print(f"Done EPUB: {OUTPUT_EPUB} ({epub_mb:.1f} MB)")


if __name__ == "__main__":
    main()
