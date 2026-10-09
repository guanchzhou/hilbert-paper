#!/usr/bin/env python3
"""Build paper.pdf: draw figures, record source digests, typeset, check the PDF."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
PAPER = ROOT / "paper.typ"
PDF = ROOT / "paper.pdf"
SLIDES = ROOT / "slides" / "slides.typ"
SLIDES_PDF = ROOT / "slides" / "slides.pdf"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def strip_orig(src: str) -> str:
    """Blank out #orig[...] spans, which hold deliberately original-language text."""
    out, i = [], 0
    while True:
        j = src.find("#orig", i)
        if j < 0:
            out.append(src[i:])
            return "".join(out)
        k = src.find("[", j)
        out.append(src[i:j])
        depth, m = 0, k
        while m < len(src):
            depth += {"[": 1, "]": -1}.get(src[m], 0)
            if depth == 0:
                break
            m += 1
        out.append("\n" * src.count("\n", j, m + 1))
        i = m + 1


def check_english() -> None:
    """Prose written for the paper must be English; original-language names and titles are allowed when marked.

    refs.bib is not checked: a reference may carry its original title.
    """
    bad = []
    for path in (PAPER, SLIDES, ROOT / "figures.py"):
        text = strip_orig(path.read_text()) if path == PAPER else path.read_text()
        for n, line in enumerate(text.splitlines(), 1):
            if CYRILLIC.search(line):
                bad.append(f"{path.name}:{n}: {line.strip()[:80]}")
    if bad:
        fail("non-English text outside #orig[...]:\n" + "\n".join(bad))


NOTE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_.-]*)+$")
PRIVATE_TEXT = re.compile(r"alpha-?sense|\bCC-\d+\b", re.IGNORECASE)


def check_private() -> None:
    """The corpus is private: result files may hold counts and scores, never note names or note-derived text."""
    bad = []

    def walk(o, path: Path) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k, path)
                walk(v, path)
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        elif isinstance(o, str):
            slug = NOTE_SLUG.match(o) and not o.startswith("bench/") and not re.search(r"\.(json|py|md)$", o)
            if slug or PRIVATE_TEXT.search(o):
                bad.append(f"{path.relative_to(ROOT)}: {o[:80]}")

    for p in sorted(BENCH.rglob("*.json")):
        walk(json.loads(p.read_text()), p)
    if bad:
        fail(f"{len(bad)} private strings in result files:\n" + "\n".join(bad[:20]))


def main() -> None:
    check_english()
    check_private()
    if subprocess.run([sys.executable, str(ROOT / "figures.py")], cwd=ROOT).returncode:
        fail("figures.py failed")

    src = PAPER.read_text()
    for fig in re.findall(r'image\("([^"]+)"', src):
        if not (ROOT / fig).is_file():
            fail(f"missing figure {fig}")
    for data in re.findall(r'json\("([^"]+)"\)', src):
        if data != "bench/sources.json" and not (ROOT / data).is_file():
            fail(f"missing result file {data}")

    files = []
    for p in sorted(BENCH.iterdir()):
        if p.name == "sources.json" or p.suffix not in {".json", ".py"}:
            continue
        files.append({"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (BENCH / "sources.json").write_text(json.dumps({"files": files}, indent=2) + "\n")

    proc = subprocess.run(["typst", "compile", "--root", str(ROOT), str(PAPER), str(PDF)],
                          capture_output=True, text=True)
    if proc.returncode:
        fail("typst:\n" + proc.stderr)
    if "warning" in proc.stderr:
        fail("typst warnings:\n" + proc.stderr)

    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True).stdout
    info = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    if "612 x 792 pts" not in info:
        fail("page size is not US Letter")
    n_images = len(re.findall(r'image\("', src))
    captions = set(re.findall(r"Figure (\d+):", text))
    if len(captions) != n_images:
        fail(f"{n_images} images but {len(captions)} figure captions in the PDF")
    for needle in ("Hypotheses", "Materials and methods", "Results", "Discussion", "Threats to validity", "Conclusion", "Data and code availability", "References", "Index", "Source files", "hk1:8:8:9e3779b97f4a7c15:2e240e0214b885c0"):
        if needle not in text:
            fail(f"PDF lacks {needle!r}")
    for bad in ("Not measured", "not been run", "??"):
        if bad in text:
            fail(f"PDF contains {bad!r}")
    pages = re.search(r"Pages:\s+(\d+)", info).group(1)

    proc = subprocess.run(["typst", "compile", "--root", str(ROOT), "--font-path", str(SLIDES.parent / "fonts"),
                           str(SLIDES), str(SLIDES_PDF)],
                          capture_output=True, text=True)
    if proc.returncode or "warning" in proc.stderr:
        fail("slides:\n" + proc.stderr)
    for fig in re.findall(r'image\("/([^"]+)"', SLIDES.read_text()):
        if not (ROOT / fig).is_file():
            fail(f"slides: missing figure {fig}")
    sinfo = subprocess.run(["pdfinfo", str(SLIDES_PDF)], capture_output=True, text=True).stdout
    slides = re.search(r"Pages:\s+(\d+)", sinfo).group(1)
    print(f"ok: {PDF.name}, {pages} pages, {n_images} figures, {len(files)} source files; slides {slides} pages")


if __name__ == "__main__":
    main()
