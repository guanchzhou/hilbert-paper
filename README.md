# hilbert-paper

Dense, lexical, and space-filling-curve retrieval in a personal knowledge base: a controlled investigation of recall, token cost, and the limits of a Hilbert key as a candidate filter. Storage only. No CI.

Open [paper.pdf](paper.pdf).

- [slides/slides.pdf](slides/slides.pdf) is the talk: what was tested and what was found, built from the same result files.
- [paper.typ](paper.typ) is the Typst source. Every number in it is read from a file in `bench/`.
- [figures.py](figures.py) draws every figure in `figures/` from `bench/`.
- [build.py](build.py) draws the figures, records the source digests in `bench/sources.json`, typesets, and refuses to finish if a figure or result file is missing, a result file names a note, or the PDF fails a check. Run `python3 build.py`.
- [refs.bib](refs.bib) holds the references.
- [bench/](bench/) holds the result files and the scripts that wrote them. The notes that were searched are private and are not in this repository; the result files hold only measurements.
- [bench/agents/](bench/agents/) holds the agent-harness scripts.
- [plan.md](plan.md) is the measurement plan.
