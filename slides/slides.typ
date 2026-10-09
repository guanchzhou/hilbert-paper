// Build from the repository root:
// typst compile --root . --font-path slides/fonts slides/slides.typ slides/slides.pdf

#let grid-data = json("/bench/retrieval-grid.json").cells
#let units = json("/bench/evidence-units.json").units
#let sweep = json("/bench/hilbert-sweep.json")
#let man = json("/bench/manifest.json")
#let ho = json("/bench/heldout.json")
#let sc = json("/bench/scale-200k.json")
#let live = json("/bench/pgvector-5143.json")
#let rtk = json("/bench/rtk-bytes.json")
#let hp = json("/bench/harness-pages.json")
#let inv = json("/bench/investigation.json")
#let lqa = json("/bench/lsh-qa.json")
#let av = json("/bench/arango-vector.json")
#let ah = json("/bench/arango-hop.json")
#let ideas = json("/bench/ideas/summary.json")
#let i1 = json("/bench/ideas/1-rerank.json")
#let i2 = json("/bench/ideas/2-prune.json")
#let pq = json("/bench/per-query.json")
#let eff = json("/bench/hk1-efficiency.json")
#let e12 = json("/bench/engines-d12.json")
#let es = json("/bench/efficiency/summary.json")
#let m9 = json("/bench/efficiency/m9.json")
#let fsh = json("/bench/factorial/level1-shapley.json").points
#let fcells = json("/bench/factorial/level1-cells.json")
#let fho = json("/bench/factorial/heldout-level1.json")
#let fjp = json("/bench/factorial/judge-packs.json")
#let fuc = json("/bench/followup/compositional.json").summary
#let fus = json("/bench/followup/compositional-sentences.json")
#let fqe = json("/bench/followup/qrels-extended.json")
#let fpw = json("/bench/followup/power.json")
#let c1 = json("/bench/followup/cell1.json")
#let c2 = json("/bench/followup/cell2.json")
#let c3 = json("/bench/followup/cell3.json")
#let c5 = json("/bench/followup/cell5.json")
#let c8 = json("/bench/followup/cell8.json")
#let c9 = json("/bench/followup/cell9.json")
#let c16 = json("/bench/followup/cell16.json")
#let s9a = json("/bench/followup/s9a.json")
#let S9(a) = s9a.summary.at(a)
#let s9b = json("/bench/followup/s9b.json")
#let S9R = ("area_type_months", "area_months", "type_months").map(sh => s9b.results.at("200000").summary.at(sh).acceptance.buffer_ratio)
#let s10 = json("/bench/followup/s10.json")
#let SA(a) = s10.arms.at(a).filter.primary_330.at("R@10")
#let SQ(a) = s10.arms.at(a).qa_same_cell_observed / s10.arms.at(a).qa_same_cell_theory
#let fl2 = json("/bench/factorial/level2-summary.json")
#let FCTX = fl2.marginals.at("num_ctx (local)")
#let EF(name) = eff.conditions.find(c => c.condition == name)

#let fx(x, d: 3) = {
  let s = str(calc.round(x, digits: d))
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  while frac.len() < d { frac += "0" }
  let head = parts.at(0).replace("-", "−")
  if d == 0 { head } else { head + "." + frac }
}
#let num(n) = {
  let s = str(int(calc.round(n)))
  let out = ()
  for (i, c) in s.clusters().rev().enumerate() {
    if i > 0 and calc.rem(i, 3) == 0 { out.push(",") }
    out.push(c)
  }
  out.rev().join()
}
#let C(k) = inv.cells.at(k)
#let I(k) = ideas.hypotheses.find(h => h.id == k)
#let A(engine, key, val) = av.results.find(r => r.engine == engine and r.at(key, default: none) == val)

#let blue = rgb("#0160FF")
#let ink = rgb("#0E1318")
#let slate = rgb("#334252")
#let pale = rgb("#EBF5FF")
#let muted = rgb("#757F8A")
#let hair = rgb("#C8CDD3")
#let foot = rgb("#888888")

#let W = 33.867cm
#let H = 19.05cm
#let mx = 0.9cm

#set document(title: "Recall and tokens in a personal AI knowledge base", author: "Andrey Maltsev", date: none)
#set page(
  width: W, height: H,
  margin: (x: mx, top: 0.9cm, bottom: 1.4cm),
  footer: context {
    set text(size: 9pt, fill: foot)
    [Recall and tokens in a personal AI knowledge base #h(1fr) #counter(page).display()]
  },
)
#set text(font: "Inter Tight", weight: 500, size: 18pt, fill: ink, lang: "en")
#show strong: set text(weight: 700)
#show raw: set text(font: "Menlo", size: 0.85em, weight: 400)
#set par(leading: 0.5em, spacing: 0.85em, justify: false)
#set list(marker: text(fill: blue, size: 0.8em, baseline: -0.1em)[■], indent: 0.1em, body-indent: 0.55em, spacing: 0.7em)
#set footnote(numbering: "1")
#set super(typographic: false)
#set footnote.entry(separator: line(length: 22%, stroke: 0.6pt + hair), gap: 0.3em, clearance: 0.6em, indent: 0pt)
#show footnote.entry: set text(size: 9pt, weight: 400, fill: muted)

#let hl(t) = text(fill: blue)[#t]
#let title(t) = block(below: 0.8cm, width: 78%)[#text(size: 32pt, weight: 600, fill: ink)[#t]]
#let tag(t) = place(top + right, box(fill: blue, inset: (x: 12pt, y: 7pt))[
  #text(size: 12pt, weight: 600, fill: white)[#t]
])
#let big(value, label, color: blue, size: 54pt) = block(below: 0.5cm)[
  #text(size: size, weight: 600, fill: color)[#value] \
  #v(-0.35cm)
  #text(size: 15pt, fill: muted)[#label]
]
#let card(head, body, h: auto) = stack(
  block(width: 100%, fill: slate, inset: (x: 13pt, y: 12pt), below: 0pt)[
    #text(size: 17pt, weight: 600, fill: white)[#head]
  ],
  v(0.22cm),
  block(width: 100%, height: h, fill: pale, inset: (x: 13pt, top: 12pt, bottom: 14pt))[
    #text(size: 14pt, weight: 400)[#body]
    #place(bottom + left, dy: 4pt, line(length: 1.1cm, stroke: 2.5pt + blue))
  ],
)
#let point(lead, body) = block(width: 100%, below: 0.45cm,
  stroke: (left: 1.5pt + hair), inset: (left: 12pt, y: 4pt))[
  #hl[#lead] #body
]
#let note(body) = text(size: 13pt, weight: 400, fill: muted)[#body]
#let slide(body) = page[#counter(footnote).update(0)#body]
#let divider(kick, body) = page(fill: rgb("#F3F8FF"), footer: none)[
  #place(top + left, dx: 0.85cm - mx, dy: 10.33cm - 0.9cm, image("/slides/network-wide.png", width: 33.02cm))
  #v(4.6cm)
  #text(size: 14pt, weight: 600, fill: muted, tracking: 0.06em)[#upper(kick)]
  #v(0.2cm)
  #text(size: 62pt, weight: 600, fill: blue)[#body]
]
#let network = image("/slides/network.png", height: H)

#let hil-line = rgb("#4C8DF6")
#let hil-grey = rgb("#D5D9DE")
#let hil-greydot = rgb("#BFC4CA")
#let hil-xy(n, d) = {
  let (x, y, t, s) = (0, 0, d, 1)
  while s < n {
    let rx = calc.rem(calc.quo(t, 2), 2)
    let ry = if calc.rem(t, 2) == rx { 0 } else { 1 }
    if ry == 0 {
      if rx == 1 { x = s - 1 - x; y = s - 1 - y }
      (x, y) = (y, x)
    }
    x += s * rx
    y += s * ry
    t = calc.quo(t, 4)
    s *= 2
  }
  (x, y)
}
#let hil-step(n, cell) = range(n * n).position(d => hil-xy(n, d) == cell)
#let hil-at(n, size, cell) = (size / n * (cell.at(0) + 0.5), size - size / n * (cell.at(1) + 0.5))
#let hil-path(n, size, stroke, dot, r) = {
  let pts = range(n * n).map(d => hil-at(n, size, hil-xy(n, d)))
  for i in range(pts.len() - 1) { place(line(start: pts.at(i), end: pts.at(i + 1), stroke: stroke)) }
  for p in pts { place(dx: p.at(0) - r, dy: p.at(1) - r, circle(radius: r, fill: dot, stroke: none)) }
}
#let hilbert-art(order, size, under: none, mark: ()) = box(width: size, height: size, {
  let n = calc.pow(2, order)
  if under != none {
    let m = calc.pow(2, under)
    for k in range(1, m) {
      place(line(start: (size * k / m, 0pt), end: (size * k / m, size), stroke: (paint: hil-grey, thickness: 0.8pt, dash: "dashed")))
      place(line(start: (0pt, size * k / m), end: (size, size * k / m), stroke: (paint: hil-grey, thickness: 0.8pt, dash: "dashed")))
    }
    hil-path(m, size, 1pt + hil-grey, hil-greydot, 4.5pt)
  }
  hil-path(n, size, 1.1pt + hil-line, blue, calc.max(2.4pt, 5.5pt - order * 1pt))
  for c in mark {
    let p = hil-at(n, size, c)
    place(dx: p.at(0) - 6.5pt, dy: p.at(1) - 6.5pt, circle(radius: 6.5pt, fill: ink, stroke: 2pt + white))
  }
})

// ------------------------------------------------------------------ title

#page(footer: none)[
  #place(top + right, dx: mx, dy: -0.9cm, network)
  #v(1fr)
  #block(width: 20cm)[
    #text(size: 14pt, weight: 600, fill: muted, tracking: 0.06em)[RESEARCH TALK · OCTOBER 2026]
    #v(0.4cm)
    #text(size: 48pt, weight: 600, fill: rgb("#262626"))[#hl[Recall and tokens] in a personal AI knowledge base]
    #v(0.5cm)
    #text(size: 20pt, weight: 400)[What decides whether the right note reaches the model, and what it costs to get it there]
    #v(1.2cm)
    #text(size: 17pt, weight: 600)[Andrey Maltsev] \
    #text(size: 13pt, fill: muted)[#raw("github.com/guanchzhou/hilbert-paper")]
  ]
  #v(1.2fr)
]

// ------------------------------------------------------------------ question

#slide[
  #title[The #hl[question]]
  #grid(columns: (1.15fr, 1fr), gutter: 1.2cm,
    [
      An AI assistant answers from my own notes. Two numbers matter:
      #v(0.2cm)
      #point[Recall:][is the note that answers the question among what the model is given?]
      #point[Cost:][how many tokens does the model read to get it?]
      #v(0.2cm)
      A switch that saves tokens but drops the right note is a regression, not a saving.

      #v(0.3cm)
      #note[Rule of the study: test every available combination, measure real data, never fit results to an expected answer.]
    ],
    [
      #card[Measured on one frozen corpus][
        #num(man.pages_live) notes, #num(man.chunk_count) chunks \
        #num(man.qrels_count) development questions with known answers \
        #num(ho.n) sealed held-out questions, opened once
      ]
      #v(0.4cm)
      #card[Everything is reproducible][
        Every number on these slides is read from a result file in the repository.
      ]
    ],
  )
]

#divider[Part 1][What was tested]

// ------------------------------------------------------------------ storage and search

#slide[
  #title[Storage and #hl[search]]
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.5cm, row-gutter: 0.45cm,
    card(h: 3.6cm)[gbrain][The personal knowledge base: notes mirrored from Obsidian into Postgres, chunked, embedded, searchable over MCP by Cursor and Claude Code.#footnote[gbrain, #raw("github.com/garrytan/gbrain"), version 0.60.64.]],
    card(h: 3.6cm)[Postgres + pgvector][The database. pgvector adds vector columns and an HNSW nearest-neighbour index.#footnote[pgvector 0.8.6, #raw("github.com/pgvector/pgvector")\; HNSW: Malkov and Yashunin, IEEE TPAMI 2020.]],
    card(h: 3.6cm)[Qwen3-Embedding][Turns each chunk and each question into a 1,024-number vector; runs locally through MLX on the laptop GPU.#footnote[Zhang et al., Qwen3 Embedding, arXiv:2506.05176, 2025.]],
    card(h: 3.6cm)[Keyword search][Postgres full-text search: matches words, ranks by term density. The classic baseline next to vectors.#footnote[PostgreSQL 16 documentation, chapter 12, Full Text Search.]],
    card(h: 3.6cm)[Reranker][A small cross-encoder (Qwen3-Reranker 0.6B) that reads the question and each candidate together and re-orders them.#footnote[Nogueira and Cho, Passage re-ranking with BERT, arXiv:1901.04085, 2019.]],
    card(h: 3.6cm)[ArangoDB][A document and graph database, tested on the same vectors as a possible second engine.#footnote[ArangoDB 3.12 documentation, vector indexes.]],
  )
]

// ------------------------------------------------------------------ key and agent side

#slide[
  #title[The key and the #hl[agent side]]
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.5cm, row-gutter: 0.45cm,
    card(h: 3.6cm)[zig-hilbert and hk1][Built for this study: turns a vector into a short sortable text key on a Hilbert curve, so a database prefix range could act as a cheap search filter.#footnote[Maltsev, zig-hilbert, #raw("github.com/guanchzhou/zig-hilbert"), v0.2.1; Hilbert, Mathematische Annalen 38, 1891.]],
    card(h: 3.6cm)[Evidence unit][What is handed to the model under a 6,000-token budget: a chunk, a window, a section, or the whole note.],
    card(h: 3.6cm)[RTK][A filter that shortens shell command output before an AI agent reads it; hooks into Cursor and Claude Code.#footnote[RTK 0.51.0, #raw("www.rtk-ai.app").]],
    card(h: 3.6cm)[Cursor CLI][The Cursor agent run headless from the terminal, with a choice of models.],
    card(h: 3.6cm)[Claude Code][Anthropic's coding agent, run headless with the same tasks.],
    card(h: 3.6cm)[Local model][qwen3.8, a 27B model in Ollama on the laptop, with context windows from 8K to 128K tokens.#footnote[Ollama, #raw("ollama.com").]],
  )
]

// ------------------------------------------------------------------ hilbert curve

#slide[
  #title[A Hilbert curve: one line through #hl[every cell]]
  #let pair = ((3, 0), (4, 0))
  #let gap = calc.abs(hil-step(8, pair.at(1)) - hil-step(8, pair.at(0)))
  #grid(columns: (auto, auto, 1fr), column-gutter: 0.9cm,
    stack(spacing: 0.3cm,
      hilbert-art(1, 5cm),
      note[4 cells: one level],
      v(0.5cm),
      hilbert-art(2, 5cm),
      note[16 cells: two levels],
    ),
    stack(spacing: 0.3cm,
      hilbert-art(3, 12cm, under: 1, mark: pair),
      note[64 cells: three levels. Grey: the first level's four cells.],
    ),
    [
      #set text(size: 0.9em)
      #point[One line, every cell.][The curve passes through every cell of a grid exactly once, without jumps. A cell's step number along it is one sortable number.#footnote[Hilbert, Mathematische Annalen 38, 1891; any number of dimensions: Skilling, AIP Conference Proceedings 707, 2004.]]
      #point[Steps stay neighbours.][Two consecutive steps are always touching cells, so a range of step numbers is one compact patch of space.]
      #point[Not the other way round.][Touching cells can be far apart on the line: the two dark dots share an edge, yet lie #gap steps apart, in different first-level cells.]
      #point[The hk1 key.][The same curve in 8 dimensions: 8 random projections, each cut into 256 steps. The step number becomes a text key; its first byte names one of 256 first-level cells.]
    ],
  )
]

// ------------------------------------------------------------------ method

#slide[
  #title[How it was #hl[measured]]
  #grid(columns: (1.1fr, 1fr), gutter: 1cm,
    image("/figures/fig01-pipeline.svg", width: 100%),
    [
      #point[Preregistered.][Hypotheses and pass criteria were written down and committed before any measurement.]
      #point[Paired.][One switch changes at a time; every comparison is made question by question.]
      #point[Corrected.][Paired statistical tests with correction for testing many things at once.#footnote[Wilcoxon, Biometrics Bulletin 1945; McNemar, Psychometrika 1947; Holm, Scandinavian Journal of Statistics 1979.]]
      #point[Sealed.][The held-out questions stay closed until a configuration has won on the development set.]
    ],
  )
]

#divider[Part 2][Results]

// ------------------------------------------------------------------ result 1

#slide[
  #tag[Result 1]
  #title[#hl[Vectors] beat keywords by a wide margin]
  #let rel = inv.dataset.relevant_per_question.hist
  #grid(columns: (1fr, 1.5fr), gutter: 1cm,
    [
      #big(fx(grid-data.at("vector-chunk").R), [recall at 10, vector search over chunks], size: 44pt)
      #big(fx(grid-data.at("keyword-page").R), [keyword search over whole notes], color: muted, size: 44pt)
      #note[Keyword search returned nothing at all for #num(pq.empty_result_lists.at("keyword-page")) of #num(man.qrels_count) questions: it joins every word of a long question with AND.]
    ],
    align(right, image("/figures/fig02-retrieval.svg", height: 7.4cm)),
  )
  #v(1fr)
  #text(size: 15pt, weight: 600)[How the scores are computed]
  #v(-0.1cm)
  #text(size: 13pt, weight: 400)[Each search returns a ranked list of notes; a chunk hit counts for its note. Each of the #num(man.qrels_count) questions has 1 to #(rel.len() - 1) notes known to answer it (exactly one for #rel.at(1)). Every score runs from 0 to 1, is computed per question, and is averaged over all #num(man.qrels_count).]
  #v(0.15cm)
  #set text(size: 13pt, weight: 400)
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.6cm,
    point[Recall at 10:][the share of a question's answering notes that appear in the first ten results.],
    point[MRR, mean reciprocal rank:][1 divided by the rank of the first answering note: 1 at rank 1, 0.5 at rank 2, 0 if it is not returned.#footnote[Voorhees, The TREC-8 question answering track report, TREC 1999.]],
    point[nDCG at 10, normalised discounted cumulative gain:][each answering note in the top ten earns #box[1 / log#sub[2]\(rank + 1)]\; the sum is divided by the best possible sum.#footnote[Järvelin and Kekäläinen, Cumulated gain-based evaluation of IR techniques, ACM TOIS 20(4), 2002.]],
  )
  #note[Example: one answering note, found at rank 3, gives recall 1, MRR 0.33 and nDCG 0.50.]
]

// ------------------------------------------------------------------ keyword fixes

#slide[
  #tag[Result 1]
  #title[Fixing keywords does #hl[not change the winner]]
  #grid(columns: (1.5fr, 1fr), gutter: 1cm,
    image("/figures/fig12-keyword.svg", width: 100%),
    [
      #point[AND to OR.][Switching keyword search from AND to OR lifts it from #fx(C("keyword-and-chunk").recall) to #fx(C("keyword-or-chunk").recall).]
      #point[Rank fusion.][Mixing keywords and vectors ties vectors on recall (#fx(C("hybrid-rrf-chunk").recall) vs #fx(C("vector-chunk").recall)) and lowers the top-rank quality.#footnote[Reciprocal rank fusion: Cormack, Clarke and Büttcher, SIGIR 2009.]]
      #point[Net effect.][On this corpus, keywords add nothing that vectors miss often enough to help.]
    ],
  )
]

// ------------------------------------------------------------------ result 2

#slide[
  #tag[Result 2]
  #title[Return #hl[chunks], not whole notes]
  #grid(columns: (1fr, 1.5fr), gutter: 1cm,
    [
      Under a 6,000-token budget, how often is the right note inside what the model gets?
      #v(0.3cm)
      #big(fx(units.chunk.relevant_in_pack), [chunks])
      #big(fx(units.page.relevant_in_pack), [whole notes], color: muted)
      #note[Same tokens spent; smaller units let more distinct notes fit. No question was better served by whole notes.]
      #v(0.1cm)
      #let bs = json("/bench/budget-sweep.json")
      #let BS(b, u) = bs.rows.find(r => r.budget == b and r.unit == u)
      #note[6,000 tokens is gbrain's default. No unit beats chunks at any budget tried, from #num(bs.budgets.first()) (#fx(BS(bs.budgets.first(), "chunk").relevant_in_pack) vs #fx(BS(bs.budgets.first(), "page").relevant_in_pack)) to #num(bs.budgets.last()) (#fx(BS(bs.budgets.last(), "chunk").relevant_in_pack) vs #fx(BS(bs.budgets.last(), "page").relevant_in_pack)).]
    ],
    image("/figures/fig06-units.svg", width: 100%),
  )
]

// ------------------------------------------------------------------ result 3

#slide[
  #tag[Result 3]
  #title[A Hilbert key is a #hl[label], not an index]
  #let P(l, r) = sweep.cells.find(c => c.level == l and c.ranges == r)
  #let keyed = man.chunk_hilbert_count
  #grid(columns: (1.5fr, 1fr), gutter: 1cm,
    image("/figures/fig07-hk1-sweep.svg", height: 9.4cm),
    [
      #big(fx(C("hk1-L1-R16").recall), [best Hilbert-key filter: level 1, 16 ranges], color: muted, size: 44pt)
      #big(fx(C("vector-chunk").recall), [plain vector search], size: 44pt)
      #note[The key does carry signal: about twice a random filter of the same size. It does not come close to exhaustive search.]
      #v(0.1cm)
      #note[Per token the model reads, hk1 is half as efficient: a relevant note per 1k delivered tokens #fx(EF("hk1 level 1, 16 ranges").survival_per_1k_delivered) vs #fx(EF("cosine, every chunk").survival_per_1k_delivered). Per token of candidates read it is #calc.round(EF("hk1 level 1, 16 ranges").survival_per_100k_processed / EF("cosine, every chunk").survival_per_100k_processed)× better than reading everything.]
    ],
  )
  #v(1fr)
  #text(size: 15pt, weight: 600)[How to read the chart]
  #v(0.1cm)
  #set text(size: 12.5pt, weight: 400)
  #grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 0.5cm,
    point[hk1 key:][each chunk's vector becomes one 64-bit text key: 8 random projections, each cut into 256 steps, ordered along a Hilbert curve so that nearby keys should mean nearby vectors.],
    point[Probe level:][how many leading bytes of a chunk's key must equal the question's. Level 1 (8 bits) splits the #num(keyed) keyed chunks into 256 cells of about #calc.round(keyed / 256) each; level 2 (16 bits) into 65,536 cells, almost all empty; level 0 keeps every chunk.],
    point[Range count (number on a point):][the question's own cell plus up to r − 1 neighbouring cells: 1, 4, 8 or 16. Level 1 grows from #calc.round(P(1, 1).mean_candidates) to #calc.round(P(1, 16).mean_candidates) candidates per question\; level 2 stays below #calc.ceil(P(2, 16).mean_candidates).],
    point[Score of a point:][the candidates are re-ranked by exact cosine and scored by recall at 10. Grey crosses use the same number of random chunks (the floor); the star is cosine over all chunks (the ceiling).],
  )
]

// ------------------------------------------------------------------ mechanism

#slide[
  #tag[Result 3]
  #title[Why the key fails: it is an #hl[8-bit hash]]
  #grid(columns: (1.4fr, 1fr), gutter: 1cm,
    image("/figures/fig14-lsh-model.svg", width: 100%),
    [
      #set text(size: 0.86em)
      #point[SimHash.][The first level of the key is the sign pattern of 8 random projections: an 8-bit SimHash.#footnote[Charikar, STOC 2002; Indyk and Motwani, STOC 1998.]]
      #point[Narrow cone.][Embeddings crowd into a narrow cone; after removing the mean direction, the theory for random projections predicts the collisions: #fx(lqa.centred.same_cell_observed) observed vs #fx(lqa.centred.same_cell_predicted) predicted.#footnote[Ethayarajh, EMNLP 2019; Mu and Viswanath, ICLR 2018.]]
      #point[Too far apart.][For one key on random axes a question and its answer are too far apart: about #calc.round(lqa.centred.at("independent_8bit_tables_for_0.9")) independent keys would be needed for 90% recall.]
      #point[Axes fitted to the notes.][On rotated principal axes a question and its answer share a cell #fx(SQ("C_itq"), d: 0)× as often as the theory says. Recall #fx(SA("C_itq")) vs #fx(SA("A_random_axes")), still below #fx(s10.at("exhaustive_R@10")) for every chunk.#footnote[Gong, Lazebnik, Gordo and Perronnin, iterative quantization, TPAMI 35(12), 2013.]]
    ],
  )
]

// ------------------------------------------------------------------ result 4

#slide[
  #tag[Result 4]
  #title[#hl[Postgres] is enough]
  #grid(columns: (1.4fr, 1fr), gutter: 1cm,
    image("/figures/fig18-engines.svg", height: 7.2cm),
    [
      #point[Vector search.][pgvector #fx(A("pgvector", "ef_search", 50).median_ms, d: 1) ms vs ArangoDB #fx(A("arangodb", "nProbe", 16).median_ms, d: 1) ms at matched quality.]
      #point[Link expansion.][Identical neighbours, #fx(ah.postgres_sql.median_ms, d: 2) ms in SQL vs #fx(ah.arangodb_aql.median_ms, d: 2) ms in AQL.]
      #point[Scale.][At 200,000 vectors pgvector still answers in #fx(sc.in_session.median_ms, d: 2) ms.]
      #let pg18 = json("/bench/pg18/comparison.json")
      #let PR(m) = pg18.rows.find(r => r.measure == m)
      #note[Measured on PostgreSQL 16.15 and ArangoDB 3.12.10. Re-run on 18.6 and 3.12.12: same quality, same order (#fx(PR("pgvector ef_search 50, median").pg18, d: 1) vs #fx(PR("arango nProbe 16, median").pg18, d: 1) ms; hop #fx(PR("SQL median").pg18, d: 2) vs #fx(PR("AQL median").pg18, d: 2) ms).]
    ],
  )
  #v(1fr)
  #text(size: 15pt, weight: 600)[Why ArangoDB first, and what the other engines did]
  #v(0.1cm)
  #set text(size: 12.5pt, weight: 400)
  #grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 0.5cm,
    point[ArangoDB:][a native graph of documents plus a vector index, so both operations the knowledge base needs, vector search and link expansion, are measured in one engine. It already runs on this machine for a genealogy database.],
    point[SQLite + sqlite-vec:][a single file, exact scan only, same SQL for links. Measured in process: #fx(e12.sqlite_vec.vector.at(0).median_ms, d: 1) ms, same recall. gbrain cannot run on it.#footnote[sqlite-vec, #raw("github.com/asg017/sqlite-vec")\; gbrain supports only Postgres and PGLite.]],
    point[Turso (libSQL):][a SQLite fork with a vector index. Measured in process: #fx(e12.libsql.vector.at(0).median_ms, d: 1) ms, same recall. Not a gbrain backend.#footnote[libSQL, #raw("github.com/tursodatabase/libsql").]],
    point[MongoDB:][Community #e12.mongodb.version.mongodb with mongot, measured: #fx(e12.mongodb.vector.at(0).median_ms, d: 1) ms vs pgvector #fx(e12.pgvector.vector.at(0).median_ms, d: 1) ms, same recall; links #fx(e12.mongodb.hop.median_ms, d: 2) vs #fx(e12.pgvector.hop.median_ms, d: 2) ms.#footnote[MongoDB, Vector Search on self-managed deployments, #raw("mongodb.com/docs/search/self-managed").]],
  )
]

// ------------------------------------------------------------------ agent side

#slide[
  #tag[First look]
  #title[Agent side: #hl[first observations]]
  #let R(n) = rtk.commands.find(c => c.name == n)
  #let B(n) = [#R(n).bytes_before → #R(n).bytes_after]
  #grid(columns: (1.4fr, 1fr), gutter: 1cm,
    image("/figures/fig10-harness.svg", height: 7.4cm),
    [
      #point[RTK.][Cut five shell commands from #num(rtk.bytes_before) to #num(rtk.bytes_after) bytes, but made one of them longer.]
      #point[gbrain search.][On #hp.n questions it found the right note #calc.round(hp.gbrain_hit_rate * hp.n) times; exact text search found it #calc.round(hp.vault_hit_rate * hp.n) times.]
      #note[Local 27B model: a 32K window used #fx(FCTX.at("32768").input_tokens.mean / FCTX.at("8192").input_tokens.mean, d: 1)× the tokens of 8K with no measured gain; 128K did not fit in 32 GB. Cloud agents were not run.]
    ],
  )
  #v(1fr)
  #text(size: 15pt, weight: 600)[What RTK is]
  #v(0.1cm)
  #set text(size: 12.5pt, weight: 400)
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.5cm,
    point[What it does:][a small command-line proxy between the AI agent and the shell. A hook in Cursor or Claude Code rewrites a command such as `git status` into `rtk git status`, runs it, and hands the agent a condensed version of the output, so the model reads fewer tokens.#footnote[RTK 0.51.0, #raw("www.rtk-ai.app")\; Cursor hook registered with #raw("rtk init -g --agent cursor").]],
    point[What it does not touch:][only shell commands pass through it. The agent's own file-read and search tools, MCP calls such as gbrain, and the model's replies are not filtered. If the hook fails, the raw command runs.],
    point[How it was measured:][five fixed commands, each run raw and through RTK, counting output bytes: #box[`ls -la`] #B("ls_la_hilbert_paper"), #box[`git status`] #B("git_status_hilbert_paper"), `rg` #B("rg_typst_hilbert_paper"), #box[`git log`] unchanged, `find` #B("find_md_depth2_zig_hilbert") (longer).],
  )
]

// ------------------------------------------------------------------ twelve ideas

#slide[
  #tag[Exploratory]
  #title[Twelve ideas from recent papers, #hl[tested]]
  #grid(columns: (2fr, 1fr), gutter: 0.8cm,
    image("/figures/fig19-ideas.svg", width: 100%),
    [
      #big([+#fx(I("H1a").diff)], [recall from reranking the vector top 50 — the only idea accepted])
      #note[#i1.secondary.default.near_misses_moved_into_top10 near-misses move into the top ten; cost #fx(i1.secondary.default.latency_seconds.median, d: 1) s per question.]
      #v(0.2cm)
      #note[Sentence-level packing kept more notes but a blind judge found the text answered less often (#fx(i2.judge_guard.yes_pruned * 100, d: 0)% vs #fx(i2.judge_guard.yes_chunk * 100, d: 0)%), so it failed.]
    ],
  )
]

// ------------------------------------------------------------------ per-token methods

#slide[
  #tag[Exploratory]
  #title[More relevant notes per token? #hl[Not with these]]
  #grid(columns: (1.35fr, 1fr), gutter: 0.9cm,
    [
      #set text(size: 15pt)
      #table(
        columns: (auto, 1fr, auto),
        column-gutter: 0.6cm,
        stroke: (x, y) => if y == 0 { (bottom: 2pt + blue) } else { (bottom: 0.6pt + hair) },
        inset: (x: 0pt, y: 6.5pt),
        align: (left, left, right),
        table.header([], [Method], hl[Change in survival]),
        [M1], [Relevance per token, greedy fill], [#fx(es.family.M1.diff)],
        [M2], [Budgeted coverage (submodular)], [#fx(es.family.M2.diff)],
        [M3], [Diversity: maximal marginal relevance], [#fx(es.family.M3.diff)],
        [M4], [Diversity: determinantal point process], [#fx(es.family.M4.diff)],
        [M5], [Heat diffusion over links], [#fx(es.family.M5.diff)],
        [M6], [Manifold ranking], [#fx(es.family.M6.diff)],
        [M7], [Advection–diffusion over links], [#fx(es.family.M7.diff)],
      )
      #note[Survival: a relevant note is in the pack, averaged over budgets of 1,000 to 6,000 tokens; confirmation half, #m9.per_question_confirm.idx.len() questions. None reached the pre-registered +0.02.]
    ],
    [
      #big([−#fx(m9.primary.relative_fall * 100, d: 0)%], [tokens from a calibrated early stop (M9), survival unchanged])
      #note[Conformal risk control picks the stop: #num(m9.tokens.confirm) instead of #num(m9.tokens.baseline_confirm) tokens per question, survival #fx(m9.survival.confirm) vs #fx(m9.survival.baseline_confirm). The bar was 20%, so it is not accepted.]
      #v(0.3cm)
      #point[Why.][Every method's best setting was the one closest to plain cosine order. Repetition and links are not what loses notes here.]
    ],
  )
]

// ------------------------------------------------------------------ scoreboard

#slide[
  #title[Scoreboard #hl[so far]]
  #set text(size: 13.5pt)
  #table(
    columns: (1.25fr, 0.75fr, 1.9fr),
    column-gutter: 0.8cm,
    stroke: (x, y) => if y == 0 {
      (bottom: 2pt + if x == 0 { ink } else { blue })
    } else {
      (bottom: 0.6pt + slate)
    },
    inset: (x: 0pt, y: 5.5pt),
    table.header([Question], hl[Winner], hl[Measured]),
    [Keywords or vectors?], [vectors], [recall #fx(grid-data.at("vector-chunk").R) vs #fx(grid-data.at("keyword-page").R)],
    [What to hand the model?], [chunks], [right note kept #fx(units.chunk.relevant_in_pack) vs #fx(units.page.relevant_in_pack) at equal tokens],
    [Hybrid search?], [no gain], [#fx(C("hybrid-rrf-chunk").recall) vs #fx(C("vector-chunk").recall)],
    [Hilbert key instead of vector search?], [no], [#fx(C("hk1-L1-R16").recall) vs #fx(C("vector-chunk").recall)],
    [Hilbert key, per token the model reads?], [cosine], [#fx(EF("hk1 level 1, 16 ranges").survival_per_1k_delivered) vs #fx(EF("cosine, every chunk").survival_per_1k_delivered) relevant per 1k delivered tokens],
    [Hilbert key, per token of candidates read?], [hk1], [#calc.round(EF("hk1 level 1, 16 ranges").survival_per_100k_processed / EF("cosine, every chunk").survival_per_100k_processed)× reading everything; about 2× random],
    [Second database engine?], [Postgres], [faster at both operations tested],
    [Smarter packing per token?], [plain cosine], [#es.family.len() methods tested, none better; an early stop saves #fx(m9.primary.relative_fall * 100, d: 0)% of tokens],
    [Reranker?], [yes, on vectors], [+#fx(I("H1a").diff) recall for #fx(i1.secondary.default.latency_seconds.median, d: 1) s],
    [Holds on unseen questions?], [yes], [held-out recall #fx(ho.at("R@10"))],
    [Sentence packs on unseen questions?], [yes], [right note kept #fx(fho.runs.best_survival.survival) vs #fx(fho.runs.reference.survival), fewer tokens],
    [Sentence packs answer the question?], [chunks], [judged sufficient #fx(fjp.yes.A) vs #fx(fjp.yes.B), #fjp.mcnemar_two_sided.new_only vs #fjp.mcnemar_two_sided.comparator_only questions],
    [Hilbert key as a facet map?], [no], [found #fx(fuc.at("hk1_facets").recall) vs #fx(fuc.at("dense_facets").recall), with #fx(fus.acceptance.at("chunk-hk1 vs chunk-dense").token_ratio, d: 1)× the tokens per answer],
    [Reranker null on unseen questions?], [underpowered], [power #fx(fpw.curves.at("held-out (+0.016)").curve.find(c => c.n == 800).power * 100, d: 0)%; with judge labels +#fx(fqe.reranker_gain.at("extended").diff)],
    [Hilbert key over area, type, month?], [B-tree], [#fx(calc.min(..S9R), d: 2) to #fx(calc.max(..S9R), d: 2)× its buffers; area filter #fx(S9("dense_title_area_filter").recall) vs area words #fx(S9("dense_title_and_area_words").recall)],
    [Key rebuilt on axes fitted to the notes?], [no], [#fx(SA("C_itq")) vs #fx(SA("A_random_axes")) on random axes and #fx(s10.at("exhaustive_R@10")) for every chunk],
  )
  #v(1fr)
  #align(right)[#text(size: 18pt)[These are #text(fill: blue, style: "italic")[measured findings, not decisions].]]
]

#divider[Part 3][Every combination]

// ------------------------------------------------------------------ factorial

#slide[
  #tag[Exploratory]
  #title[Every combination, then #hl[unseen questions]]
  #grid(columns: (1fr, 1fr), gutter: 0.9cm,
    [
      #big([#fx(fho.runs.best_survival.survival)], [held-out questions with a relevant note in the pack, packing the best sentences of the top 30 notes (chunks: #fx(fho.runs.reference.survival))])
      #note[#fho.paired_vs_reference.best_survival.survival.new_only questions favour sentence packs, #fho.paired_vs_reference.best_survival.survival.comparator_only favour chunks; #num(-fho.paired_vs_reference.best_survival.tokens_diff) fewer tokens. A pre-registered judge found them sufficient to answer less often (#fx(fjp.yes.A) vs #fx(fjp.yes.B) on #fjp.n questions), so this is reaching the note, not answering.]
    ],
    [
      #point[#num(fcells.n_cells) combinations.][Eight switches crossed on #fcells.n_questions_analysis_set questions; each switch scored by its average contribution (Shapley value).]
      #point[Reranker.][Largest recall gain on development questions (+#fx(fsh.at("F3=on").at("R@10").shapley)) for #fx(fsh.at("F3=on").latency.shapley / 1000, d: 1) s; on unseen questions the best combination's gain shrank to +#fx(fho.paired_vs_reference.best_recall.at("R@10").diff), not significant.]
      #point[Hilbert key filter.][#fx(fsh.at("F7=hk1").at("R@10").shapley) recall in every combination.]
    ],
  )
]

// ------------------------------------------------------------------ specified cells

#slide[
  #tag[Exploratory]
  #title[The same switches, #hl[measured again]]
  #grid(columns: (1fr, 1fr), gutter: 0.9cm,
    [
      #point[Best sentence.][Recall at 10 #fx(c1.at("R@10").best_sentence) against #fx(c1.at("R@10").best_chunk).]
      #point[Seed clamp.][Recall changed by #fx(c2.diff). The original top note stayed in the top ten.]
      #point[Forward push.][#fx(c3.at("R@10").forward_push) against #fx(c3.at("R@10").existing_pagerank), and the walk is slower.]
      #point[Reranker on exact ties.][#fx(c5.seconds.selective, d: 2) s against #fx(c5.seconds.always_on, d: 2) s. Confirmation MRR stays inside the always-on interval.]
    ],
    [
      #big([#fx((1 - c8.token_ratio) * 100, d: 0)%], [fewer tokens when a separate checker stops the pack], size: 48pt)
      #note[The 27B judge calls those packs sufficient for #fx(c8.sufficiency.yes) of 150 questions, against #fx(c8.sufficiency.chunk_pack) for the full chunk pack.]
      #v(0.35cm)
      #point[Winning sentence and its headings.][#fx(c9.tokens.sentence_headings, d: 0) tokens, sufficient for #fx(c9.sufficiency.yes).]
      #point[Cover both pages.][Set recall #fx(c16.set_recall) on the 60 two-page questions, above #fx(c16.bar).]
      #v(0.15cm)
      #note[The held-out questions stay sealed.]
    ],
  )
]

// ------------------------------------------------------------------ close

#page(footer: none)[
  #place(top + right, dx: mx, dy: -0.9cm, network)
  #v(1fr)
  #text(size: 50pt, weight: 600)[Thank #hl[you]]
  #v(0.5cm)
  #text(size: 18pt, weight: 400)[Paper, measured results and every script:] \
  #text(size: 16pt)[#raw("github.com/guanchzhou/hilbert-paper")]
  #v(0.4cm)
  #note[The notes themselves are private and are not in the repository.]
  #v(1.2fr)
]
