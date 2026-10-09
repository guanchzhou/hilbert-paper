// Build: python3 build.py   (figures, checks, then typst compile)

#let grid-data = json("bench/retrieval-grid.json").cells
#let units = json("bench/evidence-units.json").units
#let sweep = json("bench/hilbert-sweep.json")
#let rnd = json("bench/random-filter.json").cells
#let unc = json("bench/uncertainty.json")
#let pq = json("bench/per-query.json")
#let man = json("bench/manifest.json")
#let ho = json("bench/heldout.json")
#let sc = json("bench/scale-200k.json")
#let live = json("bench/pgvector-5143.json")
#let rtk = json("bench/rtk-bytes.json")
#let hp = json("bench/harness-pages.json")
#let inv = json("bench/investigation.json")
#let cen = json("bench/centering.json")
#let lqa = json("bench/lsh-qa.json")
#let h4t = json("bench/h4-ttest.json")
#let ideas = json("bench/ideas/summary.json")
#let I(k) = ideas.hypotheses.find(h => h.id == k)
#let i1 = json("bench/ideas/1-rerank.json")
#let i2 = json("bench/ideas/2-prune.json")
#let i6 = json("bench/ideas/6-partitions.json")
#let i8 = json("bench/ideas/8-synopsis.json")
#let i9d = json("bench/ideas/9-links-diagnostic.json")
#let i10 = json("bench/ideas/10-hyde.json")
#let av = json("bench/arango-vector.json")
#let al = json("bench/arango-load.json")
#let ah = json("bench/arango-hop.json")
#let pg18 = json("bench/pg18/comparison.json")
#let PR(m) = pg18.rows.find(r => r.measure == m)
#let A(engine, key, val) = av.results.find(r => r.engine == engine and r.at(key, default: none) == val)
#let AX(engine) = av.results.find(r => r.engine == engine and r.index.starts-with("none"))
#let sources = json("bench/sources.json")
#let top-ho = json("bench/factorial/heldout-level1.json")
#let top-jp = json("bench/factorial/judge-packs.json")
#let top-es = json("bench/efficiency/summary.json")

#let fx(x, d: 3) = {
  let v = calc.round(x, digits: d)
  let s = str(v)
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
#let fp(p) = {
  if p <= 0 {
    [$<$ 0.001]
  } else if p < 0.001 {
    let e = calc.floor(calc.log(p, base: 10))
    let m = p / calc.pow(10.0, e)
    [#fx(m, d: 1) × 10#super[#str(e).replace("-", "−")]]
  } else { fx(p) }
}
#let ci(c) = [#fx(c.mean) (#fx(c.lo) to #fx(c.hi))]
#let dci(t) = [#fx(t.diff) (95% CI #fx(t.ci_lo) to #fx(t.ci_hi))]
#let H(k) = inv.confirmatory.at(k)
#let E(k) = inv.exploratory.at(k)
#let C(k) = inv.cells.at(k)
#let bs = json("bench/budget-sweep.json")
#let BS(b, u) = bs.rows.find(r => r.budget == b and r.unit == u)
#let eff = json("bench/hk1-efficiency.json")
#let EF(name) = eff.conditions.find(c => c.condition == name)
#let hk-gain = (1, 4, 8, 16).map(r => {
  let h = EF("hk1 level 1, " + str(r) + " ranges")
  let x = EF("random, same size as hk1 " + str(r) + " ranges")
  (h.survival_per_1k_delivered / x.survival_per_1k_delivered, h.survival_per_100k_processed / x.survival_per_100k_processed)
}).flatten()
#let LABELS = (
  H1: "Dense chunk vs lexical note (AND)", H2: "Dense chunk vs dense note mean",
  H3: "Chunk pack vs page pack (survival)", H4: "hk1 level 1, 16 ranges vs dense chunk (non-inferiority)",
  E1: "Lexical note, OR vs AND", E2: "RRF hybrid (note lexical) vs dense chunk",
  E3: "Chunk pack vs section pack (survival)", E4: "Lexical chunk OR vs dense chunk",
  E5: "RRF hybrid (chunk lexical) vs dense chunk",
)

// Original-language names and titles, for example a Russian source cited in its own script.
#let orig(lang: "ru", body) = text(lang: lang, body)

#let index(term) = [#metadata(term)<idx>#term]
#let mark(term) = [#metadata(term)<idx>]
#let make-index() = context {
  let rows = ()
  for it in query(<idx>) { rows.push((str(it.value), it.location().page())) }
  let keys = rows.map(r => r.at(0)).sorted(key: k => lower(k)).dedup()
  set par(justify: false, first-line-indent: 0pt)
  set text(size: 8.5pt)
  columns(2, gutter: 14pt)[
    #for key in keys {
      let pages = rows.filter(r => r.at(0) == key).map(r => r.at(1)).sorted().dedup()
      [#key #box(width: 1fr, repeat[.]) #pages.map(str).join(", ") \ ]
    }
  ]
}

#let boxed(title, body) = block(
  width: 100%, inset: (x: 9pt, y: 7pt), stroke: (left: 1.2pt + rgb("#1f4e79")),
  fill: rgb("#f3f6fa"),
)[#set par(first-line-indent: 0pt); *#title* #body]

#let rule = table.hline(stroke: 0.5pt)
#let thin = table.hline(stroke: 0.3pt)

#set document(title: "Dense, lexical, and space-filling-curve retrieval in a personal knowledge base", author: "Andrey Maltsev", date: none)
// arXiv preprint layout (NeurIPS preprint style): US Letter, 5.5 in by 9 in text block, Times 10 pt.
#set page(
  paper: "us-letter",
  margin: (left: 1.5in, right: 1.5in, top: 1in, bottom: 1in),
  footer: context {
    set text(size: 10pt)
    align(center)[#counter(page).display()]
  },
)
#set text(font: "Times New Roman", size: 10pt, lang: "en", hyphenate: true)
#show math.equation: set text(font: "New Computer Modern Math")
#show raw: set text(size: 9pt)
#set par(justify: true, leading: 0.5em, spacing: 0.9em, first-line-indent: 0pt)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.4em, below: 0.8em)
#show heading.where(level: 1): set text(size: 12pt, weight: "bold")
#show heading.where(level: 2): set text(size: 10pt, weight: "bold")
#show heading.where(level: 3): set text(size: 10pt, weight: "bold")
#show heading: it => {
  if it.numbering == none { it } else {
    block(below: 0.8em)[#counter(heading).display(it.numbering)#h(1em)#it.body]
  }
}
#set math.equation(numbering: "(1)")
#show figure: set block(spacing: 1.4em)
#show figure.caption: set text(size: 10pt)
#show figure.caption: set par(justify: true)
#show figure.where(kind: table): set figure.caption(position: top)
#set figure.caption(separator: [: ])
#show table: set text(size: 9pt)
#show table.cell: set par(justify: false)
#show cite: set text(fill: rgb(0, 128, 0))
#show ref: set text(fill: rgb(0, 0, 200))
#show link: set text(fill: rgb(0, 0, 200))
#set footnote.entry(separator: line(length: 2in, stroke: 0.4pt))
#show footnote.entry: set text(size: 9pt)
#set list(indent: 1em)
#set enum(indent: 1em)

// ---------------------------------------------------------------- title block

#v(0.15in)
#line(length: 100%, stroke: 4pt)
#v(0.18in)
#align(center)[#text(size: 17pt, weight: "bold")[Dense, Lexical, and Space-Filling-Curve Retrieval\ in a Personal Knowledge Base]]
#v(0.12in)
#line(length: 100%, stroke: 1pt)
#v(0.3in)
#align(center)[
  #text(weight: "bold")[Andrey Maltsev] \
  Independent researcher \
  #raw("github.com/guanchzhou/hilbert-paper")
]
#v(0.3in)

#align(center)[#text(size: 12pt, weight: "bold")[Abstract]]
#v(0.05in)
#pad(x: 0.5in)[
  Personal knowledge bases retrieve notes for language models under a token budget, yet it is not known which design choices change whether the right note reaches the model and which only change cost. We test four hypotheses, stated before data collection, on one frozen corpus of #num(man.pages_live) notes and #num(man.chunk_count) chunks with #num(man.qrels_count) development questions and #num(ho.n) sealed held-out questions, using paired Wilcoxon and exact McNemar tests with Holm correction. Dense chunk retrieval exceeds lexical note retrieval by #fx(H("H1 vector-chunk > keyword-and-page").diff) in recall at 10, while chunk-level and note-level dense indexing do not differ. Under a 6,000-token budget, returning chunks keeps the relevant note for #fx(inv.survival.chunk) of questions against #fx(inv.survival.page) for whole notes, and no question favours whole notes.

  We also built a Hilbert-curve key for this study and evaluated it as a candidate filter. Its best setting reaches a recall of #fx(C("hk1-L1-R16").recall) against #fx(C("vector-chunk").recall) for exhaustive cosine and fails non-inferiority. We show that its first level is an eight-bit sign hash; after mean-centring the anisotropic embeddings, angular hashing theory predicts the observed question-to-answer collision rate (#fx(lqa.centred.same_cell_observed) observed, #fx(lqa.centred.same_cell_predicted) predicted) and implies that about #calc.round(lqa.centred.at("independent_8bit_tables_for_0.9")) independent keys would be needed for 90% recall. Compared on the same vectors, pgvector's HNSW index is faster than ArangoDB's vector index at matched quality, and both engines return identical one-hop neighbourhoods. On held-out questions the dense chunk configuration reaches a recall at 10 of #fx(ho.at("R@10")).

  Two exploratory studies follow. None of the #top-es.family.len() methods tested for putting more relevant notes into the same token budget beat plain cosine order. In a full factorial of eight pipeline switches, packing the best sentences of the top 30 notes kept a relevant note for #fx(top-ho.runs.best_survival.survival) of held-out questions against #fx(top-ho.runs.reference.survival) for chunk packs, with fewer tokens, yet a pre-registered local judge found those packs sufficient to answer less often (#fx(top-jp.yes.A) against #fx(top-jp.yes.B)); the best-recall combination's development gain did not replicate on held-out questions, which follow-up checks trace to low power and to incomplete relevance labels. Used as a map whose facet regions an agent intersects, the Hilbert key also failed a pre-registered cost rule.
]

#place(bottom + left, float: true, scope: "parent", clearance: 0pt)[#text(size: 9pt)[Preprint.]]

// ---------------------------------------------------------------- 1

= Introduction <sec-intro>

Retrieval-augmented use of language models has moved from web-scale collections to small personal ones: a few thousand notes, written by one person, searched by that person's assistant. At this scale the classical constraints change. Exhaustive similarity search over every chunk is cheap, so the index structure matters less than in web search. What matters is whether the right note is among the few passages a model is given, and how many tokens those passages cost, because the context window is the scarce resource and long contexts are used unevenly @liu2024.

Practitioners face a set of design switches with little controlled evidence for small corpora: lexical or dense scoring @robertson2009 @karpukhin2020, note-level or chunk-level representation @khattab2020 @jayaram2024, the unit of text returned to the model, and the index structure. This study adds one candidate of its own: a compact text key that places every vector on a Hilbert curve @hilbert1891 @moon2001, so that the key can be stored in an ordinary database column and a prefix range of keys can act as a candidate filter. To test whether such a key is a useful addition to knowledge management for AI systems, we implemented one, the hk1 key in the zig-hilbert library @zighilbert, and evaluate it here.

This study measures those switches on one frozen personal corpus, under a protocol fixed in advance, and investigates the mechanisms behind each result. Five research questions are posed.

/ RQ1: Does dense retrieval find relevant notes that lexical retrieval misses, and the reverse?
/ RQ2: Is the chunk or the note the better unit of indexing for dense retrieval?
/ RQ3: Under a fixed token budget, which unit of returned text best preserves the relevant note?
/ RQ4: Can a prefix range of Hilbert keys replace exhaustive cosine search as a candidate filter, and if not, why not?
/ RQ5: At this corpus size and at forty times it, does query latency or a graph operation justify an engine other than Postgres, specifically ArangoDB?

The contributions are: a controlled comparison with pre-stated hypotheses, paired tests, and a sealed held-out set (@sec-results); an analytical account of the Hilbert key as an eight-bit angular hash, validated against observation after correcting for embedding anisotropy (@sec-mech-hk1); a failure taxonomy of dense retrieval on this corpus (@sec-failures); and a fully reproducible artefact in which every number in the text is read from a result file (see Data and code availability).

// ---------------------------------------------------------------- 2

= Background and related work <sec-related>

== Lexical and dense retrieval

#index("BM25") scores a document by term frequency, inverse document frequency, and length normalisation @robertson2009. Dense passage retrieval encodes the question and the passage separately and ranks by inner product @karpukhin2020. Across the eighteen BEIR collections neither family dominates, and BM25 is a strong zero-shot baseline @thakur2021. Hybrid retrieval combines the two, commonly by #index("reciprocal rank fusion") @cormack2009, and has been reported to be more robust than either alone @sawarkar2024. Embedding models such as #index("Qwen3-Embedding") are trained with instruction prefixes for the query side @zhang2025.

== Documents as sets of vectors

Late-interaction models score a passage by matching each query token to its best document token @khattab2020. #index("MUVERA") compresses a multi-vector document into a single fixed-dimensional encoding whose inner product approximates the set similarity @jayaram2024. A note represented by the mean of its chunk vectors is the simplest single-vector summary; a note scored by its best chunk is the simplest set score. RQ2 compares these two.

== Approximate search, hashing, and space-filling curves

Graph indexes such as #index("HNSW") make exhaustive search unnecessary at large scale @malkov2018; pgvector implements HNSW inside Postgres @pgvector. Inverted-file (IVF) indexes partition the vectors into lists around learned centroids and search only the lists nearest the query @sivic2003 @jegou2011; the Faiss library implements them @douze2024. Quantisation-based methods protect the neighbours that matter @guo2020, and redundant assignment recovers neighbours near partition boundaries @sun2023. Random-hyperplane hashing preserves angular similarity: two vectors at angle $theta$ fall on the same side of a random hyperplane with probability $1 - theta \/ pi$ @charikar2002, and concatenating and repeating such hashes yields locality-sensitive hashing with tunable recall @indyk1998. A #index("Hilbert curve") maps a grid to a line while preserving locality better than other simple curves @hilbert1891 @moon2001. The hk1 key built for this study composes a fixed sign projection, a quantiser, and an eight-dimensional Hilbert index @zighilbert.

== Geometry of embedding spaces

Contextual embeddings are anisotropic: they occupy a narrow cone, so that unrelated texts have positive cosine similarity @ethayarajh2019. Removing the common mean direction is a standard correction @mu2018. @sec-mech-hk1 shows that this property governs the behaviour of a sign-projection key.

== Evaluation methodology

Normalised discounted cumulative gain discounts rank $i$ by $log_2(i+1)$ @jarvelin2002, and mean reciprocal rank, introduced for the TREC-8 question-answering track, averages the reciprocal rank of the first correct answer @voorhees1999. For paired comparisons of retrieval systems over the same questions, randomisation, bootstrap, and signed-rank tests agree closely, while the sign test and paired $t$-test can mislead @smucker2007. This study uses the Wilcoxon signed-rank test @wilcoxon1945 for graded per-question scores, the exact McNemar test @mcnemar1947 for binary outcomes, percentile bootstrap intervals @efron1979, and Holm's step-down correction for multiple tests @holm1979. A non-inferiority test asks whether a cheaper method is worse than a reference by no more than a margin fixed in advance, rather than whether the two differ @walker2011. @sec-stats explains each procedure as applied here.

// ---------------------------------------------------------------- 3

= Hypotheses <sec-hypotheses>

The measurement plan (`plan.md` in the repository) was written and committed before any development cell was run. It fixed the corpus, the question set, the metrics, the budget, and the following confirmatory hypotheses with their acceptance criteria. Analyses not listed here are labelled exploratory throughout.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    stroke: none,
    rule,
    [*Id*], [*Hypothesis and acceptance criterion*], [*Test*],
    thin,
    [H1], [Dense chunk retrieval has higher recall at 10 than lexical whole-note retrieval.], [Wilcoxon @wilcoxon1945, one-sided],
    [H2], [Dense chunk retrieval and dense note-mean retrieval differ in recall at 10.], [Wilcoxon @wilcoxon1945, two-sided],
    [H3], [At a 6,000-token budget, a chunk pack contains the relevant note more often than a whole-note pack. A cheaper unit that loses the note counts as a regression.], [exact McNemar @mcnemar1947, one-sided],
    [H4], [The best Hilbert-key filter is non-inferior to exhaustive cosine: recall at 10 no more than 0.03 below it, with median candidates at most 20 percent of the keyed chunks.], [larger $p$ of Wilcoxon and paired $t$ on shifted differences, one-sided @walker2011],
    rule,
  ),
  caption: [Confirmatory hypotheses, family-wise $alpha$ = 0.05 with Holm correction @holm1979. Each test is described in @sec-stats.],
) <tab-hyp>

Exploratory analyses: E1, AND against OR semantics for lexical queries; E2 and E5, reciprocal-rank-fusion hybrids against dense retrieval; E3, chunk against section packs; E4, the best lexical arm against dense retrieval; a random-subset control for the key filter; the hashing model and the mean-centring intervention of @sec-mech-hk1; and the failure taxonomy of @sec-failures. Exploratory tests are Holm-corrected within their own family.

// ---------------------------------------------------------------- 4

= Materials and methods <sec-methods>

== Corpus

The corpus is the author's #index("gbrain") knowledge base @gbrain at the time of measurement: markdown notes on software engineering, infrastructure, research, and genealogy, mirrored from an Obsidian @obsidian vault into Postgres 16 @postgres16. Notes are split into chunks by gbrain's chunker and each chunk is embedded with Qwen3-Embedding-0.6B @zhang2025 at #man.dimensions dimensions, run locally; gbrain records the model as `ollama:qwen3-embedding-8k`, after its Ollama provider @ollama. Question vectors for this study were computed through a local endpoint that serves the same model with MLX @mlx. The note title is prefixed to each chunk before embedding, a minimal form of the chunk contextualisation proposed as contextual retrieval @anthropic2024. @tab-corpus and @fig-dataset describe the corpus and the questions.

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    stroke: none,
    rule,
    [*Property*], [*Value*],
    thin,
    [Live notes], [#num(man.pages_live)],
    [Chunks with a vector (stored)], [#num(man.chunk_count) (#num(man.chunks_all))],
    [Chunks per note, median (90th percentile)], [#fx(inv.dataset.chunks_per_page.median, d: 0) (#fx(inv.dataset.chunks_per_page.p90, d: 0))],
    [Tokens per chunk, median (90th percentile)], [#num(inv.dataset.chunk_tokens.median) (#num(inv.dataset.chunk_tokens.p90))],
    [Chunks with an hk1 key], [#num(man.chunk_hilbert_count)],
    [Development questions], [#num(inv.dataset.questions)],
    [Words per question, median (interquartile range)], [#fx(inv.dataset.question_words.median, d: 0) (#fx(inv.dataset.question_words.p25, d: 0) to #fx(inv.dataset.question_words.p75, d: 0))],
    [Relevant notes per question, mean], [#fx(inv.dataset.relevant_per_question.mean, d: 2)],
    [Relevant slugs absent from the live corpus], [#inv.dataset.relevant_slugs_missing_from_corpus],
    [Held-out questions], [#num(ho.n)],
    rule,
  ),
  caption: [Corpus and question set. Source: `bench/manifest.json`, `bench/investigation.json`.],
) <tab-corpus>

#figure(
  image("figures/fig00-dataset.svg", width: 100%),
  caption: [Distributions of question length, relevant notes per question, and chunk length (every fifth chunk). The bimodal chunk length reflects short structural chunks and long body chunks.],
) <fig-dataset>

== Questions and relevance

The question set is gbrain's evaluation set: natural-language questions written against the brain, each with the slugs of the notes that answer it, split into a development half and a held-out half of #num(ho.n) questions each. The held-out file was not read until every development analysis was frozen; it was then opened once, for the winning dense cell (@sec-heldout). Relevance is binary and scored at note level after deduplication.

== Retrieval conditions

@fig-pipeline shows the system. Each condition changes one factor.

#figure(
  image("figures/fig01-pipeline.svg", width: 100%),
  caption: [The pipeline. Solid arrows are the production path; the keyword path ranks by Postgres full-text search; the dashed path is the Hilbert-key filter under test.],
) <fig-pipeline>

- *Lexical, whole note (AND or OR).* Postgres full-text search over the note text @postgres16. The function `websearch_to_tsquery`, with the English configuration, stems the question's words, drops stop words, and joins the remaining terms with AND; notes are ranked by `ts_rank_cd`, a cover-density score that rewards matched terms occurring close together @clarke2000. The OR variant replaces conjunctions by disjunctions in the parsed query.
- *Lexical, chunk (AND or OR).* The same, over each chunk's stored text-search vector; a note is ranked by its best chunk.
- *Dense, note mean.* Cosine between the question vector and the mean of the note's chunk vectors.
- *Dense, chunk.* Cosine between the question vector and every chunk; a note is ranked by its best chunk.
- *Hybrid.* Reciprocal rank fusion @cormack2009 over the top 50 of a lexical arm and of dense chunk retrieval: a note at rank $r$ in an arm receives $1 \/ (60 + r)$ from that arm, the scores are summed, and the constant 60 limits the weight of the very top ranks.
- *Hilbert filter.* The probe of a question is the set of key ranges searched for it: the cell that contains the question's own key at a chosen level, plus neighbouring cells (@sec-key). Only chunks whose key falls in the probe ranges are rescored by exact cosine.

Question vectors were computed once with the live instruction prefix (`Instruct: Given a web search query, retrieve relevant passages that answer the query`, then `Query:` on a new line), the query format of Qwen3-Embedding @zhang2025, and reused by every condition. No condition calls a reranker.

== The hk1 key <sec-key>

#mark("hk1")
For a vector $e in RR^1024$, zig-hilbert 0.2.1 computes the key in four steps @zighilbert: (i) project onto eight axes $z_j = s_j dot e$, where $s_j in {-1, +1}^1024$ are fixed by the seed `9e3779b97f4a7c15` (the 64-bit golden-ratio constant) through #index("SplitMix64"), a fast 64-bit pseudorandom generator @steele2014; (ii) divide by $norm(e)$; (iii) map each axis into 256 cells through the logistic function $sigma(z) = 1 \/ (1 + e^(-1.702 z))$, which with the scale 1.702 stays within 0.01 of the standard normal distribution function @camilli1994 @bowling2009, so that the cells would be nearly equally likely if the normalised projections were standard normal; (iv) index the eight cell coordinates on the eight-dimensional Hilbert curve with Skilling's transpose algorithm @skilling2004, giving 64 bits written as `hk1:8:8:9e3779b97f4a7c15:` and sixteen hex digits, for example `hk1:8:8:9e3779b97f4a7c15:2e240e0214b885c0`. A #index("probe level") $ell$ keeps the leading $8 ell$ bits; a probe with $r$ ranges adds up to $r - 1$ neighbouring cells across the nearest boundaries.

*Proposition.* The leading eight bits of an hk1 key identify the orthant of $(z_1, dots, z_8)$. Because $sigma(z) >= 1\/2$ exactly when $z >= 0$, the top bit of each cell coordinate is the sign of $z_j$, and the first level of the Hilbert curve is a bijection between the $2^8$ sub-cubes and the leading eight index bits. A level-1 cell is therefore an eight-bit random-sign hash. If the projections behave as random hyperplanes, two unit vectors at angle $theta$ share a level-1 cell with probability

$ P_1(theta) = (1 - theta / pi)^8 , $ <eq-lsh>

and a filter built from $L$ independent such keys finds a neighbour with probability $1 - (1 - P_1)^L$, so that a target recall $rho$ requires

$ L = ln(1 - rho) / ln(1 - P_1) $ <eq-tables>

independent keys @charikar2002 @indyk1998. The first claim was verified empirically: on every sampled pair, equality of the leading eight bits coincided with equality of the eight signs (@sec-mech-hk1).

== Evidence packing

From the dense chunk ranking, four packs were built: the winning chunk; the chunk with one neighbour on each side (window); the chunk extended to the enclosing markdown heading (section); and the whole note (page). Units are walked in rank order and kept whole if they fit in the remaining 6,000 tokens, skipped otherwise. The budget is gbrain's default delivery budget (`DEFAULT_RETURN_BUDGET` in its source), the amount of evidence it hands an agent per search; it was adopted as the operating point, not tuned, and @tab-budgets repeats the packing at other budgets. Tokens follow gbrain's counter: one token per CJK (Chinese, Japanese, or Korean) character and one per four other characters.

== Measures

For a question with relevant set $R$ and a deduplicated ranked list $L$:

$ "R@10" = (|R inter L_(1..10)|) / (|R|) , $ <eq-recall>

$ "MRR" = 1 / (min{i : L_i in R}) , $ <eq-mrr>

$ "nDCG@10" = (sum_(i=1)^10 bb(1)[L_i in R] \/ log_2(i+1)) / (sum_(i=1)^(min(10,|R|)) 1 \/ log_2(i+1)) . $ <eq-ndcg>

#mark("recall at 10")#mark("MRR")#mark("nDCG")
Recall at 10 is the share of a question's relevant notes found among the first ten. Mean reciprocal rank (MRR) scores the first relevant note at rank $i$ as $1 \/ i$, and zero if none is returned @voorhees1999. Normalised discounted cumulative gain (nDCG) at 10 discounts each relevant note by the logarithm of its rank and divides by the best attainable value @jarvelin2002. Survival is the indicator that a pack contains at least one relevant note. All figures are means over questions.

== Statistical analysis <sec-stats>

Every comparison is paired: both conditions answer the same questions, and each question serves as its own control. Per-question recall differences were tested with the Wilcoxon signed-rank test @wilcoxon1945, a nonparametric test of whether paired differences are centred on zero: it ranks the absolute values of the non-zero differences and compares the rank sums of the positive and the negative ones. Zero differences were discarded. Survival, which is binary, was compared with the exact McNemar test @mcnemar1947. It uses only the discordant questions, those where exactly one of the two packs kept the relevant note, and asks whether they split evenly between the two packs; the exact form takes the $p$-value from the binomial distribution with probability one half.

Effect sizes are mean paired differences with 95 percent percentile #index("bootstrap interval")s @efron1979: questions were resampled with replacement 10,000 times, and the interval runs from the 2.5th to the 97.5th percentile of the resampled mean differences. For Wilcoxon comparisons we also report the matched-pairs rank-biserial correlation @kerby2014: the rank sum of the positive differences minus the rank sum of the negative ones, divided by the total rank sum, with the ranks and discarded zeros of the test itself; it runs from −1 to 1.

H4 is a non-inferiority hypothesis @walker2011. It does not ask whether the key filter differs from exhaustive cosine, but whether it is worse by more than a margin fixed in advance, 0.03 in recall at 10. The paired differences were shifted by adding the margin, and one-sided tests were applied to the shifted values; rejecting their null hypothesis would have shown the filter to be at most 0.03 worse. The shifted Wilcoxon test alone is unreliable when most paired differences are zero: the shift turns every tie into a small positive value, and those values can dominate the positive rank sum even for a clearly worse method. A one-sided paired $t$-test on the same shifted differences does not have this defect, so the reported $p$ is the larger of the two. This can only make non-inferiority harder to show. The same defect was found and handled in the same way in the exploratory study of @sec-ideas.

Holm's step-down procedure @holm1979 controls the family-wise error rate, the probability of at least one false rejection among several tests. The $m$ $p$-values are sorted, the smallest is multiplied by $m$, the next by $m - 1$, and so on, and each adjusted value is capped at one and kept at least as large as the one before it. It was applied within the confirmatory family and, separately, within the exploratory family. Tests were computed with SciPy @virtanen2020 and resampling with NumPy @harris2020. Latency is reported descriptively.

== Controls

One factor changes per condition. The corpus is not re-embedded. Scratch databases for the scale experiment were separate databases, dropped in the step that recorded their numbers. The run-to-run stability of the dense cell was checked by re-embedding twenty questions five times (@sec-heldout).

// ---------------------------------------------------------------- 5

= Results <sec-results>

== RQ1: lexical and dense retrieval <sec-rq1>

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    rule,
    [*Condition*], [*R\@10*], [*95% CI*], [*MRR*], [*nDCG\@10*],
    thin,
    ..(("keyword-page", "Lexical, note, AND"), ("keyword-chunk", "Lexical, chunk, AND"),
       ("vector-page", "Dense, note mean"), ("vector-chunk", "Dense, chunk")).map(((k, name)) => {
      let c = grid-data.at(k)
      let u = unc.recall_at_10.at(k)
      ([#name], [#fx(c.R)], [#fx(u.lo) to #fx(u.hi)], [#fx(c.MRR)], [#fx(c.nDCG)])
    }).flatten(),
    rule,
  ),
  caption: [Primary retrieval conditions on #num(man.qrels_count) development questions.],
) <tab-grid>

#figure(
  image("figures/fig02-retrieval.svg", width: 100%),
  caption: [Recall at 10, MRR, and nDCG at 10 for the primary conditions.],
) <fig-retrieval>

Dense chunk retrieval reached a recall at 10 of #fx(grid-data.at("vector-chunk").R) against #fx(grid-data.at("keyword-page").R) for lexical whole-note retrieval (@tab-grid). The paired difference was #dci(H("H1 vector-chunk > keyword-and-page")), matched-pairs rank-biserial correlation #fx(H("H1 vector-chunk > keyword-and-page").rank_biserial, d: 2), Holm $p$ = #fp(H("H1 vector-chunk > keyword-and-page").p_holm). H1 is supported. The advantage holds at every depth (@fig-hitk).

#figure(
  image("figures/fig03-hit-at-k.svg", width: 100%),
  caption: [Share of questions with a relevant note in the top $k$.],
) <fig-hitk>

#let ov = pq.overlap_keyword_page_vs_vector_chunk
Per question (@fig-overlap), both conditions found a relevant note for #ov.both questions, dense retrieval alone for #ov.at("only_vector-chunk"), lexical retrieval alone for #ov.at("only_keyword-page"), and neither for #ov.neither. Lexical retrieval returned no note at all for #num(pq.empty_result_lists.at("keyword-page")) questions.

#figure(
  image("figures/fig04-overlap.svg", width: 100%),
  caption: [Per-question agreement between pairs of conditions on whether a relevant note is in the top 10.],
) <fig-overlap>

== RQ2: note or chunk <sec-rq2>

Dense chunk retrieval and dense note-mean retrieval reached #fx(grid-data.at("vector-chunk").R) and #fx(grid-data.at("vector-page").R). The difference was #dci(H("H2 vector-chunk != vector-page")), Holm $p$ = #fx(H("H2 vector-chunk != vector-page").p_holm). H2 is not supported: at this corpus size the two representations rank notes equally well. The two disagreed on only #(pq.overlap_vector_page_vs_vector_chunk.at("only_vector-page") + pq.overlap_vector_page_vs_vector_chunk.at("only_vector-chunk")) questions.

== RQ3: the evidence unit under a budget <sec-rq3>

#figure(
  image("figures/fig05-tokens-recall.svg", width: 100%),
  caption: [Recall at 10 against the tokens in the full text of the ten notes returned (mean, log scale). The 6,000-token budget is marked.],
) <fig-tokens>

The ten notes returned by dense chunk retrieval contain on average #num(grid-data.at("vector-chunk").tokens) tokens, about fifteen times the budget (@fig-tokens), so every pack must select.

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    rule,
    [*Unit*], [*Tokens*], [*Hits skipped*], [*Survival*], [*95% CI*],
    thin,
    ..("chunk", "window", "section", "page").map(u => {
      let r = units.at(u)
      let s = unc.survival.at(u)
      ([#u], [#num(r.tokens_delivered)], [#fx(r.hits_dropped, d: 2)], [#fx(r.relevant_in_pack)], [#fx(s.lo) to #fx(s.hi)])
    }).flatten(),
    rule,
  ),
  caption: [Evidence units packed from the same ranking under a 6,000-token budget. Means per question.],
) <tab-units>

#figure(
  image("figures/fig06-units.svg", width: 100%),
  caption: [Survival of the relevant note, and ranked hits skipped by the budget, per unit.],
) <fig-units>

All four packs spend nearly the whole budget (@tab-units); they differ in how many distinct notes fit. The chunk pack kept the relevant note in #fx(inv.survival.chunk) of questions and the page pack in #fx(inv.survival.page): a difference of #dci(H("H3 chunk pack > page pack (survival)")). Of the discordant questions, #H("H3 chunk pack > page pack (survival)").discordant_a_only favoured the chunk pack and #H("H3 chunk pack > page pack (survival)").discordant_b_only the page pack; exact McNemar Holm $p$ = #fp(H("H3 chunk pack > page pack (survival)").p_holm). H3 is supported. The upper bound for any pack is the share of questions with a relevant note in the top ten, #fx(pq.hit_at_k.at("vector-chunk").at(9))\; the chunk pack attains #fx(inv.survival.chunk / pq.hit_at_k.at("vector-chunk").at(9) * 100, d: 0) percent of it.

*Other budgets.* The same packing at #bs.budgets.map(b => num(b)).join(", ", last: " and ") tokens (@tab-budgets) gives the same order at every budget: chunks keep the relevant note at least as often as any other unit, and whole notes never more often. The gap is largest when the budget is small: at #num(bs.budgets.first()) tokens chunks keep it for #fx(BS(bs.budgets.first(), "chunk").relevant_in_pack) of questions and whole notes for #fx(BS(bs.budgets.first(), "page").relevant_in_pack). At large budgets all units approach the ceiling set by the ten notes walked, and whole notes remain lowest while delivering the most text: at #num(bs.budgets.last()) tokens, #fx(BS(bs.budgets.last(), "page").relevant_in_pack) in #num(BS(bs.budgets.last(), "page").tokens_delivered) tokens against #fx(BS(bs.budgets.last(), "chunk").relevant_in_pack) in #num(BS(bs.budgets.last(), "chunk").tokens_delivered) for chunks. This sweep was run after the hypotheses were tested and is exploratory.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (right, right, right, right, right),
    stroke: none,
    rule,
    [*Budget*], [*Chunk*], [*Window*], [*Section*], [*Whole note*],
    thin,
    ..bs.budgets.map(b => ([#num(b)], ..("chunk", "window", "section", "page").map(u => [#fx(BS(b, u).relevant_in_pack)]))).flatten(),
    rule,
  ),
  caption: [Share of the #bs.questions development questions whose relevant note is in the pack, by token budget and evidence unit (top 10 notes of the dense chunk ranking, units kept whole if they fit). 6,000 is gbrain's default. Source: `bench/budget-sweep.json`.],
) <tab-budgets>

== RQ4: the Hilbert-key filter <sec-rq4>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (center, center, right, right, right, right),
    stroke: none,
    rule,
    [*Level*], [*Ranges*], [*Median cand.*], [*R\@10*], [*nDCG\@10*], [*Range ms/q.*],
    thin,
    ..sweep.cells.map(c => (
      [#c.level], [#c.ranges], [#num(c.median_candidates)],
      [#fx(c.at("R@10"))], [#fx(c.at("nDCG@10"))], [#fx(c.scan_seconds / 817 * 1000, d: 3)],
    )).flatten(),
    thin,
    [—], [—], [#num(sweep.cosine.candidates)], [#fx(sweep.cosine.at("R@10"))], [#fx(sweep.cosine.at("nDCG@10"))], [—],
    rule,
  ),
  caption: [All sixteen probe settings on chunk keys. Last row: exhaustive cosine. Range time is the Postgres range count per question on the live key table.],
) <tab-sweep>

Level 0 covers the whole key space and equals exhaustive search. Levels 2 and 3 return almost no candidates, because $2^16$ and $2^24$ cells hold #num(man.chunk_hilbert_count) chunks. Level 1 is the informative level (@tab-sweep, @fig-sweep). Its best setting, sixteen ranges, read a median of 329 chunks and reached #fx(C("hk1-L1-R16").recall)\; the difference from exhaustive cosine was #dci(H("H4 hk1 L1x16 non-inferior to cosine")), non-inferiority Holm $p$ = #fx(H("H4 hk1 L1x16 non-inferior to cosine").p_holm). H4 is rejected. The paired $t$-test on the shifted differences was added after this run and computed in a separate run of the same script on the corpus as it stood then, #num(h4t.corpus.chunks) chunks against #num(h4t.frozen_run_corpus.chunks) in the main run: the difference was #fx(h4t.H4.diff) (95% CI #fx(h4t.H4.ci_lo) to #fx(h4t.H4.ci_hi)), with $p$ = #fx(h4t.H4.p_wilcoxon_shifted) for the shifted Wilcoxon test and $p$ = #fx(h4t.H4.p_t_shifted) for the paired $t$-test. The conclusion is unchanged.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (right, right, right, right),
    stroke: none,
    rule,
    [*Candidates*], [*Random subset R\@10*], [*hk1 level 1 R\@10*], [*Ratio*],
    thin,
    ..rnd.map(r => {
      let h = sweep.cells.find(c => c.level == 1 and c.median_candidates == r.candidates)
      ([#num(r.candidates)], [#fx(r.at("R@10_mean"))],
       if h == none { [—] } else { [#fx(h.at("R@10"))] },
       if h == none { [—] } else { [#fx(h.at("R@10") / r.at("R@10_mean"), d: 1)] })
    }).flatten(),
    rule,
  ),
  caption: [Exploratory control: uniformly random subsets of keyed chunks of the same size, rescored by cosine, mean of three seeds.],
) <tab-random>

#figure(
  image("figures/fig07-hk1-sweep.svg", width: 100%),
  caption: [Recall at 10 after rescore against candidates per question, for level-1 and level-2 probes, the random control, level 0, and exhaustive cosine.],
) <fig-sweep>

At equal candidate counts the level-1 filter recalls 1.8 to 2.8 times as much as a #index("random subset") (@tab-random): the key carries angular information. Shared prefix length rises with cosine (@fig-neighbour), but a chunk's true nearest neighbour, at median cosine #fx(inv.lsh.nearest_neighbour_pairs.median_cos, d: 2), shares the first eight bits in only #fx(inv.lsh.nearest_neighbour_pairs.observed * 100, d: 0) percent of cases. @sec-mech-hk1 explains why.

*Efficiency per token.* An index is judged here by what it delivers per token, on two cost axes. The first is the tokens processed: the text of the candidate chunks. The second is the tokens delivered: the 6,000-token chunk pack the model reads, built by the rule of @sec-rq3 (@tab-hk1-tokens). With no filter the pack reproduces the survival of #fx(units.chunk.relevant_in_pack) reported there.

Per token delivered, the filter halves efficiency. With sixteen ranges a relevant note survives in #fx(EF("hk1 level 1, 16 ranges").survival) of packs, #fx(EF("hk1 level 1, 16 ranges").survival_per_1k_delivered) per 1,000 delivered tokens, against #fx(EF("cosine, every chunk").survival_per_1k_delivered) for cosine over every chunk. The pack fills to almost the same size either way (#num(EF("hk1 level 1, 16 ranges").tokens_delivered_mean) against #num(EF("cosine, every chunk").tokens_delivered_mean) tokens), so the filter changes which notes fill it, not how much the model reads.

Per token processed, the order reverses. The filter reads about 1/#calc.round(EF("cosine, every chunk").tokens_processed_mean / EF("hk1 level 1, 16 ranges").tokens_processed_mean) of the text and keeps a relevant note at #fx(EF("hk1 level 1, 16 ranges").survival_per_100k_processed) per 100,000 processed tokens, against #fx(EF("cosine, every chunk").survival_per_100k_processed). Against random subsets of the same size for each question, hk1 is #fx(calc.min(..hk-gain), d: 1) to #fx(calc.max(..hk-gain), d: 1) times as efficient on both axes, so the key carries information on either measure.

Which axis applies depends on the reader. When a vector search ranks the candidates and only the pack is read, the delivered axis applies and the filter loses. When something has to read the candidates themselves, the processed axis applies and the filter wins against reading everything. These figures were computed after the hypotheses were tested, on the corpus as it stood on 6 October (#num(eff.chunks) chunks, #num(eff.keyed_chunks) keyed), and are exploratory; the pre-registered factorial study measures the same pack per filter.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    rule,
    [*Condition*], [*Candidates*], [*Survival*], [*Delivered*], [*Per 1k delivered*], [*Processed*], [*Per 100k processed*],
    thin,
    ..eff.conditions.map(c => (
      [#c.condition], [#num(c.mean_candidates)], [#fx(c.survival)], [#num(c.tokens_delivered_mean)],
      [#fx(c.survival_per_1k_delivered)], [#num(c.tokens_processed_mean)], [#fx(c.survival_per_100k_processed)],
    )).flatten(),
    rule,
  ),
  caption: [Efficiency of hk1 level 1 per token. Survival: a relevant note is in the 6,000-token chunk pack. Delivered: tokens in that pack; processed: tokens in the candidate chunks; both are means per question. Random rows use, for each question, as many keyed chunks as the hk1 setting returned, averaged over three seeds. Source: `bench/hk1-efficiency.json`.],
) <tab-hk1-tokens>

#figure(
  image("figures/fig08-neighbour.svg", width: 100%),
  caption: [Left: mean pair cosine against shared prefix bits. Right: shared bits between each chunk and its true nearest neighbour; 214 duplicated chunks share all 64.],
) <fig-neighbour>

== RQ5: engines <sec-rq5>

#figure(
  image("figures/fig09-latency.svg", width: 100%),
  caption: [Median latency of a pgvector top-10 and of one hk1 prefix range. Grey and red bars include starting a client per query.],
) <fig-latency>

#mark("Postgres")
An HNSW top-10 in #index("pgvector") took #fx(live.median_ms, d: 2) ms median (95th percentile #fx(live.p95_ms, d: 2) ms) on the live chunks and #fx(sc.in_session.median_ms, d: 2) ms (#fx(sc.in_session.p95_ms, d: 2) ms) on a scratch set of 200,000 vectors, inside one session (@fig-latency). The 200,000-vector index occupied #fx(sc.in_session.index_bytes / 1e6, d: 0) MB and built in #fx(sc.in_session.index_build_s, d: 1) s. The one hk1 range timed at that scale, #fx(sc.range_scan.hilbert_one_range_ms, d: 0) ms, was a sequential scan because the query used the default collation; it is an upper bound. Latency does not motivate leaving Postgres at this size or at forty times it.

#mark("ArangoDB")
*Choice of comparison engine.* RQ5 asks whether query latency or a graph operation justifies an engine other than Postgres, so the comparison engine had to support both operations the knowledge base performs: nearest-neighbour search over chunk vectors and expansion along the links between notes. ArangoDB stores documents and a graph of edges natively, queries both in one language, and since version 3.12 has a vector index @arangodb. It is also the engine the author already runs on the same machine for a genealogy database, where it was chosen for documents and a graph together, which makes it a realistic alternative rather than a hypothetical one. Three other stores were considered and not measured. SQLite with the sqlite-vec extension @sqlitevec and libSQL, Turso's fork of SQLite with a native vector type and an approximate vector index @libsql, are embedded single-file stores with the same relational model as Postgres: a one-hop expansion is the same SQL join, so they would answer a question about deployment, one file against a server, rather than about the graph operation. gbrain's storage layer supports only Postgres and its WebAssembly build, PGLite @gbrain, so neither could have served the live corpus without a port. MongoDB was set aside because its vector search was taken to require the hosted Atlas service. That premise is out of date: vector search is available for the self-managed Community Edition through a separate search process, mongot, that requires a replica set @mongodbvs. All three were measured afterwards, in throwaway instances, with the same vectors, links and questions (@tab-engines2).

*Engine comparison.* #index("ArangoDB") #al.server.version @arangodb was run in a separate, throwaway container on its own port, with no persistent volume; the genealogy instance on the same machine was never started or contacted. The #num(al.documents) live chunk vectors loaded in #fx(al.load_seconds, d: 1) s, and its vector index built in #fx(al.vector_index.build_seconds, d: 1) s. ArangoDB's vector index is a Faiss inverted-file (IVF) index @arangodb @douze2024 @jegou2011: it partitions the vectors into nLists lists around learned centroids, here #al.vector_index.nLists lists with the cosine metric, and a query searches only the nProbe lists whose centroids are nearest to it. Both engines answered the #num(av.queries) development questions through one persistent client connection, fetching the top #av.k_fetched chunks; quality is the share of the exact top ten chunks found, and note-level recall at 10 after collapsing chunks to notes (@tab-engines, @fig-engines).

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    rule,
    [*Engine and setting*], [*Median ms*], [*p95 ms*], [*Exact top-10 found*], [*R\@10*],
    thin,
    ..av.results.map(r => {
      let name = if r.engine == "arangodb" {
        if r.index.starts-with("none") [ArangoDB, full scan] else [ArangoDB, nProbe #r.nProbe]
      } else {
        if r.index.starts-with("none") [pgvector, full scan] else [pgvector HNSW, ef_search #r.ef_search]
      }
      ([#name], [#fx(r.median_ms, d: 2)], [#fx(r.p95_ms, d: 2)], [#fx(r.chunk_recall_vs_exact_at_10)], [#fx(r.page_recall_at_10)])
    }).flatten(),
    rule,
  ),
  caption: [ArangoDB against pgvector on the same #num(al.documents) vectors and #num(av.queries) questions. Client-observed latency over one persistent connection; ArangoDB is reached over HTTP with JSON, Postgres over its wire protocol. Source: `bench/arango-vector.json`.],
) <tab-engines>

#figure(
  image("figures/fig18-engines.svg", width: 100%),
  caption: [Left: share of the exact top-10 chunks found against median latency, for ArangoDB's vector index at increasing nProbe (lists searched) and pgvector's HNSW at increasing ef_search (size of the candidate list kept during the graph search @pgvector); open markers are full scans. Right: one-hop link expansion with hubs capped, in AQL and SQL.],
) <fig-engines>

At matched quality pgvector was faster. Its HNSW index @malkov2018, whose search effort is set by ef_search, the size of the candidate list kept while walking the graph @pgvector, found #fx(A("pgvector", "ef_search", 50).chunk_recall_vs_exact_at_10) of the exact top-10 chunks at #fx(A("pgvector", "ef_search", 50).median_ms, d: 2) ms median, and #fx(A("pgvector", "ef_search", 400).chunk_recall_vs_exact_at_10, d: 4) at #fx(A("pgvector", "ef_search", 400).median_ms, d: 2) ms. ArangoDB reached #fx(A("arangodb", "nProbe", 16).chunk_recall_vs_exact_at_10) at #fx(A("arangodb", "nProbe", 16).median_ms, d: 2) ms, and exact results only by probing all lists, at #fx(A("arangodb", "nProbe", al.vector_index.nLists).median_ms, d: 2) ms. Full scans took #fx(AX("arangodb").median_ms, d: 1) ms in ArangoDB and #fx(AX("pgvector").median_ms, d: 1) ms in Postgres. Note-level recall is the same #fx(AX("pgvector").page_recall_at_10) in every exact configuration, so neither engine changes what is retrieved, only how fast.

For the graph operation, the #num(ah.edges) links between #num(ah.pages) live notes were loaded as an edge collection, and one-hop neighbours of #ah.starts random linked notes were expanded with hubs of more than #ah.hub_cap neighbours excluded (#ah.hubs_excluded notes). AQL, ArangoDB's query language @arangodb, and SQL returned identical neighbour sets for all #ah.identical_sets starts (mean Jaccard similarity, the size of the intersection over the size of the union @jaccard1912, #fx(ah.neighbour_set_jaccard_mean, d: 2)); the median time was #fx(ah.arangodb_aql.median_ms, d: 2) ms in AQL and #fx(ah.postgres_sql.median_ms, d: 2) ms in SQL. The throwaway database was dropped and the container deleted after the run. A repeat after upgrading to PostgreSQL 18.6, pgvector 0.8.7 and ArangoDB 3.12.12 kept the same ordering (@app-pg18).

#let e12 = json("bench/engines-d12.json")
#let E12(sec) = e12.at(sec)
#let erow(name, r) = ([#name], [#fx(r.median_ms, d: 2)], [#fx(r.p95_ms, d: 2)], [#fx(r.chunk_recall_vs_exact_at_10)], [#fx(r.page_recall_at_10)])
*Three more engines.* SQLite #E12("sqlite_vec").version.sqlite with sqlite-vec #E12("sqlite_vec").version.sqlite_vec @sqlitevec, libSQL @libsql (SQLite #E12("libsql").version.libsql_sqlite) with its approximate vector index, and MongoDB Community #E12("mongodb").version.mongodb with mongot @mongodbvs were each loaded with the live chunk vectors and links and asked the same #num(e12.queries) questions, with pgvector re-measured in the same session (@tab-engines2). SQLite and libSQL run inside the client process, so their latency contains no network round trip and is not comparable with the server engines; MongoDB, a single-node replica set in a container beside its mongot, is. Every engine returned the same note-level recall within a thousandth, and every one returned the true one-hop neighbourhood for all #E12("pgvector").hop.starts starts. MongoDB's vector index answered in #fx(E12("mongodb").vector.at(0).median_ms, d: 1) to #fx(E12("mongodb").vector.at(2).median_ms, d: 1) ms against #fx(E12("pgvector").vector.at(0).median_ms, d: 1) ms for pgvector's HNSW, and its one-hop expansion took #fx(E12("mongodb").hop.median_ms, d: 2) ms against #fx(E12("pgvector").hop.median_ms, d: 2) ms in SQL. The containers were removed after the run; the genealogy instance was not touched.

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    rule,
    [*Engine and setting*], [*Median ms*], [*p95 ms*], [*Exact top-10 found*], [*R\@10*],
    thin,
    ..erow([pgvector HNSW, ef_search 100], E12("pgvector").vector.at(0)),
    ..erow([pgvector, full scan], E12("pgvector").vector.at(1)),
    ..erow([SQLite + sqlite-vec, exact scan (in process)], E12("sqlite_vec").vector.at(0)),
    ..erow([libSQL vector index (in process)], E12("libsql").vector.at(0)),
    ..erow([libSQL, full scan (in process)], E12("libsql").vector.at(1)),
    ..E12("mongodb").vector.filter(r => r.at("numCandidates", default: none) != none).map(r => erow([MongoDB vectorSearch, numCandidates #r.numCandidates], r)).flatten(),
    ..erow([MongoDB vectorSearch, exact], E12("mongodb").vector.at(3)),
    rule,
  ),
  caption: [The remaining engines on the same live vectors and #num(e12.queries) questions, with pgvector re-measured in the same session. One-hop expansion, median ms: SQL in Postgres #fx(E12("pgvector").hop.median_ms, d: 2), SQLite #fx(E12("sqlite_vec").hop.median_ms, d: 3), libSQL #fx(E12("libsql").hop.median_ms, d: 3), MongoDB #fx(E12("mongodb").hop.median_ms, d: 2)\; identical neighbour sets throughout. Source: `bench/engines-d12.json`.],
) <tab-engines2>

On these measurements a second engine adds no capability Postgres lacks at this scale, and is slower at both operations tested.

== Summary of hypothesis tests <sec-tests>

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, right, right, left),
    stroke: none,
    rule,
    [*Id*], [*Comparison*], [*Difference (95% CI)*], [*Holm p*], [*Outcome*],
    thin,
    ..inv.confirmatory.pairs().map(((k, v)) => {
      let id = k.split(" ").at(0)
      let ok = v.p_holm < 0.05
      ([#id], [#LABELS.at(id)], [#fx(v.diff) (#fx(v.ci_lo) to #fx(v.ci_hi))], [#fp(v.p_holm)],
       [#if ok [supported] else [not supported]])
    }).flatten(),
    thin,
    ..inv.exploratory.pairs().map(((k, v)) => {
      let id = k.split(" ").at(0)
      ([#id], [#LABELS.at(id)], [#fx(v.diff) (#fx(v.ci_lo) to #fx(v.ci_hi))], [#fp(v.p_holm)],
       [#if v.p_holm < 0.05 [differs] else [no difference]])
    }).flatten(),
    rule,
  ),
  caption: [All tests. H rows are confirmatory, E rows exploratory; each family is Holm-corrected separately. Differences are in recall at 10, or in survival share for H3 and E3.],
) <tab-tests>

#figure(
  image("figures/fig13-effects.svg", width: 100%),
  caption: [Paired differences with 95 percent bootstrap intervals. Filled markers: Holm-adjusted $p < 0.05$.],
) <fig-effects>

== Generalisation and stability <sec-heldout>

#figure(
  image("figures/fig11-heldout.svg", width: 100%),
  caption: [Dense chunk retrieval on development and #index("held-out set")s, with gbrain's production hybrid on the held-out set.],
) <fig-heldout>

On the held-out questions the dense chunk cell reached recall at 10 of #fx(ho.at("R@10")), MRR #fx(ho.MRR), and nDCG at 10 of #fx(ho.at("nDCG@10")), within the development interval's upper range (@fig-heldout). gbrain's production hybrid, measured on the same held-out set by gbrain's own evaluator, reached 0.65, 0.52, and 0.53. Re-embedding twenty questions five times changed no vector component by more than #fx(unc.reembedded_spread.max_abs_component_drift_vs_saved_vectors, d: 5) and changed no ranking; sampling over questions, captured by the bootstrap intervals, is the relevant uncertainty.

// ---------------------------------------------------------------- 6

= Mechanism analyses <sec-mechanisms>

== Why lexical retrieval underperforms <sec-mech-lex>

`websearch_to_tsquery` joins every term with a conjunction, and the median question has #fx(inv.dataset.question_words.median, d: 0) words. Replacing conjunctions by disjunctions (E1) raised whole-note lexical recall by #dci(E("E1 keyword OR > keyword AND (page)")), and chunk-level lexical recall from #fx(C("keyword-and-chunk").recall) to #fx(C("keyword-or-chunk").recall) (@fig-keyword). The best lexical arm still trailed dense retrieval by #fx(-E("E4 keyword OR chunk vs vector-chunk").diff) (E4).

#figure(
  image("figures/fig12-keyword.svg", width: 100%),
  caption: [Lexical variants, reciprocal-rank-fusion hybrids, and dense chunk retrieval. Source: `bench/investigation.json`.],
) <fig-keyword>

Fusing a lexical arm with dense retrieval did not help. With the whole-note lexical arm, fusion lowered recall by #fx(-E("E2 hybrid RRF vs vector-chunk").diff) (E2). With the stronger chunk-level arm, recall was unchanged, #dci(E("E5 hybrid RRF (chunk keyword) vs vector-chunk")), while MRR fell from #fx(C("vector-chunk").mrr) to #fx(C("hybrid-rrf-chunk").mrr). On this corpus the lexical arm adds no notes that dense retrieval misses often enough to offset the rank noise it introduces. This is consistent with the production hybrid scoring below plain dense retrieval on held-out questions.

== Why the Hilbert key is a weak filter <sec-mech-hk1>

The Proposition of @sec-key predicts the level-1 collision rate from the pair angle alone. Tested on 200,000 random chunk pairs, the leading-eight-bit equality and the eight-sign equality coincided on every pair, confirming that a level-1 cell is a sign hash. The magnitude, however, deviated: on raw vectors the observed collision rate was #fx(cen.variants.raw.pair_cell_obs, d: 4) against a predicted #fx(cen.variants.raw.pair_cell_pred, d: 4), and the deviation grew at low cosine (@fig-lsh, left).

*Anisotropy.* The mean of the unit chunk vectors has norm #fx(cen.mean_unit_vector_norm_chunks): the embeddings occupy a narrow cone @ethayarajh2019. Each projection then carries a large common offset, and the eight axes are unbalanced, with between #fx(calc.min(..cen.variants.raw.axis_top_share) * 100, d: 0) and #fx(calc.max(..cen.variants.raw.axis_top_share) * 100, d: 0) percent of chunks on the positive side. Per-axis sign agreement between two chunks, #fx(cen.variants.raw.pair_axis_agree), is then no higher than for two unrelated chunks with the same marginals, although the random-hyperplane model expects #fx(cen.variants.raw.pair_axis_pred). Cell occupancy is correspondingly skewed: the largest of the 256 cells holds #cen.variants.raw.occupancy_max chunks against a uniform #fx(man.chunk_hilbert_count / 256, d: 1) (@fig-occupancy).

#figure(
  image("figures/fig14-lsh-model.svg", width: 100%),
  caption: [Share of pairs in the same level-1 cell against pair cosine, observed and predicted by @eq-lsh, for raw and mean-centred vectors. Stars and crosses: question-to-answer pairs at their median cosine.],
) <fig-lsh>

#figure(
  image("figures/fig15-occupancy.svg", width: 100%),
  caption: [Chunks per level-1 cell, sorted, for raw and mean-centred vectors.],
) <fig-occupancy>

*Intervention.* Subtracting the corpus mean before keying, as in @mu2018, balanced every axis (#fx(calc.min(..cen.variants.at("centred-corpus-mean").axis_top_share) * 100, d: 0) to #fx(calc.max(..cen.variants.at("centred-corpus-mean").axis_top_share) * 100, d: 0) percent), filled all 256 cells, and brought the model into agreement with observation: per-axis agreement #fx(cen.variants.at("centred-corpus-mean").pair_axis_agree) against #fx(cen.variants.at("centred-corpus-mean").pair_axis_pred) predicted, cell collisions #fx(cen.variants.at("centred-corpus-mean").pair_cell_obs, d: 4) against #fx(cen.variants.at("centred-corpus-mean").pair_cell_pred, d: 4) (@fig-lsh, right). For question-to-answer pairs, with median centred cosine #fx(lqa.centred.cos_quartiles.at(1), d: 2), the observed same-cell rate was #fx(lqa.centred.same_cell_observed, d: 4) and the predicted #fx(lqa.centred.same_cell_predicted, d: 4).

*Consequence.* Centring corrected the geometry but did not rescue the filter: at equal candidate counts its recall curve lies only slightly above the raw curve (@fig-centring). The limit is the angle between a question and its answer. At the median centred cosine, @eq-lsh gives a collision probability of #fx(lqa.centred.p_at_median_cos, d: 3), and @eq-tables then requires about #calc.round(lqa.centred.at("independent_8bit_tables_for_0.9")) independent eight-bit keys to find the answer in 90 percent of questions. One Hilbert key is one table. Probes across neighbouring cells recover part of the loss, which is why sixteen ranges reach #fx(C("hk1-L1-R16").recall), but a single locality-preserving key cannot approach exhaustive recall on questions whose answers are this far away in angle. The bound is for random axes. Axes fitted to the corpus raise the collision rate several times over, yet one key still stays well below exhaustive recall (@sec-followup).

#let hk2s = json("bench/hk2-sweep.json")
#let HK(L, p) = hk2s.rows.find(r => r.tables == L and r.probes == p)
#let hk-below = hk2s.rows.filter(r => r.probes == 1 and r.pair_collision_observed < 0.9).last().tables
#let hk-above = hk2s.rows.find(r => r.probes == 1 and r.pair_collision_observed >= 0.9).tables
*Many keys (exploratory).* To test @eq-tables directly, zig-hilbert 0.3.0 added a second key type, `hk2`, that stores many independent eight-bit sign keys, one per table, with optional query-directed probes @lv2007, and keeps `hk1` unchanged. On the centred vectors of the #hk2s.pairs question-to-answer pairs, the share of pairs that shared a bucket in at least one of $L$ tables followed the model at every table count: #fx(HK(16, 1).pair_collision_observed) observed against #fx(HK(16, 1).pair_collision_predicted) predicted at 16 tables, #fx(HK(71, 1).pair_collision_observed) against #fx(HK(71, 1).pair_collision_predicted) at 71, and #fx(HK(128, 1).pair_collision_observed) against #fx(HK(128, 1).pair_collision_predicted) at 128. The figure of about #calc.round(lqa.centred.at("independent_8bit_tables_for_0.9")) tables was optimistic because it used the median angle; averaged over each pair's own angle, 90 percent is reached between #hk-below and #hk-above tables. The cost is the candidate set: at 128 tables a question shares a bucket with #fx(HK(128, 1).candidates_mean_share * 100, d: 0) percent of all chunks, and with four probes per table, 32 tables recover the exhaustive recall at 10 (#fx(HK(32, 4).at("R@10_after_rescoring")) against #fx(hk2s.at("exhaustive_R@10"))) only while scanning #fx(HK(32, 4).candidates_mean_share * 100, d: 0) percent of them. The theory holds; at this angle between questions and answers, hashing saves little over reading every vector.

#let rbq = json("bench/rabitq-sweep.json")
#let RB(n, b: 1) = rbq.rows.find(r => r.candidates == n and r.code_bits_per_dim == b)
#let pqs = json("bench/pq-sweep.json")
#let PQ(c, n) = pqs.rows.find(r => r.config == c and r.candidates == n)
*A filter that keeps the angle (exploratory).* Sign hashes discard how far each projection is from zero. RaBitQ @gao2024rabitq keeps one bit per dimension of a randomly rotated vector, as a hash does, but uses all of them in an unbiased estimate of the inner product with a known error bound. With zig-rabitq @zigrabitq, written for this check, ranking every chunk by the estimate from #(rbq.code_bits)-bit codes and a #(rbq.query_bits)-bit query, and rescoring only the top #RB(100).candidates by exact cosine (#fx(RB(100).share_of_chunks * 100, d: 1) percent of the corpus), gave recall at 10 of #fx(RB(100).at("R@10_after_rescoring")), equal to exhaustive cosine (#fx(rbq.at("exhaustive_R@10"))); the top #RB(25).candidates alone gave #fx(RB(25).at("R@10_after_rescoring")). Extended RaBitQ codes @gao2025rabitq, with several bits per dimension, shorten the list: at four bits the top 25 estimates contained every exact top-10 chunk (#fx(RB(25, b: 4).exact_top10_chunks_found) against #fx(RB(25).exact_top10_chunks_found) at one bit). Product quantization @jegou2011 at the same 128 bytes per vector, with zig-pq @zigpq checked against Faiss @douze2024, performed alike (#fx(PQ("m=128", 25).exact_top10_chunks_found) of the exact top-10 chunks among the top 25, #fx(PQ("m=128", 100).exact_top10_chunks_found) among the top 100); at 64 bytes it needed about twice as many candidates, and an inverted file probing 8 of 64 lists capped at #fx(PQ("m=128,nlist=64,nprobe=8", 1600).exact_top10_chunks_found), because the true neighbours of some questions sit in lists it does not visit. The estimate still reads one code per chunk, so at this size it saves arithmetic and memory rather than scans; but as a compact key it does what the Hilbert key was meant to do, cutting the exact rescoring to about two percent of the chunks without losing recall.

#figure(
  image("figures/fig16-centring-filter.svg", width: 100%),
  caption: [Level-1 filter recall against candidates for raw and mean-centred keys, with the random control.],
) <fig-centring>

== Where dense retrieval fails <sec-failures>

Dense chunk retrieval placed no relevant note in the top ten for #inv.failures.misses questions (@fig-failures). In #inv.failures.relevant_not_in_corpus the relevant note no longer existed in the live corpus, and in #inv.failures.duplicate_in_top10 a duplicate copy of the relevant note under a second slug was retrieved instead; these are label errors, not retrieval errors. Of the rest, #(inv.failures.rank_11_20 + inv.failures.rank_21_50) had the relevant note within the top 50, within reach of a reranker or a larger budget, and #(inv.failures.rank_51_100 + inv.failures.rank_over_100) ranked it below 50, where no reordering of the top of the list helps.

#figure(
  image("figures/fig17-failures.svg", width: 100%),
  caption: [Failure taxonomy of dense chunk retrieval on development questions with no relevant note in the top 10.],
) <fig-failures>

// ---------------------------------------------------------------- 7

= Exploratory study: twelve ideas from the literature <sec-ideas>

The confirmatory study left two measured weak points: 190 questions with no relevant note in the dense top ten, of which most have it within the top 50 (@sec-failures), and the evidence pack, which fits only about ten notes into 6,000 tokens (@sec-rq3). A survey of the cited preprints and related recent work produced twelve ideas aimed at these points or at the Hilbert key. This section tests them as a separate exploratory family.

== Design

Each idea was implemented on the same 817 development questions; the sealed held-out set was not used. A pre-registration fixing every hypothesis, metric, test, threshold and tunable value was committed before any idea ran (`bench/ideas/preregistration.md`). The development questions were split once into a tuning half and a confirmation half; any choice among variants was made on the tuning half and tested on the confirmation half. The #ideas.m hypotheses of the family were Holm-corrected together, with hypotheses that could not be run entering at $p = 1$. An idea was accepted only if its point estimate met its pre-stated threshold, its family-wide Holm $p$ was below 0.05, and any guard check passed. One deviation was recorded before the affected ideas ran: the non-inferiority tests (ideas 6, 7 and 11) use the larger $p$ of the shifted Wilcoxon test and a paired $t$-test, for the reason given in @sec-stats.

#let IDEA = (
  H1a: ([Rerank dense top 50, default instruction], [@zhang2025 @nogueira2019]),
  H1b: ([Rerank dense top 50, corpus instruction], [@zhang2025]),
  H2: ([Sentence-pruned packing of the top 30 (survival)], [@chirkova2025]),
  H3: ([Corpus-specific query instruction], [@zhang2025]),
  H4a: ([Subtract the shared mean], [@ren2025 @mu2018]),
  H4b: ([Subtract separate query and chunk means], [@ren2025]),
  H4c: ([Project off the shared mean direction], [@ren2025]),
  H4d: ([Project off separate directions], [@ren2025]),
  H5: ([Rocchio feedback on vectors], [@li2022prf]),
  H6: ([Hilbert partitions with SOAR assignment, vs exhaustive], [@alqurishi2025 @sun2023]),
  H7: ([Many randomised Hilbert orders, vs exhaustive], [@imamura2025]),
  H8: ([Per-chunk synopses from a local model], [@anthropic2024]),
  H9: ([Link-graph personalised PageRank], [@gutierrez2024]),
  H10: ([HyDE draft answer fused with the question], [@gao2022 @wang2023]),
  H11: ([Adaptive cut-off at the largest score gap (survival)], [@taguchi2025]),
  H12: ([MUVERA encodings over hk1 buckets, notes with 5+ chunks], [@jayaram2024]),
)

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, left, right, right, right, left),
    stroke: none,
    rule,
    [*Id*], [*Idea and source*], [*Effect (95% CI)*], [*Threshold*], [*Holm p*], [*Outcome*],
    thin,
    ..ideas.hypotheses.map(h => {
      let (name, src) = IDEA.at(h.id)
      let eff = if h.at("diff", default: none) == none [—] else [#fx(h.diff) (#fx(h.ci95.at(0)) to #fx(h.ci95.at(1)))]
      ([#h.id], [#name #src], eff, [#fx(h.threshold, d: 2)], [#fp(h.p_holm_family)], [#h.status])
    }).flatten(),
    rule,
  ),
  caption: [The exploratory family. Effects are paired differences in recall at 10 against dense chunk retrieval unless marked; for ideas 6 and 7 the comparator is exhaustive cosine and the threshold is a non-inferiority margin. Effects for ideas 2, 3, 5, 6, 7, 9 and 10 are on the confirmation half (408 questions), idea 12 on its subset of 314 questions, the rest on all 817. Source: `bench/ideas/summary.json`.],
) <tab-ideas>

#figure(
  image("figures/fig19-ideas.svg", width: 100%),
  caption: [Each idea's effect against its own pre-stated threshold (red tick). Idea 2 exceeds its threshold but failed its guard check, so it is not accepted.],
) <fig-ideas>

== Results

#let r1 = i1.secondary.default
*Accepted: reranking the dense top 50.* Reordering the dense top 50 notes with the cross-encoder reranker and its default instruction raised recall at 10 by #fx(I("H1a").diff) (95% CI #fx(I("H1a").ci95.at(0)) to #fx(I("H1a").ci95.at(1)), family Holm $p$ = #fx(I("H1a").p_holm_family)). Of the #r1.near_miss_pool_rank_11_50 questions whose relevant note sat at ranks 11 to 50, #r1.near_misses_moved_into_top10 moved into the top ten, while #r1.top10_hits_lost earlier top-ten hits dropped out. The cost is latency: #fx(r1.latency_seconds.median, d: 1) s median and #fx(r1.latency_seconds.p95, d: 1) s at the 95th percentile per question. The corpus-specific instruction did worse than the default (#fx(I("H1b").diff), not significant). Appendix A had found no gain from the same reranker; there it reordered the weaker hybrid list, not the dense one.

*Close but not accepted.* Sentence-level pruning of the top 30 notes kept the relevant note in the pack far more often than chunk packing (+#fx(I("H2").diff) survival) with fewer tokens. Survival, however, rises almost by construction when more notes fit, so the pre-registered guard asked a blind judge, the local 27B model, whether the packed text answers the question on #i2.judge_guard.n questions. It did so for #fx(i2.judge_guard.yes_pruned * 100, d: 0) percent of pruned packs against #fx(i2.judge_guard.yes_chunk * 100, d: 0) percent of chunk packs, a loss beyond the two-point limit, so the idea is not accepted. Link-graph PageRank passed its gate (#fx(i9d.share_within_2_links * 100, d: 0) percent of the deep misses lie within two links of a top-ten note) and gained #fx(I("H9").diff) in recall at 10, below its +0.02 threshold. HyDE gained #fx(I("H10").diff) and added #fx(i10.latency_seconds.added_median, d: 0) s per question for the draft answer.

#let km = i6.curves.at("kmeans/M128/C20")
*Null and negative results.* A corpus-specific query instruction, all four variants of mean-direction removal, and Rocchio feedback changed recall by less than 0.02 in either direction. Hilbert-ordered partitions with a SOAR-style second assignment closed most of the gap left by the hk1 filter (@sec-rq4): at #fx(i6.primary.at("H6 Hilbert partitions non-inferior to exhaustive").median_candidates_confirm / i6.keyed_chunks * 100, d: 0) percent of chunks scanned they ended #fx(I("H6").diff) behind exhaustive cosine, against #fx(H("H4 hk1 L1x16 non-inferior to cosine").diff) for the hk1 probes, but missed the 0.03 margin, and ordinary k-means partitions at a similar scan (#num(km.median_candidates.confirm) candidates) reached #fx(km.metrics.confirm.at("R@10")) against #fx(i6.primary.at("H6 Hilbert partitions non-inferior to exhaustive").mean_comparator) for exhaustive search. Many randomised Hilbert orders did worse (#fx(I("H7").diff)). The adaptive cut-off removed most tokens but lost the relevant note for many questions (#fx(I("H11").diff) survival). MUVERA encodings over hk1 buckets fell far behind best-chunk scoring (#fx(I("H12").diff)): with single-vector queries the encoding reduces to scoring one bucket. Per-chunk synopses were not run: one synopsis took #fx(i8.timing.seconds_per_chunk_mean, d: 1) s on the local model, projecting to #fx(i8.timing.projected_hours, d: 0) h for the corpus, beyond the 10-hour budget. Late chunking @gunther2024 was not applicable, because Qwen3-Embedding pools only the last token.

== Implementation plans

Each plan below is stated as work to be done and verified, not as a decision taken.

- *Reranking (accepted).* Feed gbrain's reranker from the dense chunk ranking at depth 50 with the default instruction. Measure depth 30, a shorter document cap and score caching against the #fx(r1.latency_seconds.median, d: 1) s cost, confirm the configuration once on the held-out set, then measure it end to end in the agent harness.
- *Link-graph PageRank (close).* Pre-register a one-hop, hub-down-weighted variant stacked on the reranker, implemented as a gbrain post-processor over the existing link table.
- *Sentence pruning (close).* Pre-register a hybrid pack: full chunks for the top ten, pruned sentences for ranks 11 to 30, with a sentence cap that uses the whole budget (pruned packs used #fx(i2.primary.at("H2 pruned pack survival").tokens_pruned_mean_confirm / 6000 * 100, d: 0) percent of it). Validate the judge against human labels first.
- *Partitioned index (descriptive).* Not needed at five thousand chunks. If the corpus grows by an order of magnitude, test pgvector's IVF index against k-means partitions before adding a partitions command to zig-hilbert.

// ---------------------------------------------------------------- 8

#let es = json("bench/efficiency/summary.json")
#let EM(k) = json("bench/efficiency/" + lower(k) + ".json")
#let ecx = json("bench/efficiency/crosscheck.json")
#let ecc = ecx.results
#let m9 = EM("M9")
#let ebase = EM("M3").baseline.confirm
#let setting(s) = s.replace("lambda", "λ").replace("alpha", "α").replace("beta", "β").replace("gamma", "γ").replace("=", " = ").replace(",", ", ")

= Exploratory study: relevant notes per token <sec-efficiency>

The evidence pack is what the model reads, so the question here is narrower than retrieval: at the same number of tokens, can a different choice of units put a relevant note into the pack more often? Eight methods from optimisation, diversity, graph diffusion and conformal prediction were tested as a second exploratory family.

== Design

The pre-registration (`bench/efficiency/preregistration.md`) fixed every hypothesis, grid, test and threshold before any data were computed, and the implementation details it left open were committed separately before the first result (`bench/efficiency/deviations.md`). The outcome is survival, whether at least one relevant note is in the pack, at budgets of 1,000, 2,000, 3,000, 4,000 and 6,000 tokens; the primary outcome per question, $overline(S)$, is its mean over the five budgets. A ratio such as relevant notes per token is not used, because a pack of one chunk would maximise it. The baseline is the chunk pack of @sec-rq3: cosine ranking, the top ten notes walked in rank order, each winning chunk kept whole if it fits. Methods that select or reorder units worked on the winning chunks of the top 50 notes; methods that re-rank scored all notes and packed with the baseline rule. Every parameter was chosen on the tuning half of the development questions and tested once on the confirmation half, with the split of @sec-ideas. The primary test is the one-sided Wilcoxon signed-rank test on $overline(S)$, Holm-corrected over the eight ideas; an idea is accepted if its gain is at least +0.02 and its Holm $p$ is below 0.05.

The methods were implemented in Python as pre-registered. As a check on the implementation, the chosen setting of each of M1 to M7 was recomputed on #ecx.questions.len() confirmation questions with two independent libraries written for this study, zig-select @zigselect and zig-diffuse @zigdiffuse: all selections and orders were equal, and diffused scores agreed to within #fp(float(ecc.M6.max_abs_diff_F)) (the conjugate-gradient tolerance of manifold ranking; the other differences were at rounding level).

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto),
    align: (left, left, left, right, right),
    stroke: none,
    rule,
    [*Id*], [*Method and source*], [*Chosen*], [*$Delta overline(S)$ (95% CI)*], [*Holm p*],
    thin,
    [M1], [Calibrated relevance per token, greedy fill @dantzig1957 @zadrozny2002], [#setting(es.family.M1.chosen)], [#fx(es.family.M1.diff) (#fx(es.family.M1.ci95.at(0)) to #fx(es.family.M1.ci95.at(1)))], [#fp(es.family.M1.p_holm)],
    [M2], [Budgeted submodular coverage @lin2011], [#setting(es.family.M2.chosen)], [#fx(es.family.M2.diff) (#fx(es.family.M2.ci95.at(0)) to #fx(es.family.M2.ci95.at(1)))], [#fp(es.family.M2.p_holm)],
    [M3], [Maximal marginal relevance @carbonell1998], [#setting(es.family.M3.chosen)], [#fx(es.family.M3.diff) (#fx(es.family.M3.ci95.at(0)) to #fx(es.family.M3.ci95.at(1)))], [#fp(es.family.M3.p_holm)],
    [M4], [Determinantal point process, greedy MAP @kulesza2012 @chen2018dpp], [#setting(es.family.M4.chosen)], [#fx(es.family.M4.diff) (#fx(es.family.M4.ci95.at(0)) to #fx(es.family.M4.ci95.at(1)))], [#fp(es.family.M4.p_holm)],
    [M5], [Heat diffusion on the link graph @kondor2002 @chung2007], [#setting(es.family.M5.chosen)], [#fx(es.family.M5.diff) (#fx(es.family.M5.ci95.at(0)) to #fx(es.family.M5.ci95.at(1)))], [#fp(es.family.M5.p_holm)],
    [M6], [Manifold ranking on a neighbour graph @zhou2004], [#setting(es.family.M6.chosen)], [#fx(es.family.M6.diff) (#fx(es.family.M6.ci95.at(0)) to #fx(es.family.M6.ci95.at(1)))], [#fp(es.family.M6.p_holm)],
    [M7], [Advection–diffusion on directed links @chapman2011], [#setting(es.family.M7.chosen)], [#fx(es.family.M7.diff) (#fx(es.family.M7.ci95.at(0)) to #fx(es.family.M7.ci95.at(1)))], [#fp(es.family.M7.p_holm)],
    rule,
  ),
  caption: [The per-token family, M1 to M7. $Delta overline(S)$ is the paired difference in mean survival over the five budgets against the chunk pack, on the confirmation half (#m9.per_question_confirm.idx.len() questions; the baseline's $overline(S)$ there is #fx(ebase.S_bar)). None is accepted. M9 is reported in the text. Source: `bench/efficiency/summary.json`.],
) <tab-efficiency>

== Results

*No method is accepted.* Every method of M1 to M7 lowered mean survival on the confirmation half. The two diversity methods came closest and are within noise of the baseline: maximal marginal relevance (#fx(es.family.M3.diff)) and the determinantal point process (#fx(es.family.M4.diff)). Their tuned settings were the grid values that give the least weight to diversity, and the three diffusion methods likewise chose the smallest diffusion they were offered. The closer each method stayed to the plain cosine order, the less it lost. Heat diffusion and manifold ranking left recall at 10 almost unchanged (#fx(EM("M5").at("R@10").confirm) and #fx(EM("M6").at("R@10").confirm) against #fx(EM("M5").at("R@10").baseline_confirm)) but still moved relevant notes out of the packed top ten. Submodular coverage lost most (#fx(es.family.M2.diff)): rewarding coverage per token fills the budget with short chunks that resemble many others, and at 6,000 tokens its packs held #fx(EM("M2").grid.at(es.family.M2.chosen).confirm.relevant_notes_6000, d: 2) relevant notes per question against #fx(ebase.relevant_notes_6000, d: 2) for the baseline. Ranking by calibrated relevance per token (M1) favours short chunks for the same reason, which costs most at the smallest budget: survival at 1,000 tokens fell from #fx(ebase.survival.at("1000")) to #fx(EM("M1").grid.at(es.family.M1.chosen).confirm.survival.at("1000")).

*Conformal stopping saves tokens but misses its threshold.* M9 keeps the baseline order and stops the 6,000-token pack at the first unit whose cosine is below a threshold calibrated on the tuning half by conformal risk control @angelopoulos2024, with a target miss rate of #fx(m9.alpha_target) (the tuning half's baseline miss rate plus 0.02). The calibrated threshold was #fx(m9.lambda_hat). On the confirmation half the pack fell from #num(m9.tokens.baseline_confirm) to #num(m9.tokens.confirm) tokens per question, #fx(m9.primary.relative_fall * 100, d: 1) percent ($p$ #fp(m9.primary.p_holm) after Holm), and survival stayed non-inferior at the 0.02 margin (#fx(m9.survival.confirm) against #fx(m9.survival.baseline_confirm), $p$ = #fx(m9.non_inferiority.p)). The realised miss rate, #fx(m9.realised_miss.confirm), stayed below the target, consistent with the method's guarantee, which bounds the expected miss rate on new questions. The pre-registered criterion was a 20 percent saving, so M9 is not accepted.

*What this rules out (efficiency).* On this corpus the pack does not lose relevant notes because it repeats itself or because related notes are ranked too low; the plain cosine order is already the best order tested. The remaining losses are questions whose relevant note is ranked below the top ten (@sec-failures), which the reranker of @sec-ideas addresses. A calibrated stop is the only method that changed cost without changing survival; whether saving #fx(m9.primary.relative_fall * 100, d: 0) percent of the tokens is worth a threshold to maintain is a question of cost, not of retrieval, and would need its own pre-registered test at the budget gbrain uses.

// ---------------------------------------------------------------- 9

#let fsh = json("bench/factorial/level1-shapley.json").points
#let fcells = json("bench/factorial/level1-cells.json")
#let flb = json("bench/factorial/level1-leaderboard.json")
#let fho = json("bench/factorial/heldout-level1.json")
#let fl2 = json("bench/factorial/level2-summary.json")
#let fjp = json("bench/factorial/judge-packs.json")
#let FJN = fjp.at("non_inferiority_margin_0.02")
#let FJM = fjp.mcnemar_two_sided
#let FHR = fho.paired_vs_reference.best_recall
#let FHS = fho.paired_vs_reference.best_survival
#let FCTX = fl2.marginals.at("num_ctx (local)")
#let snum(x) = if x < 0 [−#num(-x)] else [#num(x)]
#let FACTOR = (
  ("F1=lexical", [Keyword search instead of dense]), ("F1=hybrid", [Hybrid of keyword and dense (RRF)]),
  ("F2=mean", [Note mean instead of best chunk]), ("F3=on", [Reranker on the top 50]),
  ("F4=on", [Link-graph PageRank]), ("F5=on", [Mean-direction removal]), ("F6=on", [Rocchio feedback]),
  ("F7=hk1", [hk1 candidate filter]), ("F7=partitions", [Hilbert partitions filter]),
  ("F8=window", [Unit: chunk with neighbours]), ("F8=section", [Unit: section]), ("F8=page", [Unit: whole note]),
  ("F8=pruned", [Unit: pruned sentences of the top 30]),
)

= Factorial study: every combination of the pipeline <sec-factorial>

The previous sections changed one switch at a time. This section crosses eight switches of the pipeline in a full factorial, so that each one's contribution can be read with every combination of the others, and then checks the two best combinations once on the held-out questions.

== Design

The pre-registration (`bench/factorial/preregistration.md`) crosses first-stage search (dense, keyword, or their reciprocal-rank fusion), note scoring, the reranker of @sec-ideas, link PageRank, mean removal, Rocchio feedback, a candidate filter, and the evidence unit, all with the parameters of @sec-ideas and none re-tuned. The valid combinations are #num(fcells.n_cells) cells, #num(fcells.n_cells / 5) rankings times five units, each evaluated on all #fcells.n_questions_analysis_set development questions. A factor's effect is summarised by its Shapley value: its average contribution to a metric over all combinations of the other factors, with a bootstrap interval over questions. Latency is composed from component timings measured once on this machine. The best configuration on each axis is chosen on the tuning half; the held-out step (deviation D8) runs the best-recall and the best-survival configurations once on the held-out questions, paired with the reference configuration of @sec-rq1, dense chunk retrieval with a chunk pack.

The study ran over two nights and recorded eleven deviations (`bench/factorial/deviations.md`). Three matter for reading it. The first night ended in a kernel panic from memory exhaustion, traced to gbrain's two MLX model servers, which kept up to 24 GB each of freed GPU memory; a 2 GB cache limit fixed it. The second level, an agent harness, ran only its local stratum: the cloud runs were dropped by the author, and runs with a 128K context window did not fit in this machine's 32 GB. And the level-2 judge ran on a work account by mistake, so its scores are not used here.

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: none,
    rule,
    [*Factor (against its reference level)*], [*R\@10*], [*Survival*], [*Tokens*], [*Latency (ms)*],
    thin,
    ..FACTOR.map(((k, name)) => {
      let v = fsh.at(k)
      ([#name], [#fx(v.at("R@10").shapley)], [#fx(v.survival.shapley)], [#snum(v.tokens.shapley)], [#snum(v.latency.shapley)])
    }).flatten(),
    rule,
  ),
  caption: [Shapley values of the factorial's factors on the #flb.n_questions development questions. Evidence units do not change the ranking, so their recall value is zero. Source: `bench/factorial/level1-shapley.json`.],
) <tab-factorial>

== Results

*What carries recall and survival.* The reranker contributes most to recall (#fx(fsh.at("F3=on").at("R@10").shapley)) at a cost of #fx(fsh.at("F3=on").latency.shapley / 1000, d: 1) s per question; link PageRank adds #fx(fsh.at("F4=on").at("R@10").shapley) for #num(fsh.at("F4=on").latency.shapley) ms, and hybrid search #fx(fsh.at("F1=hybrid").at("R@10").shapley). Mean removal, Rocchio feedback and note-mean scoring are within a few thousandths of zero. The hk1 filter costs #fx(fsh.at("F7=hk1").at("R@10").shapley) in recall in every combination, consistent with @sec-rq4. Among evidence units, packing the best sentences of the top 30 notes raises survival by #fx(fsh.at("F8=pruned").survival.shapley) with #num(-fsh.at("F8=pruned").tokens.shapley) fewer tokens, and whole notes lower it by #fx(-fsh.at("F8=page").survival.shapley).

*Held-out confirmation.* On the #fho.n held-out questions the reference configuration reached recall at 10 of #fx(fho.runs.reference.at("R@10")), the value of @sec-heldout. The best-survival configuration (#flb.best_survival.top20.at(0).label) kept a relevant note for #fx(fho.runs.best_survival.survival) of questions against #fx(fho.runs.reference.survival), a gain of #fx(FHS.survival.diff) (95% CI #fx(FHS.survival.ci95.at(0)) to #fx(FHS.survival.ci95.at(1))) with #num(-FHS.tokens_diff) fewer tokens; #FHS.survival.new_only questions favoured it and #FHS.survival.comparator_only the reference. The best-recall configuration (#flb.best_recall.top20.at(0).label), which gained #fx(flb.confirmation.at("recall axis: R@10 vs reference").diff) on the confirmation half, gained only #fx(FHR.at("R@10").diff) on held-out questions (95% CI #fx(FHR.at("R@10").ci95.at(0)) to #fx(FHR.at("R@10").ci95.at(1)), Holm $p$ = #fx(FHR.at("R@10").p_holm, d: 2)); its development gain did not replicate at significance. The survival gain of sentence packs replicates.

*Do sentence packs answer the question?* A separate pre-registered check (`bench/factorial/judge-preregistration.md`) put both packs of #fjp.n questions from the confirmation half to the judge of @sec-ideas, the local 27B model, which was asked whether the pack contains the information needed to answer; the two packs of a question were judged in a random order. The best-survival configuration's sentence packs were judged sufficient for #fx(fjp.yes.A) of questions and the reference chunk packs for #fx(fjp.yes.B), a difference of #fx(FJN.diff) (95% CI #fx(FJN.ci95.at(0)) to #fx(FJN.ci95.at(1))), so non-inferiority at the 0.02 margin is not accepted ($p$ = #fx(FJN.p, d: 2)). Of the questions on which the judge disagreed between packs, #FJM.new_only favoured the sentence pack and #FJM.comparator_only the chunk pack (exact McNemar, two-sided $p$ = #fx(FJM.p)). On the same questions the sentence packs kept a relevant note more often (#fx(fjp.survival.A) against #fx(fjp.survival.B)) with #num(calc.round(fjp.tokens.B - fjp.tokens.A)) fewer tokens, but when a relevant note was present they were judged sufficient less often (#fx(fjp.yes_given_survival.A) against #fx(fjp.yes_given_survival.B)), which suggests that the five sentences kept per note can leave out what the answer needs; which text was lost was not examined. On this corpus survival is therefore not a stand-in for answer sufficiency, and the survival gain of sentence packs does not carry over to answers.

*The local agent.* #fl2.coverage.completed_local runs of the local 27B model compared 8K and 32K context windows on the pilot tasks of the agent harness. The 32K window used #fx(FCTX.at("32768").input_tokens.mean / FCTX.at("8192").input_tokens.mean, d: 1) times the input tokens and #fx(FCTX.at("32768").wall_s.mean / FCTX.at("8192").wall_s.mean, d: 1) times the wall time of the 8K window, and found the target note in #fx(FCTX.at("32768").page_found.mean, d: 2) of runs against #fx(FCTX.at("8192").page_found.mean, d: 2). The sample is too small for a test; the larger window bought no measured gain.

// ---------------------------------------------------------------- 10

#let fuq = json("bench/followup/compositional-questions.json")
#let fuc = json("bench/followup/compositional.json")
#let FUC(a) = fuc.summary.at(a)
#let FUT = fuc.tests.at("hk1_facets vs dense_facets (primary)")
#let fls = json("bench/followup/lsh-sweep.json")
#let flm = json("bench/followup/lsh-matched.json")
#let LS(f, p, pr, L) = fls.rows.find(r => r.family == f and r.preprocessing == p and r.probes == pr and r.tables == L)
#let fpw = json("bench/followup/power.json")
#let FPW(k) = fpw.curves.at(k)
#let fua = json("bench/followup/compositional-agent.json").summary
#let fus = json("bench/followup/compositional-sentences.json")
#let FUS(a) = fus.summary.at(a)
#let FUA(k) = fus.acceptance.at(k)
#let fjl = json("bench/followup/judge-length.json")
#let fcp = json("bench/followup/compress-packs.json")
#let FCP(k) = fcp.tests.at(k)
#let fqe = json("bench/followup/qrels-extended.json")
#let FQG(k) = fqe.reranker_gain.at(k)
#let fjh = json("bench/followup/judge-human.json")
#let fgc = json("bench/followup/gold-check.json")
#let FJH(k) = fjh.s1b.at(k)
#let FGS(a) = fgc.summary.at(a)
#let FGT = fgc.tests.at("hk1_facets vs dense_facets (primary)")
#let c1 = json("bench/followup/cell1.json")
#let c2 = json("bench/followup/cell2.json")
#let c3 = json("bench/followup/cell3.json")
#let c4 = json("bench/followup/cell4.json")
#let c5 = json("bench/followup/cell5.json")
#let c6 = json("bench/followup/cell6.json")
#let c8 = json("bench/followup/cell8.json")
#let c9 = json("bench/followup/cell9.json")
#let c14 = json("bench/followup/cell14.json")
#let c15 = json("bench/followup/cell15.json")
#let c16 = json("bench/followup/cell16.json")
#let c17 = json("bench/followup/cell17.json")
#let s9a = json("bench/followup/s9a.json")
#let s9x = json("bench/followup/s9a-explore.json")
#let S9(a) = s9a.summary.at(a)
#let S9X(a) = s9x.summary.at(a)
#let S9T(k) = s9a.tests.at(k)
#let S9P = S9T("dense_title_area_filter vs dense_title_and_area_words (primary)")
#let S9Q = S9T("dense_title_area_filter vs dense_question_top30")
#let S9H = S9T("hk1_title_area_filter vs dense_title_area_filter")
#let s9b = json("bench/followup/s9b.json")
#let S9SHAPES = ("area_type_months", "area_months", "type_months")
#let S9B(sh) = s9b.results.at("200000").summary.at(sh)
#let S9R(f) = S9SHAPES.map(sh => f(S9B(sh)))
#let s10 = json("bench/followup/s10.json")
#let s10g = json("bench/followup/s10-gate.json")
#let SA(a) = s10.arms.at(a)
#let S10T(a) = s10.tests_vs_A.at(a + " vs A_random_axes")

= Follow-up checks <sec-followup>

Six further checks were pre-registered together (`bench/followup/preregistration.md`) after the factorial study, one on the author's idea for Hilbert keys and five suggested by the literature. Their departures from the pre-registration are recorded in `bench/followup/deviations.md`. Two more were pre-registered later in the same file: a Hilbert key over attributes the notes already carry, and a key rebuilt at ingest.

*Hilbert keys as a map for compositional questions.* The checks so far used a Hilbert key as a filter for one question vector. The author's idea is different: an agent splits a question such as "who in a team contributed to open source" into facets, each facet has a region of key cells, and the answer lies in the intersection of the regions. To test it, #fuq.n questions of the form "Which notes are about both A and B?" were drawn from pairs of notes A and B that share #fuq.common_range.at(0) to #fuq.common_range.at(1) linking notes (#num(fuq.qualifying_pairs) pairs qualified), and the gold answer is the set of notes linking to both. Dense retrieval of the whole question found #fx(FUC("dense_question_top10").recall) of the gold in its top 10 and #fx(FUC("dense_question_top30").recall) in its top 30. Intersecting the top 50 notes of each facet, retrieved densely, found #fx(FUC("dense_facets").recall) at a precision of #fx(FUC("dense_facets").precision). Intersecting the facets' hk1 regions (level 1, 16 ranges) found #fx(FUC("hk1_facets").recall) at a precision of #fx(FUC("hk1_facets").precision), a difference of #fx(FUT.diff) from the dense intersection (Holm $p$ #fp(FUT.p_holm)); each facet's region held on average only #fx(fuc.hk1_region_coverage_mean) of the gold notes, so the intersection loses most of them. The links themselves answer every question exactly, by construction of the gold. On these measurements an embedding-derived Hilbert key does not carry facets well enough to be intersected; a key built over explicit attributes, the classic use of the curve for multi-attribute indexing, is tested below.

*The same questions for an agent and on sentences.* The local 27B agent of level 2 answered #fua.dense.runs of these questions with three tool sets. With a tool returning the dense facet intersection it found #fx(fua.dense.recall) of the gold notes using #num(fua.dense.prompt_tokens) input tokens and #fx(fua.dense.wall_s, d: 0) s on average; with the hk1 facet tool it found #fx(fua.hk1.recall) using #num(fua.hk1.prompt_tokens) tokens and #fx(fua.hk1.wall_s, d: 0) s; with search alone #fx(fua.search.recall) using #num(fua.search.prompt_tokens) tokens (descriptive, no test). Keys on #num(fus.sentences) sentences instead of chunks raised the hk1 intersection's set recall to #fx(FUS("sentence-hk1").recall), but only because a facet's region then covers most notes: the median answer held #num(FUS("sentence-hk1").size_median) notes at a precision of #fx(FUS("sentence-hk1").precision). Before this run the author fixed an acceptance rule: a slower key is acceptable if it spends at least 25 percent fewer model tokens per correct answer and at most twice the time. Against dense intersection on the same sentences, sentence keys needed #fx(FUA("sentence-hk1 vs sentence-dense (primary)").token_ratio, d: 0) times the evidence tokens per correct answer while taking #fx(FUA("sentence-hk1 vs sentence-dense (primary)").time_ratio, d: 2) of the time; chunk keys needed #fx(FUA("chunk-hk1 vs chunk-dense").token_ratio, d: 1) times the tokens and #fx(FUA("chunk-hk1 vs chunk-dense").time_ratio, d: 1) times the time; the agent with the hk1 tool, judged after the fact, #fx(FUA("S6b agent, hk1 tool vs dense tool (after the fact)").token_ratio, d: 1) and #fx(FUA("S6b agent, hk1 tool vs dense tool (after the fact)").time_ratio, d: 1) times. The rule rejects the key in every comparison. Dense intersection on sentences itself found #fx(FUS("sentence-dense").recall) of the gold with a median of #num(FUS("sentence-dense").size_median) notes and #num(FUS("sentence-dense").evidence_tokens_per_correct) evidence tokens per correct answer; because a note that links to A usually names A in some sentence, this figure may reflect how the gold was built. The author checked the first #fgc.checked gold lists and marked #fgc.correct_and_complete of them correct and complete. On those #fgc.n questions the hk1 intersection's set recall is #fx(FGS("hk1_facets").recall) against #fx(FGS("dense_facets").recall) for dense intersection (Holm $p$ #fp(FGT.p_holm)), the same ordering as on all #fuq.n.

*A Hilbert key over explicit attributes.* The first of these gives the key attributes on which nearness is defined and nothing is learned: a note's area (the first two parts of its path), its type, and the month of its date. Retrieval comes first. Questions of the form "Which notes in area P are about B?" were drawn, where B is a note outside P and the gold is the notes in P that link to B. The caps on how often an area may recur allowed #s9a.n pairs instead of the planned 60 (deviation D13). Taking the top 50 notes of a dense search on B's title and keeping those in P found #fx(S9("dense_title_area_filter").recall) of the gold at a precision of #fx(S9("dense_title_area_filter").precision). Intersecting the same 50 with a dense search on the area's words, as the facets were intersected above, found #fx(S9("dense_title_and_area_words").recall), a difference of #fx(S9P.diff) (Holm $p$ #fp(S9P.p_holm)). Against the whole question's top 30 (#fx(S9("dense_question_top30").recall)) the filter's difference of #fx(S9Q.diff) was not significant (Holm $p$ = #fx(S9Q.p_holm, d: 2)), and its median answer held #num(S9("dense_title_area_filter").tokens_median) tokens against #num(S9("dense_question_top30").tokens_median). The level-1 hk1 region of B's title, kept to P, found #fx(S9("hk1_title_area_filter").recall) (difference #fx(S9H.diff), Holm $p$ #fp(S9H.p_holm)). With the area cap raised until #s9x.n pairs were drawn, an exploratory check, the order was the same: #fx(S9X("dense_title_area_filter").recall) for the filter, #fx(S9X("dense_title_and_area_words").recall) for the area's words and #fx(S9X("dense_question_top30").recall) for the top 30. An attribute helps as a filter, not as words to embed, and the embedding key does not help even with the filter.

Then the index. Each note's area, type and month, as ordinals of #s9b.bits_per_axis bits, were given a three-dimensional Hilbert index, checked to visit every cell of a grid #s9b.hilbert_check.grid.split("^").at(0) cells on a side once, in unit steps. It was compared with a Z-order key, a composite B-tree on (area, type, month), and three single-column B-trees combined by bitmap AND, each in a table stored in its own index order, on PostgreSQL #s9b.server_version.split(" ").at(0). A query is a box: an area and a type, an area alone, or a type alone, over 1 to 24 months, #s9b.boxes_per_shape boxes of each shape, at the #num(s9b.results.live.rows) live notes and at #num(s9b.results.at("200000").rows) rows drawn from their attributes. Curve keys cover a box with at most #s9b.max_ranges ranges. The composite B-tree touched the fewest buffers for every shape at both sizes (@tab-s9b). At 200,000 rows the Hilbert key touched #fx(calc.min(..S9R(s => s.acceptance.buffer_ratio)), d: 2) to #fx(calc.max(..S9R(s => s.acceptance.buffer_ratio)), d: 2) times the composite B-tree's buffers and took #fx(calc.min(..S9R(s => s.acceptance.time_ratio)), d: 2) to #fx(calc.max(..S9R(s => s.acceptance.time_ratio)), d: 2) times its time, so the pre-registered rule, at least 25 percent fewer buffers in at most twice the time, rejects it for every shape. Against Z-order the Hilbert key needed a median of #fx(S9B("type_months").hilbert.exact_ranges_median, d: 1) exact ranges for a type over all areas against #fx(S9B("type_months").zorder.exact_ranges_median, d: 0), and touched #fx((1 - S9B("type_months").acceptance.hilbert_vs_zorder_buffer_ratio) * 100, d: 0) percent fewer buffers. With an area fixed the two were level (ratio #fx(S9B("area_months").acceptance.hilbert_vs_zorder_buffer_ratio, d: 2)). The bitmap AND, its rows in random order, touched the most. On these attributes the curve gives nothing that a composite B-tree does not already give.

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    rule,
    [], table.cell(colspan: 2)[*Area and type*], table.cell(colspan: 2)[*Area*], table.cell(colspan: 2)[*Type*],
    [*Access path*], [*Buffers*], [*ms*], [*Buffers*], [*ms*], [*Buffers*], [*ms*],
    thin,
    ..(("hilbert", "Hilbert key"), ("zorder", "Z-order key"), ("composite", "Composite B-tree"), ("bitmap", "Bitmap AND")).map(((k, name)) => {
      (([#name],) + S9SHAPES.map(sh => ([#num(S9B(sh).at(k).buffers_median)], [#fx(S9B(sh).at(k).ms_median, d: 2)])).flatten())
    }).flatten(),
    rule,
  ),
  caption: [Attribute boxes at #num(s9b.results.at("200000").rows) rows: median shared buffers touched and median execution time over #s9b.boxes_per_shape boxes per shape, each box over 1 to 24 months. "Area" fixes one area and any type, "Type" one type and any area. Source: `bench/followup/s9b.json`.],
) <tab-s9b>

*Rebuilding the key at ingest.* The explanation in @sec-mech-hk1 holds for a key whose axes are random, and storing or ordering chunks by that key cannot change the angle between a question and its answer. The last check rebuilt the level-1 key over every chunk in two ways that change what the angle acts on: the axes, and the vector that is keyed. Every arm reads the question's own cell, then the cells whose flipped axes lie nearest their boundaries, until #SA("A_random_axes").filter.primary_330.budget chunks are candidates, and ranks them by exact cosine. On the #s10.confirmation_questions confirmation questions, eight random axes, as in hk1, reached recall at 10 of #fx(SA("A_random_axes").filter.primary_330.at("R@10")). The probing and the question set differ from those of @sec-rq4, so this figure is not comparable with the #fx(C("hk1-L1-R16").recall) reported there. Questions and their answers shared a cell #fx(SA("A_random_axes").qa_same_cell_observed) of the time, against #fx(SA("A_random_axes").qa_same_cell_theory) predicted by @eq-lsh. The top eight principal axes of the centred chunks, with each axis cut at its corpus median, reached #fx(SA("B_corpus_axes").filter.primary_330.at("R@10")). The same axes rotated by iterative quantization @gong2013itq reached #fx(SA("C_itq").filter.primary_330.at("R@10")). Their differences from random axes were #fx(S10T("B_corpus_axes").diff) and #fx(S10T("C_itq").diff) (Holm $p$ #fp(S10T("B_corpus_axes").p_holm) and #fp(S10T("C_itq").p_holm)). On these axes a question and its answer shared a cell #fx(SA("B_corpus_axes").qa_same_cell_observed) and #fx(SA("C_itq").qa_same_cell_observed) of the time, #fx(SA("B_corpus_axes").qa_same_cell_observed / SA("B_corpus_axes").qa_same_cell_theory, d: 0) and #fx(SA("C_itq").qa_same_cell_observed / SA("C_itq").qa_same_cell_theory, d: 0) times the rate that @eq-lsh gives for random axes. No arm came within the 0.03 margin of exhaustive cosine (#fx(s10.at("exhaustive_R@10"))). Scanning a fifth of the chunks, the rotated key reached #fx(SA("C_itq").filter.at("20.0%").at("R@10")), below the Hilbert and k-means partitions of @sec-ideas at a similar scan. The second rebuild keyed each chunk by the mean vector of three questions that a local four-billion-parameter model wrote for it. It stopped at its gate. On #s10g.n tuning questions, the generated questions were farther from the real question than the chunk itself (median centred cosine #fx(s10g.centred_cosine_median.question_generated) against #fx(s10g.centred_cosine_median.question_chunk)), and generating them for the corpus would have taken #fx(s10g.projected_hours, d: 1) hours against a limit of #fx(s10g.max_hours, d: 0). Fitting the axes to the corpus beats the random-hyperplane rate several times over, but a single key still stays well below exhaustive search.

*Is the judge's verdict on sentence packs about length?* Sentence packs are shorter than chunk packs, and language-model judges are known to favour longer text @zheng2023judge. For each of the #fjl.n questions of the answer-quality check, the chunk pack was cut to the token count of that question's sentence pack (#num(fjl.tokens.Bcut) tokens on average) and judged again with the same prompt, which asks whether the context is sufficient to answer @joren2024. The cut chunk packs were judged sufficient for #fx(fjl.yes.Bcut) of the questions, the same as the full chunk packs (#fx(fjl.yes.B)); against the sentence packs (#fx(fjl.yes.A)), #fjl.Bcut_vs_A.new_only questions favoured the cut chunk pack and #fjl.Bcut_vs_A.comparator_only the sentence pack (exact McNemar $p$ = #fx(fjl.Bcut_vs_A.p)). By the pre-registered reading, content and not length explains the gap, which suggests that the five sentences kept per note leave out what whole chunks contain.

*The author's labels on the same questions.* He labelled both packs of 50 of the #fjl.n questions, one pack at a time, with the condition and the judge's answer hidden. The judge agreed with him on #fx(fjh.s1b.agreement.all.agreement) of the 100 packs (Cohen's kappa #fx(fjh.s1b.agreement.all.kappa, d: 2)), #fx(fjh.s1b.agreement.A.agreement) on sentence packs and #fx(fjh.s1b.agreement.B.agreement) on chunk packs. His yes-rate was #fx(FJH("A").human.estimate) for sentence packs and #fx(FJH("B").human.estimate) for chunk packs, a paired difference of #fx(FJH("A_minus_B").human.estimate) (95% CI #fx(FJH("A_minus_B").human.ci95.at(0)) to #fx(FJH("A_minus_B").human.ci95.at(1))). Prediction-powered inference @angelopoulos2023ppi takes the judge's mean on all #FJH("A").ppi.n_judge questions and adds the mean of (author minus judge) on the #FJH("A").ppi.n_human labelled questions. For sentence packs that estimate is #fx(FJH("A").ppi.estimate) (95% CI #fx(FJH("A").ppi.ci95.at(0)) to #fx(FJH("A").ppi.ci95.at(1))), for chunk packs #fx(FJH("B").ppi.estimate) (95% CI #fx(FJH("B").ppi.ci95.at(0)) to #fx(FJH("B").ppi.ci95.at(1))), and for the difference #fx(FJH("A_minus_B").ppi.estimate) (95% CI #fx(FJH("A_minus_B").ppi.ci95.at(0)) to #fx(FJH("A_minus_B").ppi.ci95.at(1))). The judge-only difference on all #FJH("A_minus_B").judge_150.n questions is #fx(FJH("A_minus_B").judge_150.estimate) (95% CI #fx(FJH("A_minus_B").judge_150.ci95.at(0)) to #fx(FJH("A_minus_B").judge_150.ci95.at(1))). The author's labels put the two packs level, and the corrected interval for the difference includes zero.

*Compression baselines.* Three further packs were judged on the same questions. A hybrid that keeps whole chunks for the top ten notes and pruned sentences for ranks 11 to 30 was judged sufficient for #fx(fcp.yes.H) of questions with #num(fcp.tokens.H) tokens; LongLLMLingua @jiang2023longllmlingua, compressing the chunk pack to the sentence pack's length, for #fx(fcp.yes.L) with #num(fcp.tokens.L) tokens; the extractive compressor of RECOMP @xu2023recomp, choosing sentences up to the same length, for #fx(fcp.yes.R) with #num(fcp.tokens.R) tokens; the full chunk pack scored #fx(fcp.yes.B) with #num(fjp.tokens.B) tokens. None passed the pre-registered non-inferiority test against the chunk pack at the 0.02 margin (hybrid $p$ = #fx(FCP("H vs B, non-inferiority 0.02").p, d: 2), LongLLMLingua $p$ = #fx(FCP("L vs B, non-inferiority 0.02").p, d: 2)), so with #fjl.n questions none can be called as good. Two differences stand out: question-aware token compression matched the chunk pack's rate, as a point estimate, with #fx((1 - fcp.tokens.L / fjp.tokens.B) * 100, d: 0) percent fewer tokens, and an extractive compressor trained on English questions did worse than the corpus's own sentence ranking (#FCP("R vs A, McNemar").comparator_only against #FCP("R vs A, McNemar").new_only discordant questions, $p$ = #fx(FCP("R vs A, McNemar").p, d: 3)).

*Cross-polytope hashing and whitening.* Andoni et al. @andoni2015 prove that cross-polytope LSH, which hashes a rotated vector to its largest coordinate, needs fewer tables than random hyperplanes at the same recall. A new library @ziglsh implements both families over the same seeded rotations. Hyperplane tables of eight bits reproduced hk2 (R\@10 #fx(LS("hyperplane", "centred", 0, 32).at("R@10_after_rescoring")) at 32 tables). The pre-registered comparison at 10 percent of chunks scanned could not be made, because cross-polytope tables scan less: at 128 tables without probes it reads #fx(LS("cross-polytope", "centred", 0, 128).scanned_mean_share * 100, d: 1) percent. At matched cost, an exploratory comparison (deviation D4), cross-polytope reached #fx(flm.cross_polytope.at("R@10")) reading #fx(flm.cross_polytope.scanned_mean_share * 100, d: 1) percent of chunks against #fx(flm.hyperplane.at("R@10")) at #fx(flm.hyperplane.scanned_mean_share * 100, d: 1) percent for hyperplanes (difference #fx(flm.test.diff), 95% CI #fx(flm.test.ci95.at(0)) to #fx(flm.test.ci95.at(1))), at four times the rotations per query. Whitening the embeddings @su2021whitening lowered recall for both families and for exhaustive cosine (#fx(fls.exhaustive.at("whitened_R@10")) against #fx(fls.exhaustive.at("raw_R@10")), $p$ #fp(fls.exhaustive.test.p)). Removing the mean, as in @sec-mech-hk1, is what makes the tables selective: at 32 hyperplane tables, raw vectors scanned #fx(LS("hyperplane", "raw", 0, 32).scanned_mean_share * 100, d: 1) percent of chunks for R\@10 #fx(LS("hyperplane", "raw", 0, 32).at("R@10_after_rescoring")), centred ones #fx(LS("hyperplane", "centred", 0, 32).scanned_mean_share * 100, d: 1) percent for #fx(LS("hyperplane", "centred", 0, 32).at("R@10_after_rescoring")).

*How many questions the reranker comparison needs.* Only #fx(fpw.nonzero_share * 100, d: 0) percent of the development questions change recall at 10 between the reranked and the reference configuration. Detecting a gain of the held-out size (+#fx(FPW("held-out (+0.016)").effect)) with 80 percent power @webber2008, by a paired bootstrap of the two-sided Wilcoxon test, needs about #num(FPW("held-out (+0.016)").n_for_80_percent) questions; at 800 questions the power is #fx(FPW("held-out (+0.016)").curve.find(c => c.n == 800).power * 100, d: 0) percent, and the held-out set has #num(fpw.heldout_n). The development gain (+#fx(FPW("development (observed)").effect)) needs #num(FPW("development (observed)").n_for_80_percent). The held-out result is therefore uninformative rather than negative (deviation D3 explains the method).

*How incomplete are the relevance labels?* Each development question has the notes its author marked relevant, and a note the retriever found but nobody marked counts as a miss @buckley2004. For #fqe.questions_used development questions (#fqe.questions_drawn were drawn; the pre-registered cap on judge calls reduced them), the notes in the top ten of the reference configuration or of the best-recall configuration that were not labelled relevant, #num(fqe.unlabelled_judged_relevant.pairs) pairs, were shown to the local judge with a relevance prompt after Thomas et al. @thomas2023. It called #fx(fqe.unlabelled_judged_relevant.share * 100, d: 0) percent of them relevant (95% CI #fx(fqe.unlabelled_judged_relevant.ci95.at(0) * 100, d: 0) to #fx(fqe.unlabelled_judged_relevant.ci95.at(1) * 100, d: 0)), and #fqe.unlabelled_judged_relevant.questions_with_any_yes of the #fqe.questions_used questions had at least one such note. Adding them to the labels lowered the reference configuration's recall at 10 from #fx(fqe.at("R@10").reference.original) to #fx(fqe.at("R@10").reference.extended), because its denominator grew, and raised the reranked configuration's from #fx(fqe.at("R@10").best_recall.original) to #fx(fqe.at("R@10").best_recall.extended): the reranker's gain on these questions went from #fx(FQG("original").diff) ($p$ = #fx(FQG("original").p, d: 2)) to #fx(FQG("extended").diff) (95% CI #fx(FQG("extended").ci95.at(0)) to #fx(FQG("extended").ci95.at(1)), $p$ #fp(FQG("extended").p)). By the judge's labels, the reranker surfaces relevant notes that the original labels miss. Two cautions apply. The judge and the reranker are models of the same family and may share preferences. On 30 pairs the author labelled afterwards, 15 the judge called relevant and 15 it did not, he agreed with #fx(fjh.s2_agreement.judge_yes.agreement) of the relevant calls and with #fx(fjh.s2_agreement.judge_no.agreement) of the others (overall agreement #fx(fjh.s2_agreement.all.agreement), kappa #fx(fjh.s2_agreement.all.kappa, d: 2)).

= Specified cells on the same switches <sec-specified>

The literature scan of 8 October 2026 proposed further changes to this corpus. Several of them are already closed here. Late chunking @gunther2024 does not apply, because Qwen3-Embedding pools only the last token. HyDE @gao2022, link-graph PageRank, sentence pruning @chirkova2025, and a pack cut off at the largest cosine gap @taguchi2025 have been run, and their results are in @sec-ideas, @sec-efficiency and @sec-factorial. The note-mean condition of @sec-rq2 already scores each note by the mean of its current chunk vectors, so there is no separate stored mean to refresh.

Six cells on switches this paper already measures have been run on the development questions. The protocol is one factor at a time, with the reranker off except where it is the factor. The held-out questions stay sealed, because no cell raised recall at 10 on the development set.

*Ranking by the best sentence.* RQ2 compared the note mean with the best chunk. The sentence packs of @sec-factorial kept sentences of notes that chunk ranking had already selected, and those packs were judged sufficient less often than chunk packs. Here a note's score is the maximum cosine of its stored sentence vectors with the question. Recall at 10 is #fx(c1.at("R@10").best_sentence) against #fx(c1.at("R@10").best_chunk) for best-chunk ranking. The difference is #fx(c1.diff) (95% CI #fx(c1.ci95.at(0)) to #fx(c1.ci95.at(1))). The interval sits below zero. Source: Chen et al. @chen2023dxr.

*Seed-clamped smoothing.* Link PageRank in @sec-ideas gained recall below its threshold, and diffusion in @sec-efficiency moved relevant notes out of the packed top ten. From seed scores $s$, the cell iterates $p <- alpha s + (1 - alpha) W p$ on the undirected link graph and ranks by $max(p, s)$. Recall at 10 is #fx(c2.at("R@10").clamped) against #fx(c2.at("R@10").best_chunk). The difference is #fx(c2.diff) (95% CI #fx(c2.ci95.at(0)) to #fx(c2.ci95.at(1))), and the original top note stayed inside the top ten for all #c2.original_top_retained_in_10 questions. Source: Miao et al. @miao2026grapher.

*Local forward push.* The PageRank factor in @sec-factorial added #num(fsh.at("F4=on").latency.shapley) ms. A forward push from the same seeds, on the directed link graph, has recall at 10 of #fx(c3.at("R@10").forward_push) against #fx(c3.at("R@10").existing_pagerank) for the existing power iteration. The difference is #fx(c3.diff) (95% CI #fx(c3.ci95.at(0)) to #fx(c3.ci95.at(1))). The push took #fx(c3.seconds_per_question.forward_push, d: 6) s per question and the power iteration #fx(c3.seconds_per_question.existing_pagerank, d: 6) s. Source: Wang et al. @wang2019ppr.

*The reranker only when the top two scores are close.* The reranker raises recall and costs seconds (@sec-ideas). The threshold is the tuning-half gap at which the reranker is called, chosen as the smallest call rate whose MRR interval includes zero, then applied once on the confirmation half. The chosen threshold is #fx(c5.threshold, d: 0), so the reranker runs only on exact ties, on #fx(c5.confirm_call_rate, d: 3) of the confirmation questions. Confirmation MRR is #fx(c5.MRR.selective) against #fx(c5.MRR.always_on) for always calling it. The difference is #fx(c5.diff) (95% CI #fx(c5.ci95.at(0)) to #fx(c5.ci95.at(1))). Time on that half is #fx(c5.seconds.selective, d: 2) s against #fx(c5.seconds.always_on, d: 2) s. Source: Zhao et al. @zhao2026larch.

*Stopping when the packed text can answer.* Conformal stopping on cosine (@sec-efficiency) saved #fx(m9.primary.relative_fall * 100, d: 1) percent of tokens and missed a pre-registered 20 percent target, while leaving survival within 0.02. Here chunks of the reference pack are added, in best-chunk order, until JEV-9B assigns probability at least one half to the pack being able to answer. On the same 150 questions, the stopped packs use #fx(c8.tokens.stopped, d: 0) tokens against #fx(c8.tokens.full_chunk_pack, d: 0) for the full chunk pack, which is #fx((1 - c8.token_ratio) * 100, d: 0) percent fewer. The 27B judge calls the stopped packs sufficient for #fx(c8.sufficiency.yes) of the questions, against #fx(c8.sufficiency.chunk_pack) for the full chunk pack. The difference of #fx(c8.sufficiency.diff) is outside 0.02. JEV-9B reads at most 1,024 tokens of a pack, its head and its tail, so it judged longer packs on a shortened text. It stopped after the first chunk on #c8.per_question.filter(r => r.stopped_chunks == 1).len() questions. On #c8.never_yes it never reached one half, and those packs kept every chunk that fit in order. Source: Jeong et al. @jeong2025ecorag.

*The winning sentence with its headings.* Sentence packs reached the note and were judged sufficient less often than chunk packs. The section pack of RQ3 returns text through the next heading. This pack is the winning sentence of each top note, in best-chunk order, together with the markdown headings open above that sentence. No new embedding was made. The packs use #fx(c9.tokens.sentence_headings, d: 0) tokens against #fx(c9.tokens.full_chunk_pack, d: 0). The same judge calls them sufficient for #fx(c9.sufficiency.yes) of the 150 questions, against #fx(c9.sufficiency.chunk_pack) for the chunk pack. Source: Rainey et al. @rainey2026spire.

The same scan ran the remaining cells on the development questions, and several lines stopped at a count. Subtracting each note's mean cosine to the tuning-half queries lowered recall at 10 from #fx(c4.at("R@10").best_chunk) to #fx(c4.at("R@10").hubness) (difference #fx(c4.diff), 95% CI #fx(c4.ci95.at(0)) to #fx(c4.ci95.at(1))). Of #c6.facts facts, #c6.superseded_facts are marked superseded and #c6.notes_with_a_superseded_fact notes carry one, so the replacement line stops, and with it the cell that would hide a replaced fact. Pages store no mean of their chunk vectors. #c15.disagreeing_entity_kinds live facts share an entity and a kind with two values, so the disagreement line stops. #c14.facts_with_an_end_date facts carry an end date and #c14.of_those_with_a_source_note of them cites a note, which is too small a set for a sufficiency or recall comparison, so that cell is the count. Because the early stop already cut tokens by more than 20 percent, the cell that would drop non-evidence stays closed. On the 60 two-page questions, the union of the top ten notes for each facet title, with the two target pages removed and with no link join, has set recall #fx(c16.set_recall), above the bar of #fx(c16.bar). A spread threshold fixed on the tuning half marks #fx(c17.confirm_mark_on_miss) of confirmation misses and #fx(c17.confirm_mark_on_hit) of confirmation hits (difference #fx(c17.diff), 95% CI #fx(c17.ci95.at(0)) to #fx(c17.ci95.at(1))).

= Discussion <sec-discussion>

*Principal findings.* On a personal corpus of about five thousand chunks, two switches account for the measurable gains: dense scoring, and returning chunks rather than whole notes to the model. The first raises recall by about half of the question set; the second raises the share of questions whose relevant note survives a 6,000-token budget by fourteen points at equal token cost, and no question in the set was hurt by it. The note-versus-chunk choice for ranking, the hybrid, the index structure, and the database engine did not change recall.

*Relation to prior work.* The weakness of lexical retrieval here is less a property of BM25-style scoring @thakur2021 than of conjunctive query parsing applied to long natural-language questions; with disjunctive parsing the lexical arm becomes competitive at chunk level, yet still adds nothing to fusion. The equality of note-mean and best-chunk ranking contrasts with the gains reported for set-based scoring on public benchmarks @khattab2020 @jayaram2024; the notes in this corpus are short (median #fx(inv.dataset.chunks_per_page.median, d: 0) chunks), so a mean of two vectors loses little. The behaviour of the Hilbert key follows from angular hashing theory @charikar2002 @indyk1998 once the anisotropy of contextual embeddings @ethayarajh2019 is removed @mu2018, and the recall measured while the key was being built, on near-duplicate pairs at cosine 0.96 @zighilbert, is consistent with @eq-lsh: at that cosine the model gives a level-1 collision probability above one half.

*Implications.* The results suggest that, for a small personal knowledge base, what is returned to the model matters more than how it is indexed. Exhaustive cosine over every chunk costs well under a millisecond per question here and remains sub-millisecond at forty times the size with HNSW. On these measurements a Hilbert key does not work as a retrieval index; making it work as one would take tens of independent keys, which is a locality-sensitive-hashing index rather than a single column. Fitting the key's axes to the corpus narrows the gap without closing it (@sec-followup). As an index over attributes the notes already carry, area, type and month, it touched more buffers than a composite B-tree for every box shape (@sec-followup). Its value as a label for browsing notes was not measured. Six further cells on these same switches are reported in @sec-specified. Calling the reranker only on exact ties met its time bar. The other five did not meet theirs, and the held-out questions stay sealed.

*Auxiliary observation: the harness.* Two harness factors were measured outside the hypothesis framework (@app-harness). A shell-output filter, RTK @rtk, halved the bytes of five fixed commands but lengthened one. On thirty questions, gbrain search found the relevant note in #calc.round(hp.gbrain_hit_rate * 30), exact text search with ripgrep @ripgrep in #calc.round(hp.vault_hit_rate * 30), and word search in #calc.round(hp.vault_token_hit_rate * 30). The comparison of agent hosts with these tools switched on and off was not run and is left to future work.

// ---------------------------------------------------------------- 8

= Threats to validity <sec-threats>

The threats are grouped by the four validity types of Wohlin et al. @wohlin2012.

*Construct validity.* Relevance labels were generated against the same brain and are binary; #inv.dataset.relevant_slugs_missing_from_corpus relevant slugs no longer exist and a few duplicated notes split relevance. The labels are also incomplete: a local judge called #fx(fqe.unlabelled_judged_relevant.share * 100, d: 0) percent of the unlabelled notes in two configurations' top ten relevant (@sec-followup), so absolute recall values are lower bounds and comparisons that favour a method surfacing unlabelled notes, such as the reranker, are biased against it. Token counts use gbrain's character heuristic, not a model tokenizer.

*Internal validity.* The live corpus grew from the planned 5,143 to #num(man.chunk_count) chunks before the primary cells ran, and all primary conditions ran within about one hour against the same tables. The hypothesis tests and mechanism analyses reloaded the live corpus later, when it held #num(inv.dataset.chunks) chunks; the chunks-per-note and tokens-per-chunk distributions of @tab-corpus and @fig-dataset come from that load. On the four primary cells the two loads gave the same recall, MRR, and nDCG to six decimals. The hk1 range at 200,000 rows was a sequential scan. The production-hybrid held-out figure comes from a different evaluator run. The engine comparison measures client-observed latency, which includes HTTP and JSON for ArangoDB and the Postgres wire protocol for pgvector; the IVF and HNSW parameters were swept over a few values, not tuned exhaustively. In the first pgvector runs the planner chose a sequential scan and the client reused cached plans; both were corrected before the reported run.

*External validity.* One corpus, one author, one embedding model, one language. Corpora with longer notes may favour set-based scoring; corpora with more exact identifiers may favour lexical retrieval. The 200,000-vector set was synthetic, so only its latency, not its recall, is interpreted. The engine comparison covers four alternatives to Postgres, measured at this corpus size only; the embedded engines were timed in process, without the network round trip a server pays.

*Software versions.* The measurements in the body ran on PostgreSQL 16.15 with pgvector 0.8.6 and ArangoDB 3.12.10. A repeat on PostgreSQL 18.6, pgvector 0.8.7 and ArangoDB 3.12.12 gave the same retrieval quality and the same ordering of the engines (@app-pg18).

*Statistical conclusion validity.* Per-question outcomes are not independent where questions share relevant notes; the bootstrap resamples questions, not notes. Exploratory analyses, including the centring intervention, were chosen after seeing the confirmatory results and are reported as such.

// ---------------------------------------------------------------- 9

= Conclusion <sec-conclusion>

Of the configurations measured, dense chunk retrieval with chunk-level evidence performed best: it found more relevant notes than lexical search (H1), did not differ from note-level dense ranking (H2), and under a fixed budget kept the relevant note far more often than whole notes (H3). A Hilbert-curve key failed as a candidate filter (H4); its first level is an eight-bit sign hash whose collision rate, after mean-centring, matches angular hashing theory, and that theory explains the failure: a question and its answer are too far apart in angle for one key on random axes to bring them together. Axes fitted to the corpus raised the collision rate several times over, but one key still stayed well below exhaustive recall (@sec-followup). Postgres with pgvector met the latency needs of this corpus at its size and at forty times it, and ArangoDB, measured on the same vectors and links, was slower at both vector search and one-hop expansion without retrieving anything different.

The two exploratory studies sharpen where the remaining losses are. Diversity, coverage and diffusion methods did not put relevant notes into the budget more often than the plain cosine order (@sec-efficiency), so redundancy is not what loses them. Packing the best sentences of more notes did, and that gain held on held-out questions (@sec-factorial), but a pre-registered judge found those shorter packs sufficient to answer less often than chunk packs, so the gain lies in reaching the note, not in answering; a pack cut to the same length but kept as whole chunks did not lose (@sec-followup), which points to what sentence pruning leaves out. The author's labels on 50 of these questions give both packs the same yes-rate, and the prediction-powered interval for their difference includes zero (@sec-followup). A hybrid of whole chunks and sentences, and question-aware token compression with #fx((1 - fcp.tokens.L / fjp.tokens.B) * 100, d: 0) percent fewer tokens, came close to chunk packs without passing a non-inferiority test on #fjl.n questions.

The same switches were measured again after the literature scan of 8 October 2026 (@sec-specified). Best-sentence ranking reached recall at 10 of #fx(c1.at("R@10").best_sentence) against #fx(c1.at("R@10").best_chunk). Seed-clamped smoothing and the forward push stayed inside the interval of their baselines. The reranker, called only when the top two cosine scores tie, took #fx(c5.seconds.selective, d: 2) s on the confirmation half against #fx(c5.seconds.always_on, d: 2) s, with MRR inside the always-on interval. A checker that stops the chunk pack early left #fx((1 - c8.token_ratio) * 100, d: 0) percent fewer tokens, and the 27B judge called those packs sufficient for #fx(c8.sufficiency.yes) of the 150 questions against #fx(c8.sufficiency.chunk_pack). The winning sentence with its headings used #fx(c9.tokens.sentence_headings, d: 0) tokens and was sufficient for #fx(c9.sufficiency.yes). The held-out questions stay sealed.

The follow-up checks answer two open questions. The reranker's shrunken held-out gain came from a measurement unlikely to detect it reliably: at the held-out set's size the power to find a gain that size is about #fx(FPW("held-out (+0.016)").curve.find(c => c.n == 800).power * 100, d: 0) percent, and incomplete relevance labels count many of the reranker's extra notes as misses. And the Hilbert key, tried as a map whose facet regions an agent intersects, again lost to dense retrieval and failed the author's own cost rule at chunk and at sentence level; dense intersection on sentences found #fx(FUS("sentence-dense").recall) of the link-derived answers with #num(FUS("sentence-dense").evidence_tokens_per_correct) evidence tokens per correct answer, the author marked #fgc.correct_and_complete of the #fgc.checked gold lists correct and complete, and on that subset the chunk-level hk1 intersection still trails dense intersection (@sec-followup). Built over attributes the notes already carry, the key lost again. The area worked as a filter on dense retrieval, finding #fx(S9("dense_title_area_filter").recall) of the gold against #fx(S9("dense_title_and_area_words").recall) when the area was searched as words. As an index, the curve touched more buffers than a composite B-tree for every box shape (@sec-followup).

// ---------------------------------------------------------------- declarations

#heading(numbering: none)[Data and code availability] <sec-data>

All result files, scripts, and figures are in the repository `github.com/guanchzhou/hilbert-paper`. Every number in this article is read by the typesetter from a JSON file in `bench/`; `python3 build.py` redraws all figures, records file digests, typesets with Typst @typst, and refuses to finish if a figure or result file is missing. The corpus is the author's private notes and is not published, and neither are note names or question texts; the result files hold counts and scores only.

#heading(numbering: none)[Competing interests]

The author has no financial or commercial interest in any tool evaluated here. zig-hilbert was written by the author as part of this investigation, as the experimental instrument for testing whether Hilbert-curve keys are a useful addition to knowledge management for AI systems; it is not a product being promoted. zig-select and zig-diffuse @zigselect @zigdiffuse were written in the same way, as independent checks of the per-token study, and zig-rabitq @zigrabitq for the filter comparison. The author uses gbrain as his personal knowledge base. The acceptance criterion for the key was fixed before the sweep.

#heading(numbering: none)[Funding]

None.

// ---------------------------------------------------------------- appendices

#show bibliography: set text(size: 9pt)
#bibliography("refs.bib", title: [References], style: "ieee")

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])

= Earlier cells <app-earlier>

Before the protocol was frozen, two switches were measured on the same development questions with gbrain's own evaluator. Prefixing each chunk with its note title before embedding did not change recall, MRR, or nDCG beyond the second decimal. gbrain's local cross-encoder #index("reranker"), which scores each question and passage jointly @nogueira2019, raised MRR and nDCG by about 0.01 and increased latency from 0.13 s to 3.8 s per question.

#figure(
  image("figures/figA1-earlier.svg", width: 100%),
  caption: [gbrain's hybrid before and after the #index("title prefix"), and with the reranker. Means printed to two decimals by gbrain's evaluator.],
) <fig-earlier>

= Harness observations <app-harness>

#figure(
  image("figures/fig10-harness.svg", width: 100%),
  caption: [Left: output bytes of five shell commands, raw and through #index("RTK"). Right: of thirty development questions, how many found a relevant note by each method.],
) <fig-harness>

RTK reduced the combined output from #num(rtk.bytes_before) to #num(rtk.bytes_after) bytes; `ls -la` fell from 586 to 129 bytes, `git log` was unchanged, and `find` grew from 113 to 139. For the page task, thirty development questions (every 27th) were searched by exact ripgrep @ripgrep over the vault, by ripgrep for up to four words longer than five characters, and by gbrain search with a limit of ten; mean time per question was #fx(hp.vault_seconds_mean, d: 2) s for ripgrep and #fx(hp.gbrain_seconds_mean, d: 1) s for gbrain.

= Replication on PostgreSQL 18 <app-pg18>

After the measurements above, the server was upgraded from PostgreSQL #pg18.pg16.server_version with pgvector #pg18.pg16.pgvector to #pg18.pg18.server_version.split(" ").at(0) with pgvector #pg18.pg18.pgvector, and ArangoDB from #pg18.pg16.arangodb to #pg18.pg18.arangodb. Every measurement that depends on the database software was repeated on the upgraded versions with the same scripts and query vectors (@tab-pg18). This is a replication against stored results, not a side-by-side test. PostgreSQL 16 had been removed by then, the corpus had grown from #num(pg18.pg16.embedded_chunks) to #num(pg18.pg18.embedded_chunks) embedded chunks, and each setting ran once under different machine load.

Retrieval quality did not change. Keyword recall at 10 is identical (#fx(PR("keyword-page recall at 10").pg18) for whole notes), and the share of the exact top ten that the HNSW index finds differs by less than 0.001 at every ef_search. Latencies moved in both directions by up to a few milliseconds, which one run per setting cannot attribute to the version. The engine comparison holds. At ef_search 50, pgvector found #fx(PR("pgvector ef_search 50, exact top-10 found").pg18) of the exact top ten in #fx(PR("pgvector ef_search 50, median").pg18, d: 2) ms. ArangoDB at nProbe 16 found #fx(PR("arango nProbe 16, exact top-10 found").pg18) in #fx(PR("arango nProbe 16, median").pg18, d: 2) ms. One-hop expansion took #fx(PR("SQL median").pg18, d: 2) ms in SQL against #fx(PR("AQL median").pg18, d: 2) ms in AQL, with identical neighbour sets for all #PR("identical neighbour sets").pg18 starts. The 200,000-vector recall is not comparable (marked †). The code of the original in-session run was not kept, so the noisy copies were rebuilt from its description and the data set differs. That script is now part of the repository (`bench/pg18/scale_in_session.py`).

#[
#show figure: set block(breakable: true)
#figure(
  {
    set text(size: 7.5pt)
    let fv(x) = if type(x) == int { num(x) } else if calc.abs(x) >= 100 { fx(x, d: 1) } else { fx(x, d: 3) }
    let prev = none
    let cells = ()
    for r in pg18.rows {
      cells.push(if r.group != prev [#r.group] else [])
      prev = r.group
      cells.push([#r.measure#if not r.comparable [ †]])
      cells.push([#fv(r.pg16)])
      cells.push([#fv(r.pg18)])
      cells.push([#r.unit])
    }
    table(
      columns: (1fr, 1.6fr, auto, auto, auto),
      align: (left, left, right, right, left),
      stroke: none,
      inset: (x: 4pt, y: 2.2pt),
      table.header(rule, [*Group*], [*Measure*], [*16.15*], [*18.6*], [*Unit*], thin),
      ..cells,
      rule,
    )
  },
  caption: [Database-dependent measurements on PostgreSQL 16.15 with pgvector 0.8.6 (stored) and repeated on 18.6 with pgvector 0.8.7; ArangoDB rows on 3.12.10 and 3.12.12. † marks a measure whose data set differs between the runs. Source: `bench/pg18/comparison.json`.],
) <tab-pg18>
]

= Source files <app-sources>

#[
#show figure: set block(breakable: true)
#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, right, left),
    stroke: none,
    table.header(rule, [*File*], [*Bytes*], [*sha256 (first 16 hex)*], thin),
    ..sources.files.map(f => ([#raw(f.name)], [#num(f.bytes)], [#raw(f.sha256.slice(0, 16))])).flatten(),
    rule,
  ),
  caption: [Result files and the scripts that wrote them, with the first sixteen hexadecimal digits of each file's SHA-256 digest @nist2015.],
  kind: "sources",
  supplement: [List],
)
]



= Index <app-index>
#make-index()
