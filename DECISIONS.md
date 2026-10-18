# Design Decisions

This log records the significant choices in the project, so each one can be justified at the thesis defense. New entries go at the bottom.
Each entry follows the same order: context &rightarrow; options considered &rightarrow; decision &rightarrow; rationale &rightarrow; evidence and status.

## D-001: Document categories (09.10.2026, provisional)

**Context:** Dr. Lux wants a self-collected corpus of scientific papers grouped by domain, and the pipeline has to be near-perfect in at least one category. The number of categories has not been fixed yet.

**Options considered:**
- A) Medical (target), CS, Economics
- B) CS (target), Medical, Physics
- C) CS (target), Medical, Economics, Physics

**Decision:** B. Computer science is the target category. Medical and Physics are secondary.

**Rationale:**
- arXiv has plenty of openly licensed CS papers (cs.*), and their layouts are fairly regular, so near-perfect extraction is realistic there.
- Medical papers from PubMed Central open access contain many tables. They test table extraction and structure detection hard.
- Physics papers are full of formulas. I expect them to be the weakest category, which shows where the pipeline stops working.
- With three categories, collecting the corpus and checking a sample of the ground truth by hand stays manageable.

**Status:** Provisional. The number of categories is not fixed yet.

## D-002: Ground truth source for evaluation (10.10.2026)

**Context:** Dr. Lux asked for an evaluation that is transparent and trustworthy. It needs ground truth for structure, tables, figures and RAG answers in all three categories from D-001.

**Options considered:**
- A) Generate ground truth automatically from the publisher sources (arXiv LaTeX, PMC JATS XML) and check a small sample by hand
- B) Annotate a small set by hand (10 to 15 papers per category)
- C) Use only RAG question/answer pairs, without structural ground truth

**Decision:** A. The RAG question/answer pairs are built on top of it and point to source locations whose position is known.

**Rationale:**
- It scales to many papers. Anyone can reproduce it by downloading the PDF and its source from the manifest.
- Structure, tables and captions each get their own metric, in addition to answer quality.
- B does not scale and is harder to reproduce. With C, nothing could be said about structure or table quality.

**Known risks:**
- Turning LaTeX into clean structure often goes wrong because of custom macros and packages. The hand-checked sample tells us how noisy the ground truth is, and that number goes into the results.
- Some papers have no usable source because the authors submitted only a PDF. These are left out of the corpus, and each exclusion is documented.

**Status:** Proposed, to be discussed with supervisor on 29.10.2026.

## D-003: Core pipeline architecture (10.10.2026)

**Context:** Dr. Lux defines "Hybrid AI system" as local LLMs combined with classical algorithms. The GPU has 8 GB of VRAM, so only one model fits in memory at a time. The main question is where the LLM sits relative to the classical algorithms.

**Options considered:**
- A) Classical first, LLM on low confidence: heuristics make the decisions, and the LLM is called only when their confidence is low or for tasks heuristics cannot handle (figure descriptions, table repair).
- B) LLM first: a vision-language model turns rendered pages into Markdown, and classical tools are only a fallback.
- C) Ensemble: several extractors run in parallel, and an LLM or a vote picks the best result for each element.

**Decision:** A. Tables additionally get an ensemble in the style of C: Camelot, pdfplumber and PyMuPDF each produce a candidate, a heuristic picks one, and the LLM decides ties.

**Rationale:**
- Most pages should never need the LLM, so I expect the pipeline to stay fast and fit in 8 GB of VRAM.
- Each decision can be traced back to the rule or the model that made it.
- Every LLM step can be switched off, so ablation studies can measure what the LLM adds.
- B was rejected because I expect small vision models to run slowly on 8 GB and to make up content, table numbers in particular. It would also be barely hybrid and close to existing page-to-Markdown models such as Nougat and olmOCR.
- C for everything would mean aligning the outputs of several extractors, which is complex, and every page would cost N times as much. I expect extractors to disagree most on tables, so that is where the ensemble should be worth its cost.

**Status:** Proposed, to be discussed with supervisor on 29.10.2026. The confidence thresholds still have to be calibrated on development data.

## D-004: License of the package (11.10.2026)

**Context:** The repo started with an MIT license. The Ingest stage is built on PyMuPDF, which is available under AGPL-3.0 or a commercial license from Artifex. Whoever distributes a program combined with PyMuPDF has to offer the source of the whole program under AGPL to the people who receive it. For modified versions, the same applies to users who reach the program over a network.

**Options considered:**
- A) Relicense the package to AGPL-3.0 and keep PyMuPDF
- B) Keep MIT and replace PyMuPDF with permissive libraries (pdfplumber and pdfminer.six under MIT, pypdfium2 under Apache-2.0/BSD)
- C) Keep MIT and depend on PyMuPDF anyway

**Decision:** A, with the license AGPL-3.0-or-later.

**Rationale:**
- PyMuPDF delivers font names, sizes, style flags, images, vector drawings and the TOC from one fast library, and the architecture uses all of them.
- B would make ingest slower and remove PyMuPDF's `find_tables` from the table ensemble.
- C is legally possible, because MIT code may depend on an AGPL library. But the package cannot run without PyMuPDF, so anyone who distributes or deploys it is bound by AGPL anyway, and an MIT label would be misleading.
- The repo is public, so publishing the source costs nothing extra.

**Known cost:** Some companies avoid AGPL dependencies, which may limit adoption outside the thesis.

**Status:** Proposed, to be discussed with supervisor on 29.10.2026.
