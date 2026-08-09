# ClinicalTrials.gov Query-to-Visualization Agent

A backend service that turns a natural-language question about clinical trials into a **structured visualization specification** backed by live [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) data.

```
POST /api/v1/query   {"query": "How are lung cancer trials distributed across phases?"}
   ->  {"visualization": {type, title, encoding, data}, "meta": {...}}
```

**The central design rule: the language model never produces a data value.** It reads the question, classifies it, and extracts the entities the user named. Every number in every response is computed by aggregation code over real trial records, and every datum carries citations to the specific trials that produced it.

---

## Contents

- [Quick start](#quick-start)
- [Demo frontend](#demo-frontend)
- [How it works](#how-it-works)
- [Why the LLM cannot fabricate a number](#why-the-llm-cannot-fabricate-a-number)
- [Request schema](#request-schema)
- [Response schema](#response-schema)
- [Supported questions](#supported-questions)
- [The single aggregation abstraction](#the-single-aggregation-abstraction)
- [Network graphs](#network-graphs)
  - [Drug synonym resolution](#drug-synonym-resolution)
- [Deep citations](#deep-citations)
- [Validation](#validation)
- [Logging](#logging)
- [Working with the ClinicalTrials.gov API](#working-with-the-clinicaltrialsgov-api)
- [Design decisions and tradeoffs](#design-decisions-and-tradeoffs)
- [Limitations and what I would do with more time](#limitations-and-what-i-would-do-with-more-time)
- [Testing](#testing)
- [AI tool usage](#ai-tool-usage)

---

## Quick start

Requires Python 3.12+ and an OpenAI API key.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then add your key
# .env:
#   OPEN_AI_API_KEY=sk-...
#   LLM_MODEL=gpt-5.4-mini

uvicorn app.main:app --reload
```

Interactive API docs: <http://127.0.0.1:8000/docs>. Health check: `GET /health`.

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/query \
  -H 'content-type: application/json' \
  -d '{"query": "How are lung cancer trials distributed across phases?"}' | jq
```

Run the tests, or regenerate every example output against the live API:

```bash
pytest -q                              # 365 tests, no network or API key needed
python scripts/run_examples.py         # writes examples/*.json from live data
```

---

## Demo frontend

A single-file demo client lives in [`frontend/`](frontend/). It exists to demonstrate the claim this project makes about its own output — that a frontend can render it **without guessing**.

```bash
uvicorn app.main:app --reload              # terminal 1
cd frontend && python -m http.server 5500  # terminal 2
open http://localhost:5500
```

Four buttons run the documented example queries, so a walkthrough needs no typing. An "Advanced options" panel exposes the structured overrides; whenever any are set, a badge shows which ones, because a stale override silently filtering later queries is otherwise invisible.

**What it demonstrates.** The renderer contains no domain knowledge — it never names `phase`, `trial_count`, `country`, or any other field. One `buildChartConfig()` reads `encoding.x`, `encoding.y`, and `encoding.color` to place the axes and split series, and serves `bar_chart`, `grouped_bar_chart`, `geo_bar_chart`, `histogram`, and `time_series` with a single `type === "time_series" ? "line" : "bar"` branch. Network graphs go through `buildNetworkData()`, which maps nodes and edges through the `encoding.nodes` / `encoding.edges` key maps. Renaming a field in the backend would require no frontend change; a renderer with per-chart-type branches would have proved the opposite.

Clicking any bar, point, node, or edge shows that datum's citations — NCT id, the exact supporting field value, and a link to the trial. On a merged network node it also lists the RxNorm surface forms folded into it, so a click on `pembrolizumab` shows both the Keytruda-named and Pembrolizumab-named trials behind one number. `meta.query_interpretation` and every `meta.warnings` entry render below the chart, since the disclosures are part of what makes a figure trustworthy.

**It is a demo, not a product.** No build step, no framework, no router, and no styling beyond readability. `file://` also works, but serving over HTTP is the documented path because a `null` origin is a confusing thing to debug.

**CORS is development-only.** The API has no CORS by default; `app/main.py` registers permissive middleware **only when `ENV == "development"`**, with `allow_credentials=False`. That is what makes a wildcard origin acceptable here, and it is not a production CORS policy — a deployment with `ENV=production` gets no CORS at all.

---

## How it works

Six stages, each a separate module with a typed input and output:

```
POST /api/v1/query
      |
      v
1. Query understanding      app/agents/understanding.py
      |                     LLM: query type, entities, chart type. No data values.
      |                     Then: ground entities against the question text,
      |                     override an incoherent chart choice.
      v
2. API query builder        app/services/ctgov.py :: build_searches
      |                     Pure function. QueryPlan -> upstream searches.
      v
3. Fetcher + record store   app/services/ctgov.py, store.py
      |                     Pagination, retry, fetch cap. Raw records retained
      |                     for the life of the request.
      v
4. Aggregator / network     app/services/aggregate.py, network.py
      |                     The only stage that produces a number. Every value
      |                     is a set cardinality over real NCT ids.
      v
5. Spec + citations         app/services/viz.py, citations.py
      |                     Formats values; attaches per-datum citations.
      v
6. Validator                app/services/validate.py
      |                     Re-derives every value, checks every citation
      |                     against the fetched records. Fails loudly.
      v
   QueryResponse
```

The stages are separate modules because each boundary is a place where something can go wrong independently, and each is testable in isolation: the aggregator against fixture records, the query builder against plan literals, the validator against deliberately corrupted responses, the client against mocked HTTP. Only two stages touch anything untrusted — the LLM in stage 1 and the external API in stage 3 — and both are behind narrow interfaces, which is what makes the no-hallucination claim auditable rather than aspirational.

---

## Why the LLM cannot fabricate a number

Three independent mechanisms, in order of how early they act:

**1. Structural — the LLM's output type has no field for a value.** The model's entire output is a `QueryUnderstanding`: a query type, a list of entity strings, a group-by dimension, a chart type. There is no count field, no bucket, no data array. A model that tried to report "Phase 3: 41 trials" has nowhere to put it.

**2. Grounding — every field that can change what gets searched must be supported by the question.** The one way the model could still influence real data is by introducing a filter nobody asked for. Eight fields reach the query builder or the client-side filters, and each is constrained differently according to what it is:

| Field | How it is grounded |
|---|---|
| `drugs`, `conditions`, `sponsors`, `countries` | Must appear in the query text, matched on **word boundaries** so an acronym cannot match inside a longer word — the leukemia `ALL` must not match "sm**all**-cell", and `SCLC` must not match `NSCLC`, which is a different disease. A fuzzy fallback handles inflection ("lung cancers" → "lung cancer") but is skipped below 6 characters, where a single character is too much of the string to forgive. |
| `compare_entities` | Same text check. Each one drives its own upstream search, so an invented entity would not merely mislabel a series — it would fetch and chart trials nobody asked about. |
| `phases` | **Read from the query text directly**, not taken from the model: `phase 3`, `phase 1/2`, `phases 2 and 3`, `Phase II or III`. Anchoring on the word "phase" and reading only the contiguous list keeps `"phase 2 study of 3 drugs"` from producing a Phase 3 filter. |
| `statuses` | **Read from the query text directly**, mapped to the live-verified `overallStatus` vocabulary via a synonym table ("stopped early" → `TERMINATED`). Longer phrases match first, so `"not yet recruiting"` cannot also register as `RECRUITING` — they are opposite filters. A status named *negatively* is recorded as an exclusion, never as the positive filter (see [status negation](#status-negation-is-detected-but-not-filterable)). |
| `year_range` | Each bound is accepted only if that four-digit year appears literally in the query. `"the last five years"` therefore applies no filter rather than a silently computed one. |

For phases and statuses, reading the text is a *stronger* guarantee than checking the model's answer: the value provably comes from the user's words. The model's own extraction is kept only as a cross-check, and anything it proposed that the text does not support is dropped and reported in `meta.warnings` — so a divergence is visible rather than silently applied.

**One honest limit.** Year *values* are grounded, but **which bound a year maps to is still the model's reading** — "since 2015" versus "before 2015". Full directional parsing was out of scope for this pass. The applied range is stated verbatim in `meta.warnings` ("Restricted to trials starting from 2015 client-side…"), so a misreading is visible in the response rather than hidden.

**3. Validation — every published value is re-derived before the response ships.** The validator receives the aggregator's own result alongside the formatted spec and recomputes each value from the pre-truncation id sets. It rejects any citation whose `nct_id` is not among the records actually fetched. A response that fails any check is withheld with HTTP 500 rather than returned — a chart that renders but cannot be traced to source data is worse than an error, because a reader has no way to tell it is wrong.

Numbers only ever come from `len(set_of_nct_ids)` in `aggregate.py` and `network.py`.

---

## Request schema

`POST /api/v1/query`, `content-type: application/json`.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string (3–2000) | **yes** | — | The natural-language question. |
| `drug_name` | string | no | `null` | Intervention name. Overrides LLM extraction. |
| `condition` | string | no | `null` | Condition or disease. Overrides LLM extraction. |
| `sponsor` | string | no | `null` | Lead sponsor name. Overrides LLM extraction. |
| `phase` | 1 \| 2 \| 3 \| 4 | no | `null` | Trial phase as a bare number, matching the API's own filter form. |
| `country` | string | no | `null` | Location country. |
| `start_year` | int (1990–2035) | no | `null` | Earliest trial start year (inclusive). |
| `end_year` | int (1990–2035) | no | `null` | Latest trial start year (inclusive). Must be ≥ `start_year`. |
| `max_citations_per_datum` | int (0–20) | no | `3` | Citations attached per datum. The full contributor count is always reported separately. |
| `max_studies` | int (100–5000) | no | `3000` | Upper bound on trials fetched. Disclosed in `meta.warnings` whenever it truncates. |

Unknown fields are rejected (422). The optional structured fields exist so the endpoint is usable programmatically as well as conversationally; **when supplied they win over whatever the LLM extracted**, since a caller passing `drug_name` has stated the entity outright and there is nothing left to infer.

```json
{
  "query": "How are trials for this drug distributed across sponsors?",
  "drug_name": "Nivolumab",
  "phase": 3
}
```

---

## Response schema

```json
{
  "visualization": {
    "type": "bar_chart",
    "title": "Lung cancer trials by phase",
    "encoding": {
      "x": {"field": "phase", "label": "Phase", "type": "ordinal"},
      "y": {"field": "trial_count", "label": "Number of trials", "type": "quantitative"},
      "color": null, "nodes": null, "edges": null
    },
    "data": [
      {
        "phase": "Phase 3",
        "trial_count": 84,
        "citations": [
          {
            "nct_id": "NCT00003117",
            "excerpt": "\"Paclitaxel With or Without Carboplatin in Treating Patients With Advanced Non-small Cell Lung Cancer\" — designModule.phases: [PHASE3]",
            "url": "https://clinicaltrials.gov/study/NCT00003117"
          }
        ],
        "total_supporting_trials": 84
      }
    ]
  },
  "meta": {
    "filters": {"condition": "lung cancer"},
    "source": "clinicaltrials.gov",
    "data_as_of": "2026-08-07T09:00:05",
    "total_studies_processed": 1000,
    "api_urls": ["https://clinicaltrials.gov/api/v2/studies?fields=...&query.cond=lung+cancer&countTotal=true"],
    "query_interpretation": "Interpreted as a distribution question, grouped by phase, rendered as a bar chart.",
    "assumptions": ["Interpreted 'across phases' as grouping trials by phase."],
    "warnings": [
      "Fetched 1,000 of 14,425 matching trials (capped at max_studies=1,000); figures below describe that sample, not the full result set.",
      "Trials can belong to more than one bucket on this axis (for example a combined Phase 1/2 trial), so bucket totals may sum to more than the number of trials.",
      "228 of 1,000 trials had no value for this axis and are shown as a separate bucket."
    ],
    "sorting": "phase order (Early Phase 1 -> Phase 4)",
    "time_granularity": null
  }
}
```

*(Real output, abridged to one data row and one citation — the full file is [`examples/01_bar_phase.json`](examples/01_bar_phase.json).)*

### `visualization`

| Field | Type | Description |
|---|---|---|
| `type` | enum | `bar_chart`, `time_series`, `grouped_bar_chart`, `network_graph`, `geo_bar_chart`, `histogram` |
| `title` | string | Human-readable, built from a template over the query's entities — never LLM free text, so it cannot describe something other than what the data shows. |
| `encoding` | object | Maps visual channels to keys present in every data row. |
| `data` | array of objects | The rows to render. |

**`encoding` for chart-family types** — `x`, `y`, and (for grouped bars) `color`, each a `{field, label, type}` where `field` is a key in every data row and `type` is one of `nominal`/`ordinal`/`quantitative`/`temporal`.

**`encoding` for `network_graph`** — `nodes` and `edges` key maps instead:

```json
"encoding": {
  "nodes": {"id": "id", "label": "label", "size": "size", "group": "kind", "rxcui": "rxcui"},
  "edges": {"source": "source", "target": "target", "weight": "weight"}
}
```

`data` for a network is a single-element array holding both collections: `[{"nodes": [...], "edges": [...]}]`. This keeps `data` a `list[dict]` for every chart type, so one response model covers all of them. A frontend can wire the key maps straight into d3-force or cytoscape.

**Every data row, node, and edge carries:**

| Field | Type | Description |
|---|---|---|
| `citations` | array | Up to `max_citations_per_datum` entries of `{nct_id, excerpt, url}`. |
| `total_supporting_trials` | int | The **full** number of contributing trials, so a truncated citation list never reads as the complete evidence set. |

**Network nodes additionally carry:**

| Field | Type | Description |
|---|---|---|
| `rxcui` | string \| null | RxNorm ingredient id when the drug name resolved, so a reader can verify the identity against RxNav. `null` for sponsors and unresolved names. |
| `merged_from` | array | The distinct source names folded into this node. Non-empty only when a real brand/generic merge happened — e.g. `["Keytruda", "KEYTRUDA®", "Pembrolizumab (MK-3475)"]`. |

### `meta`

| Field | Description |
|---|---|
| `filters` | The filters actually applied upstream. **Shape varies by query type:** flat (`{"condition": "melanoma"}`) for a single-search query; keyed by series label (`{"Pembrolizumab": {...}, "Nivolumab": {...}}`) for a comparison, mirroring how the series themselves are keyed — a flat dict would let one series' filters overwrite another's. |
| `source` | Always `clinicaltrials.gov`. |
| `data_as_of` | `dataTimestamp` from the API's `/version` endpoint — when the registry data was current. |
| `total_studies_processed` | Trials actually fetched and aggregated. |
| `api_urls` | Every upstream URL called, verbatim, so any result can be reproduced by hand. |
| `query_interpretation` | One-line restatement of how the question was read. |
| `assumptions` | Interpretation choices made, including any chart-type correction. |
| `warnings` | Fetch truncation, multi-valued axes, missing data, client-side filtering, graph pruning. |
| `sorting` | How rows are ordered, so a frontend does not re-sort them. |
| `time_granularity` | `"year"` for time series, else `null`. |

### Errors

| Status | Code | When |
|---|---|---|
| 200 | — | Success. **Also returned for a well-formed question that matched no trials**, as an empty `data` array plus a `meta.warnings` explanation — "no trials match" is a correct answer, and the caller gets the same metadata they would for a populated chart. |
| 422 | `UNSUPPORTED_QUERY` | Not answerable from trial registry data (e.g. "what is the capital of France?"). |
| 422 | — | Request body failed schema validation. |
| 502 | `LLM_ERROR` | The model call failed or was declined. |
| 502 | `UPSTREAM_ERROR` | ClinicalTrials.gov unreachable or erroring after retries. |
| 500 | `VALIDATION_ERROR` | The response failed its grounding checks and was withheld. |

Error bodies are `{"detail": {"error": {"code", "message", "details"}}}`.

---

## Supported questions

Every example below is a real run; outputs are in [`examples/`](examples/), regenerated by `scripts/run_examples.py`.

| # | Question | Type | Output |
|---|---|---|---|
| 1 | "How are lung cancer trials distributed across phases?" | distribution | `bar_chart`, 7 buckets |
| 2 | "How has the number of trials for Pembrolizumab changed per year since 2015?" | time trend | `time_series`, 12 years |
| 3 | "Compare phases for trials involving Pembrolizumab vs Nivolumab." | comparison | `grouped_bar_chart`, 2 series |
| 4 | "Which drugs frequently co-occur in combination studies with pembrolizumab?" | relationship | `network_graph`, 30 nodes / 120 edges |
| 5 | "Show a network of sponsors and drugs for melanoma trials." | relationship | `network_graph`, 40 nodes / 66 edges |
| 6 | "Which countries have the most recruiting trials for melanoma?" | geographic | `geo_bar_chart`, 20 countries |
| 7 | `{"query": "...distributed across sponsors?", "drug_name": "Nivolumab", "phase": 3}` | structured override | `bar_chart`, 15 sponsors |

Also handled: "What are the most common intervention types for melanoma trials?" (distribution over intervention type), "Compare sponsor categories across melanoma and lung cancer" (condition-vs-condition comparison), and enrollment-size histograms.

Available group-by axes: `phase`, `start_year`, `status`, `sponsor`, `sponsor_class`, `intervention_type`, `country`, `enrollment_bucket`.

---

## The single aggregation abstraction

Bar charts, time series, grouped comparisons, geographic breakdowns, and histograms are all the same operation — *group trials by a dimension, optionally split into series, count distinct trials per bucket* — so they are one function, not five.

```python
def aggregate(
    records: Mapping[str, dict],                     # nct_id -> raw study record
    dimension: Dimension,                            # how to derive bucket keys
    *,
    series_membership: Mapping[str, AbstractSet[str]] | None = None,
    include_unknown: bool = True,
    top_n: int | None = None,
) -> AggregationResult
```

The half that varies is a `Dimension` in a registry (`app/services/dimensions.py`):

```python
@dataclass(frozen=True)
class Dimension:
    name: str
    api_fields: tuple[str, ...]                 # PascalCase names for the `fields` param
    extract: Callable[[dict], list[str]]        # record -> 0..n bucket keys
    display: Callable[[str], str]               # "PHASE3" -> "Phase 3"
    order: Callable[[str], Any]                 # clinical order, year order, or by count
    axis_label: str
    unknown_label: str
    multi_valued: bool
    sorting: str
    field_type: str
```

How each visualization maps onto the one function:

| Visualization | Call |
|---|---|
| `bar_chart` (trials by phase) | `aggregate(records, DIMENSIONS["phase"])` |
| `time_series` (trials per year) | `aggregate(records, DIMENSIONS["start_year"])` then `zero_fill_years(...)` |
| `grouped_bar_chart` (Drug A vs B) | one upstream search per entity → `series_membership={"A": ids_a, "B": ids_b}` → same call |
| `geo_bar_chart` (trials by country) | `aggregate(records, DIMENSIONS["country"], top_n=20)` |
| `histogram` (enrollment size) | `aggregate(records, DIMENSIONS["enrollment_bucket"])` |

Adding a new axis means adding a `Dimension` and nothing else — no new aggregation code, no new formatting code, no route change.

`extract` returns a *list* because several dimensions are genuinely multi-valued in the source data: a combined Phase 1/2 trial has two phases, a multinational trial has several countries. `[]` means the record has no value, which becomes an explicit bucket rather than a silent drop.

Only relationship queries take a different path, because nodes and edges are genuinely not buckets — not because the logic forked.

---

## Network graphs

Two kinds, both built from the same fetched records, both fully cited.

### `drug_drug` — co-occurrence

*"Which drugs frequently co-occur in combination studies with pembrolizumab?"*

- **Node** = a normalized drug or biological intervention. `size` = number of distinct trials containing it.
- **Edge** = an unordered pair appearing in the **same trial**. `weight` = number of shared trials, and `nct_ids` is exactly those trials — so every edge is individually checkable.

Real output from example 4 (600 pembrolizumab trials):

| Edge | Weight |
|---|---|
| carboplatin — pembrolizumab | 49 |
| cisplatin — pembrolizumab | 39 |
| paclitaxel — pembrolizumab | 35 |
| pembrolizumab — pemetrexed | 26 |
| carboplatin — paclitaxel | 25 |

These are the actual standard-of-care chemo-immunotherapy regimens in non-small-cell lung cancer, which is a useful sanity check that the graph is measuring something real. Opening the top edge's first citation, [NCT02578680](https://clinicaltrials.gov/study/NCT02578680), shows *"Pembrolizumab 200 mg, Cisplatin, Carboplatin, Pemetrexed…"* in its intervention list.

### `sponsor_drug` — bipartite

*"Show a network of sponsors and drugs for melanoma trials."*

- **Left nodes** = lead sponsors, **right nodes** = drugs, `kind` distinguishes them so a frontend can two-tone the graph.
- **Edge** = this sponsor runs trials testing this drug; `weight` = how many.

Pruning keeps the busiest sponsors first and *then* the drugs those sponsors actually study, so the surviving graph stays connected instead of being two independent top-N slices.

### Design choices

- **Intervention names are normalized.** Registry names are free text, so the same agent appears as `Pembrolizumab`, `Pembrolizumab 200 mg`, `Pembrolizumab (MK-3475)`, and `pembrolizumab IV`. Without normalization each is its own node and the graph fragments into near-duplicates. Parentheticals, dosages, and administration routes are stripped.
- **Brand and generic names are merged via RxNorm.** String normalization cannot collapse `Keytruda` and `pembrolizumab`, which share no characters, so the trials naming each were counted as evidence for two different compounds. Each cleaned name is resolved to its RxNorm **ingredient** and that ingredient becomes the node identity. See [Drug synonym resolution](#drug-synonym-resolution) below.
- **Placebo and standard-of-care are excluded.** They appear in a large fraction of trials and carry no relationship information; keeping them makes every graph a star centred on "Placebo".
- **Only `DRUG`, `BIOLOGICAL`, and `COMBINATION_PRODUCT` interventions become nodes.** "Surgery" and "Counseling" are not drugs.
- **Edges need ≥2 shared trials by default.** A single co-occurrence is overwhelmingly incidental rather than a studied combination.
- **Isolated nodes are dropped**, and pruning never leaves a dangling edge. Truncation is always reported via `truncated_to_top_n` and a `meta.warnings` entry — a graph silently showing the top 30 of 400 drugs would read as the whole picture.

### Drug synonym resolution

Node identity comes from [RxNorm](https://rxnav.nlm.nih.gov/) (NLM, no API key). RxNorm over ChEMBL because it is maintained by the same organization as ClinicalTrials.gov and is built around *clinical drug identity* — "is this the same medicine a patient receives" — rather than general chemistry, which is exactly the question a co-occurrence node asks.

**Resolution walks to the ingredient, not the matched concept.** Brand and generic are *distinct* RxNorm concepts — Keytruda is RxCUI 1547550, pembrolizumab is 1547545 — and asking RxNorm for Keytruda's preferred name returns `"Keytruda"`. Matching alone therefore does not merge them. Each name goes through two calls:

1. `approximateTerm` → a concept plus a confidence score
2. `related.json?tty=IN` → that concept's **ingredient**

Step 2 is idempotent (an ingredient relates to itself), so brand and generic input converge through one code path, and the response carries the canonical name — no third call needed.

**Ordering matters: resolution happens before the graph is built, not after pruning.** Merging changes node size and edge weight, and those are what pruning ranks on. Resolved afterwards, two fragments of one compound could each miss a node cap their merged form would clear, and a weight-1 `Keytruda—carboplatin` edge would be discarded before it could join the `pembrolizumab—carboplatin` edge it belongs to. What *is* deferred is the choice of *which* names to resolve: provisional sizes are computed from string normalization alone (no API calls), and only the top candidates are sent to RxNorm — on a 600-trial query that is 87 lookups instead of 628, **86% avoided**.

**The confidence threshold is 11.0**, and the score scale is unbounded (~0–15), not 0–1. Measured against the live API, every correct match scored ≥ 11.49 while the worst false positive scored 6.38 (`MK-3475` → an unrelated concept; `study drug` → a hand sanitizer gel at 2.63). The threshold sits inside that empty band and is deliberately strict, because the costs are asymmetric: **a wrong merge fuses two compounds into one node and is invisible in the output**, while a missed merge only leaves the pre-RxNorm behavior.

**A high score is not sufficient, which live data proved.** RxNorm resolves multi-drug strings confidently to one component and silently discards the rest — `"placebo for pembrolizumab"` → pembrolizumab (a *control arm* counted as the active drug), `"favezelimab pembrolizumab"` → pembrolizumab (moving favezelimab's trial onto pembrolizumab's node), `"ipilimumab pembrolizumab durvalumab idarubicin bevacizumab"` → durvalumab. All score 12–14. A name is therefore only sent to RxNorm when it plausibly denotes exactly one agent: no multi-agent connective (`and`, `or`, `with`, `+`, `/`), no comparator or placebo wording, not a bare research code, and at most one token that is not a known formulation qualifier — so `nab paclitaxel` and `doxorubicin hydrochloride` resolve while `gemcitabine nab-paclitaxel` does not.

**Failure is always soft.** A timeout, a 5xx, an unknown name, or a combination product with no single ingredient all yield an *unresolved* result carrying the cleaned name; the graph degrades to string normalization and says so in `meta.warnings`. `RXNORM_ENABLED=false` disables resolution entirely, producing output byte-identical to the pre-RxNorm behavior.

**Every merge is auditable.** A merged node carries `rxcui` and `merged_from` (the source names folded into it), and its `nct_ids` is the set *union* of every contributing name's trials — so `size`, edge weights, citations, and `total_supporting_trials` are all computed over that union rather than reconciled afterwards. In the live example, one node's citations span both a Keytruda trial and a Pembrolizumab trial.

---

## Deep citations

Citations are **not** assembled by searching for supporting evidence after the numbers exist. Bucket membership *is* the evidence: the aggregator already knows exactly which trials put a bar at the height it is, and the citation builder renders those. That ordering is what makes it impossible for a citation to disagree with the value it supports — both are projections of the same set of NCT ids.

```
fetch      -> raw records kept in StudyStore for the life of the request
aggregate  -> each bucket/node/edge carries the nct_ids of every contributor
format     -> build_citations(nct_ids, store, dimension, limit) renders excerpts
validate   -> every cited nct_id must exist in the store
```

**Which contributors get cited is an evenly-spaced sample, not the first few.** Contributors arrive sorted ascending and NCT ids are assigned roughly chronologically, so taking the first `limit` cited every bucket's oldest members — a 373-trial Phase 2 bucket was evidenced by three 1990s studies, which is deterministic but reads as cherry-picked from one end. A systematic sample across the sorted list keeps every property the design relies on: same contributors always give the same citations, no dependence on an optionally-missing field like enrolment (whose absence would make the choice non-deterministic), no bias toward either end, and identical behaviour when `limit >= n`. The effect on a real bucket:

```
Phase 2 (373 trials)   before  NCT00001499, NCT00002465, NCT00003154
                       after   NCT00001499, NCT03396185, NCT07751042
```

An excerpt quotes the **exact response field** that caused the membership, alongside the trial's brief title and the field's path:

```
"Paclitaxel With or Without Carboplatin in Treating Patients With Advanced
 Non-small Cell Lung Cancer" — designModule.phases: [PHASE3]
```

Including the path (`designModule.phases`) means a reader can verify the claim against the raw API response without guessing which field was used. For a network edge the evidence is the intervention list, which shows both endpoints.

No LLM is involved, and no extra API calls are made — every field quoted was already fetched by the search. Trials in an "unknown" bucket get an honest `not reported` excerpt, because being absent is precisely why they are in that bucket.

---

## Validation

`app/services/validate.py` runs before any response is returned. Every check has a corresponding test in `tests/test_validate.py` that corrupts one property of a known-good response and asserts the check fires.

| Check | Catches |
|---|---|
| **Schema conformance** | A response that resembles the contract but does not re-parse as it. |
| **Encoding ↔ data** | An encoding naming a field absent from the rows — which renders as an empty axis and looks like "no data" rather than a bug. |
| **Count integrity** | `value != len(set(nct_ids))` for any datum, node, or edge; published rows that disagree with the aggregation they came from; **published network node sizes and edge weights that disagree with the graph they were built from**; an edge referencing a node not in the graph. |
| **Citation grounding** | A citation pointing at a trial that was never fetched; **a citation naming a real trial that is not in that specific datum's contributor set**; an empty excerpt; a non-zero value with no citations (unless zero citations were requested); more citations than claimed supporters. |
| **Non-empty** | A chart with zero rows, or a network with no nodes, shipping as if it were an answer. |
| **Meta integrity** | A processed-trial count that disagrees with the store; missing upstream URLs (without which a result cannot be reproduced). |

Failures raise `ValidationFailure` → HTTP 500 with a diagnostic. The response is withheld rather than degraded.

Upstream, stage 1 applies its own constraints: entity grounding, phase range-checking, and chart-type reconciliation.

---

## Logging

Enough to answer "which stage was slow" or "why did that request 500" from the log alone, when running or debugging the service locally. It is standard-library `logging` only — no OpenTelemetry, no external service, and **not a claim of production observability infrastructure**.

Output is **JSON lines on stdout**, one object per event, with the level set by `LOG_LEVEL` (default `INFO`; use `DEBUG` for per-page and per-name detail). Messages are short constants and all variable data lives in structured fields, so lines stay both greppable and machine-readable:

```json
{"time":"2026-08-08T23:41:02Z","level":"INFO","name":"app.services.ctgov","message":"ctgov search completed",
 "request_id":"dad3a22c","records":200,"pages":1,"total_count":2922,"truncated":true,"duration_ms":292.1}
```

Every line emitted while handling one request carries the same **`request_id`**, so concurrent requests can be told apart.

| Stage | Level | Message | Key fields |
|---|---|---|---|
| Request received | INFO | `query received` | `query` (truncated), `overrides`, `max_studies` |
| LLM | INFO | `llm call started` / `llm call completed` | `model`, `query_type`, `viz_type`, `entity_counts`, `duration_ms` |
| LLM failure | **WARNING** | `llm call failed` / `llm refused request` | `model`, `error_type`, `duration_ms` |
| Fetch | INFO | `ctgov search started` / `ctgov search completed` | `params`, `records`, `pages`, `total_count`, `truncated`, `duration_ms` |
| Fetch retry | **WARNING** | `ctgov request retrying` | `attempt`, `max_retries`, `reason`, `delay_s` |
| Aggregation | INFO | `aggregation completed` | `dimension`, `buckets`, `total_studies_matched`, `unbucketed` |
| Network | INFO | `network built` | `kind`, `nodes`, `edges`, `merged_nodes`, `truncated_to_top_n` |
| Drug resolution | INFO | `drug resolution started` / `drug resolution completed` | `distinct_names`, `candidates`, `resolved`, `cache_hits`, `live_lookups`, `duration_ms` |
| Resolution degraded | **WARNING** | `drug resolution degraded` | `failed`, `of` |
| Validation | INFO | `validation passed` | `viz_type`, `rows`, `citations` |
| Request failed | **ERROR** | `request failed` | `code`, `stage`, `query`, `filters`, `detail` |
| Request finished | INFO / **ERROR** | `request completed` | `method`, `path`, `status`, `duration_ms` |

A healthy network query reads as a single legible trace:

```
INFO  dad3a22c  query received              {"query":"Which drugs co-occur ...","overrides":{}}
INFO  dad3a22c  llm call completed          {"query_type":"relationship","viz_type":"network_graph","duration_ms":1402.7}
INFO  dad3a22c  ctgov search completed      {"records":200,"pages":1,"total_count":2922,"truncated":true,"duration_ms":292.1}
INFO  dad3a22c  drug resolution completed   {"resolved":41,"live_lookups":119,"cache_hits":0,"duration_ms":2498.3}
INFO  dad3a22c  network built               {"nodes":28,"edges":58,"merged_nodes":9}
INFO  dad3a22c  validation passed           {"viz_type":"network_graph","rows":1,"citations":221}
INFO  dad3a22c  request completed           {"status":200,"duration_ms":5183.6}
```

### What is deliberately not logged

Prompts, completions, raw API responses, and trial records never reach the log — only counts, ids, durations, and query text truncated to 200 characters. Every structured payload is an explicit allowlist of fields rather than a serialized object, so config can never be dumped and a credential cannot leak by accident. Tests assert both: that no emitted line contains a configured API key, and that none contains a raw record.

### Three judgment calls

**A validation failure is logged once, not twice.** The validator logs only the pass; on failure it raises and the route handler emits the single ERROR carrying both the failing check and the query. Logging in both places would make one fault read as two.

**RxNorm retries are DEBUG, not WARNING.** Resolution fans out over hundreds of drug names, so a single outage emitted **116 near-identical WARNING lines** in testing — enough to bury everything else. The per-retry and per-name detail stays available at `DEBUG`, and the batch warns exactly once (`drug resolution degraded`) with the failure count. ClinicalTrials.gov retries *do* log at WARNING, because there are at most a handful per request.

**There is no LLM retry line.** The OpenAI SDK retries internally and exposes no per-attempt callback, so a retry log there would be invented rather than observed. Only the final failure is logged.

---

## Working with the ClinicalTrials.gov API

Findings from live verification against API v2.0.5 (see [`docs/api-notes.md`](docs/api-notes.md)). Several are places where widely-circulated third-party documentation is wrong.

**`filter.phase` does not exist.** It returns HTTP 400 `{"error": "unknown parameter"}`, despite being documented as valid by multiple third-party sources. Phase filtering goes through `aggFilters=phase:3`.

**The phase value is a bare number, and the enum form fails silently.** `aggFilters=phase:PHASE3` returns **HTTP 200 with zero results** — indistinguishable from "no matching trials" unless you already know the answer. This is the most dangerous behaviour in the API, so the code makes the wrong form unrepresentable:

```python
AggFilter.phase(3).render()   # "phase:3"
AggFilter.phase(5)            # ValueError
```

There is no code path that emits a raw filter string.

**A 200 response is not proof a filter worked.** When a filtered search returns nothing, the client re-asks without the filter and reports the unfiltered count:

> "No trials matched the filter (phase:4), though 2,922 trials matched the same search without it."

That both catches a silent-filter regression and gives the user a useful answer instead of a bare zero.

**`OR` is a real set union inside `query.*`; a comma is an intersection.** Both are silent, so the only way to distinguish them is to count. Pembrolizumab returns 2,922 trials and Nivolumab 2,016; `Pembrolizumab OR Nivolumab` returns 4,648 and `Pembrolizumab, Nivolumab` returns 290 — and `2922 + 2016 − 290 = 4648` closes exactly. This is why every extracted value reaches the query instead of only the first: asking about "pembrolizumab and nivolumab" searches for both, and the union reading is stated in `meta.assumptions` rather than applied silently. The comma form is never emitted — it reads as a list and behaves as an `AND`. Comparison queries are unaffected: each compared entity keeps its own search, which is what makes per-series membership exact. The join is capped at five values, and truncating past that is disclosed as a warning.

**Request field names do not map mechanically to response paths.** `Phase` → `designModule.phases`, `StartDate` → `statusModule.startDateStruct.date`. Every mapping used here was confirmed against a live response and is pinned by `tests/test_dimensions.py`, which runs every extractor against a captured API page and fails if any stops finding values.

**Other verified behaviour:** `pageSize` defaults to 10 if omitted (max 1000) so it is always set explicitly; pagination is `pageToken` only, with no offset; `phases` is always an array; date granularity varies (`2024-08-22`, `2024-08`, `2024` all appear); `GET /stats/field/values` returns 500.

**Server-side aggregation is deliberately not used.** Beyond being unavailable, it could not support this design even if it worked: aggregating server-side means the underlying records never pass through our hands, so there would be nothing to cite. Client-side aggregation is a requirement of the citation architecture, not a workaround.

---

## Design decisions and tradeoffs

**Plain OpenAI SDK rather than LangChain/LangGraph.** The scaffold started with LangChain. There is exactly one LLM call in this service, with a fixed schema and no tool use, chaining, or agent loop — a framework would add a dependency layer without removing any code. `client.chat.completions.parse(response_format=QueryUnderstanding)` gives a schema-guaranteed Pydantic object directly, so there is no JSON repair or retry-on-parse-failure logic anywhere.

**`gpt-5.4-mini` by default.** Intent classification and entity extraction is an easy task for a modern model; all seven query classes classified correctly on the first attempt in testing. Configurable via `LLM_MODEL` (`gpt-4.1-mini` is roughly twice as fast for a small quality tradeoff).

**Multi-phase trials are counted in every phase they belong to.** A Phase 1/2 trial appears in both buckets, so bucket totals can exceed the trial count. The alternative — a combined "Phase 1/2" bucket — makes phases non-comparable. The behaviour is disclosed in `meta.warnings`, and the validator checks a set-union invariant rather than a naive sum.

**Trials with no phase get their own visible bucket.** About a quarter of studies have no usable phase. In example 1 that is 228 of 1,000 trials — hiding them would materially misrepresent the distribution. The one exception is the time-series axis, where a "no start date" category cannot be placed on a temporal scale; those trials are excluded from the chart and reported in `meta.warnings` instead.

**Fetch is capped and the cap is always disclosed.** Broad queries match tens of thousands of trials ("lung cancer" matches 14,425). Fetching everything is slow and mostly unnecessary for a distribution, so `max_studies` defaults to 3,000 and every truncated response says so: *"Fetched 1,000 of 14,425 matching trials… figures below describe that sample, not the full result set."* An accurate statement about a sample beats an implied claim about a population.

**A trial in both comparison series is counted in both.** It genuinely is evidence for both, and series membership comes from the upstream searches themselves rather than client-side guessing.

**Year ranges, most status filters, and multi-phase requests are applied client-side.** There is no live-verified date-range parameter, and only `status:rec` was confirmed among the status codes. `aggFilters` also carries only *one* phase and repeating the key does not OR them, so a single phase is filtered upstream while two or more are fetched unfiltered and OR-ed locally — matching by intersection, since a combined Phase 1/2 trial genuinely satisfies a request for either. Sending an unverified filter risks a silently-empty result; filtering locally is exact, keeps citations intact, and is disclosed in `meta.warnings`.

<a id="status-negation-is-detected-but-not-filterable"></a>
**Status negation is detected, but not filterable — and that is disclosed.** `"trials that are not recruiting"` used to send `aggFilters=status:rec` upstream, returning precisely the opposite set, because `"recruiting"` matched as a substring. A negation cue immediately before a status now records it as an *exclusion*. Nothing in a ClinicalTrials.gov search expresses "every status except X", and enumerating the known statuses instead would be wrong — a live 1,000-trial sample returned `APPROVED_FOR_MARKETING`, which is not in the synonym table at all, so an enumerated exclusion would silently drop those trials. So a negated status applies **no** status filter and says so in `meta.warnings`. An honest superset with a disclosure beats a confidently inverted answer.

`"active, not recruiting"` and `"trials that are not recruiting"` are deliberately *not* treated as the same request: the first is the specific `ACTIVE_NOT_RECRUITING` status and is filterable; the second is an exclusion of `RECRUITING` and is not. The negation check runs after longer phrases are matched and masked, so the `"not"` belonging to `ACTIVE_NOT_RECRUITING` or `NOT_YET_RECRUITING` can never trigger it, and a clause boundary stops it spreading — in `"not completed and recruiting"`, only `COMPLETED` is excluded.

**Brand and generic drug names are merged, conservatively.** Node identity comes from RxNorm ingredients, so `Keytruda` and `pembrolizumab` are one node (see [Drug synonym resolution](#drug-synonym-resolution)). The filter is deliberately strict rather than maximal: on a 600-trial pembrolizumab query, **22% of the 628 distinct drug names resolve, producing 15 merges, all clinically correct** (`keytruda`/`pembrolizumab`, `opdivo`/`nivolumab`, `xeloda`/`capecitabine`, `abraxane`/`nab-paclitaxel`, `5-FU`/`fluorouracil`, `aldesleukin`/`recombinant human interleukin-2`) **with zero false merges**. The unresolved 78% is mostly research codes (`ACE2016`, `MK-3475`) and genuine multi-drug arms that *should not* merge. An earlier, looser filter resolved 34% but produced merges that folded placebo arms and combination partners into the active drug — so the lower rate is the feature working, not a shortfall.

**Excerpts quote fields already fetched.** Prose spans from full trial text would need a `GET /studies/{nctId}` call per cited trial. The spec explicitly allows "a specific field/value", and the field-level excerpt is more verifiable anyway — it names the exact path a reader should check.

**Empty results are a 200, not a 404.** "No trials match your question" is a correct answer to a well-formed question, and the caller still gets the filters applied, URLs called, and warnings.

**Chart titles are templated, not generated.** A title is a claim about what the reader is looking at. Templating it from the plan's own entities means it cannot describe something other than what the data shows.

---

## Limitations and what I would do with more time

- **The unresolved drug-name tail.** RxNorm resolution is conservative by design, so roughly three quarters of distinct names stay unmerged — research-stage compounds RxNorm has no entry for, and multi-drug arm descriptions that must not be collapsed onto one component. Some near-duplicate nodes therefore remain. Closing more of that gap needs per-token drug identification rather than a looser threshold, which would trade invisible false merges for visible ones.
- **The RxNorm cache is process-local.** It is a bounded in-memory singleton, so each worker warms up independently and nothing is shared across processes. Fine at this scale; a shared cache would matter under real concurrency.
- **Resolution is limited to a candidate pool.** Only the most-mentioned names are sent to RxNorm, so a merge involving two rarely-named variants can be missed. The bound is disclosed and the failure is conservative (no merge), never a wrong one.
- **Full-text excerpts for cited trials.** Deep-fetch complete records for only the ≤3 cited trials per datum — bounded cost, since the citation limit caps it — and quote a prose span from the detailed description rather than a field value.
- **Smarter sampling under the fetch cap.** Currently the first N trials the API returns are used. For a time series that biases toward whatever the API's default ordering favours. Stratified sampling by year, or using `countTotal` per year bucket to scale a sample up to a population estimate (clearly labelled as an estimate), would be more honest for very broad queries.
- **Response caching.** Identical queries re-fetch from scratch. A short-lived cache keyed on the normalized search parameters would cut both latency and upstream load; the `data_as_of` timestamp is already tracked and would make invalidation straightforward.
- **Streaming progress.** A 3,000-trial fetch takes several seconds. Server-sent events reporting "fetched 1,000 / 2,922" would make the wait legible.
- **More visualization types.** Scatter plots (enrollment vs duration) and Sankey diagrams (phase progression) both fit the existing dimension registry; neither was reachable in the time box.
- **Verify the remaining `aggFilters` codes.** Status codes beyond `rec`, and whether `filter.overallStatus` still exists, are open questions in `docs/api-notes.md`. Confirming them would let more filtering move upstream and reduce the number of records fetched.
- **A minimal frontend.** The response format is designed to be rendered without guessing; a small React + d3 demo would prove that end to end rather than asserting it.

---

## Testing

```bash
pytest -q      # 365 tests, ~9s, no network access or API key required
```

| File | Covers |
|---|---|
| `test_dimensions.py` | Every extractor, including against an unmodified live API page — this is the guard against upstream field changes. |
| `test_aggregate.py` | The core invariant (`value == len(set(nct_ids))`) across every dimension, multi-phase double counting, unknown buckets, zero-fill, series splitting, top-N, determinism. |
| `test_ctgov.py` | Parameter construction, `pageToken` pagination, fetch caps, 429/5xx retry, non-retry on 400, the empty-filtered-result probe, and the pure query builder. |
| `test_citations.py` | Excerpt construction, truncation reporting, and that citations track the aggregation exactly. |
| `test_network.py` | Name normalization, placebo exclusion, edge weights, pruning without dangling edges, bipartiteness, RxNorm merging (union of `nct_ids`, edge weights summing across merged names, `resolutions=None` reproducing prior output), and independent re-verification of edges against live data. |
| `test_drug_resolver.py` | Confidence-threshold boundary, the ingredient walk (brand → generic, generic → itself), the multi-drug and control-arm pre-filters, cache hit/miss/eviction and negative caching, and every RxNorm failure mode degrading rather than raising. Includes a captured-live-response guard asserting Keytruda and pembrolizumab reach the same RxCUI. |
| `test_validate.py` | Each check driven by a deliberately corrupted response. |
| `test_understanding.py` | Entity grounding, hallucination rejection, chart-type reconciliation, structured overrides, defaults. |
| `test_pipeline.py` | Full pipeline with LLM and HTTP mocked, plus route-level error mapping. |
| `test_logging.py` | Formatter output including `extra` fields and `request_id`, one line per stage, failure paths emitting WARNING/ERROR, and guards that no log line carries a credential, a raw trial record, or a `LogRecord`-reserved key. |

Correctness was validated three ways: unit tests on fixtures; extractors and network builders run against a **real captured API response** so a response-shape change fails loudly; and every documented example executed against the **live API**, with outputs checked for clinical plausibility (the pembrolizumab time series tracks its 2014 approval; the co-occurrence graph reproduces known NSCLC regimens).

---

## AI tool usage

Per the assignment's integrity note.

**Tools used.** Claude Code (Claude Opus 5) throughout, for both design and implementation.

**Designed deliberately.** The pipeline decomposition and the rule that the LLM never emits a data value; the three-mechanism defence (schema shape, entity grounding, validation); the single-`aggregate()`-plus-dimension-registry approach; the citation architecture that carries `nct_id` sets from bucketing so citations cannot drift from values; the network node/edge/weight semantics and pruning rules; the decision to reject server-side aggregation as citation-incompatible; and the disclosure policy (multi-valued axes, fetch caps, client-side filters, and graph truncation are all stated in `meta.warnings` rather than hidden).

**Generated and then adapted.** Module scaffolding, Pydantic model boilerplate, and the first draft of most test cases. Everything was reviewed and revised — notably the fetcher's empty-filtered-result handling, which was originally specified to raise an error but would have false-positived on the legitimate case of "this drug genuinely has no Phase 4 trials"; it became a diagnostic warning that reports the unfiltered count instead.

**Validated by hand.** Every ClinicalTrials.gov and RxNorm API claim was verified with live calls rather than trusted from documentation or model knowledge. That is how the nonexistent `filter.phase` parameter and the silently-failing `phase:PHASE3` form were caught; it is also how two RxNorm design errors were caught before they shipped. First, resolving a name to its matched concept does *not* merge brand into generic (Keytruda and pembrolizumab are separate RxCUIs), which forced the ingredient-walk design. Second, running resolution against 600 real trials — rather than only against fixtures — exposed that RxNorm resolves multi-drug strings confidently to one component, so `"placebo for pembrolizumab"` and `"favezelimab pembrolizumab"` were both merging into pembrolizumab. Fixtures alone would not have surfaced either. All example outputs were inspected for clinical plausibility, and cited NCT ids were spot-checked against the live registry.
