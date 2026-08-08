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
- [How it works](#how-it-works)
- [Why the LLM cannot fabricate a number](#why-the-llm-cannot-fabricate-a-number)
- [Request schema](#request-schema)
- [Response schema](#response-schema)
- [Supported questions](#supported-questions)
- [The single aggregation abstraction](#the-single-aggregation-abstraction)
- [Network graphs](#network-graphs)
- [Deep citations](#deep-citations)
- [Validation](#validation)
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
pytest -q                              # 166 tests, no network or API key needed
python scripts/run_examples.py         # writes examples/*.json from live data
```

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

**2. Grounding — extracted entities must appear in the question.** The one way the model could still influence real data is by naming an entity nobody asked about, which would silently change what gets searched. After the call, `ground_entities()` drops any drug, condition, sponsor, or country that does not appear in the user's text (exact substring, with a tight fuzzy fallback for inflection like "lung cancers"). Dropped entities are reported in `meta.warnings`.

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
  "nodes": {"id": "id", "label": "label", "size": "size", "group": "kind"},
  "edges": {"source": "source", "target": "target", "weight": "weight"}
}
```

`data` for a network is a single-element array holding both collections: `[{"nodes": [...], "edges": [...]}]`. This keeps `data` a `list[dict]` for every chart type, so one response model covers all of them. A frontend can wire the key maps straight into d3-force or cytoscape.

**Every data row, node, and edge carries:**

| Field | Type | Description |
|---|---|---|
| `citations` | array | Up to `max_citations_per_datum` entries of `{nct_id, excerpt, url}`. |
| `total_supporting_trials` | int | The **full** number of contributing trials, so a truncated citation list never reads as the complete evidence set. |

### `meta`

| Field | Description |
|---|---|
| `filters` | The filters actually applied upstream. |
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
- **Placebo and standard-of-care are excluded.** They appear in a large fraction of trials and carry no relationship information; keeping them makes every graph a star centred on "Placebo".
- **Only `DRUG`, `BIOLOGICAL`, and `COMBINATION_PRODUCT` interventions become nodes.** "Surgery" and "Counseling" are not drugs.
- **Edges need ≥2 shared trials by default.** A single co-occurrence is overwhelmingly incidental rather than a studied combination.
- **Isolated nodes are dropped**, and pruning never leaves a dangling edge. Truncation is always reported via `truncated_to_top_n` and a `meta.warnings` entry — a graph silently showing the top 30 of 400 drugs would read as the whole picture.

---

## Deep citations

Citations are **not** assembled by searching for supporting evidence after the numbers exist. Bucket membership *is* the evidence: the aggregator already knows exactly which trials put a bar at the height it is, and the citation builder renders those. That ordering is what makes it impossible for a citation to disagree with the value it supports — both are projections of the same set of NCT ids.

```
fetch      -> raw records kept in StudyStore for the life of the request
aggregate  -> each bucket/node/edge carries the nct_ids of every contributor
format     -> build_citations(nct_ids, store, dimension, limit) renders excerpts
validate   -> every cited nct_id must exist in the store
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
| **Count integrity** | `value != len(set(nct_ids))` for any datum, node, or edge; published rows that disagree with the aggregation they came from; an edge referencing a node not in the graph. |
| **Citation grounding** | A citation pointing at a trial that was never fetched; an empty excerpt; a non-zero value with no citations; more citations than claimed supporters. |
| **Non-empty** | A chart with zero rows, or a network with no nodes, shipping as if it were an answer. |
| **Meta integrity** | A processed-trial count that disagrees with the store; missing upstream URLs (without which a result cannot be reproduced). |

Failures raise `ValidationFailure` → HTTP 500 with a diagnostic. The response is withheld rather than degraded.

Upstream, stage 1 applies its own constraints: entity grounding, phase range-checking, and chart-type reconciliation.

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

**Year ranges and most status filters are applied client-side.** There is no live-verified date-range parameter, and only `status:rec` was confirmed among the status codes. Sending an unverified filter risks a silently-empty result; filtering locally is exact, keeps citations intact, and is disclosed in `meta.warnings`.

**Brand and generic drug names are not merged.** Mapping Keytruda → pembrolizumab needs a drug vocabulary (RxNorm or similar) that this service does not have. Guessing would silently fuse distinct agents, which is a worse failure than leaving them separate. Documented rather than hidden.

**Excerpts quote fields already fetched.** Prose spans from full trial text would need a `GET /studies/{nctId}` call per cited trial. The spec explicitly allows "a specific field/value", and the field-level excerpt is more verifiable anyway — it names the exact path a reader should check.

**Empty results are a 200, not a 404.** "No trials match your question" is a correct answer to a well-formed question, and the caller still gets the filters applied, URLs called, and warnings.

**Chart titles are templated, not generated.** A title is a claim about what the reader is looking at. Templating it from the plan's own entities means it cannot describe something other than what the data shows.

---

## Limitations and what I would do with more time

- **Drug synonym resolution.** Integrate RxNorm or the ChEMBL API so brand and generic names collapse to one node, and so `query.intr=Keytruda` and `query.intr=pembrolizumab` return the same graph. This is the single biggest quality win available for network graphs.
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
pytest -q      # 166 tests, ~3s, no network access or API key required
```

| File | Covers |
|---|---|
| `test_dimensions.py` | Every extractor, including against an unmodified live API page — this is the guard against upstream field changes. |
| `test_aggregate.py` | The core invariant (`value == len(set(nct_ids))`) across every dimension, multi-phase double counting, unknown buckets, zero-fill, series splitting, top-N, determinism. |
| `test_ctgov.py` | Parameter construction, `pageToken` pagination, fetch caps, 429/5xx retry, non-retry on 400, the empty-filtered-result probe, and the pure query builder. |
| `test_citations.py` | Excerpt construction, truncation reporting, and that citations track the aggregation exactly. |
| `test_network.py` | Name normalization, placebo exclusion, edge weights, pruning without dangling edges, bipartiteness, and independent re-verification of edges against live data. |
| `test_validate.py` | Each check driven by a deliberately corrupted response. |
| `test_understanding.py` | Entity grounding, hallucination rejection, chart-type reconciliation, structured overrides, defaults. |
| `test_pipeline.py` | Full pipeline with LLM and HTTP mocked, plus route-level error mapping. |

Correctness was validated three ways: unit tests on fixtures; extractors and network builders run against a **real captured API response** so a response-shape change fails loudly; and every documented example executed against the **live API**, with outputs checked for clinical plausibility (the pembrolizumab time series tracks its 2014 approval; the co-occurrence graph reproduces known NSCLC regimens).

---

## AI tool usage

Per the assignment's integrity note.

**Tools used.** Claude Code (Claude Opus 5) throughout, for both design and implementation.

**Designed deliberately.** The pipeline decomposition and the rule that the LLM never emits a data value; the three-mechanism defence (schema shape, entity grounding, validation); the single-`aggregate()`-plus-dimension-registry approach; the citation architecture that carries `nct_id` sets from bucketing so citations cannot drift from values; the network node/edge/weight semantics and pruning rules; the decision to reject server-side aggregation as citation-incompatible; and the disclosure policy (multi-valued axes, fetch caps, client-side filters, and graph truncation are all stated in `meta.warnings` rather than hidden).

**Generated and then adapted.** Module scaffolding, Pydantic model boilerplate, and the first draft of most test cases. Everything was reviewed and revised — notably the fetcher's empty-filtered-result handling, which was originally specified to raise an error but would have false-positived on the legitimate case of "this drug genuinely has no Phase 4 trials"; it became a diagnostic warning that reports the unfiltered count instead.

**Validated by hand.** Every ClinicalTrials.gov API claim was verified with live `curl` calls rather than trusted from documentation or model knowledge — which is how the nonexistent `filter.phase` parameter and the silently-failing `phase:PHASE3` form were caught. All example outputs were inspected for clinical plausibility, and cited NCT ids were spot-checked against the live registry.
