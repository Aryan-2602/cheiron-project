> **Historical document — not authoritative.**
>
> This is the pre-implementation planning note, kept for the reasoning it
> records. Where it differs from the shipped system, `README.md` and the
> current source and schema definitions are correct. Several decisions here
> were revised once the live API was measured — see `docs/api-notes.md` for
> what was actually verified.

# PLAN.md — ClinicalTrials.gov Query-to-Visualization Agent (Cheiron take-home)

## Context

Take-home for Backend Engineer, Applied Agents at Cheiron (~24h time box). Build a FastAPI backend that turns a natural-language clinical-trials question into a **structured visualization spec** backed by live ClinicalTrials.gov API v2 data. Evaluation: System Design 35%, AI/Agent Design 20%, Code Quality 20%, Query/Viz Coverage 15%, I/O Design 10%, + Deep Citations bonus.

**Core principle (non-negotiable):** the LLM never generates data values. It only classifies intent, extracts entities, and picks a viz type. Every number in the output is computed deterministically by our aggregation code over real API responses, and every datum carries citations (`nct_id` + exact excerpt) to the records that produced it. This is enforced structurally: the LLM stage's output schema *has no field where a data value could go*, and the validator rejects any citation `nct_id` not present in the fetched record store.

**Ground truth for API behavior:** `/docs/api-notes.md` (live-verified 2026-08-08). Where it contradicts anything else — including general knowledge of this API — api-notes.md wins. Key facts baked into this plan: `filter.phase` does not exist (400); phase filtering is `aggFilters=phase:3` with a **bare number** (`phase:PHASE3` silently returns zero rows); `pageSize` max 1000 (default 10 — always set); pagination is `pageToken` only; `countTotal=true` for totals; response keys are camelCase under `protocolSection.<module>Module` and do **not** map 1:1 to the PascalCase `fields` request names; `phases` is always an array (multi-phase trials exist); ~25% of studies have no usable phase; date granularity is inconsistent; `GET /stats/field/values` returned 500 and can't support citations anyway — all aggregation is client-side.

---

## 1) Architecture overview

```
POST /api/v1/query
      │
      ▼
┌─────────────────────┐   LLM (structured output). Intent only: query_type,
│ 1. Query            │   entities, group_by dimension, viz_type. NO data.
│    Understanding    │
└─────────┬───────────┘
          ▼
┌─────────────────────┐   Pure function: QueryPlan -> list[CTGovRequest]
│ 2. API Query        │   (verified params only: query.intr/cond/spons/locn,
│    Builder          │   aggFilters, fields, pageSize, countTotal)
└─────────┬───────────┘
          ▼
┌─────────────────────┐   httpx async client. Pagination, 429 retry/backoff,
│ 3. Fetcher +        │   silent-zero-result guard. Raw records land in a
│    StudyStore       │   per-request StudyStore keyed by nct_id.
└─────────┬───────────┘
          ▼
┌─────────────────────┐   ONE generic group-by-count function + a Dimension
│ 4. Aggregator       │   registry (phase/year/status/sponsor/country/…).
│    (+ network       │   Buckets carry contributing nct_ids. Network builder
│     builder)        │   is a sibling with node/edge output.
└─────────┬───────────┘
          ▼
┌─────────────────────┐   AggregationResult -> assignment-shaped viz spec
│ 5. Viz Spec Builder │   (type/title/encoding/data). Citation builder
│    + Citations      │   attaches excerpts per datum from StudyStore.
└─────────┬───────────┘
          ▼
┌─────────────────────┐   Schema conformance, encoding↔data consistency,
│ 6. Validator        │   count invariants, citation grounding, non-empty
└─────────┬───────────┘   checks. Fails loudly, never silently.
          ▼
   QueryResponse (JSON)
```

**Why separate modules:** each stage has a single typed input and output (Pydantic models), so each is unit-testable without the network or the LLM — the aggregator is tested on fixture records, the query builder on QueryPlan literals, the validator on hand-built responses. It also isolates the only two untrusted boundaries (the LLM and the external API) behind narrow interfaces, so the hallucination-prevention argument is auditable: data values can only originate in stage 4, which is deterministic code. This maps directly to the System Design (35%) and "avoid hallucination-prone steps" criteria.

**File layout** (extends the existing scaffold; keep `app/core/*` as-is):

```
app/
  main.py                      # existing; router already included
  api/routes.py                # POST /api/v1/query, GET /health
  models/schemas.py            # ALL Pydantic models (section 2)
  agents/understanding.py      # stage 1 (LLM)
  services/ctgov.py            # stages 2+3: CTGovClient + build_requests()
  services/store.py            # StudyStore
  services/dimensions.py       # Dimension registry + field extractors
  services/aggregate.py        # aggregate() — the single coherent abstraction
  services/network.py          # network graph builders
  services/viz.py              # viz spec builder
  services/citations.py        # excerpt builder
  services/validate.py         # validator
  pipeline.py                  # run_pipeline(request) -> QueryResponse orchestrator
tests/
  test_understanding.py  test_ctgov.py  test_aggregate.py  test_network.py
  test_citations.py  test_validate.py  test_pipeline_e2e.py  fixtures/
```

**LLM usage (stage 1 only):** use the official `anthropic` SDK directly (drop `langchain`/`langgraph`/`langchain-anthropic` from requirements — one fewer abstraction layer to review; document as a design decision). Call `client.messages.parse(model=settings.LLM_MODEL, max_tokens=2000, output_format=QueryUnderstanding, ...)` — structured outputs guarantee a schema-valid `QueryUnderstanding` with no JSON-repair code. Default `LLM_MODEL=claude-opus-5` (current recommended default; env-configurable — `claude-haiku-4-5` is the cheap swap if cost matters during dev, but that's a per-run choice, not the default). Do **not** pass `temperature` (removed on Opus 5 — sending it 400s). Handle `stop_reason == "refusal"` defensively by returning a 502-style structured error. The system prompt states explicitly: *"You never estimate or output counts, statistics, or data values. You only classify the question and extract entities that appear in it."*

---

## 2) Pydantic schemas (all in `app/models/schemas.py`, Pydantic v2)

### 2.1 Input request

```python
Phase = Literal[1, 2, 3, 4]

class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)          # required NL question
    # optional structured overrides (candidate-defined, per spec 3.1) —
    # when present they OVERRIDE whatever the LLM extracts:
    drug_name: str | None = None
    condition: str | None = None
    sponsor: str | None = None
    phase: Phase | None = None
    country: str | None = None
    start_year: int | None = Field(None, ge=1990, le=2030)
    end_year: int | None = Field(None, ge=1990, le=2030)
    # output controls:
    max_citations_per_datum: int = Field(5, ge=0, le=20)
    max_studies: int = Field(3000, ge=100, le=5000)            # fetch cap, disclosed in meta

    @model_validator(mode="after")  # start_year <= end_year
```

### 2.2 Query understanding output (the ONLY LLM-produced object)

```python
QueryType = Literal["distribution", "time_trend", "comparison",
                    "relationship", "geographic", "unsupported"]
VizType   = Literal["bar_chart", "time_series", "grouped_bar_chart",
                    "network_graph", "geo_bar_chart", "histogram", "scatter"]
GroupByDim = Literal["phase", "start_year", "status", "sponsor",
                     "sponsor_class", "intervention_type", "country"]
NetworkKind = Literal["drug_drug", "sponsor_drug"] 

class YearRange(BaseModel):
    start: int | None; end: int | None

class ExtractedEntities(BaseModel):
    drugs: list[str] = []          # intervention names as they appear in the query
    conditions: list[str] = []
    sponsors: list[str] = []
    phases: list[int] = []         # 1..4
    statuses: list[str] = []       # e.g. "recruiting"
    countries: list[str] = []
    year_range: YearRange | None = None

class QueryUnderstanding(BaseModel):
    query_type: QueryType
    entities: ExtractedEntities
    group_by: GroupByDim | None          # bar/time_series/geo: the x-dimension
    compare_entities: list[str] = []     # comparison only: the series (e.g. ["Drug A","Drug B"])
    compare_entity_kind: Literal["drug", "condition", "sponsor"] | None = None
    network_kind: NetworkKind | None = None   # relationship only
    viz_type: VizType
    assumptions: list[str] = []          # human-readable interpretation notes -> meta.notes
```

Note what's absent: no counts, no data, no labels-with-values. Structurally hallucination-proof. After the LLM call, `agents/understanding.py` runs a deterministic **grounding pass**: every extracted entity string must fuzzy-match a substring of the user query (or a structured field); non-matching entities are dropped and a warning added. Structured request fields are then merged on top (they win).

`QueryUnderstanding` + merged fields become an internal `QueryPlan` (same shape + resolved defaults, e.g. `group_by="phase"` when a distribution query didn't specify).

### 2.3 Shared aggregator output (generic across bar / time_series / comparison / geographic)

```python
class AggregatedDatum(BaseModel):
    key: str                      # bucket label, e.g. "Phase 3", "2019", "United States"
    series: str | None = None     # comparison only, e.g. "Pembrolizumab"; else None
    value: int                    # ALWAYS computed: len(nct_ids)
    nct_ids: list[str]            # every contributing trial (citation source; pre-truncation)

class AggregationResult(BaseModel):
    dimension: str                        # e.g. "phase"
    series_dimension: str | None = None   # e.g. "drug" for comparisons
    data: list[AggregatedDatum]           # sorted per-dimension rule
    total_studies_matched: int            # unique nct_ids across all buckets
    unbucketed: int                       # studies with no value for the dimension
    unbucketed_key_included: bool         # True when an explicit "Unknown" bucket is emitted
```

Invariant the validator checks: for every datum, `value == len(set(nct_ids))`.

### 2.4 Network graph output (separate, because nodes/edges ≠ buckets)

```python
class NetworkNode(BaseModel):
    id: str                                # normalized, e.g. "drug:pembrolizumab"
    label: str                             # display, e.g. "Pembrolizumab"
    kind: Literal["drug", "sponsor", "condition"]
    size: int                              # = len(nct_ids)
    nct_ids: list[str]

class NetworkEdge(BaseModel):
    source: str; target: str               # node ids
    weight: int                            # = len(nct_ids) (co-occurring studies)
    nct_ids: list[str]

class NetworkResult(BaseModel):
    kind: NetworkKind
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    truncated_to_top_n: int | None = None  # disclosed cap (e.g. 30 nodes)
```

### 2.5 Final response (matches the assignment's example JSON, extended with citations)

```python
class Citation(BaseModel):
    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    excerpt: str                  # exact text/field value from the API response

class FieldRef(BaseModel):
    field: str                    # key name present in each data row
    label: str | None = None
    type: Literal["nominal", "ordinal", "quantitative", "temporal"] | None = None

class Encoding(BaseModel):        # chart-family charts
    x: FieldRef | None = None
    y: FieldRef | None = None
    color: FieldRef | None = None       # series, for grouped_bar_chart
    # network_graph instead uses:
    nodes: dict | None = None           # {"id": "id", "label": "label", "size": "size"}
    edges: dict | None = None           # {"source": "source", "target": "target", "weight": "weight"}

class VisualizationSpec(BaseModel):
    type: VizType
    title: str                          # deterministic template, NOT LLM free text
    encoding: Encoding
    data: list[dict[str, Any]]          # rows; keys must cover encoding fields;
                                        # each row also carries:
                                        #   "citations": list[Citation]
                                        #   "total_supporting_trials": int
                                        # network: data = [{"nodes": [...], "edges": [...]}]-style
                                        # → for network we emit data as
                                        #   {"nodes": [node rows], "edges": [edge rows]} (dict, see 4)

class Meta(BaseModel):
    filters: dict[str, str]             # applied filters, e.g. {"drug_name": "Pembrolizumab"}
    source: str = "clinicaltrials.gov"
    data_as_of: str | None              # dataTimestamp from GET /version
    total_studies_processed: int
    api_urls: list[str]                 # exact request URLs (transparency/reproducibility)
    query_interpretation: str           # one-line restatement
    assumptions: list[str]
    warnings: list[str]                 # e.g. "fetch capped at 3000 studies", "12 trials had no phase"
    sorting: str | None = None          # frontend hints per spec 3.2
    time_granularity: str | None = None # "year" for time_series

class QueryResponse(BaseModel):
    visualization: VisualizationSpec
    meta: Meta
```

For network graphs `VisualizationSpec.data` is `[{"nodes": [...node dicts...], "edges": [...edge dicts...]}]` (single row holding both lists) so `data` stays `list[dict]` and one response model covers everything; `encoding.nodes/edges` documents the key mapping. Citations attach per **node and per edge** dict.

Error shape (documented in README): `{"error": {"code": "NO_RESULTS" | "UNSUPPORTED_QUERY" | "UPSTREAM_ERROR" | "LLM_ERROR", "message": str, "details": {...}}}` with appropriate HTTP status (422/502/200-with-empty-explanation — decide: NO_RESULTS returns 200 with an `empty_result` payload + explanation, since "zero trials" is a valid answer).

---

## 3) The single coherent approach: one aggregation abstraction

Everything that is "count trials grouped by X (optionally split by series S)" — bar, time series, comparison, geographic — is **one function + a dimension registry**. No per-chart aggregation code exists anywhere.

```python
# services/dimensions.py
ExtractFn = Callable[[dict], list[str]]   # raw study record -> 0..n bucket keys
                                          # (list because phases/countries are multi-valued;
                                          #  [] means "no value" -> unbucketed/Unknown)

@dataclass(frozen=True)
class Dimension:
    name: str                    # matches GroupByDim literal
    api_fields: tuple[str, ...]  # PascalCase names for the `fields` request param
    extract: ExtractFn           # navigates protocolSection.<module>Module (camelCase!)
    sort_key: Callable[[AggregatedDatum], Any]   # phase order / year asc / value desc
    display: Callable[[str], str]                # "PHASE3" -> "Phase 3"
    emit_unknown_bucket: bool    # True for phase (25% missing — must be visible)

DIMENSIONS: dict[str, Dimension] = {
    "phase":       Dimension(..., api_fields=("Phase",),
                             extract=lambda r: r["protocolSection"]["designModule"].get("phases", []) or []),
    "start_year":  Dimension(..., api_fields=("StartDate",), ...),   # tolerant parse: "2019", "2019-05", "2019-05-01"
    "status":      Dimension(..., api_fields=("OverallStatus",), ...),
    "sponsor":     Dimension(..., api_fields=("LeadSponsorName",), ...),
    "sponsor_class": Dimension(..., api_fields=("LeadSponsorClass",), ...),
    "intervention_type": Dimension(..., api_fields=("InterventionType",), ...),  # multi-valued
    "country":     Dimension(..., api_fields=("LocationCountry",),
                             extract=<unique countries from contactsLocationsModule.locations>),
}
```

(Exact request-name → response-path mapping is confirmed per field during implementation with one live call each — api-notes.md warns it is not a mechanical case conversion. Extractors use `.get()` chains and never KeyError on sparse records.)

```python
# services/aggregate.py — THE function
def aggregate(
    records: Mapping[str, dict],                    # StudyStore.records: nct_id -> raw record
    dimension: Dimension,
    series_membership: Mapping[str, AbstractSet[str]] | None = None,
        # series label -> set of nct_ids belonging to that series
        # (built by the pipeline from per-entity fetch result sets)
    unknown_label: str = "No phase data",           # per-dimension override
) -> AggregationResult:
    # for each (series or the single implicit series):
    #   for nct_id in membership: keys = dimension.extract(record) or [UNKNOWN]
    #     bucket[(series, key)].add(nct_id)
    # datum.value = len(bucket set); sort by dimension.sort_key
```

How each viz maps onto it (this table goes in the README too):

| Viz | Call |
|---|---|
| `bar_chart` (trials by phase) | `aggregate(store.records, DIMENSIONS["phase"])` |
| `time_series` (trials per year) | `aggregate(store.records, DIMENSIONS["start_year"])` + zero-fill missing years between min/max |
| `grouped_bar_chart` (Drug A vs B) | one fetch per compared entity → `series_membership={"Drug A": ids_a, "Drug B": ids_b}` → same call |
| `geo_bar_chart` (recruiting trials by country) | `aggregate(store.records, DIMENSIONS["country"])` (recruiting via `aggFilters=status:rec`, the one verified status code) |
| `histogram` (enrollment sizes) — stretch | same function with a binning `extract` (`lambda r: [bin_label(enrollment)]`) |

Multi-valued semantics (documented): a Phase 1/2 trial counts in both the "Phase 1" and "Phase 2" buckets; a trial in both comparison series counts in both. Sum of bucket values may exceed `total_studies_matched` — the validator checks the set-union invariant instead of a naive sum, and `meta.warnings` notes it when it happens.

Only `relationship` queries bypass `aggregate()` and use `services/network.py` — justified because its output shape (nodes/edges) is genuinely different, not because the logic forked.

---

## 4) Network graph design (highest-value piece)

Two supported network kinds, both built from the same fetched records, both citation-carrying.

**Common machinery (`services/network.py`):**

```python
def normalize_intervention(name: str) -> str
    # lowercase, strip parentheticals "(MK-3475)", strip dosage suffixes ("200 mg", "IV"),
    # collapse whitespace. Display label = most frequent original casing.

def extract_drugs(record: dict) -> list[tuple[str, str]]   # (norm_id, display) pairs
    # armsInterventionsModule.interventions where type in {"DRUG", "BIOLOGICAL"}
    # fields param: InterventionName, InterventionType

def build_cooccurrence_network(records, extract, *, min_edge_weight=2, max_nodes=30) -> NetworkResult
def build_bipartite_network(records, left_extract, right_extract, *, max_nodes=40) -> NetworkResult
```

**Kind 1 — `drug_drug` co-occurrence** (appendix: *"Which drugs frequently co-occur in combination studies?"*):
- **Node** = normalized drug/biological intervention name. `size` = number of distinct studies containing it; `nct_ids` = those studies.
- **Edge** = unordered pair of drugs appearing in the **same study**. `weight` = count of shared studies; `nct_ids` = exactly those shared studies (this is the citation set — each edge is verifiable).
- Anchoring: query like "…with pembrolizumab" → fetch `query.intr=Pembrolizumab` (plus condition filter if given), so the graph is the anchor drug's co-occurrence neighborhood. Un-anchored ("in [condition] trials") → fetch by condition.
- Pruning (disclosed in `meta.warnings` + `truncated_to_top_n`): drop edges with `weight < min_edge_weight` (default 2 — a single shared trial is noise), keep top `max_nodes=30` nodes by size, drop dangling edges. Deterministic tie-break by name.

**Kind 2 — `sponsor_drug` bipartite** (appendix: *"Show a network of sponsors ↔ drugs for [condition] trials"*):
- Left nodes = `leadSponsor.name` (`kind="sponsor"`); right nodes = drugs (`kind="drug"`).
- Edge sponsor→drug when a study led by that sponsor tests that drug; `weight` = study count; `nct_ids` = those studies.
- Same top-N pruning; keep top sponsors by study count first, then their drugs.

**Viz spec emission:** `type="network_graph"`, `encoding.nodes={"id":"id","label":"label","size":"size","group":"kind"}`, `encoding.edges={"source":"source","target":"target","weight":"weight"}`, `data=[{"nodes":[...], "edges":[...]}]`. Every node/edge dict carries `citations` (truncated) + `total_supporting_trials`. A frontend can render this with d3-force or cytoscape without guessing — README shows the mapping.

**Example queries it must answer** (become e2e tests / README example runs):
1. "Which drugs frequently co-occur in combination studies with pembrolizumab?" → drug_drug anchored
2. "Show a network of sponsors and drugs for melanoma trials." → sponsor_drug
3. "Drug co-occurrence network for NSCLC trials." → drug_drug by condition

---

## 5) Citation wiring (built-in, not bolt-on)

Citations flow through the pipeline as **nct_id sets first, text last**:

1. **Fetcher (stage 3)** stores every raw record in `StudyStore` (`records: dict[str, dict]`, plus `request_urls: list[str]`). Records are never discarded before the response is built.
2. **Aggregator / network builder (stage 4)** never copies record text — every `AggregatedDatum`, `NetworkNode`, `NetworkEdge` carries the full `nct_ids` list of contributors. This is cheap (ids only) and exact (the bucket membership *is* the provenance).
3. **Citation builder (stage 5, `services/citations.py`)** runs while the viz rows are emitted:

```python
def build_citations(
    nct_ids: list[str], store: StudyStore,
    dimension: str, key: str,            # what this datum claims, e.g. ("phase", "PHASE3")
    limit: int,                          # request.max_citations_per_datum
) -> tuple[list[Citation], int]:         # (citations, total_supporting_trials)

def build_excerpt(record: dict, dimension: str, key: str) -> str
    # Deterministic: quotes the EXACT field value that caused bucket membership,
    # plus briefTitle for context. E.g. for ("phase", "PHASE3"):
    #   '"Phase 3 randomized study evaluating pembrolizumab..." — phases: ["PHASE3"]'
    # For an edge (drug_drug): '"<briefTitle>" — interventions: ["Pembrolizumab", "Lenvatinib"]'
```

   This satisfies the spec's definition — *"an exact text excerpt from the API response (or a specific field/value) that supports the datum"* — with zero LLM involvement and no extra per-study API calls (excerpts come from fields already fetched; `BriefTitle` is always in the `fields` list). `total_supporting_trials` discloses truncation.
4. **Validator (stage 6)** enforces grounding: every `citation.nct_id` ∈ `store.records`, every excerpt is non-empty, every datum with `value > 0` has ≥1 citation (when `max_citations_per_datum > 0`).

Because ids ride along from the moment of bucketing, there is no reconciliation step where citations could drift from the numbers they support.

---

## 6) Validation layer (`services/validate.py`)

`validate_response(resp: QueryResponse, store: StudyStore, plan: QueryPlan) -> QueryResponse` — raises `ValidationFailure` (→ 500 with diagnostic, never a silently-wrong chart). Checks, in order:

1. **Schema conformance** — final `QueryResponse.model_validate()` round-trip (strict mode).
2. **Encoding↔data consistency** — every `FieldRef.field` named in `encoding` exists as a key in every data row (network: node/edge key mappings exist in node/edge dicts).
3. **Count invariants** — for every datum/node/edge: `value == len(set(nct_ids))` recomputed from the pre-truncation id lists (the aggregator hands the validator its raw `AggregationResult`/`NetworkResult` alongside the spec); `total_studies_processed == len(store.records)`.
4. **Citation grounding** — every cited `nct_id` matches `^NCT\d{8}$` AND exists in `store.records`; excerpt is a non-empty string actually derivable from that record (spot-check: briefTitle substring present).
5. **Non-empty / empty-is-explained** — if `data` is empty, response must instead be the structured `NO_RESULTS` payload with explanation; a chart with zero rows never ships.
6. **Silent-filter guard** (api-notes gotcha #3) — enforced upstream in the fetcher but re-asserted here: if `aggFilters` was used and results were 0 while the same query unfiltered had `totalCount > 0`, the pipeline must have either errored or added a warning; validator checks the warning exists.
7. **LLM-output sanity (run in stage 1, listed here for the README's validation story)** — entity grounding vs. query text; `phases ⊆ {1,2,3,4}`; `viz_type` compatible with `query_type` (e.g. `relationship` → `network_graph`), else deterministic correction + assumption note.

Unit tests feed hand-corrupted responses (wrong count, alien nct_id, missing encoding field) and assert each check fires. This section is written to map 1:1 onto the "include validation or constraints" AI/Agent Design bullet — say so in the README.

---

## 7) Task checklist (~24h)

Ordered; each task is independently implementable and testable before the next. Commit after each (per CLAUDE.md). Hours are budget, not estimates of luck.

**Phase A — Foundations (≈4.5h)**
- [ ] A1 (0.5h) `models/schemas.py`: all section-2 models + unit tests for validators (year range, citation regex).
- [ ] A2 (1.5h) `services/ctgov.py`: `CTGovClient` (async httpx) — `get_version()`, `search()` with `fields`, `pageSize=1000`, `countTotal`, `pageToken` loop, `max_studies` cap, 429 retry w/ backoff, and `aggFilters` **bare-number phase guard** (if filtered result empty but unfiltered `totalCount>0` → raise `SilentFilterError`). Record every request URL. Tests with `respx`-mocked responses using a real captured fixture payload.
- [ ] A3 (0.5h) `services/store.py`: `StudyStore` (records dict, request_urls, `add_page()`); trivial tests.
- [ ] A4 (1h) `services/dimensions.py`: registry with `phase`, `start_year`, `status`, `sponsor` extractors; confirm each request-field→response-path mapping with one live curl; tolerant date parsing ("2019", "2019-05", "2019-05-01"); tests on sparse/multi-phase fixture records.
- [ ] A5 (1h) `services/aggregate.py`: `aggregate()` + tests: multi-phase double-count, unknown bucket, series membership, sort orders, value==len(ids) invariant.

**Phase B — Vertical slice: bar chart end-to-end (≈4h)** ← first demoable milestone
- [ ] B1 (1.5h) `agents/understanding.py`: system prompt, `messages.parse()` call, grounding pass, structured-field merge → `QueryPlan`. Tests mock the Anthropic client.
- [ ] B2 (0.5h) query builder in `ctgov.py`: `build_requests(plan) -> list[CTGovRequest]` (pure; verified params only). Tests are string assertions on URLs.
- [ ] B3 (1h) `services/viz.py` + `services/citations.py` for chart-family; `pipeline.py` orchestration; `api/routes.py` POST endpoint.
- [ ] B4 (1h) `services/validate.py` checks 1–5 + corruption tests; live e2e: *"How are lung cancer trials distributed across phases?"* → save actual JSON as `examples/01_bar_phase.json`.

**Phase C — Time series + comparison (≈2.5h)** (nearly free on top of B)
- [ ] C1 (1h) time series: `start_year` dimension already exists; add zero-fill + `time_granularity` meta; e2e: *"How has the number of pembrolizumab trials changed per year since 2015?"* → `examples/02_time_series.json`.
- [ ] C2 (1.5h) comparison: pipeline fetches per compared entity, builds `series_membership`, `grouped_bar_chart` encoding with `color`; overlap-counting note in meta; e2e: *"Compare phases for trials involving pembrolizumab vs nivolumab."* → `examples/03_comparison.json`.

**Phase D — Network graph (≈5h, protected budget — the differentiator)**
- [ ] D1 (1h) `normalize_intervention()` + `extract_drugs()` + tests against messy real intervention names (fixtures captured live).
- [ ] D2 (1.5h) `build_cooccurrence_network()` + pruning + tests (weights, top-N determinism, dangling-edge removal).
- [ ] D3 (1h) `build_bipartite_network()` + tests.
- [ ] D4 (1.5h) network viz spec emission + node/edge citations + validator extensions + two live e2e runs → `examples/04_network_drug_drug.json`, `examples/05_network_sponsor_drug.json`.

**Phase E — Robustness + submission (≈5h)**
- [ ] E1 (1h) error paths: NO_RESULTS explanation payload, UPSTREAM_ERROR on CTGov 5xx, LLM refusal/failure handling, request timeout budget.
- [ ] E2 (0.5h) `meta` completeness: `data_as_of` from `/version`, api_urls, warnings everywhere they're promised.
- [ ] E3 (2h) README: how to run; full request/response schema docs; design decisions & tradeoffs (pull from section 8); limitations/next steps; AI-tools disclosure (integrity note section 8 of spec); the 3–5 example runs with **actual** outputs.
- [ ] E4 (0.5h) polish: ruff/format pass, docstrings on public functions, `pytest -q` green, `.env.example` updated (remove OPENAI key, add `LLM_MODEL`).
- [ ] E5 (1h) buffer / zip packaging / final live smoke test of all 5 examples.

**Phase F — Only if time remains (explicitly cut-able)**
- [ ] F1 geographic (`country` dimension is already in the registry — mostly a fields/extractor test + example run).
- [ ] F2 histogram (enrollment bins via the same `aggregate()`).
- [ ] F3 deterministic regex fallback parser when `ANTHROPIC_API_KEY` is absent (nice for the reviewer running without a key — at minimum, return a clear error telling them the key is required).

Total core (A–E): ~21h, leaving ~3h of real-world slack.

---

## 8) Known risks / open design questions (→ README "Design Decisions")

1. **Multi-phase counting.** A `["PHASE1","PHASE2"]` trial counts in both buckets. Alternative (a combined "Phase 1/2" bucket) hides comparability. Decision: double-count + disclose in `meta.warnings`; validator checks set-union, not sum.
2. **Missing phase data (~25%).** Emitted as an explicit "No phase data" bucket rather than silently dropped — data honesty over chart prettiness.
3. **Fetch cap.** Broad queries (e.g. "cancer") can match 10⁵ trials. Cap at `max_studies` (default 3000 = 3 pages), always disclosed via `meta.warnings` + `totalCount` comparison. Aggregations are then *over the fetched sample* — stated plainly. Judgment call: correctness-of-claim over false completeness.
4. **Comparison overlap.** A trial testing both compared drugs counts in both series (it *is* evidence for both). Disclosed per response.
5. **Drug-name normalization is heuristic.** Brand/generic synonyms (Keytruda vs pembrolizumab) are NOT merged (would require a drug-synonym source; out of 24h scope — listed as "with more time"). Parenthetical/dosage stripping only.
6. **Unverified API params.** Only verified params from api-notes.md are used. `filter.overallStatus` stays unused unless the A2 live check confirms it; status filtering beyond `status:rec` falls back to **client-side post-filtering** on `overallStatus` (we already have the records — filtering locally is always safe and citation-compatible).
7. **Date granularity.** Year-precision parse of `startDateStruct.date`; trials with unparseable/missing dates go to `unbucketed` with a warning. Time series buckets by year only (no month granularity in v1).
8. **Server-side aggregation rejected deliberately.** `/stats/field/values` 500s live, and even working it can't yield per-record citations. Client-side aggregation is a requirement of the citation design, not a workaround — say this in the README (it reads as a design insight, evaluators reward it).
9. **LLM dependency.** Stage 1 requires `ANTHROPIC_API_KEY`. Mitigation: it's the only LLM call; everything else runs and tests offline; mocked in all tests; F3 fallback if time allows.
10. **Excerpt fidelity.** Excerpts are exact field values + briefTitle rather than prose spans from full-text (which would need per-study `GET /studies/{nctId}` calls — N extra requests). The spec explicitly allows "a specific field/value". If time allows, deep-fetch full records for only the ≤5 cited studies per datum (bounded cost) — noted as an upgrade path.
11. **Ambiguous viz choice** (e.g. "most common intervention types" could be bar or pie). LLM proposes, deterministic compatibility table corrects, `meta.assumptions` records the interpretation. No pie charts — bar covers it.
12. **Rate limits.** ~50 req/min. Worst case (comparison, 2 entities × 3 pages + version) ≈ 8 requests/query — fine; backoff on 429 anyway.

---
*Prepared 2026-08-08 against ClinicalTrials.gov API v2.0.5 (dataTimestamp 2026-08-07). API facts per docs/api-notes.md (live-verified).*
