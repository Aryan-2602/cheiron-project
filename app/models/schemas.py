"""All request/response and intermediate models for the query-to-visualization pipeline.

Design note — hallucination prevention
--------------------------------------
The models in this file are split into three groups, and the split is the
enforcement mechanism for the project's core rule that *the LLM never produces a
data value*:

1. ``QueryUnderstanding`` and its children are the **only** models the LLM ever
   produces. They contain no numeric measure, no bucket, no count -- there is
   structurally nowhere for a fabricated data value to live.
2. ``AggregationResult`` / ``NetworkResult`` are produced by deterministic code
   in ``app.services.aggregate`` / ``app.services.network`` and are the only
   place values come from. Every value is a set cardinality over real NCT ids.
3. ``QueryResponse`` is the frontend-facing contract, assembled from (2) and
   validated against the fetched records by ``app.services.validate``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# Shared vocabularies
# --------------------------------------------------------------------------

QueryType = Literal[
    "distribution",
    "time_trend",
    "comparison",
    "relationship",
    "geographic",
    "unsupported",
]

VizType = Literal[
    "bar_chart",
    "time_series",
    "grouped_bar_chart",
    "network_graph",
    "geo_bar_chart",
    "histogram",
]

GroupByDim = Literal[
    "phase",
    "start_year",
    "status",
    "sponsor",
    "sponsor_class",
    "intervention_type",
    "country",
    "enrollment_bucket",
]

NetworkKind = Literal["drug_drug", "sponsor_drug"]

CompareEntityKind = Literal["drug", "condition", "sponsor"]

NCT_ID_PATTERN = r"^NCT\d{8}$"


# --------------------------------------------------------------------------
# 1. Input
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for ``POST /api/v1/query``.

    ``query`` is the only required field. The optional structured fields are
    candidate-defined overrides: when supplied they take precedence over
    whatever the LLM extracts from the natural-language query, which makes the
    endpoint usable both conversationally and programmatically.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=2000,
        description="Natural-language question about clinical trials.",
    )

    # Optional structured overrides.
    drug_name: str | None = Field(
        default=None, description="Intervention name; overrides LLM extraction."
    )
    condition: str | None = Field(
        default=None, description="Condition/disease; overrides LLM extraction."
    )
    sponsor: str | None = Field(
        default=None, description="Lead sponsor name; overrides LLM extraction."
    )
    phase: Literal[1, 2, 3, 4] | None = Field(
        default=None, description="Trial phase as a bare number, matching the API."
    )
    country: str | None = Field(default=None, description="Location country.")
    start_year: int | None = Field(default=None, ge=1990, le=2035)
    end_year: int | None = Field(default=None, ge=1990, le=2035)

    # Output controls.
    max_citations_per_datum: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Citations attached per datum. Full contributor counts are "
        "always reported via total_supporting_trials.",
    )
    max_studies: int = Field(
        default=3000,
        ge=100,
        le=5000,
        description="Upper bound on studies fetched. Disclosed in meta.warnings "
        "whenever it truncates the result set.",
    )

    @model_validator(mode="after")
    def _check_year_range(self) -> QueryRequest:
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must be less than or equal to end_year")
        return self


# --------------------------------------------------------------------------
# 2. Query understanding (the ONLY LLM-produced models)
# --------------------------------------------------------------------------


class YearRange(BaseModel):
    """Inclusive year bounds mentioned in the query. Either side may be absent."""

    model_config = ConfigDict(extra="forbid")

    start: int | None
    end: int | None


class ExtractedEntities(BaseModel):
    """Entities the LLM found *in the text of the query*.

    Every list holds surface strings copied from the question. A deterministic
    grounding pass (``app.agents.understanding.ground_entities``) drops anything
    that does not actually appear in the user's text, so the LLM cannot invent
    a drug or sponsor that was never asked about.
    """

    model_config = ConfigDict(extra="forbid")

    drugs: list[str]
    conditions: list[str]
    sponsors: list[str]
    phases: list[int]
    statuses: list[str]
    countries: list[str]
    year_range: YearRange | None


class QueryUnderstanding(BaseModel):
    """The complete output of the LLM stage.

    Note what is *absent*: no counts, no buckets, no data rows, no totals. The
    LLM classifies and extracts; it never measures.
    """

    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    entities: ExtractedEntities
    group_by: GroupByDim | None
    compare_entities: list[str]
    compare_entity_kind: CompareEntityKind | None
    network_kind: NetworkKind | None
    viz_type: VizType
    assumptions: list[str]


class QueryPlan(BaseModel):
    """Post-processed understanding: grounded, merged with structured overrides,
    defaults resolved. This is what the query builder and pipeline consume."""

    model_config = ConfigDict(extra="forbid")

    query: str
    query_type: QueryType
    entities: ExtractedEntities
    group_by: GroupByDim | None
    compare_entities: list[str] = Field(default_factory=list)
    compare_entity_kind: CompareEntityKind | None = None
    network_kind: NetworkKind | None = None
    viz_type: VizType
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    max_citations_per_datum: int = 3
    max_studies: int = 3000


# --------------------------------------------------------------------------
# 3. Aggregation output (shared across bar / time_series / comparison / geo)
# --------------------------------------------------------------------------


class AggregatedDatum(BaseModel):
    """One bucket. ``value`` is always ``len(set(nct_ids))`` -- computed, never
    asserted -- which is the invariant the validator re-checks."""

    model_config = ConfigDict(extra="forbid")

    key: str
    series: str | None = None
    value: int
    nct_ids: list[str]


class AggregationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    series_dimension: str | None = None
    data: list[AggregatedDatum]
    total_studies_matched: int
    unbucketed: int = 0
    unbucketed_key_included: bool = False
    multi_valued: bool = False
    #: Distinct categories before the display cap, and how many survived it.
    #: Carried so the pipeline can disclose a truncation it would otherwise
    #: present as the whole picture -- 20 of 47 countries reads as "all
    #: countries" unless the other 27 are named.
    total_categories: int = 0
    displayed_categories: int = 0
    category_limit: int | None = None

    @property
    def omitted_categories(self) -> int:
        return max(0, self.total_categories - self.displayed_categories)


# --------------------------------------------------------------------------
# 4. Network output
# --------------------------------------------------------------------------


class DrugResolution(BaseModel):
    """One drug name resolved (or not) to its RxNorm ingredient.

    ``rxcui`` is always an *ingredient* concept, never a brand: brand and
    generic are distinct RxNorm concepts (Keytruda is 1547550, pembrolizumab is
    1547545), so resolution walks to the ingredient to make them the same node.

    An unresolved result is a normal outcome, not an error -- research compounds
    and combination products legitimately have no single ingredient. It carries
    the cleaned name so callers can use it unconditionally.
    """

    model_config = ConfigDict(extra="forbid")

    rxcui: str | None
    canonical_name: str
    original_names: set[str] = Field(default_factory=set)
    resolved: bool
    #: approximateTerm score that produced the match, kept for auditability.
    score: float | None = None


class NetworkNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    kind: Literal["drug", "sponsor", "condition"]
    size: int
    nct_ids: list[str]
    #: RxNorm ingredient id, when the name resolved. Lets a reader verify the
    #: merge against RxNav.
    rxcui: str | None = None
    #: The distinct surface forms folded into this node. More than one entry
    #: means an actual brand/generic merge happened.
    merged_from: list[str] = Field(default_factory=list)


class NetworkEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    weight: int
    nct_ids: list[str]


class NetworkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: NetworkKind
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    truncated_to_top_n: int | None = None
    min_edge_weight: int = 1


# --------------------------------------------------------------------------
# 5. Response
# --------------------------------------------------------------------------


class Citation(BaseModel):
    """A deep citation: the trial record that contributed to one datum, plus the
    exact field value from the API response that supports it."""

    model_config = ConfigDict(extra="forbid")

    nct_id: str = Field(pattern=NCT_ID_PATTERN)
    excerpt: str = Field(min_length=1)
    url: str


class FieldRef(BaseModel):
    """Maps a visual channel to a key present in every row of ``data``."""

    model_config = ConfigDict(extra="forbid")

    field: str
    label: str | None = None
    type: Literal["nominal", "ordinal", "quantitative", "temporal"] | None = None


class Encoding(BaseModel):
    """Channel mapping. Chart-family visualizations use x/y/color; network
    graphs use the nodes/edges key maps instead."""

    model_config = ConfigDict(extra="forbid")

    x: FieldRef | None = None
    y: FieldRef | None = None
    color: FieldRef | None = None
    nodes: dict[str, str] | None = None
    edges: dict[str, str] | None = None


class VisualizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: VizType
    title: str
    encoding: Encoding
    data: list[dict[str, Any]]


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: dict[str, Any] = Field(default_factory=dict)
    source: str = "clinicaltrials.gov"
    data_as_of: str | None = None
    total_studies_processed: int = 0
    api_urls: list[str] = Field(default_factory=list)
    query_interpretation: str = ""
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sorting: str | None = None
    time_granularity: str | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visualization: VisualizationSpec
    meta: Meta


# --------------------------------------------------------------------------
# 6. Errors
# --------------------------------------------------------------------------

ErrorCode = Literal[
    "NO_RESULTS",
    "UNSUPPORTED_QUERY",
    "UPSTREAM_ERROR",
    "LLM_ERROR",
    "VALIDATION_ERROR",
]


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
