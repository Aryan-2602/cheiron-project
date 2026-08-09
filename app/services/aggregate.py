"""The single aggregation abstraction.

Every non-network visualization this service supports -- distribution bars, time
series, grouped comparisons, geographic breakdowns, enrollment histograms -- is
the same operation: *group trials by a dimension, optionally split into series,
and count distinct trials per bucket*. That operation lives in exactly one
function, :func:`aggregate`. Chart types differ only in which
:class:`~app.services.dimensions.Dimension` they pass in and how the resulting
buckets are encoded visually.

Two properties make this the trustworthy half of the pipeline:

* **Values are never asserted, only counted.** ``datum.value`` is the
  cardinality of a set of NCT ids collected from real records. There is no code
  path that writes a number from any other source.
* **Provenance rides along.** Each bucket keeps the ids of every trial that put
  it there, so citations are a projection of the aggregation rather than a
  later reconciliation step that could drift from it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any

from app.models.schemas import AggregatedDatum, AggregationResult
from app.services.dimensions import Dimension


def aggregate(
    records: Mapping[str, dict[str, Any]],
    dimension: Dimension,
    *,
    series_membership: Mapping[str, AbstractSet[str]] | None = None,
    include_unknown: bool = True,
    top_n: int | None = None,
) -> AggregationResult:
    """Group ``records`` by ``dimension`` and count distinct trials per bucket.

    Args:
        records: NCT id -> raw study record (a :class:`~app.services.store.StudyStore`
            record map).
        dimension: How to derive bucket keys from a record.
        series_membership: Optional series label -> set of NCT ids. Supplied for
            comparison queries, where each series is the result set of its own
            upstream search. A trial appearing in two searches is counted in
            both series -- it genuinely is evidence for both.
        include_unknown: Emit an explicit bucket for records with no value on
            this dimension. On by default: ~25% of studies have no usable phase,
            and hiding them would misrepresent the distribution.
        top_n: Keep only the ``top_n`` largest buckets (by trial count, summed
            across series so grouped charts stay aligned). Used for
            high-cardinality dimensions like sponsor or country.

    Returns:
        An :class:`AggregationResult` whose ``data`` is sorted by the
        dimension's own ordering, then by descending count, then by key.
    """
    if series_membership is None:
        series_membership = {None: set(records)}  # type: ignore[dict-item]

    # (series, raw_key) -> set of contributing NCT ids
    buckets: dict[tuple[str | None, str], set[str]] = defaultdict(set)
    unbucketed: set[str] = set()
    matched: set[str] = set()
    unknown_emitted = False

    for series, nct_ids in series_membership.items():
        for nct_id in nct_ids:
            record = records.get(nct_id)
            if record is None:
                continue
            matched.add(nct_id)
            keys = dimension.extract(record)
            if not keys:
                unbucketed.add(nct_id)
                if include_unknown:
                    buckets[(series, dimension.unknown_label)].add(nct_id)
                    unknown_emitted = True
                continue
            for key in keys:
                buckets[(series, key)].add(nct_id)

    # Counted across series, because the display cap applies to categories on
    # the axis, not to rows.
    total_categories = len({key for _series, key in buckets})
    if top_n is not None:
        totals: dict[str, int] = defaultdict(int)
        for (_series, key), ids in buckets.items():
            totals[key] += len(ids)
        keep = {
            key
            for key, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        }
        buckets = {k: v for k, v in buckets.items() if k[1] in keep}
        unknown_emitted = unknown_emitted and dimension.unknown_label in keep
    displayed_categories = len({key for _series, key in buckets})

    # Sort on the *raw* key (the dimension defines order in its own terms), then
    # by descending count, then by label -- fully deterministic, no ties left to
    # dict iteration order. The unknown bucket always sorts last.
    rows = []
    for (series, key), ids in buckets.items():
        is_unknown = key == dimension.unknown_label
        rows.append(
            (
                float("inf") if is_unknown else dimension.order(key),
                -len(ids),
                key if is_unknown else dimension.display(key),
                series or "",
                sorted(ids),
                series,
            )
        )
    rows.sort(key=lambda r: r[:4])

    data = [
        AggregatedDatum(key=label, series=series, value=-neg_count, nct_ids=ids)
        for _order, neg_count, label, _series_sort, ids, series in rows
    ]

    # A series whose search returned nothing produced no rows at all, so it
    # vanished from a chart whose title still named it. Zero-fill it across the
    # keys the other series produced: "this series exists and is empty" is the
    # answer, and it keeps the encoding honest for a frontend that splits on
    # the series channel.
    if None not in series_membership:
        charted = {d.series for d in data}
        missing = [s for s in series_membership if s not in charted]
        if missing and data:
            keys = list(dict.fromkeys(d.key for d in data))
            data += [
                AggregatedDatum(key=key, series=series, value=0, nct_ids=[])
                for series in missing
                for key in keys
            ]

    return AggregationResult(
        dimension=dimension.name,
        series_dimension="series" if None not in series_membership else None,
        data=data,
        total_studies_matched=len(matched),
        unbucketed=len(unbucketed),
        unbucketed_key_included=unknown_emitted,
        multi_valued=dimension.multi_valued,
        total_categories=total_categories,
        displayed_categories=displayed_categories,
        category_limit=top_n,
    )


#: A guard against an absurd axis if a bound is ever wrong; the fetch cap means
#: no realistic request spans anything like this many years.
MAX_FILLED_YEARS = 120


def zero_fill_years(
    result: AggregationResult,
    *,
    start: int | None = None,
    end: int | None = None,
) -> AggregationResult:
    """Insert empty year buckets so a time series has no phantom gaps.

    A year with no trials is real information: leaving it out would make a
    line chart interpolate straight through it and imply activity that did not
    happen. Filled years carry ``value=0`` and no citations, which is honest --
    there are no records to cite for a year with no trials.

    ``start``/``end`` are the bounds the *question* asked for. When both are
    known the axis spans exactly that interval, so "2020 through 2024" with
    data only in 2021-2023 shows the empty years at both ends rather than
    quietly cropping to the data. Without them the observed range is used, as
    before -- inferring a bound the question never gave would be inventing one.
    """
    years = [d.key for d in result.data if d.key.isdigit()]
    explicit = start is not None and end is not None
    if not years or (len(years) < 2 and not explicit):
        return result

    low = start if start is not None else int(min(years))
    high = end if end is not None else int(max(years))
    if high < low or high - low + 1 > MAX_FILLED_YEARS:
        return result

    series_labels = sorted({d.series for d in result.data})
    present = {(d.series, d.key) for d in result.data}
    filled = list(result.data)
    for series in series_labels:
        for year in range(low, high + 1):
            if (series, str(year)) not in present:
                filled.append(
                    AggregatedDatum(key=str(year), series=series, value=0, nct_ids=[])
                )

    filled.sort(
        key=lambda d: (
            0 if d.key.isdigit() else 1,
            int(d.key) if d.key.isdigit() else 0,
            d.series or "",
        )
    )
    return result.model_copy(update={"data": filled})
