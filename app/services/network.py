"""Relationship queries: entity co-occurrence networks over trial records.

Networks are the one visualization family that does not reduce to
"group and count", because the output is nodes and edges rather than buckets.
The provenance discipline is identical though: a node's ``size`` and an edge's
``weight`` are set cardinalities over real NCT ids, and those id sets are what
the citation builder renders. An edge in particular is a strong claim -- "these
two drugs are studied together" -- so it carries exactly the trials containing
both endpoints, and a reader can open any of them and see both drugs listed.

Two network kinds are supported:

* ``drug_drug`` -- undirected co-occurrence. Answers "which drugs are combined
  with X" and "which drugs co-occur in combination studies".
* ``sponsor_drug`` -- bipartite. Answers "show a network of sponsors and drugs
  for [condition] trials".
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from itertools import combinations
from typing import Any

from app.models.schemas import (
    DrugResolution,
    NetworkEdge,
    NetworkNode,
    NetworkResult,
)

#: Intervention types that name a therapeutic agent. PROCEDURE, DEVICE,
#: BEHAVIORAL and friends are excluded: "Placebo administration" and
#: "Standard of care" are not drugs and would dominate a co-occurrence graph.
DRUG_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}

#: Interventions that appear in a large share of trials and carry no
#: relationship information -- keeping them makes every graph a star centred on
#: "Placebo".
STOPWORD_DRUGS = {
    "placebo",
    "saline",
    "normal saline",
    "standard of care",
    "best supportive care",
    "matching placebo",
    "placebos",
    "vehicle",
    "no intervention",
}

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_DOSAGE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|mg/kg|mg/m2|iu|units?|%)\b", re.IGNORECASE
)
_ROUTE = re.compile(
    r"\b(iv|im|sc|po|oral(ly)?|intravenous(ly)?|subcutaneous(ly)?|"
    r"injection|infusion|tablets?|capsules?|solution)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-z0-9\s\-+]")
_WS = re.compile(r"\s+")


def normalize_intervention(name: str) -> str:
    """Reduce an intervention name to a comparison key.

    Registry intervention names are free text, so the same agent appears as
    "Pembrolizumab", "Pembrolizumab 200 mg", "Pembrolizumab (MK-3475)" and
    "pembrolizumab IV". Without normalization each is its own node and the
    graph fragments into near-duplicates.

    This deliberately does *not* map brand names to generic names (Keytruda ->
    pembrolizumab): that needs a drug vocabulary the service does not have, and
    guessing would merge distinct agents. Unmerged synonyms are a documented
    limitation, not a silent one.
    """
    text = name.lower()
    text = _PARENTHETICAL.sub(" ", text)
    text = _DOSAGE.sub(" ", text)
    text = _ROUTE.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip(" -+")
    return text


def extract_drugs(
    record: dict[str, Any],
    *,
    resolutions: Mapping[str, DrugResolution] | None = None,
) -> list[tuple[str, str]]:
    """``(identity_key, display_label)`` for each therapeutic agent in a trial.

    Deduplicated within a trial, so a drug listed once per study arm counts once.

    ``resolutions`` maps a string-normalized name to its RxNorm ingredient. It
    is applied *here*, at key construction, which is what makes merging work
    end to end: two names sharing an ingredient produce the same key, so the
    node's ``nct_ids`` become the union of both names' trials and every derived
    figure -- size, edge weight, citations, ``total_supporting_trials`` -- is
    computed over that union rather than reconciled afterwards.

    Passing ``None`` reproduces the pre-RxNorm behaviour exactly.
    """
    interventions = (
        record.get("protocolSection", {})
        .get("armsInterventionsModule", {})
        .get("interventions")
        or []
    )
    seen: dict[str, str] = {}
    for intervention in interventions:
        if not isinstance(intervention, dict):
            continue
        if intervention.get("type") not in DRUG_TYPES:
            continue
        name = intervention.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = normalize_intervention(name)
        if not key or key in STOPWORD_DRUGS:
            continue
        display = name.strip()
        if resolutions:
            resolution = resolutions.get(key)
            if resolution is not None and resolution.resolved and resolution.rxcui:
                key, display = f"rxcui:{resolution.rxcui}", resolution.canonical_name
        seen.setdefault(key, display)
    return sorted(seen.items())


def rank_candidate_names(
    records: Mapping[str, dict[str, Any]], *, top_k: int
) -> list[str]:
    """The ``top_k`` most-mentioned drug names, by provisional (unmerged) size.

    Resolving every distinct name is mostly wasted work: a 600-trial query
    yields hundreds of names, but only a few dozen nodes survive pruning. This
    ranks names *before* any API call so resolution can be limited to the ones
    that could plausibly reach the final graph.

    The pool is deliberately several times larger than the node cap, because
    merging only ever *increases* a node's size -- a name that looks marginal
    on its own may be part of a compound that ranks highly once merged. Names
    outside the pool are simply left unresolved, which is the same conservative
    failure as any other unresolved name.
    """
    counts: dict[str, int] = defaultdict(int)
    for record in records.values():
        for key, _label in extract_drugs(record):
            counts[key] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:top_k]]


def extract_sponsors(record: dict[str, Any]) -> list[tuple[str, str]]:
    name = (
        record.get("protocolSection", {})
        .get("sponsorCollaboratorsModule", {})
        .get("leadSponsor", {})
        .get("name")
    )
    if not isinstance(name, str) or not name.strip():
        return []
    return [(name.strip().lower(), name.strip())]


ExtractEntities = Callable[..., list[tuple[str, str]]]


def _merge_sources(
    resolutions: Mapping[str, DrugResolution] | None,
) -> dict[str, set[str]]:
    """Node key -> the distinct source names that resolved to it.

    Taken from the resolution map rather than from node labels: after a merge
    every contributing name shares one canonical label, so the labels can no
    longer show what was combined.
    """
    sources: dict[str, set[str]] = defaultdict(set)
    for cleaned, resolution in (resolutions or {}).items():
        if resolution.resolved and resolution.rxcui:
            sources[f"rxcui:{resolution.rxcui}"] |= (
                resolution.original_names or {cleaned}
            )
    return sources


def _extract(
    extract: ExtractEntities,
    record: dict[str, Any],
    resolutions: Mapping[str, DrugResolution] | None,
) -> list[tuple[str, str]]:
    """Call an extractor, passing ``resolutions`` only to those that accept it.

    Keeps custom extractors (and ``extract_sponsors``) usable unchanged.
    """
    if resolutions is None:
        return extract(record)
    try:
        return extract(record, resolutions=resolutions)
    except TypeError:
        return extract(record)


def _label_for(candidates: Iterable[str]) -> str:
    """Pick the display label for a node: the most common surface form, with
    ties broken alphabetically so output is deterministic."""
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate] += 1
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _build_nodes(
    members: Mapping[str, set[str]],
    labels: Mapping[str, list[str]],
    kind: str,
    prefix: str,
    originals: Mapping[str, set[str]] | None = None,
) -> dict[str, NetworkNode]:
    """Materialize nodes. ``originals`` records the distinct source names folded
    into each key, so a merge is visible and auditable in the output."""
    nodes: dict[str, NetworkNode] = {}
    for key, ids in members.items():
        # A resolved key is "rxcui:<id>"; keep the RxCUI itself on the node so a
        # reader can check the merge against RxNav.
        rxcui = key.split("rxcui:", 1)[1] if key.startswith("rxcui:") else None
        merged = sorted(originals.get(key, set())) if originals else []
        nodes[f"{prefix}:{key}"] = NetworkNode(
            id=f"{prefix}:{key}",
            label=_label_for(labels[key]),
            kind=kind,  # type: ignore[arg-type]
            size=len(ids),
            nct_ids=sorted(ids),
            rxcui=rxcui,
            # Only meaningful when more than one name collapsed here.
            merged_from=merged if len(merged) > 1 else [],
        )
    return nodes


def _prune(
    nodes: dict[str, NetworkNode],
    edges: list[NetworkEdge],
    *,
    max_nodes: int,
) -> tuple[list[NetworkNode], list[NetworkEdge], int | None]:
    """Keep the ``max_nodes`` largest nodes and drop edges left dangling.

    Truncation is reported by the caller rather than applied quietly: a graph
    that silently shows the top 30 of 400 drugs would read as the whole picture.
    """
    if len(nodes) <= max_nodes:
        kept_ids = set(nodes)
        truncated = None
    else:
        ranked = sorted(nodes.values(), key=lambda n: (-n.size, n.id))[:max_nodes]
        kept_ids = {n.id for n in ranked}
        truncated = max_nodes

    kept_nodes = sorted(
        (n for n in nodes.values() if n.id in kept_ids), key=lambda n: (-n.size, n.id)
    )
    kept_edges = sorted(
        (e for e in edges if e.source in kept_ids and e.target in kept_ids),
        key=lambda e: (-e.weight, e.source, e.target),
    )
    return kept_nodes, kept_edges, truncated


def build_cooccurrence_network(
    records: Mapping[str, dict[str, Any]],
    *,
    extract: ExtractEntities = extract_drugs,
    kind: str = "drug",
    min_edge_weight: int = 2,
    max_nodes: int = 30,
    resolutions: Mapping[str, DrugResolution] | None = None,
) -> NetworkResult:
    """Undirected co-occurrence graph over entities sharing a trial.

    Args:
        min_edge_weight: Minimum shared trials for an edge to be kept. The
            default of 2 drops pairs that co-occur exactly once, which are
            overwhelmingly incidental rather than a studied combination.
        max_nodes: Cap on graph size, reported via ``truncated_to_top_n``.
        resolutions: Optional RxNorm ingredient map. Applied before sizes and
            weights are computed, so a merged compound competes for a place in
            the graph at its merged size rather than as separate fragments.
    """
    members: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, list[str]] = defaultdict(list)
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    originals = _merge_sources(resolutions)

    for nct_id, record in records.items():
        entities = _extract(extract, record, resolutions)
        for key, label in entities:
            members[key].add(nct_id)
            labels[key].append(label)
        # An unordered pair, keyed consistently so (a,b) and (b,a) are one edge.
        for (left, _), (right, _) in combinations(entities, 2):
            pairs[(min(left, right), max(left, right))].add(nct_id)

    nodes = _build_nodes(members, labels, kind, kind, originals)
    edges = [
        NetworkEdge(
            source=f"{kind}:{left}",
            target=f"{kind}:{right}",
            weight=len(ids),
            nct_ids=sorted(ids),
        )
        for (left, right), ids in pairs.items()
        if len(ids) >= min_edge_weight
    ]

    # Drop nodes with no surviving edge: an isolated point in a relationship
    # graph shows nothing about relationships.
    connected = {e.source for e in edges} | {e.target for e in edges}
    nodes = {nid: node for nid, node in nodes.items() if nid in connected}

    kept_nodes, kept_edges, truncated = _prune(nodes, edges, max_nodes=max_nodes)
    return NetworkResult(
        kind="drug_drug",
        nodes=kept_nodes,
        edges=kept_edges,
        truncated_to_top_n=truncated,
        min_edge_weight=min_edge_weight,
    )


def build_bipartite_network(
    records: Mapping[str, dict[str, Any]],
    *,
    left_extract: ExtractEntities = extract_sponsors,
    right_extract: ExtractEntities = extract_drugs,
    left_kind: str = "sponsor",
    right_kind: str = "drug",
    max_left: int = 15,
    max_right: int = 25,
    resolutions: Mapping[str, DrugResolution] | None = None,
) -> NetworkResult:
    """Bipartite graph linking each sponsor to the agents it studies.

    Pruning keeps the busiest sponsors first, then the drugs those sponsors
    actually study, so the surviving graph stays connected and interpretable
    rather than being a top-N slice of each side independently.
    """
    left_members: dict[str, set[str]] = defaultdict(set)
    right_members: dict[str, set[str]] = defaultdict(set)
    left_labels: dict[str, list[str]] = defaultdict(list)
    right_labels: dict[str, list[str]] = defaultdict(list)
    links: dict[tuple[str, str], set[str]] = defaultdict(set)

    right_originals = _merge_sources(resolutions)

    for nct_id, record in records.items():
        lefts = left_extract(record)
        # Only the drug side is resolvable; sponsors are already canonical.
        rights = _extract(right_extract, record, resolutions)
        for key, label in lefts:
            left_members[key].add(nct_id)
            left_labels[key].append(label)
        for key, label in rights:
            right_members[key].add(nct_id)
            right_labels[key].append(label)
        for left_key, _ in lefts:
            for right_key, _ in rights:
                links[(left_key, right_key)].add(nct_id)

    top_left = {
        key
        for key, _ in sorted(
            left_members.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )[:max_left]
    }
    # Rank drugs by how much of the *kept* sponsors' work they account for.
    right_scores: dict[str, int] = defaultdict(int)
    for (left_key, right_key), ids in links.items():
        if left_key in top_left:
            right_scores[right_key] += len(ids)
    top_right = {
        key
        for key, _ in sorted(right_scores.items(), key=lambda kv: (-kv[1], kv[0]))[
            :max_right
        ]
    }

    nodes: dict[str, NetworkNode] = {}
    nodes.update(
        _build_nodes(
            {k: v for k, v in left_members.items() if k in top_left},
            left_labels,
            left_kind,
            left_kind,
        )
    )
    nodes.update(
        _build_nodes(
            {k: v for k, v in right_members.items() if k in top_right},
            right_labels,
            right_kind,
            right_kind,
            right_originals,
        )
    )

    edges = sorted(
        (
            NetworkEdge(
                source=f"{left_kind}:{left_key}",
                target=f"{right_kind}:{right_key}",
                weight=len(ids),
                nct_ids=sorted(ids),
            )
            for (left_key, right_key), ids in links.items()
            if left_key in top_left and right_key in top_right
        ),
        key=lambda e: (-e.weight, e.source, e.target),
    )

    connected = {e.source for e in edges} | {e.target for e in edges}
    kept_nodes = sorted(
        (n for n in nodes.values() if n.id in connected),
        key=lambda n: (n.kind, -n.size, n.id),
    )
    truncated = (
        max_left + max_right
        if len(left_members) > max_left or len(right_members) > max_right
        else None
    )
    return NetworkResult(
        kind="sponsor_drug",
        nodes=kept_nodes,
        edges=edges,
        truncated_to_top_n=truncated,
        min_edge_weight=1,
    )
