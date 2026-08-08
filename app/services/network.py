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

from app.models.schemas import NetworkEdge, NetworkNode, NetworkResult

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


def extract_drugs(record: dict[str, Any]) -> list[tuple[str, str]]:
    """``(normalized_key, display_label)`` for each therapeutic agent in a trial.

    Deduplicated within a trial, so a drug listed once per study arm counts once.
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
        seen.setdefault(key, name.strip())
    return sorted(seen.items())


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


ExtractEntities = Callable[[dict[str, Any]], list[tuple[str, str]]]


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
) -> dict[str, NetworkNode]:
    return {
        f"{prefix}:{key}": NetworkNode(
            id=f"{prefix}:{key}",
            label=_label_for(labels[key]),
            kind=kind,  # type: ignore[arg-type]
            size=len(ids),
            nct_ids=sorted(ids),
        )
        for key, ids in members.items()
    }


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
) -> NetworkResult:
    """Undirected co-occurrence graph over entities sharing a trial.

    Args:
        min_edge_weight: Minimum shared trials for an edge to be kept. The
            default of 2 drops pairs that co-occur exactly once, which are
            overwhelmingly incidental rather than a studied combination.
        max_nodes: Cap on graph size, reported via ``truncated_to_top_n``.
    """
    members: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, list[str]] = defaultdict(list)
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)

    for nct_id, record in records.items():
        entities = extract(record)
        for key, label in entities:
            members[key].add(nct_id)
            labels[key].append(label)
        # An unordered pair, keyed consistently so (a,b) and (b,a) are one edge.
        for (left, _), (right, _) in combinations(entities, 2):
            pairs[(min(left, right), max(left, right))].add(nct_id)

    nodes = _build_nodes(members, labels, kind, kind)
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

    for nct_id, record in records.items():
        lefts = left_extract(record)
        rights = right_extract(record)
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
