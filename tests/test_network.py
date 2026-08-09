"""Network graphs: normalization, co-occurrence, bipartite, and pruning."""

import pytest

from app.services.network import (
    build_bipartite_network,
    build_cooccurrence_network,
    extract_drugs,
    normalize_intervention,
    rank_candidate_names,
)
from tests.conftest import make_record


def trial(nct_id, drugs, sponsor="Acme Pharma"):
    return make_record(
        nct_id, interventions=[("DRUG", d) for d in drugs], sponsor=sponsor
    )


class TestNormalization:
    """Registry intervention names are free text; without normalization the same
    agent becomes several nodes and the graph fragments."""

    def test_dosage_route_and_parentheticals_are_stripped(self):
        for variant in [
            "Pembrolizumab",
            "Pembrolizumab 200 mg",
            "Pembrolizumab (MK-3475)",
            "pembrolizumab IV",
            "PEMBROLIZUMAB 200mg IV infusion",
        ]:
            assert normalize_intervention(variant) == "pembrolizumab"

    def test_distinct_drugs_stay_distinct(self):
        assert normalize_intervention("Pembrolizumab") != normalize_intervention(
            "Nivolumab"
        )

    def test_brand_names_are_deliberately_not_merged(self):
        """Merging Keytruda->pembrolizumab needs a drug vocabulary we do not
        have; guessing would silently fuse distinct agents."""
        assert normalize_intervention("Keytruda") != normalize_intervention(
            "Pembrolizumab"
        )


class TestExtractDrugs:
    def test_only_therapeutic_agents_are_nodes(self):
        record = make_record(
            interventions=[
                ("DRUG", "Pembrolizumab"),
                ("BIOLOGICAL", "Nivolumab"),
                ("PROCEDURE", "Surgery"),
                ("DEVICE", "Stent"),
                ("BEHAVIORAL", "Counseling"),
            ]
        )
        assert [label for _, label in extract_drugs(record)] == [
            "Nivolumab",
            "Pembrolizumab",
        ]

    def test_placebo_is_excluded(self):
        """Placebo appears in a large share of trials and would make every
        graph a star centred on it."""
        record = make_record(
            interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Placebo")]
        )
        assert [label for _, label in extract_drugs(record)] == ["Pembrolizumab"]

    def test_same_drug_across_arms_counts_once(self):
        record = make_record(
            interventions=[("DRUG", "Pembrolizumab 200 mg"), ("DRUG", "Pembrolizumab")]
        )
        assert len(extract_drugs(record)) == 1

    def test_trial_with_no_interventions(self):
        assert extract_drugs(make_record(interventions=None)) == []


class TestCooccurrence:
    def build(self, **kwargs):
        records = {
            "NCT00000001": trial("NCT00000001", ["Pembrolizumab", "Carboplatin"]),
            "NCT00000002": trial("NCT00000002", ["Pembrolizumab", "Carboplatin"]),
            "NCT00000003": trial("NCT00000003", ["Pembrolizumab", "Lenvatinib"]),
        }
        return records, build_cooccurrence_network(records, **kwargs)

    def test_edge_weight_is_the_number_of_shared_trials(self):
        _, result = self.build(min_edge_weight=1)
        weights = {(e.source, e.target): e.weight for e in result.edges}
        assert weights[("drug:carboplatin", "drug:pembrolizumab")] == 2
        assert weights[("drug:lenvatinib", "drug:pembrolizumab")] == 1

    def test_edge_cites_exactly_the_trials_containing_both_endpoints(self):
        """An edge is a strong claim -- its citations must show both drugs."""
        records, result = self.build(min_edge_weight=1)
        edge = next(
            e for e in result.edges if e.target == "drug:pembrolizumab"
            and e.source == "drug:lenvatinib"
        )
        assert edge.nct_ids == ["NCT00000003"]
        drugs = {k for k, _ in extract_drugs(records["NCT00000003"])}
        assert {"pembrolizumab", "lenvatinib"} <= drugs

    def test_node_size_is_distinct_trial_count(self):
        _, result = self.build(min_edge_weight=1)
        sizes = {n.id: n.size for n in result.nodes}
        assert sizes["drug:pembrolizumab"] == 3
        assert sizes["drug:carboplatin"] == 2

    def test_min_edge_weight_drops_incidental_pairs(self):
        _, result = self.build(min_edge_weight=2)
        pairs = {(e.source, e.target) for e in result.edges}
        assert ("drug:carboplatin", "drug:pembrolizumab") in pairs
        assert ("drug:lenvatinib", "drug:pembrolizumab") not in pairs
        # Lenvatinib is now isolated, so it is not a node either.
        assert "drug:lenvatinib" not in {n.id for n in result.nodes}

    def test_edges_are_undirected_and_deduplicated(self):
        records = {
            "NCT00000001": trial("NCT00000001", ["A drug", "B drug"]),
            "NCT00000002": trial("NCT00000002", ["B drug", "A drug"]),
        }
        result = build_cooccurrence_network(records, min_edge_weight=1)
        assert len(result.edges) == 1
        assert result.edges[0].weight == 2

    def test_isolated_nodes_are_dropped(self):
        records = {"NCT00000001": trial("NCT00000001", ["Solo drug"])}
        result = build_cooccurrence_network(records, min_edge_weight=1)
        assert result.nodes == []

    def test_truncation_is_reported_not_silent(self):
        records = {
            f"NCT{i:08d}": trial(f"NCT{i:08d}", ["Anchor", f"Drug {i}", "Shared"])
            for i in range(40)
        }
        result = build_cooccurrence_network(records, min_edge_weight=1, max_nodes=10)
        assert len(result.nodes) == 10
        assert result.truncated_to_top_n == 10

    def test_no_dangling_edges_after_pruning(self):
        records = {
            f"NCT{i:08d}": trial(f"NCT{i:08d}", ["Anchor", f"Drug {i}"])
            for i in range(30)
        }
        result = build_cooccurrence_network(records, min_edge_weight=1, max_nodes=5)
        node_ids = {n.id for n in result.nodes}
        for edge in result.edges:
            assert edge.source in node_ids and edge.target in node_ids

    def test_deterministic(self):
        records, first = self.build(min_edge_weight=1)
        second = build_cooccurrence_network(records, min_edge_weight=1)
        assert first.model_dump() == second.model_dump()


class TestBipartite:
    def records(self):
        return {
            "NCT00000001": trial("NCT00000001", ["Pembrolizumab"], sponsor="Merck"),
            "NCT00000002": trial("NCT00000002", ["Pembrolizumab"], sponsor="Merck"),
            "NCT00000003": trial("NCT00000003", ["Nivolumab"], sponsor="BMS"),
        }

    def test_links_sponsors_to_the_drugs_they_study(self):
        result = build_bipartite_network(self.records())
        edges = {(e.source, e.target): e.weight for e in result.edges}
        assert edges[("sponsor:merck", "drug:pembrolizumab")] == 2
        assert edges[("sponsor:bms", "drug:nivolumab")] == 1

    def test_nodes_are_typed_so_a_frontend_can_two_tone_them(self):
        result = build_bipartite_network(self.records())
        kinds = {n.id: n.kind for n in result.nodes}
        assert kinds["sponsor:merck"] == "sponsor"
        assert kinds["drug:pembrolizumab"] == "drug"

    def test_graph_is_bipartite_no_sponsor_sponsor_edges(self):
        result = build_bipartite_network(self.records())
        kinds = {n.id: n.kind for n in result.nodes}
        for edge in result.edges:
            assert kinds[edge.source] != kinds[edge.target]

    def test_pruning_keeps_the_graph_connected(self):
        records = {
            f"NCT{i:08d}": trial(f"NCT{i:08d}", [f"Drug {i}"], sponsor=f"Sponsor {i}")
            for i in range(30)
        }
        result = build_bipartite_network(records, max_left=5, max_right=5)
        node_ids = {n.id for n in result.nodes}
        for edge in result.edges:
            assert edge.source in node_ids and edge.target in node_ids
        assert result.truncated_to_top_n is not None


class TestAgainstRealData:
    def test_cooccurrence_over_a_live_page_is_grounded(self, live_page):
        records = {
            s["protocolSection"]["identificationModule"]["nctId"]: s
            for s in live_page["studies"]
        }
        result = build_cooccurrence_network(records, min_edge_weight=1)
        for node in result.nodes:
            assert node.size == len(set(node.nct_ids))
            assert all(i in records for i in node.nct_ids)
        for edge in result.edges:
            assert edge.weight == len(set(edge.nct_ids))
            # Independently re-verify: both endpoints really are in each trial.
            for nct_id in edge.nct_ids:
                keys = {k for k, _ in extract_drugs(records[nct_id])}
                assert edge.source.split(":", 1)[1] in keys
                assert edge.target.split(":", 1)[1] in keys


class TestRxNormMerging:
    """Brand/generic merging via a resolution map.

    The load-bearing property is that merging happens at *key construction*, so
    a merged node's ``nct_ids`` is a genuine set union of every contributing
    name's trials -- which is what makes size, edge weight, citations and
    total_supporting_trials all agree without a reconciliation step.
    """

    PEMBRO = "1547545"

    def resolutions(self):
        from app.models.schemas import DrugResolution

        return {
            "keytruda": DrugResolution(
                rxcui=self.PEMBRO, canonical_name="pembrolizumab",
                original_names={"Keytruda"}, resolved=True, score=14.25,
            ),
            "pembrolizumab": DrugResolution(
                rxcui=self.PEMBRO, canonical_name="pembrolizumab",
                original_names={"Pembrolizumab"}, resolved=True, score=13.69,
            ),
        }

    def records(self):
        return {
            "NCT00000001": trial("NCT00000001", ["Keytruda", "Carboplatin"]),
            "NCT00000002": trial("NCT00000002", ["Pembrolizumab", "Carboplatin"]),
            "NCT00000003": trial("NCT00000003", ["Pembrolizumab", "Cisplatin"]),
        }

    def test_two_names_collapse_into_one_node(self):
        result = build_cooccurrence_network(
            self.records(), min_edge_weight=1, resolutions=self.resolutions()
        )
        drug_nodes = [n for n in result.nodes if n.kind == "drug"]
        ids = {n.id for n in drug_nodes}
        assert f"drug:rxcui:{self.PEMBRO}" in ids
        # The two source names no longer exist as separate nodes.
        assert "drug:keytruda" not in ids
        assert "drug:pembrolizumab" not in ids

    def test_merged_node_nct_ids_are_the_union_of_both_names(self):
        result = build_cooccurrence_network(
            self.records(), min_edge_weight=1, resolutions=self.resolutions()
        )
        node = next(n for n in result.nodes if n.rxcui == self.PEMBRO)
        assert node.nct_ids == ["NCT00000001", "NCT00000002", "NCT00000003"]
        assert node.size == 3

    def test_merged_node_carries_rxcui_and_its_source_names(self):
        result = build_cooccurrence_network(
            self.records(), min_edge_weight=1, resolutions=self.resolutions()
        )
        node = next(n for n in result.nodes if n.rxcui == self.PEMBRO)
        assert node.label == "pembrolizumab"
        assert node.merged_from == ["Keytruda", "Pembrolizumab"]

    def test_unmerged_nodes_have_no_rxcui_or_merged_from(self):
        result = build_cooccurrence_network(
            self.records(), min_edge_weight=1, resolutions=self.resolutions()
        )
        carbo = next(n for n in result.nodes if n.label == "Carboplatin")
        assert carbo.rxcui is None
        assert carbo.merged_from == []

    def test_edge_weights_sum_across_merged_names(self):
        """The reason merging must precede pruning: unmerged, keytruda-carboplatin
        and pembrolizumab-carboplatin are two weight-1 edges that min_edge_weight
        would discard. Merged, they are one weight-2 edge that survives."""
        records = self.records()
        unmerged = build_cooccurrence_network(records, min_edge_weight=2)
        assert unmerged.edges == []

        merged = build_cooccurrence_network(
            records, min_edge_weight=2, resolutions=self.resolutions()
        )
        edge = next(
            e for e in merged.edges if "carboplatin" in (e.source + e.target).lower()
        )
        assert edge.weight == 2
        assert edge.nct_ids == ["NCT00000001", "NCT00000002"]

    def test_citations_union_across_merged_names(self):
        """The requirement moving resolution could have broken: a merged node's
        citations must draw from both original names' trials."""
        from app.services.citations import build_citations
        from app.services.store import StudyStore

        records = self.records()
        store = StudyStore()
        store.add_records(records.values())
        result = build_cooccurrence_network(
            records, min_edge_weight=1, resolutions=self.resolutions()
        )
        node = next(n for n in result.nodes if n.rxcui == self.PEMBRO)

        citations, total = build_citations(node.nct_ids, store, "drug", limit=10)
        assert total == 3, "total_supporting_trials must count the union"
        cited = {c.nct_id for c in citations}
        # NCT...001 named the brand, NCT...002/003 the generic: all three cited.
        assert cited == {"NCT00000001", "NCT00000002", "NCT00000003"}

    def test_resolutions_none_is_byte_identical_to_before(self):
        records = self.records()
        assert (
            build_cooccurrence_network(records, min_edge_weight=1).model_dump()
            == build_cooccurrence_network(
                records, min_edge_weight=1, resolutions=None
            ).model_dump()
        )

    def test_unresolved_entries_do_not_merge(self):
        from app.models.schemas import DrugResolution

        resolutions = {
            "keytruda": DrugResolution(
                rxcui=None, canonical_name="keytruda", resolved=False
            )
        }
        result = build_cooccurrence_network(
            self.records(), min_edge_weight=1, resolutions=resolutions
        )
        ids = {n.id for n in result.nodes}
        assert "drug:keytruda" in ids and "drug:pembrolizumab" in ids

    def test_bipartite_merges_the_drug_side_only(self):
        result = build_bipartite_network(
            self.records(), resolutions=self.resolutions()
        )
        drug_ids = {n.id for n in result.nodes if n.kind == "drug"}
        assert f"drug:rxcui:{self.PEMBRO}" in drug_ids
        sponsors = [n for n in result.nodes if n.kind == "sponsor"]
        assert sponsors and all(n.rxcui is None for n in sponsors)
        edge = next(e for e in result.edges if e.target.endswith(self.PEMBRO))
        assert edge.weight == 3


class TestCandidatePool:
    def test_ranks_names_by_how_many_trials_mention_them(self):
        records = {
            "NCT00000001": trial("NCT00000001", ["Common", "Rare"]),
            "NCT00000002": trial("NCT00000002", ["Common"]),
            "NCT00000003": trial("NCT00000003", ["Common"]),
        }
        assert rank_candidate_names(records, top_k=1) == ["common"]

    def test_pool_is_capped(self):
        records = {
            f"NCT{i:08d}": trial(f"NCT{i:08d}", [f"Drug{i}"]) for i in range(50)
        }
        assert len(rank_candidate_names(records, top_k=10)) == 10

    def test_ranking_is_deterministic_on_ties(self):
        records = {"NCT00000001": trial("NCT00000001", ["Bbb", "Aaa", "Ccc"])}
        assert rank_candidate_names(records, top_k=3) == ["aaa", "bbb", "ccc"]


class TestExtractorErrorsSurface:
    """A signature mismatch and a bug are different things. Deciding by
    catching TypeError could not tell them apart."""

    def records(self):
        return {
            f"NCT0000000{i}": trial(f"NCT0000000{i}", ["Alpha", "Beta"]) for i in (1, 2)
        }

    def resolutions(self):
        from app.models.schemas import DrugResolution

        return {
            "alpha": DrugResolution(
                rxcui="111", canonical_name="alpha", resolved=True
            )
        }

    def test_a_bug_in_the_resolution_branch_is_not_swallowed(self):
        """The dangerous shape: the extractor accepts `resolutions` but its
        resolution-handling branch is broken. The old catch-and-retry swallowed
        the error, silently re-ran without resolutions, and RxNorm merging
        quietly stopped working with nothing reported."""

        def buggy(record, *, resolutions=None):
            if resolutions is not None:
                len(None)  # genuine bug, unrelated to the signature
            return [("alpha", "Alpha"), ("beta", "Beta")]

        with pytest.raises(TypeError, match="NoneType"):
            build_cooccurrence_network(
                self.records(),
                extract=buggy,
                resolutions=self.resolutions(),
                min_edge_weight=1,
            )

    def test_an_extractor_without_the_keyword_is_still_supported(self):
        """Older/simpler extractors keep working -- decided by signature, not
        by provoking an exception."""

        def simple(record):
            return [("alpha", "Alpha"), ("beta", "Beta")]

        result = build_cooccurrence_network(
            self.records(),
            extract=simple,
            resolutions=self.resolutions(),
            min_edge_weight=1,
        )
        assert {n.label for n in result.nodes} == {"Alpha", "Beta"}

    def test_an_extractor_with_the_keyword_receives_resolutions(self):
        seen = {}

        def capturing(record, *, resolutions=None):
            seen["got"] = resolutions is not None
            return [("alpha", "Alpha"), ("beta", "Beta")]

        build_cooccurrence_network(
            self.records(),
            extract=capturing,
            resolutions=self.resolutions(),
            min_edge_weight=1,
        )
        assert seen["got"] is True

    def test_kwargs_only_extractor_is_treated_as_accepting_resolutions(self):
        seen = {}

        def flexible(record, **kwargs):
            seen["got"] = "resolutions" in kwargs
            return [("alpha", "Alpha"), ("beta", "Beta")]

        build_cooccurrence_network(
            self.records(),
            extract=flexible,
            resolutions=self.resolutions(),
            min_edge_weight=1,
        )
        assert seen["got"] is True


class TestNetworkKinds:
    """node_kind labels the nodes; result_kind names the graph. They are
    different concepts, and the result kind used to be hardcoded while the
    signature implied otherwise."""

    def records(self):
        return {
            "NCT00000001": trial("NCT00000001", ["Alpha", "Beta"]),
            "NCT00000002": trial("NCT00000002", ["Alpha", "Beta"]),
        }

    def test_result_kind_is_honoured(self):
        result = build_cooccurrence_network(
            self.records(), result_kind="sponsor_drug", min_edge_weight=1
        )
        assert result.kind == "sponsor_drug"

    def test_result_kind_defaults_to_drug_drug(self):
        assert build_cooccurrence_network(self.records(), min_edge_weight=1).kind == (
            "drug_drug"
        )

    def bipartite_records(self):
        return {
            "NCT00000001": trial("NCT00000001", ["Pembrolizumab"], sponsor="Merck"),
            "NCT00000002": trial("NCT00000002", ["Nivolumab"], sponsor="BMS"),
        }

    def test_bipartite_result_kind_is_honoured(self):
        """The sibling gained this last pass and the bipartite builder did not,
        so its returned kind contradicted whatever pair it was given."""
        result = build_bipartite_network(
            self.bipartite_records(), result_kind="drug_drug"
        )
        assert result.kind == "drug_drug"

    def test_bipartite_result_kind_defaults_to_sponsor_drug(self):
        assert build_bipartite_network(self.bipartite_records()).kind == "sponsor_drug"

    def test_both_builders_expose_the_same_kind_parameters(self):
        """Guards the asymmetry itself, not just one instance of it."""
        import inspect

        for fn in (build_cooccurrence_network, build_bipartite_network):
            assert "result_kind" in inspect.signature(fn).parameters

    def test_node_kind_drives_node_ids_and_kinds(self):
        result = build_cooccurrence_network(
            self.records(), node_kind="sponsor", min_edge_weight=1
        )
        assert all(n.id.startswith("sponsor:") for n in result.nodes)
        assert all(n.kind == "sponsor" for n in result.nodes)
        assert all(e.source.startswith("sponsor:") for e in result.edges)
