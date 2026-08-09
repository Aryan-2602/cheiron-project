"""Network graphs: normalization, co-occurrence, bipartite, and pruning."""

import pytest

from app.services.network import (
    MAX_ENTITIES_PER_RECORD,
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


class TestPostPruneNodeMembership:
    """A node in a relationship graph answers "how much of the drawn structure
    rests on this node". Global membership answered a different question: a hub
    showed 40 while its drawn edges accounted for 8, and clicking cited trials
    whose only connection had been pruned away."""

    def hub_graph(self, partners=40, max_nodes=5):
        records = {}
        for i in range(partners):
            for rep in (0, 1):  # twice, so each pair clears min_edge_weight
                nct = f"NCT{i * 2 + rep:08d}"
                records[nct] = trial(nct, ["HubDrug", f"Partner{i}"])
        return build_cooccurrence_network(records, max_nodes=max_nodes)

    def visible_trials(self, network, node_id):
        seen = set()
        for edge in network.edges:
            if node_id in (edge.source, edge.target):
                seen |= set(edge.nct_ids)
        return seen

    def test_node_size_matches_its_visible_edges(self):
        network = self.hub_graph()
        hub = next(n for n in network.nodes if n.label.lower() == "hubdrug")
        assert hub.size == len(self.visible_trials(network, hub.id))

    def test_every_cited_trial_is_on_a_visible_edge(self):
        network = self.hub_graph()
        hub = next(n for n in network.nodes if n.label.lower() == "hubdrug")
        assert set(hub.nct_ids) <= self.visible_trials(network, hub.id)

    def test_the_size_identity_still_holds(self):
        network = self.hub_graph()
        assert all(n.size == len(set(n.nct_ids)) for n in network.nodes)

    def test_pruning_shrinks_the_hub_rather_than_leaving_it_global(self):
        wide = self.hub_graph(partners=40, max_nodes=5)
        whole = self.hub_graph(partners=40, max_nodes=100)
        wide_hub = next(n for n in wide.nodes if n.label.lower() == "hubdrug")
        whole_hub = next(n for n in whole.nodes if n.label.lower() == "hubdrug")
        assert wide_hub.size < whole_hub.size

    def test_an_isolated_node_is_dropped_rather_than_shown_at_zero(self):
        """A point with no edge asserts a relationship that is not there."""
        network = self.hub_graph()
        assert all(
            any(n.id in (e.source, e.target) for e in network.edges)
            for n in network.nodes
        )

    def test_bipartite_nodes_follow_the_same_rule(self):
        records = {
            f"NCT{i:08d}": make_record(
                f"NCT{i:08d}", sponsor="Merck", interventions=[("DRUG", "Pembrolizumab")]
            )
            for i in range(3)
        }
        network = build_bipartite_network(records)
        assert all(n.size == len(set(n.nct_ids)) for n in network.nodes)
        for node in network.nodes:
            assert set(node.nct_ids) == self.visible_trials(network, node.id)


class TestDenseRecordsAreSkippedNotTruncated:
    """Pairs grow as the square of a trial's agent count, and this was the one
    unbounded resource in a project that caps the fetch, the candidate pool,
    the node count, the join and the title. Measured before the cap: 200 trials
    x 400 interventions took 46s and 863MB.

    The whole trial is skipped rather than its first N agents kept. Keeping a
    prefix would invent a co-occurrence structure that is an artifact of sort
    order, and the *absence* of a pair is something the drawing cannot show
    and the reader cannot question.
    """

    def records(self, dense=1, dense_size=100, normal=2):
        records = {
            f"NCT9{i:07d}": trial(
                f"NCT9{i:07d}", [f"Agent{j:03d}" for j in range(dense_size)]
            )
            for i in range(dense)
        }
        for i in range(normal):
            for rep in (0, 1):
                nct = f"NCT1{i * 2 + rep:07d}"
                records[nct] = trial(nct, ["Alpha", "Beta"])
        return records

    def test_a_dense_trial_contributes_no_pairs(self):
        network = build_cooccurrence_network(self.records())
        assert network.dense_records_skipped == 1
        # Only the ordinary trials' pair survives.
        assert [(e.source, e.target) for e in network.edges] == [
            ("drug:alpha", "drug:beta")
        ]

    def test_the_skip_is_order_independent(self):
        """A prefix cap would give different graphs for different orderings of
        the same corpus. Skipping cannot."""
        records = self.records(dense=2)
        forward = build_cooccurrence_network(records)
        reversed_ = build_cooccurrence_network(dict(reversed(list(records.items()))))
        assert [(e.source, e.target, e.weight) for e in forward.edges] == [
            (e.source, e.target, e.weight) for e in reversed_.edges
        ]
        assert forward.dense_records_skipped == reversed_.dense_records_skipped == 2

    def test_a_trial_at_the_cap_is_still_paired(self):
        """The bound must not touch real combination therapy."""
        network = build_cooccurrence_network(
            self.records(dense=2, dense_size=MAX_ENTITIES_PER_RECORD, normal=0)
        )
        assert network.dense_records_skipped == 0
        assert network.edges

    def test_nothing_is_skipped_in_an_ordinary_corpus(self):
        network = build_cooccurrence_network(self.records(dense=0))
        assert network.dense_records_skipped == 0


class TestCompoundDosageUnits:
    """"mg/kg" was listed after "mg" in the alternation, so the shorter branch
    matched first and left the divisor behind: "Pembrolizumab 200 mg/kg"
    normalised to "pembrolizumab kg", a different node from plain
    "pembrolizumab" -- exactly the fragmentation this function prevents."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Pembrolizumab 200 mg/kg", "pembrolizumab"),
            ("Pembrolizumab 200mg/kg", "pembrolizumab"),
            ("Drug 5 mg/m2", "drug"),
            ("Drug 3 mg/kg/day", "drug"),
            ("Drug 5 g/kg", "drug"),
            ("Drug 10 ml", "drug"),
            ("Drug 20 units", "drug"),
            ("Vitamin D 1000 IU", "vitamin d"),
            ("Nab-paclitaxel 125 mg/m2", "nab-paclitaxel"),
        ],
    )
    def test_a_dose_is_stripped_whole(self, name, expected):
        assert normalize_intervention(name) == expected

    def test_every_dose_form_of_one_agent_is_one_node(self):
        forms = [
            "Pembrolizumab",
            "Pembrolizumab 200 mg",
            "Pembrolizumab 200 mg/kg",
            "Pembrolizumab 2 mg/kg/day",
        ]
        assert len({normalize_intervention(f) for f in forms}) == 1

    @pytest.mark.parametrize(
        "name", ["Interleukin 2", "COVID 19 vaccine", "Carboplatin AUC 5"]
    )
    def test_a_number_that_is_not_a_dose_is_kept(self, name):
        assert normalize_intervention(name) == name.lower()
