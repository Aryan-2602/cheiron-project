"""Network graphs: normalization, co-occurrence, bipartite, and pruning."""

from app.services.network import (
    build_bipartite_network,
    build_cooccurrence_network,
    extract_drugs,
    normalize_intervention,
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
