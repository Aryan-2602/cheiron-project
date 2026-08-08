"""The validator, driven by deliberately corrupted responses.

Each test breaks one specific property of a known-good response and asserts the
matching check fires. This is the test file that backs the claim that a
fabricated or drifted value cannot reach a caller.
"""


import pytest

from app.models.schemas import (
    Encoding,
    ExtractedEntities,
    FieldRef,
    Meta,
    QueryPlan,
    QueryResponse,
    VisualizationSpec,
)
from app.services.aggregate import aggregate
from app.services.dimensions import DIMENSIONS
from app.services.network import build_cooccurrence_network
from app.services.store import StudyStore
from app.services.validate import ValidationFailure, validate_response
from app.services.viz import build_chart_spec, build_network_spec


def plan(**kwargs):
    return QueryPlan(
        query=kwargs.pop("query", "How are trials distributed across phases?"),
        query_type=kwargs.pop("query_type", "distribution"),
        entities=ExtractedEntities(
            drugs=[], conditions=[], sponsors=[], phases=[], statuses=[],
            countries=[], year_range=None,
        ),
        group_by=kwargs.pop("group_by", "phase"),
        viz_type=kwargs.pop("viz_type", "bar_chart"),
        **kwargs,
    )


@pytest.fixture
def good(records):
    """A known-good response plus the artefacts the validator checks against."""
    store = StudyStore()
    store.add_records(records.values())
    store.add_url("https://clinicaltrials.gov/api/v2/studies?query.cond=test")
    result = aggregate(records, DIMENSIONS["phase"])
    spec = build_chart_spec(result, plan(), DIMENSIONS["phase"], store)
    response = QueryResponse(
        visualization=spec,
        meta=Meta(
            total_studies_processed=len(store),
            api_urls=store.api_urls,
        ),
    )
    return response, store, result


class TestAcceptsValidResponses:
    def test_known_good_response_passes(self, good):
        response, store, result = good
        assert validate_response(response, store, aggregation=result) is response

    def test_network_response_passes(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        network = build_cooccurrence_network(store.records, min_edge_weight=1)
        spec = build_network_spec(network, plan(query_type="relationship"), store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        assert validate_response(response, store, network=network) is response


class TestCountIntegrity:
    def test_rejects_a_value_that_does_not_match_its_contributors(self, good):
        """The core anti-hallucination check: a number nobody can back up."""
        response, store, result = good
        tampered = result.model_copy(deep=True)
        tampered.data[0].value = 999
        with pytest.raises(ValidationFailure, match="claims 999 trials"):
            validate_response(response, store, aggregation=tampered)

    def test_rejects_a_published_row_edited_after_aggregation(self, good):
        """Catches drift introduced between measuring and formatting."""
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data[0]["trial_count"] += 1
        with pytest.raises(ValidationFailure, match="do not match the aggregation"):
            validate_response(response, store, aggregation=result)

    def test_rejects_an_invented_extra_row(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data.append(
            {
                "phase": "Phase 99",
                "trial_count": 42,
                "citations": [],
                "total_supporting_trials": 42,
            }
        )
        with pytest.raises(ValidationFailure):
            validate_response(response, store, aggregation=result)

    def test_rejects_a_network_node_size_that_does_not_match(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        network = build_cooccurrence_network(store.records, min_edge_weight=1)
        spec = build_network_spec(network, plan(query_type="relationship"), store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        tampered = network.model_copy(deep=True)
        tampered.nodes[0].size = 500
        with pytest.raises(ValidationFailure, match="claims size 500"):
            validate_response(response, store, network=tampered)

    def test_rejects_an_edge_pointing_at_a_missing_node(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        network = build_cooccurrence_network(store.records, min_edge_weight=1)
        spec = build_network_spec(network, plan(query_type="relationship"), store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        tampered = network.model_copy(deep=True)
        tampered.edges[0].target = "drug:does-not-exist"
        with pytest.raises(ValidationFailure, match="not in the graph"):
            validate_response(response, store, network=tampered)


class TestCitationGrounding:
    def test_rejects_a_citation_to_a_trial_we_never_fetched(self, good):
        """An id outside the store cannot have come from the aggregation."""
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data[0]["citations"][0]["nct_id"] = "NCT12345678"
        with pytest.raises(ValidationFailure, match="not among the"):
            validate_response(response, store, aggregation=result)

    def test_rejects_an_empty_excerpt(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data[0]["citations"][0]["excerpt"] = "   "
        with pytest.raises(ValidationFailure, match="empty excerpt"):
            validate_response(response, store, aggregation=result)

    def test_rejects_a_value_with_no_supporting_citations(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data[0]["citations"] = []
        with pytest.raises(ValidationFailure, match="no citations backing it"):
            validate_response(response, store, aggregation=result)

    def test_rejects_more_citations_than_claimed_supporters(self, good):
        """total_supporting_trials is what tells a reader the citation list was
        truncated, so it must never understate the evidence."""
        response, store, result = good
        response = response.model_copy(deep=True)
        row = response.visualization.data[0]
        assert row["citations"], "fixture should have at least one citation"
        row["total_supporting_trials"] = len(row["citations"]) - 1
        with pytest.raises(ValidationFailure, match="citations but claims only"):
            validate_response(response, store, aggregation=result)

    def test_zero_valued_rows_may_have_no_citations(self, good):
        """A zero-filled year has nothing to cite -- and that is correct."""
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data.append(
            {"phase": "Phase 9", "trial_count": 0, "citations": [], "total_supporting_trials": 0}
        )
        # Fails on the row-match check, not the citation check.
        with pytest.raises(ValidationFailure, match="do not match the aggregation"):
            validate_response(response, store, aggregation=result)


class TestEncodingContract:
    def test_rejects_an_encoding_field_absent_from_the_data(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.encoding.x = FieldRef(field="nonexistent")
        with pytest.raises(ValidationFailure, match="absent from data row"):
            validate_response(response, store, aggregation=result)

    def test_rejects_a_network_encoding_key_absent_from_nodes(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        network = build_cooccurrence_network(store.records, min_edge_weight=1)
        spec = build_network_spec(network, plan(query_type="relationship"), store)
        spec.encoding.nodes = {"id": "id", "size": "not_a_key"}
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        with pytest.raises(ValidationFailure, match="absent from a node"):
            validate_response(response, store, network=network)


class TestEmptyAndMeta:
    def test_rejects_a_chart_with_no_rows(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data = []
        with pytest.raises(ValidationFailure, match="no data rows"):
            validate_response(response, store, aggregation=result)

    def test_rejects_a_network_with_no_nodes(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        response = QueryResponse(
            visualization=VisualizationSpec(
                type="network_graph",
                title="empty",
                encoding=Encoding(nodes={"id": "id"}, edges={}),
                data=[{"nodes": [], "edges": []}],
            ),
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        with pytest.raises(ValidationFailure, match="no nodes"):
            validate_response(response, store)

    def test_rejects_a_processed_count_that_disagrees_with_the_store(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.meta.total_studies_processed = 9999
        with pytest.raises(ValidationFailure, match="studies processed"):
            validate_response(response, store, aggregation=result)

    def test_requires_the_upstream_urls_to_be_recorded(self, good):
        """Without the URLs a reader cannot reproduce the query."""
        response, store, result = good
        response = response.model_copy(deep=True)
        response.meta.api_urls = []
        with pytest.raises(ValidationFailure, match="upstream API URLs"):
            validate_response(response, store, aggregation=result)


class TestMergeInvariant:
    """A merge asserts that several names are one compound. The validator
    refuses to publish that claim without the identity that justifies it."""

    def network_response(self, records, mutate=None):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        network = build_cooccurrence_network(store.records, min_edge_weight=1)
        if mutate:
            mutate(network)
        spec = build_network_spec(network, plan(query_type="relationship"), store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        return response, store, network

    def test_rejects_a_merge_with_no_rxnorm_identity(self, records):
        def mutate(network):
            network.nodes[0].merged_from = ["Keytruda", "Pembrolizumab"]
            network.nodes[0].rxcui = None

        response, store, network = self.network_response(records, mutate)
        with pytest.raises(ValidationFailure, match="no RxNorm identity"):
            validate_response(response, store, network=network)

    def test_rejects_a_merge_of_a_single_name(self, records):
        def mutate(network):
            network.nodes[0].merged_from = ["Pembrolizumab"]
            network.nodes[0].rxcui = "1547545"

        response, store, network = self.network_response(records, mutate)
        with pytest.raises(ValidationFailure, match="fewer than two names"):
            validate_response(response, store, network=network)

    def test_a_resolved_node_without_a_merge_is_fine(self, records):
        """One name resolving to an ingredient is normal -- not every resolution
        collapses two names."""
        def mutate(network):
            network.nodes[0].rxcui = "1547545"
            network.nodes[0].merged_from = []

        response, store, network = self.network_response(records, mutate)
        assert validate_response(response, store, network=network) is response
