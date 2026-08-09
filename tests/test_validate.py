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
from tests.conftest import make_record


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


class TestZeroCitationRequests:
    """0 is a documented value for max_citations_per_datum, so an empty
    citation list is legitimate -- but only when it was actually asked for."""

    def test_empty_citations_pass_when_none_were_requested(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        for row in response.visualization.data:
            row["citations"] = []
        assert (
            validate_response(
                response, store, aggregation=result, max_citations_per_datum=0
            )
            is response
        )

    def test_empty_citations_still_fail_when_citations_were_requested(self, good):
        response, store, result = good
        response = response.model_copy(deep=True)
        response.visualization.data[0]["citations"] = []
        with pytest.raises(ValidationFailure, match="no citations backing it"):
            validate_response(
                response, store, aggregation=result, max_citations_per_datum=3
            )

    def test_citations_present_despite_a_zero_request_is_rejected(self, good):
        """Suppression must be honest in both directions."""
        response, store, result = good
        with pytest.raises(ValidationFailure, match="asked for none"):
            validate_response(
                response, store, aggregation=result, max_citations_per_datum=0
            )

    def test_other_citation_rules_still_apply_at_zero(self, good):
        """Zero citations is not a way past the rest of the validator."""
        response, store, result = good
        response = response.model_copy(deep=True)
        for row in response.visualization.data:
            row["citations"] = []
        response.meta.api_urls = []
        with pytest.raises(ValidationFailure, match="upstream API URLs"):
            validate_response(
                response, store, aggregation=result, max_citations_per_datum=0
            )


class TestPublishedNetworkValues:
    """The internal checks prove the graph is self-consistent. These prove
    formatting did not alter it on the way out."""

    def network_case(self):
        records = {
            "NCT00000001": make_record(
                "NCT00000001", interventions=[("DRUG", "A"), ("DRUG", "B")]
            ),
            "NCT00000002": make_record(
                "NCT00000002", interventions=[("DRUG", "A"), ("DRUG", "B")]
            ),
            "NCT00000003": make_record(
                "NCT00000003", interventions=[("DRUG", "A"), ("DRUG", "C")]
            ),
        }
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        network = build_cooccurrence_network(store.records, min_edge_weight=1)
        spec = build_network_spec(network, plan(query_type="relationship"), store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        return response, store, network

    def test_known_good_network_passes(self):
        response, store, network = self.network_case()
        assert validate_response(response, store, network=network) is response

    def test_rejects_an_inflated_published_edge_weight(self):
        """A weight of 2 published as 70 previously shipped unnoticed."""
        response, store, network = self.network_case()
        response.visualization.data[0]["edges"][0]["weight"] = 70
        with pytest.raises(ValidationFailure, match="reports weight 70"):
            validate_response(response, store, network=network)

    def test_rejects_an_altered_published_node_size(self):
        response, store, network = self.network_case()
        response.visualization.data[0]["nodes"][0]["size"] = 999
        with pytest.raises(ValidationFailure, match="reports size 999"):
            validate_response(response, store, network=network)

    def test_rejects_a_published_node_absent_from_the_graph(self):
        response, store, network = self.network_case()
        response.visualization.data[0]["nodes"][0]["id"] = "drug:invented"
        # Caught by the set-equality check, which names both sides of the
        # difference rather than only the unexpected id.
        with pytest.raises(ValidationFailure, match="drug:invented"):
            validate_response(response, store, network=network)


class TestCitationsBelongToTheirDatum:
    """A citation naming a real trial is not enough -- it has to be one of the
    trials that produced *this* number."""

    def test_rejects_a_real_trial_from_the_wrong_bucket(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        result = aggregate(records, DIMENSIONS["phase"])
        spec = build_chart_spec(result, plan(), DIMENSIONS["phase"], store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )

        # Find a row and a fetched trial that did NOT contribute to it.
        row_index, datum = next(
            (i, d) for i, d in enumerate(result.data) if d.nct_ids
        )
        intruder = next(n for n in store.nct_ids if n not in datum.nct_ids)
        response.visualization.data[row_index]["citations"] = [
            {"nct_id": intruder, "excerpt": "real trial, wrong bucket",
             "url": store.study_url(intruder)}
        ]

        # It is a genuine fetched record, so the store-membership check passes...
        assert intruder in store
        # ...but it did not produce this datum.
        with pytest.raises(ValidationFailure, match="not one of the"):
            validate_response(response, store, aggregation=result)

    def test_rejects_a_wrong_bucket_citation_on_a_network_node(self):
        response, store, network = TestPublishedNetworkValues().network_case()
        # Drug C appears in one trial only, so another fetched trial is
        # guaranteed to be a real record that did not produce this node.
        node_row = min(
            response.visualization.data[0]["nodes"], key=lambda n: n["size"]
        )
        source = next(n for n in network.nodes if n.id == node_row["id"])
        intruder = next(n for n in sorted(store.nct_ids) if n not in set(source.nct_ids))

        node_row["citations"] = [
            {"nct_id": intruder, "excerpt": "real trial, wrong node",
             "url": store.study_url(intruder)}
        ]
        assert intruder in store
        with pytest.raises(ValidationFailure, match="not one of the"):
            validate_response(response, store, network=network)

    def test_correct_citations_still_pass(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        result = aggregate(records, DIMENSIONS["phase"])
        spec = build_chart_spec(result, plan(), DIMENSIONS["phase"], store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        assert validate_response(response, store, aggregation=result) is response


class TestNetworkCompleteness:
    """The validator checked that every *published* item matched a computed
    one, but never the reverse -- so a dropped node or a duplicated edge passed
    silently. Both directions are now asserted against the final pruned graph,
    so intentional pruning stays valid."""

    network_case = TestPublishedNetworkValues.network_case

    def test_a_valid_graph_passes(self):
        response, store, network = self.network_case()
        validate_response(response, store, network=network)

    def test_rejects_a_missing_node(self):
        response, store, network = self.network_case()
        response.visualization.data[0]["nodes"].pop()
        with pytest.raises(ValidationFailure, match="differs from the computed graph"):
            validate_response(response, store, network=network)

    def test_rejects_a_duplicated_node(self):
        response, store, network = self.network_case()
        nodes = response.visualization.data[0]["nodes"]
        nodes.append(dict(nodes[0]))
        with pytest.raises(ValidationFailure, match="repeats a node id"):
            validate_response(response, store, network=network)

    def test_rejects_a_missing_edge(self):
        response, store, network = self.network_case()
        response.visualization.data[0]["edges"].pop()
        with pytest.raises(ValidationFailure, match="distinct edges"):
            validate_response(response, store, network=network)

    def test_rejects_a_duplicated_edge(self):
        response, store, network = self.network_case()
        edges = response.visualization.data[0]["edges"]
        edges.append(dict(edges[0]))
        with pytest.raises(ValidationFailure, match="repeats an edge"):
            validate_response(response, store, network=network)

    def test_rejects_an_extra_invented_edge(self):
        response, store, network = self.network_case()
        edges = response.visualization.data[0]["edges"]
        edges.append({**edges[0], "source": "drug:a", "target": "drug:invented"})
        with pytest.raises(ValidationFailure):
            validate_response(response, store, network=network)

    def test_rejects_a_wrong_node_size(self):
        response, store, network = self.network_case()
        response.visualization.data[0]["nodes"][0]["size"] += 1
        with pytest.raises(ValidationFailure, match="reports size"):
            validate_response(response, store, network=network)

    def test_rejects_a_wrong_edge_weight(self):
        response, store, network = self.network_case()
        response.visualization.data[0]["edges"][0]["weight"] += 1
        with pytest.raises(ValidationFailure, match="reports weight"):
            validate_response(response, store, network=network)

    def test_rejects_more_than_one_data_container(self):
        response, store, network = self.network_case()
        response.visualization.data.append(dict(response.visualization.data[0]))
        with pytest.raises(ValidationFailure, match="exactly one data container"):
            validate_response(response, store, network=network)


class TestSupportingTotalIsExact:
    """total_supporting_trials tells a reader how much evidence a truncated
    citation list stands for. Checking only citations <= total let a wrong
    total misstate that while every individual citation still checked out."""

    def case(self, records):
        store = StudyStore()
        store.add_records(records.values())
        store.add_url("https://example.test")
        result = aggregate(records, DIMENSIONS["phase"])
        spec = build_chart_spec(result, plan(), DIMENSIONS["phase"], store)
        response = QueryResponse(
            visualization=spec,
            meta=Meta(total_studies_processed=len(store), api_urls=store.api_urls),
        )
        return response, store, result

    def test_a_correct_total_passes(self, records):
        response, store, aggregation = self.case(records)
        validate_response(response, store, aggregation=aggregation)

    def test_rejects_a_total_one_too_small(self, records):
        response, store, aggregation = self.case(records)
        row = response.visualization.data[0]
        row["total_supporting_trials"] -= 1
        with pytest.raises(ValidationFailure, match="supporting trials"):
            validate_response(response, store, aggregation=aggregation)

    def test_rejects_a_total_one_too_large(self, records):
        response, store, aggregation = self.case(records)
        response.visualization.data[0]["total_supporting_trials"] += 1
        with pytest.raises(ValidationFailure, match="supporting trials"):
            validate_response(response, store, aggregation=aggregation)

    def test_a_sampled_citation_list_with_a_correct_total_passes(self, records):
        """The normal case: fewer citations than contributors."""
        response, store, aggregation = self.case(records)
        row = response.visualization.data[0]
        row["citations"] = row["citations"][:1]
        validate_response(response, store, aggregation=aggregation)

    def test_zero_citation_limit_still_requires_an_exact_total(self, records):
        response, store, aggregation = self.case(records)
        for row in response.visualization.data:
            row["citations"] = []
        validate_response(response, store, aggregation=aggregation, max_citations_per_datum=0)
        response.visualization.data[0]["total_supporting_trials"] += 1
        with pytest.raises(ValidationFailure, match="supporting trials"):
            validate_response(
                response, store, aggregation=aggregation, max_citations_per_datum=0
            )
