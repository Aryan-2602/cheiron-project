"""Deep citations: every datum traceable to the records that produced it."""

from app.services.aggregate import aggregate
from app.services.citations import build_citations, build_excerpt
from app.services.dimensions import DIMENSIONS
from app.services.store import StudyStore
from tests.conftest import make_record


def store_from(records):
    store = StudyStore()
    store.add_records(records.values())
    return store


class TestExcerpts:
    """Excerpts quote the exact response field that caused bucket membership,
    so a reader can verify the claim against the raw API record."""

    def test_names_the_field_path_and_its_value(self):
        record = make_record(title="A phase 3 study", phases=["PHASE3"])
        excerpt = build_excerpt(record, "phase")
        assert "A phase 3 study" in excerpt
        assert "designModule.phases" in excerpt
        assert "PHASE3" in excerpt

    def test_multi_valued_fields_list_every_value(self):
        record = make_record(phases=["PHASE1", "PHASE2"])
        assert "[PHASE1, PHASE2]" in build_excerpt(record, "phase")

    def test_missing_field_is_reported_honestly(self):
        """A trial in the 'no phase data' bucket is there *because* the field is
        absent; the excerpt should say so rather than fabricate one."""
        excerpt = build_excerpt(make_record(phases=None), "phase")
        assert "not reported" in excerpt

    def test_long_titles_are_clipped_not_dropped(self):
        record = make_record(title="X" * 400, phases=["PHASE1"])
        excerpt = build_excerpt(record, "phase")
        assert "..." in excerpt
        assert len(excerpt) < 400

    def test_every_dimension_produces_a_non_empty_excerpt(self, records):
        for name in DIMENSIONS:
            for record in records.values():
                assert build_excerpt(record, name).strip()

    def test_drug_excerpt_lists_the_interventions(self):
        record = make_record(
            interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Lenvatinib")]
        )
        excerpt = build_excerpt(record, "drug")
        assert "Pembrolizumab" in excerpt and "Lenvatinib" in excerpt


class TestBuildCitations:
    def test_reports_full_contributor_count_even_when_truncated(self, records):
        store = store_from(records)
        ids = sorted(records)
        citations, total = build_citations(ids, store, "phase", limit=2)
        assert len(citations) == 2
        assert total == len(ids) == 4

    def test_limit_zero_still_reports_the_total(self, records):
        store = store_from(records)
        citations, total = build_citations(sorted(records), store, "phase", limit=0)
        assert citations == []
        assert total == 4

    def test_ids_absent_from_the_store_are_never_cited(self, records):
        """An unverifiable citation is worse than a missing one."""
        store = store_from(records)
        citations, total = build_citations(["NCT99999999"], store, "phase", limit=5)
        assert citations == []
        assert total == 1

    def test_citations_carry_a_resolvable_url(self, records):
        store = store_from(records)
        citations, _ = build_citations(["NCT00000001"], store, "phase", limit=1)
        assert citations[0].url == "https://clinicaltrials.gov/study/NCT00000001"


class TestCitationsTrackAggregation:
    """Citations are a projection of the aggregation, not a separate lookup --
    so they cannot disagree with the number they support."""

    def test_every_bucket_cites_only_trials_that_are_in_it(self, records):
        store = store_from(records)
        result = aggregate(records, DIMENSIONS["phase"])
        for datum in result.data:
            citations, total = build_citations(
                datum.nct_ids, store, "phase", limit=10
            )
            assert total == datum.value
            for citation in citations:
                assert citation.nct_id in datum.nct_ids
                # The cited record really does carry the value claimed.
                record = store.get(citation.nct_id)
                phases = DIMENSIONS["phase"].extract(record)
                labels = [DIMENSIONS["phase"].display(p) for p in phases]
                assert datum.key in labels or datum.key == "No phase data"

    def test_citations_from_real_api_data_are_all_grounded(self, live_page):
        records = {
            s["protocolSection"]["identificationModule"]["nctId"]: s
            for s in live_page["studies"]
        }
        store = store_from(records)
        result = aggregate(records, DIMENSIONS["phase"])
        for datum in result.data:
            citations, total = build_citations(datum.nct_ids, store, "phase", limit=3)
            assert total == datum.value
            assert all(c.nct_id in store for c in citations)
            assert all(c.excerpt.strip() for c in citations)


class TestCitationSelection:
    """Contributors arrive sorted ascending and NCT ids are assigned roughly
    chronologically, so taking the first N cited every bucket's oldest members
    -- a 373-trial bucket evidenced by three 1990s studies."""

    def bucket(self, n):
        return [f"NCT{i:08d}" for i in range(1, n + 1)]

    def store_for(self, ids):
        return store_from({i: make_record(i, phases=["PHASE2"]) for i in ids})

    def cite(self, ids, limit):
        citations, total = build_citations(
            ids, self.store_for(ids), "phase", limit=limit
        )
        return [c.nct_id for c in citations], total

    def test_sample_spans_the_range_instead_of_clustering_at_the_start(self):
        ids = self.bucket(373)
        cited, total = self.cite(ids, 3)
        assert total == 373
        assert cited == [ids[0], ids[186], ids[-1]]

    def test_the_full_contributor_count_is_still_reported(self):
        _, total = self.cite(self.bucket(373), 3)
        assert total == 373

    def test_selection_is_stable_across_repeated_calls(self):
        ids = self.bucket(200)
        assert self.cite(ids, 4)[0] == self.cite(ids, 4)[0]

    def test_limit_at_or_above_the_bucket_size_returns_everything_in_order(self):
        ids = self.bucket(3)
        assert self.cite(ids, 3)[0] == ids
        assert self.cite(ids, 10)[0] == ids

    def test_limit_of_one_returns_exactly_one(self):
        assert len(self.cite(self.bucket(50), 1)[0]) == 1

    def test_limit_of_zero_returns_none_but_still_reports_the_total(self):
        cited, total = self.cite(self.bucket(50), 0)
        assert cited == []
        assert total == 50

    def test_never_emits_more_citations_than_requested(self):
        ids = self.bucket(97)
        for limit in range(1, 21):
            assert len(self.cite(ids, limit)[0]) <= limit

    def test_every_cited_id_is_a_real_contributor(self):
        """The validator rejects a citation outside the datum's contributor
        set, so this is the property the whole selection must preserve."""
        ids = self.bucket(97)
        for limit in (1, 3, 7, 20):
            assert set(self.cite(ids, limit)[0]) <= set(ids)


class TestEdgeEvidenceCoversBothEndpoints:
    """An edge asserts a link between two things, so its citation has to
    evidence both. A sponsor-drug edge cited with intervention names alone
    proved the drug end and left the sponsor end on trust."""

    def graph(self, kind):
        from app.models.schemas import ExtractedEntities, QueryPlan
        from app.services.network import (
            build_bipartite_network,
            build_cooccurrence_network,
        )
        from app.services.viz import build_network_spec

        records = [
            make_record(
                f"NCT0000000{i}",
                sponsor="Merck",
                interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")],
            )
            for i in (1, 2)
        ]
        store = StudyStore()
        store.add_records(records)
        store.add_url("https://example.test")
        network = (
            build_bipartite_network(store.records)
            if kind == "sponsor_drug"
            else build_cooccurrence_network(store.records, min_edge_weight=1)
        )
        plan = QueryPlan(
            query="q", query_type="relationship", group_by=None,
            viz_type="network_graph", network_kind=kind,
            entities=ExtractedEntities(
                drugs=[], conditions=[], sponsors=[], phases=[], statuses=[],
                countries=[], year_range=None,
            ),
        )
        return build_network_spec(network, plan, store), store, network

    def test_a_sponsor_drug_edge_cites_the_sponsor_field(self):
        spec, _, _ = self.graph("sponsor_drug")
        excerpt = spec.data[0]["edges"][0]["citations"][0]["excerpt"]
        assert "leadSponsor.name" in excerpt
        assert "Merck" in excerpt

    def test_a_sponsor_drug_edge_still_cites_the_drug_field(self):
        spec, _, _ = self.graph("sponsor_drug")
        excerpt = spec.data[0]["edges"][0]["citations"][0]["excerpt"]
        assert "interventions[].name" in excerpt

    def test_the_cited_trial_belongs_to_the_edge(self):
        spec, _, network = self.graph("sponsor_drug")
        edge_row = spec.data[0]["edges"][0]
        edge = next(
            e for e in network.edges
            if (e.source, e.target) == (edge_row["source"], edge_row["target"])
        )
        for citation in edge_row["citations"]:
            assert citation["nct_id"] in edge.nct_ids

    def test_a_drug_drug_edge_is_unchanged(self):
        """The intervention list already names both endpoints there."""
        spec, _, _ = self.graph("drug_drug")
        excerpt = spec.data[0]["edges"][0]["citations"][0]["excerpt"]
        assert "interventions[].name" in excerpt
        assert "leadSponsor" not in excerpt

    def test_the_excerpt_quotes_real_fields_not_prose(self):
        spec, _, _ = self.graph("sponsor_drug")
        excerpt = spec.data[0]["edges"][0]["citations"][0]["excerpt"]
        # Every fragment is a path: value pair from the record itself.
        assert excerpt.count(":") >= 2
