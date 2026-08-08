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
