"""Stage 1: grounding, reconciliation, and structured-override merging.

The LLM call itself is mocked; what is tested here is the deterministic layer
around it -- the part that decides what the model is *allowed* to influence.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agents.understanding import (
    UnderstandingError,
    _appears_in,
    _match_statuses,
    build_plan,
    call_llm,
    extract_phases_from_query,
    extract_statuses_from_query,
    extract_years_from_query,
    ground_compare_entities,
    ground_entities,
    reconcile_viz_type,
)
from app.models.schemas import (
    ExtractedEntities,
    QueryPlan,
    QueryRequest,
    QueryUnderstanding,
    YearRange,
)
from app.services.ctgov import build_searches


def understanding(**kwargs):
    entities = ExtractedEntities(
        drugs=kwargs.pop("drugs", []),
        conditions=kwargs.pop("conditions", []),
        sponsors=kwargs.pop("sponsors", []),
        phases=kwargs.pop("phases", []),
        statuses=kwargs.pop("statuses", []),
        countries=kwargs.pop("countries", []),
        year_range=kwargs.pop("year_range", None),
    )
    return QueryUnderstanding(
        query_type=kwargs.pop("query_type", "distribution"),
        entities=entities,
        group_by=kwargs.pop("group_by", "phase"),
        compare_entities=kwargs.pop("compare_entities", []),
        compare_entity_kind=kwargs.pop("compare_entity_kind", None),
        network_kind=kwargs.pop("network_kind", None),
        viz_type=kwargs.pop("viz_type", "bar_chart"),
        assumptions=kwargs.pop("assumptions", []),
    )


class TestEntityGrounding:
    """The one way a language model could still affect real data is by naming an
    entity nobody asked about. Grounding removes that."""

    def test_keeps_entities_present_in_the_question(self):
        entities, warnings = ground_entities(
            understanding(drugs=["Pembrolizumab"]),
            "How many Pembrolizumab trials are there?",
        )
        assert entities.drugs == ["Pembrolizumab"]
        assert warnings == []

    def test_drops_a_hallucinated_entity_and_says_so(self):
        entities, warnings = ground_entities(
            understanding(drugs=["Keytruda"]),
            "How many nivolumab trials are there?",
        )
        assert entities.drugs == []
        assert any("Keytruda" in w for w in warnings)

    def test_matching_is_case_insensitive(self):
        entities, _ = ground_entities(
            understanding(drugs=["pembrolizumab"]), "Trials for PEMBROLIZUMAB"
        )
        assert entities.drugs == ["pembrolizumab"]

    def test_tolerates_minor_inflection(self):
        entities, _ = ground_entities(
            understanding(conditions=["lung cancer"]), "trials in lung cancers"
        )
        assert entities.conditions == ["lung cancer"]

    def test_unrelated_drug_names_do_not_pass_the_fuzzy_match(self):
        entities, _ = ground_entities(
            understanding(drugs=["Nivolumab"]), "trials for pembrolizumab"
        )
        assert entities.drugs == []

    def test_phases_come_from_the_query_text(self):
        """Out-of-range values are discarded, and so is any in-range phase the
        question never mentions."""
        entities, _ = ground_entities(
            understanding(phases=[3, 7, 0]), "phase 3 trials"
        )
        assert entities.phases == [3]

    def test_grounding_applies_to_every_free_text_entity_kind(self):
        entities, warnings = ground_entities(
            understanding(
                drugs=["Ghost"], conditions=["Phantom"], sponsors=["Nobody"],
                countries=["Atlantis"],
            ),
            "a question about nothing in particular",
        )
        assert (entities.drugs, entities.conditions, entities.sponsors,
                entities.countries) == ([], [], [], [])
        assert len(warnings) == 4


class TestVizReconciliation:
    """The model proposes a chart type; a deterministic table decides."""

    @pytest.mark.parametrize(
        "query_type,proposed,expected",
        [
            ("distribution", "network_graph", "bar_chart"),
            ("time_trend", "network_graph", "time_series"),
            ("comparison", "bar_chart", "grouped_bar_chart"),
            ("relationship", "bar_chart", "network_graph"),
            ("geographic", "network_graph", "geo_bar_chart"),
        ],
    )
    def test_incoherent_choice_is_corrected(self, query_type, proposed, expected):
        viz, notes = reconcile_viz_type(
            understanding(query_type=query_type, viz_type=proposed)
        )
        assert viz == expected
        assert notes and expected in notes[0]

    def test_legitimate_alternatives_are_left_alone(self):
        viz, notes = reconcile_viz_type(
            understanding(query_type="distribution", viz_type="histogram")
        )
        assert viz == "histogram"
        assert notes == []

    def test_correction_is_recorded_as_an_assumption(self):
        plan = build_plan(
            QueryRequest(query="Show a network of sponsors and drugs for melanoma"),
            understanding(query_type="relationship", viz_type="bar_chart"),
        )
        assert plan.viz_type == "network_graph"
        assert any("network_graph" in a for a in plan.assumptions)


class TestStructuredOverrides:
    """An explicitly-passed field is a stated fact; it beats inference."""

    def test_drug_name_overrides_extraction(self):
        plan = build_plan(
            QueryRequest(query="trials for pembrolizumab", drug_name="Nivolumab"),
            understanding(drugs=["pembrolizumab"]),
        )
        assert plan.entities.drugs == ["Nivolumab"]

    def test_override_survives_grounding_even_though_it_is_not_in_the_text(self):
        plan = build_plan(
            QueryRequest(query="how are these trials distributed?", condition="melanoma"),
            understanding(),
        )
        assert plan.entities.conditions == ["melanoma"]

    def test_year_range_from_structured_fields(self):
        plan = build_plan(
            QueryRequest(query="trials over time", start_year=2015, end_year=2020),
            understanding(query_type="time_trend", group_by="start_year"),
        )
        assert plan.entities.year_range == YearRange(start=2015, end=2020)

    def test_output_controls_flow_into_the_plan(self):
        plan = build_plan(
            QueryRequest(query="a question", max_citations_per_datum=7, max_studies=500),
            understanding(),
        )
        assert plan.max_citations_per_datum == 7
        assert plan.max_studies == 500


class TestDefaults:
    def test_time_trend_always_groups_by_year(self):
        plan = build_plan(
            QueryRequest(query="trials per year"),
            understanding(query_type="time_trend", group_by="phase"),
        )
        assert plan.group_by == "start_year"

    def test_missing_group_by_falls_back_per_query_type(self):
        plan = build_plan(
            QueryRequest(query="which countries run the most trials"),
            understanding(query_type="geographic", group_by=None, viz_type="geo_bar_chart"),
        )
        assert plan.group_by == "country"

    def test_relationship_defaults_to_drug_cooccurrence_and_says_so(self):
        plan = build_plan(
            QueryRequest(query="show the relationships between things"),
            understanding(query_type="relationship", viz_type="network_graph", group_by=None),
        )
        assert plan.network_kind == "drug_drug"
        assert any("co-occurrence" in a for a in plan.assumptions)

    def test_comparison_with_nothing_to_compare_degrades_to_a_distribution(self):
        """Better to answer the answerable part than to error out."""
        plan = build_plan(
            QueryRequest(query="compare these trials"),
            understanding(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                compare_entities=["Pembrolizumab"],
                compare_entity_kind="drug",
            ),
        )
        assert plan.viz_type == "bar_chart"
        assert plan.compare_entities == []
        assert any("Fewer than two" in a for a in plan.assumptions)


class TestRequestValidation:
    def test_rejects_an_inverted_year_range(self):
        with pytest.raises(ValueError, match="start_year"):
            QueryRequest(query="trials", start_year=2020, end_year=2015)

    def test_rejects_an_unknown_field(self):
        with pytest.raises(ValueError):
            QueryRequest(query="trials", not_a_field="x")

    def test_rejects_an_out_of_range_phase(self):
        with pytest.raises(ValueError):
            QueryRequest(query="trials", phase=9)


class TestDeterministicPhaseExtraction:
    """Phases are read from the query text rather than trusted from the model,
    so a phase filter provably comes from the user's words."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Phase 3 melanoma trials", [3]),
            ("phase 1/2 studies", [1, 2]),
            ("phases 2 and 3", [2, 3]),
            ("Phase II or III trials", [2, 3]),
            ("phase I trials", [1]),
            ("trials in phase 4", [4]),
        ],
    )
    def test_reads_phases_the_user_wrote(self, query, expected):
        assert extract_phases_from_query(query) == expected

    def test_only_the_contiguous_list_after_phase_counts(self):
        """Anchoring on the word "phase" is what stops an unrelated number from
        becoming a filter."""
        assert extract_phases_from_query("phase 2 study of 3 drugs") == [2]

    @pytest.mark.parametrize(
        "query",
        [
            "melanoma trials",
            "How are lung cancer trials distributed across phases?",
            "Compare phases for trials involving Drug A vs Drug B",
        ],
    )
    def test_no_phase_stated_means_no_phase_filter(self, query):
        assert extract_phases_from_query(query) == []

    def test_a_year_is_never_read_as_a_phase(self):
        assert extract_phases_from_query("phase 3 trials started in 2020") == [3]


class TestDeterministicStatusExtraction:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("recruiting melanoma trials", ["RECRUITING"]),
            ("completed studies", ["COMPLETED"]),
            ("trials that were stopped early", ["TERMINATED"]),
            ("enrolling patients", ["RECRUITING"]),
            ("active not recruiting", ["ACTIVE_NOT_RECRUITING"]),
        ],
    )
    def test_maps_plain_words_to_the_api_vocabulary(self, query, expected):
        assert extract_statuses_from_query(query) == expected

    def test_longer_phrases_win_over_the_words_inside_them(self):
        """"not yet recruiting" must not also register as "recruiting" -- they
        are opposite filters."""
        assert extract_statuses_from_query("not yet recruiting trials") == [
            "NOT_YET_RECRUITING"
        ]

    def test_several_statuses(self):
        assert set(extract_statuses_from_query("terminated or suspended")) == {
            "TERMINATED",
            "SUSPENDED",
        }

    def test_no_status_stated(self):
        assert extract_statuses_from_query("melanoma trials") == []


class TestYearGrounding:
    def test_reads_years_present_in_the_query(self):
        assert extract_years_from_query("between 2010 and 2020") == {2010, 2020}

    def test_a_dosage_is_not_a_year(self):
        assert extract_years_from_query("Pembrolizumab 200 mg trials") == set()

    def test_a_computed_year_is_not_grounded(self):
        """"the last five years" has no literal year, so no filter is applied
        rather than one the user never stated."""
        assert extract_years_from_query("trials over the last five years") == set()


class TestOverExtractionIsDroppedAndReported:
    """The model plausibly over-extracts on a bare query. Every such field
    changes which trials are fetched, so each must be dropped and reported."""

    BARE = "How are melanoma trials distributed?"

    def test_phase_not_in_the_question_is_dropped(self):
        entities, warnings = ground_entities(
            understanding(conditions=["melanoma"], phases=[3]), self.BARE
        )
        assert entities.phases == []
        assert any("phase 3" in w for w in warnings)

    def test_status_not_in_the_question_is_dropped(self):
        entities, warnings = ground_entities(
            understanding(conditions=["melanoma"], statuses=["recruiting"]), self.BARE
        )
        assert entities.statuses == []
        assert any("status" in w for w in warnings)

    def test_year_not_in_the_question_is_dropped(self):
        entities, warnings = ground_entities(
            understanding(conditions=["melanoma"], year_range=YearRange(start=2015, end=None)),
            self.BARE,
        )
        assert entities.year_range is None
        assert any("2015" in w for w in warnings)

    def test_an_unrecognised_status_string_never_reaches_the_filter(self):
        """Previously any non-empty string was accepted and then matched against
        overallStatus, silently filtering everything out."""
        entities, warnings = ground_entities(
            understanding(conditions=["melanoma"], statuses=["ongoing-ish"]), self.BARE
        )
        assert entities.statuses == []
        assert warnings

    def test_stated_values_survive(self):
        query = "recruiting phase 3 melanoma trials since 2015"
        entities, warnings = ground_entities(
            understanding(
                conditions=["melanoma"],
                phases=[3],
                statuses=["recruiting"],
                year_range=YearRange(start=2015, end=None),
            ),
            query,
        )
        assert entities.phases == [3]
        assert entities.statuses == ["RECRUITING"]
        assert entities.year_range == YearRange(start=2015, end=None)
        assert warnings == []

    def test_partial_year_range_keeps_the_grounded_bound(self):
        entities, warnings = ground_entities(
            understanding(year_range=YearRange(start=2015, end=2099)),
            "trials since 2015",
        )
        assert entities.year_range == YearRange(start=2015, end=None)
        assert any("2099" in w for w in warnings)


class TestCompareEntityGrounding:
    """Each comparison entity drives its own upstream search, so an invented one
    would fetch and chart trials nobody asked about."""

    def test_entities_named_in_the_question_survive(self):
        kept, warnings = ground_compare_entities(
            ["Pembrolizumab", "Nivolumab"],
            "Compare phases for Pembrolizumab vs Nivolumab.",
        )
        assert kept == ["Pembrolizumab", "Nivolumab"]
        assert warnings == []

    def test_an_invented_entity_is_dropped_and_reported(self):
        kept, warnings = ground_compare_entities(
            ["Pembrolizumab", "Keytruda"],
            "Compare phases for Pembrolizumab vs Nivolumab.",
        )
        assert kept == ["Pembrolizumab"]
        assert any("Keytruda" in w for w in warnings)

    def test_grounding_runs_through_build_plan(self):
        plan = build_plan(
            QueryRequest(query="Compare phases for Pembrolizumab vs Nivolumab."),
            understanding(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                compare_entities=["Pembrolizumab", "Atezolizumab"],
                compare_entity_kind="drug",
            ),
        )
        assert "Atezolizumab" not in plan.compare_entities
        assert any("Atezolizumab" in w for w in plan.warnings)
        # Only one entity survives, so the existing degrade path takes over.
        assert plan.viz_type == "bar_chart"


class TestEntityMatchingPrecision:
    """The matching heuristic, investigated rather than assumed safe.

    Two defects were found and fixed: an unanchored substring test that let an
    acronym match inside a longer word, and a fuzzy fallback whose
    length-relative threshold was too forgiving for short strings. These cases
    stay in the suite as the record of what was checked.
    """

    @pytest.mark.parametrize(
        "entity,query,why",
        [
            ("SCLC", "NSCLC trials", "small-cell vs non-small-cell: different diseases"),
            ("NSCLC", "SCLC trials", "and in the other direction"),
            ("ALL", "SMALL cell lung cancer trials", "a leukemia inside the word 'small'"),
            ("HCC", "HCV trials", "liver cancer vs hepatitis C"),
        ],
    )
    def test_acronyms_do_not_match_inside_longer_words(self, entity, query, why):
        assert _appears_in(entity, query) is False, why

    @pytest.mark.parametrize(
        "entity,query",
        [
            # Distinct agents with shared suffixes -- the -mab and -nib families
            # are where a loose threshold would do real damage.
            ("Nivolumab", "pembrolizumab trials"),
            ("Atezolizumab", "durvalumab trials"),
            ("Ipilimumab", "nivolumab trials"),
            ("Trastuzumab", "pertuzumab trials"),
            ("Olaparib", "niraparib trials"),
            ("Imatinib", "dasatinib trials"),
            ("Axitinib", "afatinib trials"),
            ("Lenvatinib", "lorlatinib trials"),
            ("Dabrafenib", "vemurafenib trials"),
            ("Carboplatin", "cisplatin trials"),
            ("Sunitinib", "sorafenib trials"),
            ("AML", "ALL trials"),
            ("CLL", "CML trials"),
        ],
    )
    def test_similar_names_are_not_confused(self, entity, query):
        assert _appears_in(entity, query) is False

    @pytest.mark.parametrize(
        "entity,query",
        [
            ("lung cancer", "trials in lung cancers"),
            ("melanoma", "melanomas trials"),
            ("breast cancer", "breast cancers"),
            ("glioblastoma", "glioblastomas"),
            # 7 characters: a fuzzy floor of 8 would have broken this, which is
            # why the floor is 6.
            ("sarcoma", "sarcomas"),
            ("non-small cell lung cancer", "non small cell lung cancer trials"),
            ("pembrolizumab", "PEMBROLIZUMAB trials"),
            ("5-FU", "5-FU and leucovorin"),
            ("HER2", "HER2 positive breast cancer"),
            ("NSCLC", "NSCLC trials"),
        ],
    )
    def test_legitimate_matches_still_work(self, entity, query):
        assert _appears_in(entity, query) is True

    def test_a_wrong_disease_is_dropped_and_reported(self):
        """End to end: the model extracts SCLC from an NSCLC question."""
        entities, warnings = ground_entities(
            understanding(conditions=["SCLC"]), "How are NSCLC trials distributed?"
        )
        assert entities.conditions == []
        assert any("SCLC" in w for w in warnings)


class TestMalformedCompletion:
    """A completion with no usable choices must fail as a typed LLM error, not
    as an IndexError that escapes to an unhandled 500."""

    def stub(self, choices):
        client = MagicMock()
        client.chat.completions.parse.return_value = SimpleNamespace(choices=choices)
        return client

    @pytest.mark.parametrize("choices", [[], None])
    def test_absent_choices_raise_understanding_error(self, choices):
        with (
            patch("app.agents.understanding._client", return_value=self.stub(choices)),
            pytest.raises(UnderstandingError, match="no completion choices"),
        ):
            call_llm("how many trials by phase")

    def test_route_returns_502_llm_error_not_500(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with (
            patch("app.agents.understanding._client", return_value=self.stub([])),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post("/api/v1/query", json={"query": "trials by phase"})
        assert response.status_code == 502
        assert response.json()["detail"]["error"]["code"] == "LLM_ERROR"


class TestStatusNegation:
    """A status named negatively must never become the positive filter. Before
    this guard, "trials that are not recruiting" sent aggFilters=status:rec --
    precisely the opposite set, applied upstream and silently."""

    def agg_filters_for(self, query):
        """The wire value is where the harm was, so assert on it directly."""
        statuses, _ = _match_statuses(query)
        plan = QueryPlan(
            query=query,
            query_type="distribution",
            group_by="phase",
            viz_type="bar_chart",
            entities=ExtractedEntities(
                drugs=["aspirin"], conditions=[], sponsors=[], phases=[],
                statuses=statuses, countries=[], year_range=None,
            ),
        )
        searches, _ = build_searches(plan)
        return searches[0].to_params(page_size=10).get("aggFilters")

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("recruiting studies", ["RECRUITING"]),
            ("completed", ["COMPLETED"]),
            ("terminated", ["TERMINATED"]),
            ("withdrawn", ["WITHDRAWN"]),
            ("enrolling by invitation", ["ENROLLING_BY_INVITATION"]),
            ("suspended trials", ["SUSPENDED"]),
        ],
    )
    def test_positive_statuses_are_unchanged(self, query, expected):
        assert _match_statuses(query) == (expected, [])

    @pytest.mark.parametrize(
        "query",
        [
            "not recruiting studies",
            "show trials that are not recruiting",
            "non-recruiting studies",
            "trials that aren't recruiting",
            "no longer recruiting",
            "trials not currently recruiting",
            "trials other than recruiting",
            "all trials excluding recruiting ones",
        ],
    )
    def test_negated_recruiting_is_never_read_as_recruiting(self, query):
        positive, negated = _match_statuses(query)
        assert "RECRUITING" not in positive
        assert negated == ["RECRUITING"]
        # The real damage was upstream, so pin the emitted filter too.
        assert self.agg_filters_for(query) is None

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("active, not recruiting", "ACTIVE_NOT_RECRUITING"),
            ("active not recruiting", "ACTIVE_NOT_RECRUITING"),
            ("not yet recruiting", "NOT_YET_RECRUITING"),
            ("not-yet-recruiting trials", "NOT_YET_RECRUITING"),
        ],
    )
    def test_specific_statuses_are_not_confused_with_recruiting(self, query, expected):
        """These contain "recruiting" and a "not", but are their own statuses --
        the negation check must not fire on the "not" that belongs to them."""
        positive, negated = _match_statuses(query)
        assert positive == [expected]
        assert negated == []
        assert self.agg_filters_for(query) is None

    def test_plain_recruiting_still_reaches_the_upstream_filter(self):
        """The one live-verified status filter must keep working."""
        assert self.agg_filters_for("recruiting melanoma trials") == "status:rec"

    def test_a_clause_boundary_stops_the_negation_spreading(self):
        """In "not completed and recruiting" the "and" starts a new clause, so
        only completed is excluded."""
        assert _match_statuses("not completed and recruiting") == (
            ["RECRUITING"], ["COMPLETED"]
        )

    def test_negation_after_a_positive_status_applies_only_to_its_own_term(self):
        assert _match_statuses("completed but not recruiting") == (
            ["COMPLETED"], ["RECRUITING"]
        )

    def test_a_status_both_requested_and_excluded_is_not_filtered_on(self):
        """Contradictory, so the safe reading wins: acting on the positive is
        the failure mode this exists to prevent."""
        positive, negated = _match_statuses("recruiting and not recruiting")
        assert "RECRUITING" not in positive
        assert "RECRUITING" in negated

    def test_the_exclusion_is_disclosed_rather_than_silently_dropped(self):
        grounded, warnings = ground_entities(
            understanding(conditions=["melanoma"]),
            "show me melanoma trials that are not recruiting",
        )
        assert grounded.statuses == []
        assert any("exclusion is not supported" in w for w in warnings)


class TestCompareEntityDeduplication:
    """Series membership and provenance are keyed by label in fetch(), so two
    entities normalising to one label made the second overwrite the first --
    losing a series and misattributing the survivor's ids."""

    QUERY = "compare Aspirin and aspirin and Ibuprofen trials"

    def test_exact_duplicates_collapse_to_one_series(self):
        kept, warnings = ground_compare_entities(["Aspirin", "Aspirin"], self.QUERY)
        assert kept == ["Aspirin"]
        assert any("duplicate" in w for w in warnings)

    def test_case_only_duplicates_collapse(self):
        """query.intr is case-insensitive upstream (Aspirin/aspirin/ASPIRIN all
        return 2,172), so these fetch identical trials."""
        kept, _ = ground_compare_entities(["Aspirin", "aspirin"], self.QUERY)
        assert kept == ["Aspirin"]
        kept, _ = ground_compare_entities(["Aspirin", "ASPIRIN"], self.QUERY)
        assert kept == ["Aspirin"]

    @pytest.mark.parametrize(
        "entities",
        [
            ["Aspirin", "Aspirin"],
            ["Aspirin", "aspirin"],
            ["Aspirin", " aspirin "],
            ["  Aspirin  ", "ASPIRIN"],
        ],
    )
    def test_every_equivalent_pair_from_the_brief_collapses(self, entities):
        """Whitespace, casing, and internal-space variants all name one series."""
        kept, warnings = ground_compare_entities(entities, self.QUERY)
        assert len(kept) == 1
        assert len(warnings) == 1

    def test_whitespace_variants_collapse(self):
        kept, _ = ground_compare_entities(["Aspirin", "  Aspirin  "], self.QUERY)
        assert kept == ["Aspirin"]

    def test_distinct_entities_are_never_merged(self):
        """The guard against over-merging: only provably-equivalent names go."""
        kept, warnings = ground_compare_entities(["Aspirin", "Ibuprofen"], self.QUERY)
        assert kept == ["Aspirin", "Ibuprofen"]
        assert warnings == []

    def test_first_occurrence_wins_and_order_is_deterministic(self):
        kept, _ = ground_compare_entities(
            ["aspirin", "Ibuprofen", "Aspirin"], self.QUERY
        )
        assert kept == ["aspirin", "Ibuprofen"]
        for _ in range(5):
            assert ground_compare_entities(
                ["aspirin", "Ibuprofen", "Aspirin"], self.QUERY
            )[0] == ["aspirin", "Ibuprofen"]

    def test_duplicates_produce_one_search_not_two(self):
        """The consequence that mattered: one series, one upstream call."""
        kept, _ = ground_compare_entities(["Aspirin", "aspirin"], self.QUERY)
        plan = QueryPlan(
            query=self.QUERY,
            query_type="comparison",
            group_by="phase",
            viz_type="grouped_bar_chart",
            compare_entities=kept,
            compare_entity_kind="drug",
            entities=ExtractedEntities(
                drugs=[], conditions=[], sponsors=[], phases=[], statuses=[],
                countries=[], year_range=None,
            ),
        )
        searches, _ = build_searches(plan)
        labels = [s.label for s in searches]
        assert len(labels) == len(set(labels)), "labels must be unique keys"

    def test_a_negated_status_is_not_also_reported_as_unsupported(self):
        """The question does name it, negatively -- so the generic "does not
        support it" message would be wrong alongside the exclusion notice."""
        _, warnings = ground_entities(
            understanding(conditions=["melanoma"], statuses=["recruiting"]),
            "melanoma trials that are not recruiting",
        )
        assert not any("does not support" in w for w in warnings)
        assert any("exclusion is not supported" in w for w in warnings)


class TestStatusWordBoundaries:
    """Matching was a raw substring search, so a vocabulary phrase buried
    inside an unrelated English word invented a filter the question never
    asked for: "unavailable" read as AVAILABLE, "incomplete" as COMPLETED."""

    @pytest.mark.parametrize(
        "query,leaked",
        [
            ("unavailable", "AVAILABLE"),
            ("temporarily unavailable", "AVAILABLE"),
            ("incomplete", "COMPLETED"),
            ("incompleted data", "COMPLETED"),
            ("uncompleted", "COMPLETED"),
            ("noncompleted", "COMPLETED"),
            ("preterminated", "TERMINATED"),
            ("reenrolling", "RECRUITING"),
            ("withdrawnness", "WITHDRAWN"),
        ],
    )
    def test_a_phrase_inside_a_longer_word_does_not_match(self, query, leaked):
        positive, negated = _match_statuses(query)
        assert leaked not in positive
        assert leaked not in negated

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("recruiting studies", "RECRUITING"),
            ("Recruiting studies", "RECRUITING"),
            ("completed", "COMPLETED"),
            ("terminated", "TERMINATED"),
            ("withdrawn", "WITHDRAWN"),
            ("enrolling by invitation", "ENROLLING_BY_INVITATION"),
            ("available", "AVAILABLE"),
            ("no longer available", "NO_LONGER_AVAILABLE"),
            ("suspended", "SUSPENDED"),
            ("on hold", "SUSPENDED"),
            ("stopped early", "TERMINATED"),
            ("unknown status", "UNKNOWN"),
        ],
    )
    def test_genuine_whole_word_mentions_still_match(self, query, expected):
        """The boundary guard must not cost any real match."""
        assert _match_statuses(query)[0] == [expected]

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("active, not recruiting", "ACTIVE_NOT_RECRUITING"),
            ("active not recruiting", "ACTIVE_NOT_RECRUITING"),
            ("not yet recruiting", "NOT_YET_RECRUITING"),
        ],
    )
    def test_shorter_phrases_do_not_leak_out_of_longer_ones(self, query, expected):
        """"recruiting" lives inside all three, and must not also register."""
        positive, negated = _match_statuses(query)
        assert positive == [expected]
        assert "RECRUITING" not in positive
        assert negated == []

    def test_punctuated_and_hyphenated_phrases_still_match(self):
        """The \\b anchors sit outside the comma and hyphens, so these are
        unaffected by the boundary change."""
        assert _match_statuses("active, not recruiting")[0] == [
            "ACTIVE_NOT_RECRUITING"
        ]
        assert _match_statuses("not-yet-recruiting trials")[0] == [
            "NOT_YET_RECRUITING"
        ]
