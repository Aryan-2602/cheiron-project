"""Stage 1: grounding, reconciliation, and structured-override merging.

The LLM call itself is mocked; what is tested here is the deterministic layer
around it -- the part that decides what the model is *allowed* to influence.
"""

import pytest

from app.agents.understanding import (
    build_plan,
    ground_entities,
    reconcile_viz_type,
)
from app.models.schemas import (
    ExtractedEntities,
    QueryRequest,
    QueryUnderstanding,
    YearRange,
)


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

    def test_phases_are_range_checked_rather_than_text_matched(self):
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
