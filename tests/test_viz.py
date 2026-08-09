"""Chart formatting: the canonical applied scope behind every title.

Titles are built from the plan, never from the LLM, so a caption cannot
describe something other than what was queried.
"""

from app.models.schemas import ExtractedEntities, QueryPlan, YearRange
from app.services.dimensions import get_dimension
from app.services.viz import applied_scope, build_title


class TestAppliedScope:
    """One canonical description of the population being measured, reused by
    the chart title and the network title. Naming only the first drug captioned
    a pembrolizumab-or-nivolumab chart "Pembrolizumab trials"."""

    def plan(self, **kwargs):
        entities = ExtractedEntities(
            drugs=kwargs.pop("drugs", []),
            conditions=kwargs.pop("conditions", []),
            sponsors=kwargs.pop("sponsors", []),
            phases=kwargs.pop("phases", []),
            statuses=kwargs.pop("statuses", []),
            countries=kwargs.pop("countries", []),
            year_range=kwargs.pop("year_range", None),
        )
        return QueryPlan(
            query=kwargs.pop("query", "a question"),
            query_type=kwargs.pop("query_type", "distribution"),
            entities=entities,
            group_by=kwargs.pop("group_by", "phase"),
            viz_type=kwargs.pop("viz_type", "bar_chart"),
            **kwargs,
        )

    def title(self, **kwargs):
        plan = self.plan(**kwargs)
        return build_title(plan, get_dimension(plan.group_by or "phase"))

    def test_a_single_entity_is_unchanged(self):
        assert self.title(drugs=["Pembrolizumab"]) == "Pembrolizumab trials by phase"

    def test_a_two_drug_union_names_both(self):
        assert (
            self.title(drugs=["Pembrolizumab", "Nivolumab"])
            == "Pembrolizumab or Nivolumab trials by phase"
        )

    def test_a_two_condition_union_names_both(self):
        assert "Melanoma or lung cancer trials" in self.title(
            conditions=["melanoma", "lung cancer"]
        )

    def test_country_is_reflected_and_reads_as_a_qualifier(self):
        title = self.title(drugs=["Pembrolizumab"], countries=["United States"])
        assert title == "Pembrolizumab trials in United States by phase"

    def test_a_year_range_is_reflected(self):
        title = self.title(
            conditions=["melanoma"], year_range=YearRange(start=2018, end=2024)
        )
        assert "2018" in title and "2024" in title

    def test_a_comparison_names_both_entities(self):
        title = self.title(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
        )
        assert "Pembrolizumab vs Nivolumab" in title

    def test_an_excluded_status_is_reflected(self):
        title = self.title(conditions=["melanoma"], excluded_statuses=["RECRUITING"])
        assert "not recruiting" in title

    def test_many_values_are_truncated_and_the_remainder_counted(self):
        """A title has to stay readable; the count keeps it honest."""
        title = self.title(drugs=["A", "B", "C", "D", "E"])
        assert "A, B, C or 2 more trials" in title

    def test_no_entities_falls_back_to_a_generic_subject(self):
        assert self.title() == "Clinical trials by phase"

    def test_the_scope_is_a_pure_function_of_the_plan(self):
        """Never asked of the LLM, so the caption cannot drift from the query."""
        plan = self.plan(drugs=["Pembrolizumab", "Nivolumab"])
        assert applied_scope(plan) == applied_scope(plan)
        assert "Nivolumab" in applied_scope(plan)
