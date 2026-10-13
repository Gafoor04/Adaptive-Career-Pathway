"""Unit tests for skill readiness, career-role rankings, skill gaps, and roadmaps."""

import pytest

from app.domain.competency_loader import CompetencyRepository
from app.domain.recommendation import (
    calculate_skill_readiness,
    rank_career_roles,
    generate_learning_roadmap,
)


class TestRecommendationEngine:
    def test_weighted_skill_readiness_calculation(self, real_repository: CompetencyRepository):
        # Empty mastery dictionary falls back gracefully to concept default baseline p_l0
        readiness_list = calculate_skill_readiness(real_repository, {})
        assert len(readiness_list) == len(real_repository.list_skills())
        for item in readiness_list:
            assert 0.0 <= item["readiness"] <= 1.0
            assert item["readiness_percent"] == round(item["readiness"] * 100, 1)

    def test_skill_readiness_with_custom_mastery(self, real_repository: CompetencyRepository):
        # Give concept in core programming high mastery
        skill = real_repository.get_skill("skill-core-programming")
        assert skill is not None
        concepts_for_skill = real_repository.get_concepts_for_skill(skill.id)
        assert len(concepts_for_skill) > 0

        # Set all concepts for this skill to 0.90
        custom_mastery = {concept.id: 0.90 for concept, _ in concepts_for_skill}
        readiness_list = calculate_skill_readiness(real_repository, custom_mastery)
        core_skill = next(s for s in readiness_list if s["skill_id"] == "skill-core-programming")
        assert core_skill["readiness"] == 0.90

    def test_career_role_ranking_competency_fit(self, real_repository: CompetencyRepository):
        rankings = rank_career_roles(real_repository, {})
        assert len(rankings) == len(real_repository.list_roles())

        # Verify sorted in descending order of fit_score
        scores = [r["fit_score"] for r in rankings]
        assert scores == sorted(scores, reverse=True)

        for r in rankings:
            assert 0.0 <= r["fit_score"] <= 1.0
            assert "fit_percent" in r
            assert "competency-fit" in r["description"].lower()

    def test_skill_gaps_identified_accurately(self, real_repository: CompetencyRepository):
        # With default baseline (around 0.10), almost all skills have gaps against target >= 0.70
        rankings = rank_career_roles(real_repository, {})
        top_role = rankings[0]
        assert len(top_role["skill_gaps"]) > 0

        for gap in top_role["skill_gaps"]:
            assert gap["current_readiness"] < gap["target_mastery"]
            expected_gap = round(gap["target_mastery"] - gap["current_readiness"], 4)
            assert gap["gap"] == expected_gap
            assert gap["importance"] > 0

    def test_skill_gaps_empty_when_all_skills_exceed_target(self, real_repository: CompetencyRepository):
        # Set all concepts to 1.0 mastery
        all_mastered = {c.id: 1.0 for c in real_repository.list_concepts()}
        rankings = rank_career_roles(real_repository, all_mastered)
        for r in rankings:
            assert r["fit_score"] == 1.0
            assert len(r["skill_gaps"]) == 0

    def test_roadmap_unknown_role_raises_value_error(self, real_repository: CompetencyRepository):
        with pytest.raises(ValueError, match="Unknown career role"):
            generate_learning_roadmap(real_repository, {}, role_id="role-non-existent")

    def test_roadmap_excludes_already_mastered_concepts(self, real_repository: CompetencyRepository):
        role_id = "role-backend-eng"
        # If all concepts are mastered, roadmap should be empty
        all_mastered = {c.id: 1.0 for c in real_repository.list_concepts()}
        roadmap = generate_learning_roadmap(
            real_repository,
            all_mastered,
            role_id=role_id,
            mastery_threshold=0.70,
        )
        assert roadmap == []

    def test_roadmap_strictly_respects_prerequisite_constraints(self, real_repository: CompetencyRepository):
        role_id = "role-backend-eng"
        dag = real_repository.prerequisite_dag

        # Low initial mastery
        low_mastery = {c.id: 0.10 for c in real_repository.list_concepts()}
        roadmap = generate_learning_roadmap(
            real_repository,
            low_mastery,
            role_id=role_id,
            mastery_threshold=0.70,
            limit=30,
        )

        assert len(roadmap) > 0
        scheduled_ids = [item["concept_id"] for item in roadmap]
        pos = {cid: idx for idx, cid in enumerate(scheduled_ids)}

        # Verify that for any pair (u, v) in the roadmap where u is a prerequisite of v,
        # pos[u] < pos[v] strictly!
        for u in scheduled_ids:
            for v in scheduled_ids:
                if u != v and real_repository.prerequisite_dag.has_edge(u, v):
                    assert pos[u] < pos[v], (
                        f"Prerequisite violation: '{u}' is prerequisite for '{v}', "
                        f"but '{u}' is at pos {pos[u]} and '{v}' is at pos {pos[v]}"
                    )

    def test_roadmap_includes_weak_prerequisites_before_dependents(self, real_repository: CompetencyRepository):
        # Test specific known dependency: 'c-prog-fundamentals' -> 'c-linear-data-structures'
        role_id = "role-backend-eng"
        low_mastery = {c.id: 0.15 for c in real_repository.list_concepts()}

        roadmap = generate_learning_roadmap(
            real_repository,
            low_mastery,
            role_id=role_id,
            mastery_threshold=0.70,
            limit=25,
        )
        scheduled_ids = [item["concept_id"] for item in roadmap]
        pos = {cid: idx for idx, cid in enumerate(scheduled_ids)}

        if "c-prog-fundamentals" in pos and "c-linear-data-structures" in pos:
            assert pos["c-prog-fundamentals"] < pos["c-linear-data-structures"]

    def test_roadmap_reasons_and_metadata(self, real_repository: CompetencyRepository):
        role_id = "role-backend-eng"
        roadmap = generate_learning_roadmap(
            real_repository,
            {},
            role_id=role_id,
            mastery_threshold=0.70,
            limit=5,
        )
        assert len(roadmap) == 5
        for item in roadmap:
            assert "concept_id" in item
            assert "concept_name" in item
            assert "mastery_gap" in item
            assert item["mastery_gap"] > 0
            assert "reason" in item
            assert len(item["reason"]) > 10

    def test_concurrency_cannot_precede_programming_foundations_despite_larger_gap(
        self, real_repository: CompetencyRepository
    ):
        """Regression test: Even when Concurrency has a much larger mastery gap than Foundations,
        Foundations is a strict prerequisite and MUST be scheduled first in the roadmap.
        """
        role_id = "role-backend-eng"
        # Foundations is partially mastered (gap = 0.10), Concurrency has baseline (gap = 0.60)
        custom_mastery = {
            "c-prog-fundamentals": 0.60,
            "c-async-concurrency": 0.10,
        }
        roadmap = generate_learning_roadmap(
            real_repository,
            custom_mastery,
            role_id=role_id,
            mastery_threshold=0.70,
            limit=25,
        )
        scheduled_ids = [item["concept_id"] for item in roadmap]
        assert "c-prog-fundamentals" in scheduled_ids
        assert "c-async-concurrency" in scheduled_ids

        pos_foundations = scheduled_ids.index("c-prog-fundamentals")
        pos_concurrency = scheduled_ids.index("c-async-concurrency")

        assert pos_foundations < pos_concurrency, (
            f"Prerequisite failure: Programming Foundations (pos={pos_foundations}) "
            f"must precede Concurrency (pos={pos_concurrency}) despite gap differences."
        )
