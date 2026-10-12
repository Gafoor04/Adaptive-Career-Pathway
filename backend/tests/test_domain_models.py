"""Unit tests for Pydantic domain models and field validation rules."""

import pytest
from pydantic import ValidationError

from app.domain.models import (
    AssessmentAttempt,
    AssessmentItemResponse,
    AssessmentType,
    BKTParameters,
    CareerRole,
    Concept,
    LearnerInteractionEvent,
    PrerequisiteRelation,
    PrerequisiteType,
    Question,
    RoleSkillRequirement,
)


class TestBKTParameters:
    def test_valid_default_parameters(self):
        params = BKTParameters()
        assert params.p_l0 == 0.10
        assert params.p_transition == 0.15
        assert params.p_guess == 0.20
        assert params.p_slip == 0.10

    def test_custom_valid_parameters(self):
        params = BKTParameters(p_l0=0.30, p_transition=0.20, p_guess=0.25, p_slip=0.05)
        assert params.p_l0 == 0.30
        assert params.p_guess == 0.25

    def test_guess_boundary_rejection(self):
        # Guess must be strictly less than 0.5
        with pytest.raises(ValidationError):
            BKTParameters(p_guess=0.55)

    def test_slip_boundary_rejection(self):
        # Slip must be strictly less than 0.5
        with pytest.raises(ValidationError):
            BKTParameters(p_slip=0.50)

    def test_monotonicity_constraint(self):
        # Parameters must satisfy P(correct | mastered) > P(correct | unmastered), i.e., 1 - S > G
        params = BKTParameters(p_guess=0.20, p_slip=0.10)
        assert (1.0 - params.p_slip) > params.p_guess


class TestConceptModel:
    def test_valid_concept_creation(self):
        c = Concept(
            id="c-recursion",
            slug="recursion-basics",
            name="Recursion Basics",
            description="Base cases and recurrence relations.",
            category="Algorithms",
        )
        assert c.id == "c-recursion"
        assert c.slug == "recursion-basics"
        assert c.default_bkt_params.p_l0 == 0.10

    def test_invalid_slug_with_uppercase_or_spaces(self):
        with pytest.raises(ValidationError):
            Concept(
                id="c-recursion",
                slug="Recursion Basics",
                name="Recursion Basics",
                description="Valid description.",
                category="Algorithms",
            )

    def test_empty_string_rejection(self):
        with pytest.raises(ValidationError):
            Concept(
                id="   ",
                slug="valid-slug",
                name="Valid Name",
                description="Valid description.",
                category="Algorithms",
            )


class TestPrerequisiteRelationModel:
    def test_valid_relation(self):
        rel = PrerequisiteRelation(
            concept_id="c-dp",
            prerequisite_id="c-recursion",
            relation_type=PrerequisiteType.STRICT,
            rationale="DP memoization requires recursion understanding.",
        )
        assert rel.concept_id == "c-dp"
        assert rel.prerequisite_id == "c-recursion"

    def test_self_dependency_raises_error(self):
        with pytest.raises(ValidationError) as excinfo:
            PrerequisiteRelation(
                concept_id="c-dp",
                prerequisite_id="c-dp",
                relation_type=PrerequisiteType.STRICT,
            )
        assert "Self-prerequisite detected" in str(excinfo.value)


class TestQuestionModel:
    def test_valid_question(self):
        q = Question(
            id="q-001",
            concept_id="c-sql",
            prompt="What is the result of a LEFT JOIN when right table has no match?",
            options=["NULL values", "Zero", "Error is thrown", "Row is omitted"],
            correct_option_index=0,
            difficulty=0.40,
            explanation="Unmatched rows populate with NULL values.",
        )
        assert q.id == "q-001"
        assert q.correct_option_index == 0

    def test_correct_index_out_of_bounds(self):
        with pytest.raises(ValidationError) as excinfo:
            Question(
                id="q-002",
                concept_id="c-sql",
                prompt="Valid prompt here?",
                options=["Option A", "Option B"],
                correct_option_index=5,
                explanation="Valid explanation.",
            )
        assert "out of bounds" in str(excinfo.value)

    def test_empty_option_rejection(self):
        with pytest.raises(ValidationError) as excinfo:
            Question(
                id="q-003",
                concept_id="c-sql",
                prompt="Valid prompt here?",
                options=["Option A", "   "],
                correct_option_index=0,
                explanation="Valid explanation.",
            )
        assert "cannot be blank" in str(excinfo.value)


class TestAssessmentAndInteractionModels:
    def test_learner_interaction_event_creation(self):
        event = LearnerInteractionEvent(
            id="evt-101",
            attempt_id="att-01",
            learner_id="user-1",
            concept_id="c-sql",
            question_id="q-001",
            is_correct=True,
            prior_mastery=0.20,
            posterior_mastery=0.474,
            post_transition_mastery=0.603,
        )
        assert event.is_correct is True
        assert event.prior_mastery == 0.20
        assert event.post_transition_mastery == 0.603

    def test_interaction_event_probability_bounds(self):
        with pytest.raises(ValidationError):
            LearnerInteractionEvent(
                id="evt-102",
                attempt_id="att-01",
                learner_id="user-1",
                concept_id="c-sql",
                question_id="q-001",
                is_correct=False,
                prior_mastery=1.5,  # Out of range > 1.0
                posterior_mastery=0.2,
                post_transition_mastery=0.3,
            )

    def test_assessment_attempt_creation(self):
        attempt = AssessmentAttempt(
            id="att-01",
            learner_id="user-1",
            assessment_type=AssessmentType.DIAGNOSTIC,
            responses=[
                AssessmentItemResponse(question_id="q-001", selected_option_index=0, response_time_ms=1200)
            ],
            total_items=1,
            score=1.0,
        )
        assert attempt.score == 1.0
        assert attempt.total_items == 1
