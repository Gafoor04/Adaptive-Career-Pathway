"""Unit tests for the CompetencyValidator engine."""

import pytest

from app.domain.models import (
    CareerRole,
    CompetencyDataset,
    Concept,
    ConceptSkillMapping,
    PrerequisiteRelation,
    PrerequisiteType,
    Question,
    RoleSkillRequirement,
    Skill,
)
from app.domain.validator import CompetencyValidationError, CompetencyValidator


def test_valid_dataset_passes(valid_mini_dataset: CompetencyDataset):
    report = CompetencyValidator.validate(valid_mini_dataset)
    assert report.is_valid is True
    assert len(report.errors) == 0


def test_detect_duplicate_concept_id(valid_mini_dataset: CompetencyDataset):
    dup_concept = Concept(
        id="c-alpha",  # Duplicate ID
        slug="alpha-duplicate",
        name="Duplicate Alpha",
        description="Duplicate description",
        category="Foundations",
    )
    dataset = valid_mini_dataset.model_copy(
        update={"concepts": valid_mini_dataset.concepts + [dup_concept]}
    )
    report = CompetencyValidator.validate(dataset)
    assert report.is_valid is False
    assert any("Duplicate concept ID detected: 'c-alpha'" in err for err in report.errors)


def test_detect_duplicate_skill_id(valid_mini_dataset: CompetencyDataset):
    dup_skill = Skill(
        id="skill-test",  # Duplicate ID
        name="Another Skill",
        category="Testing",
    )
    dataset = valid_mini_dataset.model_copy(
        update={"skills": valid_mini_dataset.skills + [dup_skill]}
    )
    report = CompetencyValidator.validate(dataset)
    assert report.is_valid is False
    assert any("Duplicate skill ID detected: 'skill-test'" in err for err in report.errors)


def test_detect_missing_concept_reference_in_prerequisite(valid_mini_dataset: CompetencyDataset):
    broken_prereq = PrerequisiteRelation(
        concept_id="c-beta",
        prerequisite_id="c-nonexistent-ghost",
        relation_type=PrerequisiteType.STRICT,
    )
    dataset = valid_mini_dataset.model_copy(
        update={"prerequisites": [broken_prereq]}
    )
    report = CompetencyValidator.validate(dataset)
    assert report.is_valid is False
    assert any("references non-existent prerequisite 'c-nonexistent-ghost'" in err for err in report.errors)


def test_detect_missing_skill_reference_in_mapping(valid_mini_dataset: CompetencyDataset):
    broken_mapping = ConceptSkillMapping(
        concept_id="c-alpha",
        skill_id="skill-ghost",
        contribution_weight=1.0,
    )
    dataset = valid_mini_dataset.model_copy(
        update={"concept_skill_mappings": [broken_mapping]}
    )
    report = CompetencyValidator.validate(dataset)
    assert report.is_valid is False
    assert any("references non-existent skill 'skill-ghost'" in err for err in report.errors)


def test_detect_orphan_skill(valid_mini_dataset: CompetencyDataset):
    orphan_skill = Skill(
        id="skill-orphan",
        name="Orphan Skill",
        category="Empty",
    )
    dataset = valid_mini_dataset.model_copy(
        update={"skills": valid_mini_dataset.skills + [orphan_skill]}
    )
    report = CompetencyValidator.validate(dataset)
    assert report.is_valid is False
    assert any("Orphan skill: skill 'skill-orphan' has no mapped concepts" in err for err in report.errors)


def test_detect_prerequisite_cycles():
    """Verify that a cycle A -> B -> C -> A is detected and raises error."""
    cA = Concept(id="cA", slug="ca", name="Concept A", description="Description for concept A", category="Cat")
    cB = Concept(id="cB", slug="cb", name="Concept B", description="Description for concept B", category="Cat")
    cC = Concept(id="cC", slug="cc", name="Concept C", description="Description for concept C", category="Cat")

    # Cycle: A requires C, C requires B, B requires A
    # Flow of learning (prereq -> concept): A -> B -> C -> A
    p1 = PrerequisiteRelation(concept_id="cB", prerequisite_id="cA") # A -> B
    p2 = PrerequisiteRelation(concept_id="cC", prerequisite_id="cB") # B -> C
    p3 = PrerequisiteRelation(concept_id="cA", prerequisite_id="cC") # C -> A

    skill = Skill(id="s1", name="Skill 1", category="Cat", description="Skill description here")
    m1 = ConceptSkillMapping(concept_id="cA", skill_id="s1", contribution_weight=1.0)
    role = CareerRole(id="r1", slug="r1", title="Role 1", description="Description for role 1 here", category="Cat")
    req = RoleSkillRequirement(role_id="r1", skill_id="s1", importance_weight=1.0, target_mastery=0.8)

    dataset = CompetencyDataset(
        version="1.0.0",
        description="Cyclic dataset",
        last_updated="2026-10-12",
        concepts=[cA, cB, cC],
        prerequisites=[p1, p2, p3],
        skills=[skill],
        concept_skill_mappings=[m1],
        career_roles=[role],
        role_skill_requirements=[req],
    )

    report = CompetencyValidator.validate(dataset)
    assert report.is_valid is False
    assert any("Prerequisite cycle detected" in err for err in report.errors)

    with pytest.raises(CompetencyValidationError) as excinfo:
        report.raise_if_invalid()
    assert "Prerequisite cycle detected" in str(excinfo.value)


def test_detect_question_missing_concept_reference(valid_mini_dataset: CompetencyDataset):
    invalid_question = Question(
        id="q-bad",
        concept_id="c-unknown-concept",
        prompt="Sample question prompt here?",
        options=["Option 1", "Option 2"],
        correct_option_index=0,
        explanation="Explanation text",
    )
    report = CompetencyValidator.validate(valid_mini_dataset, questions=[invalid_question])
    assert report.is_valid is False
    assert any("references non-existent concept 'c-unknown-concept'" in err for err in report.errors)
