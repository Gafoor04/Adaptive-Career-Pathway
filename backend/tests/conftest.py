"""Pytest configuration and shared test fixtures."""

import sys
from pathlib import Path
import pytest

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.domain.competency_loader import CompetencyRepository
from app.domain.models import (
    BKTParameters,
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


@pytest.fixture
def valid_mini_dataset() -> CompetencyDataset:
    """Fixture providing a minimal, structurally valid competency dataset."""
    c1 = Concept(
        id="c-alpha",
        slug="alpha-concept",
        name="Alpha Concept",
        description="Foundational concept alpha",
        category="Foundations",
    )
    c2 = Concept(
        id="c-beta",
        slug="beta-concept",
        name="Beta Concept",
        description="Dependent concept beta",
        category="Advanced",
    )
    prereq = PrerequisiteRelation(
        concept_id="c-beta",
        prerequisite_id="c-alpha",
        relation_type=PrerequisiteType.STRICT,
        rationale="Alpha must precede Beta",
    )
    skill = Skill(
        id="skill-test",
        name="Test Skill",
        category="Testing",
        description="A skill for unit testing",
    )
    mapping = ConceptSkillMapping(
        concept_id="c-alpha",
        skill_id="skill-test",
        contribution_weight=1.0,
    )
    role = CareerRole(
        id="role-test",
        title="Test Engineer",
        slug="test-engineer",
        description="A role for running automated testing",
        category="Engineering",
    )
    requirement = RoleSkillRequirement(
        role_id="role-test",
        skill_id="skill-test",
        importance_weight=1.0,
        target_mastery=0.80,
    )
    return CompetencyDataset(
        version="1.0.0",
        description="Minimal test fixture dataset",
        last_updated="2026-10-12",
        concepts=[c1, c2],
        prerequisites=[prereq],
        skills=[skill],
        concept_skill_mappings=[mapping],
        career_roles=[role],
        role_skill_requirements=[requirement],
    )


@pytest.fixture
def real_repository() -> CompetencyRepository:
    """Fixture loading the complete curated competency dataset and question bank from disk."""
    return CompetencyRepository.load_from_files()
