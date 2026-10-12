"""Domain models for Concept-Centric Learner Modeling and Career Pathways.

All models use Pydantic V2 with strict type validation, field boundary checks,
and immutability/serialization configurations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PrerequisiteType(str, Enum):
    """Classification of prerequisite dependency."""

    STRICT = "strict"          # Hard requirement; dependent concept cannot be scheduled before
    RECOMMENDED = "recommended" # Advisory requirement


class AssessmentType(str, Enum):
    """Type of assessment session."""

    DIAGNOSTIC = "diagnostic"
    CONCEPT_QUIZ = "concept_quiz"
    PRACTICE = "practice"


class BKTParameters(BaseModel):
    """Bayesian Knowledge Tracing prior and transition parameters for a concept."""

    model_config = ConfigDict(frozen=True)

    p_l0: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Prior probability that a learner has mastered the concept initially P(L0)."
    )
    p_transition: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Probability of transitioning from unmastered to mastered state T."
    )
    p_guess: float = Field(
        default=0.20,
        ge=0.0,
        lt=0.5,
        description="Probability of a correct response despite lacking mastery G (< 0.5)."
    )
    p_slip: float = Field(
        default=0.10,
        ge=0.0,
        lt=0.5,
        description="Probability of an incorrect response despite having mastered concept S (< 0.5)."
    )

    @model_validator(mode="after")
    def validate_monotonicity(self) -> "BKTParameters":
        """Ensure G + S < 1.0 to preserve cognitive monotonicity."""
        if (self.p_guess + self.p_slip) >= 1.0:
            raise ValueError(
                f"Invalid BKT parameters: p_guess ({self.p_guess}) + p_slip ({self.p_slip}) "
                "must be strictly less than 1.0 for valid Bayesian updating."
            )
        return self


class Concept(BaseModel):
    """An atomic, measurable technical concept."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=2, description="Unique concept identifier, e.g., 'c-sql-basics'")
    slug: str = Field(..., pattern=r"^[a-z0-9-]+$", description="URL-friendly identifier")
    name: str = Field(..., min_length=2, max_length=150, description="Display name of the concept")
    description: str = Field(..., min_length=5, description="Pedagogical description of the concept")
    category: str = Field(..., min_length=2, description="Broad knowledge domain, e.g., 'Databases'")
    default_bkt_params: BKTParameters = Field(
        default_factory=BKTParameters,
        description="Default prior BKT parameters for this concept"
    )
    tags: List[str] = Field(default_factory=list, description="Categorical tags for filtering")

    @field_validator("id", "name", "category")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Field cannot be empty or whitespace only")
        return v_stripped


class PrerequisiteRelation(BaseModel):
    """A directed dependency indicating prerequisite_id must precede concept_id."""

    model_config = ConfigDict(frozen=True)

    concept_id: str = Field(..., description="The dependent target concept ID")
    prerequisite_id: str = Field(..., description="The prerequisite concept ID required beforehand")
    relation_type: PrerequisiteType = Field(
        default=PrerequisiteType.STRICT,
        description="Strictness level of dependency"
    )
    rationale: str = Field(
        default="",
        description="Pedagogical explanation of why this prerequisite is required"
    )

    @model_validator(mode="after")
    def prevent_self_dependency(self) -> "PrerequisiteRelation":
        if self.concept_id == self.prerequisite_id:
            raise ValueError(
                f"Self-prerequisite detected: concept '{self.concept_id}' cannot depend on itself."
            )
        return self


class Skill(BaseModel):
    """A higher-level technical capability composed of atomic concepts."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=2, description="Unique skill ID, e.g., 'skill-relational-db'")
    name: str = Field(..., min_length=2, max_length=150, description="Display name of the skill")
    category: str = Field(..., min_length=2, description="Domain group, e.g., 'Data Engineering'")
    description: str = Field(default="", description="Detailed skill overview")

    @field_validator("id", "name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Field cannot be empty or whitespace only")
        return v_stripped


class ConceptSkillMapping(BaseModel):
    """Defines how much an atomic concept contributes to a higher-level skill."""

    model_config = ConfigDict(frozen=True)

    concept_id: str = Field(..., description="ID of the contributing concept")
    skill_id: str = Field(..., description="ID of the parent skill")
    contribution_weight: float = Field(
        ...,
        gt=0.0,
        description="Relative positive contribution weight of this concept to the skill"
    )
    rationale: str = Field(
        default="",
        description="Justification for concept relevance to skill"
    )


class CareerRole(BaseModel):
    """A target industry career pathway (e.g., Backend Software Engineer)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=2, description="Unique role ID, e.g., 'role-backend-eng'")
    title: str = Field(..., min_length=2, max_length=150, description="Professional job title")
    slug: str = Field(..., pattern=r"^[a-z0-9-]+$", description="URL-friendly slug")
    description: str = Field(..., min_length=10, description="Comprehensive role responsibilities")
    category: str = Field(..., min_length=2, description="Engineering discipline")
    version: str = Field(default="1.0.0", description="Semver version of this competency profile")

    @field_validator("id", "title")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Field cannot be empty or whitespace only")
        return v_stripped


class RoleSkillRequirement(BaseModel):
    """Defines target mastery threshold and importance weight of a skill for a role."""

    model_config = ConfigDict(frozen=True)

    role_id: str = Field(..., description="Target career role ID")
    skill_id: str = Field(..., description="Required skill ID")
    importance_weight: float = Field(
        ...,
        gt=0.0,
        description="Relative importance weight of this skill for the role (> 0)"
    )
    target_mastery: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum mastery threshold required for role readiness (e.g., 0.80)"
    )
    rationale: str = Field(
        default="",
        description="Why this skill level is essential for the role"
    )


class Question(BaseModel):
    """An assessment item measuring mastery of a specific concept."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=2, description="Unique question ID, e.g., 'q-sql-001'")
    concept_id: str = Field(..., description="Atomic concept tested by this question")
    prompt: str = Field(..., min_length=10, description="The question text or code snippet")
    options: List[str] = Field(..., min_length=2, description="List of answer choices")
    correct_option_index: int = Field(
        ...,
        ge=0,
        description="0-indexed position of the single correct option"
    )
    difficulty: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Calibrated item difficulty parameter in [0.0, 1.0]"
    )
    explanation: str = Field(
        ...,
        min_length=5,
        description="Pedagogical explanation revealed after attempt"
    )
    tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_correct_index_in_bounds(self) -> "Question":
        if self.correct_option_index >= len(self.options):
            raise ValueError(
                f"Question '{self.id}': correct_option_index {self.correct_option_index} "
                f"out of bounds for options list of length {len(self.options)}."
            )
        # Ensure all options are non-empty
        for idx, opt in enumerate(self.options):
            if not opt.strip():
                raise ValueError(f"Question '{self.id}': option at index {idx} cannot be blank.")
        return self


# ---------------------------------------------------------------------------
# Learner Assessment & Interaction Event Models
# ---------------------------------------------------------------------------

class AssessmentItemResponse(BaseModel):
    """Single item response submitted by a learner during an assessment."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(..., description="Attempted question ID")
    selected_option_index: int = Field(..., ge=0, description="Selected option index")
    response_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time spent on the item in milliseconds"
    )


class AssessmentAttempt(BaseModel):
    """A full assessment session attempt (diagnostic or practice quiz)."""

    id: str = Field(..., description="Unique attempt ID")
    learner_id: str = Field(..., description="Learner user ID")
    assessment_type: AssessmentType = Field(default=AssessmentType.DIAGNOSTIC)
    responses: List[AssessmentItemResponse] = Field(..., min_length=1)
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_items: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0, description="Fraction of correct answers")


class LearnerInteractionEvent(BaseModel):
    """An immutable audit event recording a BKT Bayesian observation and update step."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Event unique ID")
    attempt_id: str = Field(..., description="Parent assessment attempt ID")
    learner_id: str = Field(..., description="Learner ID")
    concept_id: str = Field(..., description="Concept evaluated")
    question_id: str = Field(..., description="Question answered")
    is_correct: bool = Field(..., description="Observed binary outcome: True (1) or False (0)")
    prior_mastery: float = Field(..., ge=0.0, le=1.0, description="P(L_{t-1}) before observation")
    posterior_mastery: float = Field(..., ge=0.0, le=1.0, description="P(L_{t} | obs) after Bayesian update")
    post_transition_mastery: float = Field(..., ge=0.0, le=1.0, description="P(L_t) after learning transition T")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConceptMasterySnapshot(BaseModel):
    """Current persisted knowledge state for a learner on a concept."""

    learner_id: str = Field(..., description="Learner ID")
    concept_id: str = Field(..., description="Concept ID")
    p_mastery: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Current estimated latent mastery probability P(L)"
    )
    interaction_count: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Full Competency Dataset Container
# ---------------------------------------------------------------------------

class CompetencyDataset(BaseModel):
    """Top-level container for a versioned career competency graph specification."""

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Dataset semantic version")
    description: str = Field(..., description="Documentation of dataset source and domain scope")
    last_updated: str = Field(..., description="ISO date of last update")
    concepts: List[Concept] = Field(..., min_length=1)
    prerequisites: List[PrerequisiteRelation] = Field(default_factory=list)
    skills: List[Skill] = Field(..., min_length=1)
    concept_skill_mappings: List[ConceptSkillMapping] = Field(..., min_length=1)
    career_roles: List[CareerRole] = Field(..., min_length=1)
    role_skill_requirements: List[RoleSkillRequirement] = Field(..., min_length=1)
