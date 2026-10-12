"""Bayesian Knowledge Tracing (BKT) Parameter Schema and Validation.

Encapsulates the standard four-parameter BKT tuple:
    P(L0): Initial mastery probability
    T:     Learning transition probability
    G:     Guess probability
    S:     Slip probability
    F:     Forget probability (default 0.0, knowledge retention assumption)

Enforces boundary constraints and cognitive monotonicity invariants:
    - G in [0, 0.5)
    - S in [0, 0.5)
    - P(correct | mastered) = (1 - S) > G = P(correct | unmastered)
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BKTParameters(BaseModel):
    """Immutable parameter container for a concept's BKT model."""

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
        description="Probability of answering correctly despite lacking mastery G (< 0.5)."
    )
    p_slip: float = Field(
        default=0.10,
        ge=0.0,
        lt=0.5,
        description="Probability of answering incorrectly despite having mastered concept S (< 0.5)."
    )
    p_forget: float = Field(
        default=0.0,
        ge=0.0,
        le=0.2,
        description="Probability of forgetting mastered knowledge F (0.0 in standard BKT)."
    )
    version: str = Field(
        default="1.0.0",
        description="Model parameter version."
    )
    source: str = Field(
        default="default_baseline",
        description="Source of parameters: e.g. 'default_baseline', 'expert_configured', 'em_fitted'."
    )

    @model_validator(mode="after")
    def validate_monotonicity(self) -> "BKTParameters":
        """Enforce P(Correct | Mastered) > P(Correct | Unmastered), i.e., 1 - S > G."""
        p_correct_mastered = 1.0 - self.p_slip
        p_correct_unmastered = self.p_guess
        if p_correct_mastered <= p_correct_unmastered:
            raise ValueError(
                f"Cognitive monotonicity violated: P(Correct|Mastered)={p_correct_mastered:.4f} "
                f"must be strictly greater than P(Correct|Unmastered)={p_correct_unmastered:.4f}. "
                f"(p_guess={self.p_guess}, p_slip={self.p_slip})"
            )
        return self

    def to_pybkt_dict(self) -> Dict[str, float]:
        """Export parameters in standard pyBKT parameter dictionary format."""
        return {
            "prior": self.p_l0,
            "learns": self.p_transition,
            "guesses": self.p_guess,
            "slips": self.p_slip,
            "forgets": self.p_forget,
        }

    @classmethod
    def from_pybkt_dict(
        cls,
        d: Dict[str, float],
        version: str = "1.0.0",
        source: str = "pybkt_imported",
    ) -> "BKTParameters":
        """Instantiate parameters from a pyBKT dictionary."""
        return cls(
            p_l0=float(d.get("prior", 0.10)),
            p_transition=float(d.get("learns", 0.15)),
            p_guess=float(d.get("guesses", 0.20)),
            p_slip=float(d.get("slips", 0.10)),
            p_forget=float(d.get("forgets", 0.0)),
            version=version,
            source=source,
        )


# Explicitly documented baseline parameter profiles (not claimed as empirically learned)
BASELINE_PROFILES: Dict[str, BKTParameters] = {
    "standard_default": BKTParameters(
        p_l0=0.15,
        p_transition=0.15,
        p_guess=0.20,
        p_slip=0.10,
        source="literature_baseline_corbett_anderson",
    ),
    "foundational_easy": BKTParameters(
        p_l0=0.30,
        p_transition=0.20,
        p_guess=0.25,
        p_slip=0.08,
        source="literature_baseline_easy",
    ),
    "advanced_rigorous": BKTParameters(
        p_l0=0.08,
        p_transition=0.10,
        p_guess=0.15,
        p_slip=0.12,
        source="literature_baseline_advanced",
    ),
}
