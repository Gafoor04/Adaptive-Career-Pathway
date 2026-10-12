"""Bayesian Knowledge Tracing (BKT) Core Mathematical Engine.

Provides mathematically pure, stateless update routines alongside stateful tracking:
    1. predict_correct_probability: P(C_t) = P(L_{t-1})(1 - S) + (1 - P(L_{t-1}))G
    2. compute_posterior: P(L_t | y_t) via Bayes' Theorem
    3. apply_transition: P(L_t) = P(L_t | y_t) + (1 - P(L_t | y_t))T
    4. step: Atomic single-observation execution pipeline
    5. process_sequence / replay: Deterministic sequence processing

Strictly distinguishes observation posterior from state transition.
Does not perform database persistence or side-effects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

from ml.bkt.parameters import BKTParameters


@dataclass(frozen=True)
class BKTStepResult:
    """Detailed audit record of an individual BKT update step."""

    step_index: int
    is_correct: bool
    prior_mastery: float
    predicted_p_correct: float
    posterior_mastery: float
    post_transition_mastery: float

    def __repr__(self) -> str:
        outcome = "Correct" if self.is_correct else "Incorrect"
        return (
            f"<BKTStepResult #{self.step_index} [{outcome}]: "
            f"prior={self.prior_mastery:.4f}, "
            f"pred_p_correct={self.predicted_p_correct:.4f}, "
            f"posterior={self.posterior_mastery:.4f}, "
            f"post_transition={self.post_transition_mastery:.4f}>"
        )


class BKTEngine:
    """Stateless pure mathematical calculations for Bayesian Knowledge Tracing."""

    EPSILON: float = 1e-12

    @classmethod
    def predict_correct_probability(
        cls,
        current_mastery: float,
        params: BKTParameters,
    ) -> float:
        """Predict the probability of a correct response given current latent mastery P(L_{t-1}).

        P(Correct_t) = P(L_{t-1}) * (1 - S) + (1 - P(L_{t-1})) * G
        """
        p_l = float(np.clip(current_mastery, 0.0, 1.0))
        p_correct = p_l * (1.0 - params.p_slip) + (1.0 - p_l) * params.p_guess
        return float(np.clip(p_correct, 0.0, 1.0))

    @classmethod
    def compute_posterior(
        cls,
        current_mastery: float,
        is_correct: bool,
        params: BKTParameters,
    ) -> float:
        """Compute posterior mastery P(L_t | y_t) after observing binary response y_t.

        If Correct (y=1):
            P(L | 1) = [P(L) * (1 - S)] / [P(L) * (1 - S) + (1 - P(L)) * G]

        If Incorrect (y=0):
            P(L | 0) = [P(L) * S] / [P(L) * S + (1 - P(L)) * (1 - G)]

        Boundary properties:
            - If P(L) == 0: P(L | y) == 0 (unmastered prior cannot become posterior-mastered without transition)
            - If P(L) == 1: P(L | y) == 1 (absorbing mastery state)
        """
        p_l = float(np.clip(current_mastery, 0.0, 1.0))

        if is_correct:
            numerator = p_l * (1.0 - params.p_slip)
            denominator = numerator + (1.0 - p_l) * params.p_guess
        else:
            numerator = p_l * params.p_slip
            denominator = numerator + (1.0 - p_l) * (1.0 - params.p_guess)

        if denominator < cls.EPSILON:
            # Degenerate case guard: preserve prior
            return p_l

        posterior = numerator / denominator
        return float(np.clip(posterior, 0.0, 1.0))

    @classmethod
    def apply_transition(
        cls,
        posterior_mastery: float,
        params: BKTParameters,
    ) -> float:
        """Apply learning transition T and optional forget rate F after observation.

        P(L_t) = P(L_t | y_t) * (1 - F) + (1 - P(L_t | y_t)) * T

        In standard BKT (F = 0):
        P(L_t) = P(L_t | y_t) + (1 - P(L_t | y_t)) * T
        """
        post = float(np.clip(posterior_mastery, 0.0, 1.0))
        t = params.p_transition
        f = params.p_forget

        post_transition = post * (1.0 - f) + (1.0 - post) * t
        return float(np.clip(post_transition, 0.0, 1.0))

    @classmethod
    def step(
        cls,
        current_mastery: float,
        is_correct: bool,
        params: BKTParameters,
        step_index: int = 1,
    ) -> Tuple[float, BKTStepResult]:
        """Perform a single complete atomic BKT observation-update-transition step.

        Order of operations:
            1. Predict probability of correct response under current mastery P(L_{t-1})
            2. Compute Bayesian observation update to obtain posterior P(L_t | y_t)
            3. Apply learning transition T exactly once to obtain post-transition mastery P(L_t)

        Returns:
            (post_transition_mastery, BKTStepResult)
        """
        prior = float(np.clip(current_mastery, 0.0, 1.0))
        p_pred = cls.predict_correct_probability(prior, params)
        posterior = cls.compute_posterior(prior, is_correct, params)
        post_trans = cls.apply_transition(posterior, params)

        result = BKTStepResult(
            step_index=step_index,
            is_correct=is_correct,
            prior_mastery=prior,
            predicted_p_correct=p_pred,
            posterior_mastery=posterior,
            post_transition_mastery=post_trans,
        )
        return post_trans, result

    @classmethod
    def process_sequence(
        cls,
        responses: Sequence[bool],
        params: BKTParameters,
        initial_mastery: Optional[float] = None,
    ) -> Tuple[float, List[BKTStepResult]]:
        """Process an ordered sequence of binary responses.

        If initial_mastery is None, params.p_l0 is used.
        Returns final mastery and full step audit log.
        """
        mastery = params.p_l0 if initial_mastery is None else float(initial_mastery)
        audit_log: List[BKTStepResult] = []

        for idx, resp in enumerate(responses, start=1):
            mastery, step_result = cls.step(
                current_mastery=mastery,
                is_correct=bool(resp),
                params=params,
                step_index=idx,
            )
            audit_log.append(step_result)

        return mastery, audit_log

    @classmethod
    def replay(
        cls,
        responses: Sequence[bool],
        params: BKTParameters,
        initial_mastery: Optional[float] = None,
    ) -> Tuple[float, List[BKTStepResult]]:
        """Deterministically replay a recorded interaction history from identical starting state.

        Guarantees bit-for-bit reproducible mastery trajectory.
        """
        return cls.process_sequence(
            responses=responses,
            params=params,
            initial_mastery=initial_mastery,
        )


class BKTTracker:
    """Stateful concept mastery tracker for an individual learner."""

    def __init__(
        self,
        concept_id: str,
        params: Optional[BKTParameters] = None,
        initial_mastery: Optional[float] = None,
    ):
        self.concept_id = concept_id
        self.params = params or BKTParameters()
        self._current_mastery = (
            self.params.p_l0 if initial_mastery is None else float(initial_mastery)
        )
        self._history: List[BKTStepResult] = []

    @property
    def current_mastery(self) -> float:
        return self._current_mastery

    @property
    def interaction_count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> List[BKTStepResult]:
        return list(self._history)

    def predict_next(self) -> float:
        """Predict probability of correct response on next interaction opportunity."""
        return BKTEngine.predict_correct_probability(self._current_mastery, self.params)

    def update(self, is_correct: bool) -> BKTStepResult:
        """Observe response, update latent state, and record audit step."""
        next_step_idx = len(self._history) + 1
        new_mastery, step_result = BKTEngine.step(
            current_mastery=self._current_mastery,
            is_correct=is_correct,
            params=self.params,
            step_index=next_step_idx,
        )
        self._current_mastery = new_mastery
        self._history.append(step_result)
        return step_result

    def update_sequence(self, responses: Sequence[bool]) -> List[BKTStepResult]:
        """Process multiple observations sequentially."""
        results = []
        for r in responses:
            results.append(self.update(r))
        return results

    def reset(self, initial_mastery: Optional[float] = None) -> None:
        """Reset tracker state to initial prior."""
        self._current_mastery = (
            self.params.p_l0 if initial_mastery is None else float(initial_mastery)
        )
        self._history.clear()
