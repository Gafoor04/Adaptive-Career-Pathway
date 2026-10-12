"""Unit tests for next-response prediction probability in BKT."""

import math
import pytest

from ml.bkt.parameters import BKTParameters
from ml.bkt.model import BKTEngine


class TestBKTPredictions:
    @pytest.fixture
    def params(self) -> BKTParameters:
        return BKTParameters(
            p_l0=0.20,
            p_transition=0.15,
            p_guess=0.25,
            p_slip=0.10,
        )

    def test_hand_calculated_prediction(self, params: BKTParameters):
        """Analytical verification:
        P(Correct) = P(L) * (1 - S) + (1 - P(L)) * G
                   = 0.20 * 0.90 + 0.80 * 0.25
                   = 0.18 + 0.20 = 0.38
        """
        p_correct = BKTEngine.predict_correct_probability(current_mastery=0.20, params=params)
        assert math.isclose(p_correct, 0.38, rel_tol=1e-9)

    def test_prediction_at_zero_mastery(self, params: BKTParameters):
        """At P(L) = 0, P(Correct) must equal the guess parameter G."""
        p_correct = BKTEngine.predict_correct_probability(current_mastery=0.0, params=params)
        assert math.isclose(p_correct, params.p_guess, rel_tol=1e-9)

    def test_prediction_at_full_mastery(self, params: BKTParameters):
        """At P(L) = 1, P(Correct) must equal 1 - S (mastery minus slip)."""
        p_correct = BKTEngine.predict_correct_probability(current_mastery=1.0, params=params)
        assert math.isclose(p_correct, 1.0 - params.p_slip, rel_tol=1e-9)

    def test_prediction_range_bounds(self, params: BKTParameters):
        """For all valid mastery values, P(Correct) is bounded within [G, 1 - S]."""
        lower_bound = params.p_guess
        upper_bound = 1.0 - params.p_slip

        for p_l in [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]:
            p_c = BKTEngine.predict_correct_probability(current_mastery=p_l, params=params)
            assert lower_bound <= p_c <= upper_bound, (
                f"Prediction {p_c} out of expected bounds [{lower_bound}, {upper_bound}]"
            )

    def test_prediction_strictly_monotonic_in_mastery(self, params: BKTParameters):
        """Prediction must strictly increase as mastery increases:
        d(P(Correct))/d(P(L)) = 1 - S - G > 0
        """
        masteries = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        preds = [
            BKTEngine.predict_correct_probability(m, params) for m in masteries
        ]

        for i in range(len(preds) - 1):
            assert preds[i] < preds[i + 1], (
                f"Prediction monotonicity failed: {preds[i]} not < {preds[i+1]}"
            )
