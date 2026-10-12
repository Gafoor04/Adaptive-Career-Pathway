"""Unit tests for Bayesian observation updates and learning transition steps.

Validates exact analytical closed-form calculations with high precision (1e-9 tolerance),
distinguishes posterior from post-transition states, and checks boundary conditions.
"""

import math
import pytest

from ml.bkt.parameters import BKTParameters
from ml.bkt.model import BKTEngine


class TestBKTUpdatesAnalytical:
    """Tests validated against step-by-step hand-calculated numerical derivations."""

    @pytest.fixture
    def standard_params(self) -> BKTParameters:
        """Known reference parameter set:
        P(L0) = 0.20
        T     = 0.15
        G     = 0.25
        S     = 0.10
        """
        return BKTParameters(
            p_l0=0.20,
            p_transition=0.15,
            p_guess=0.25,
            p_slip=0.10,
        )

    def test_correct_response_hand_calculated(self, standard_params: BKTParameters):
        """Analytical verification for Correct Response (y = 1):

        Prior: P(L_0) = 0.20
        Numerator   = 0.20 * (1 - 0.10) = 0.18
        Denominator = 0.18 + (1 - 0.20) * 0.25 = 0.18 + 0.20 = 0.38
        Posterior P(L_1 | y=1) = 0.18 / 0.38 = 9 / 19 ≈ 0.4736842105263158

        Transition:
        P(L_1) = (9/19) + (1 - 9/19) * 0.15 = (9/19) + (10/19) * 0.15
               = (9 + 1.5) / 19 = 10.5 / 19 = 21 / 38 ≈ 0.5526315789473685
        """
        prior = 0.20

        # 1. Test isolated posterior update
        posterior = BKTEngine.compute_posterior(prior, is_correct=True, params=standard_params)
        expected_posterior = 9.0 / 19.0
        assert math.isclose(posterior, expected_posterior, rel_tol=1e-9), (
            f"Posterior mismatch: got {posterior}, expected {expected_posterior}"
        )

        # 2. Test isolated transition update
        post_trans = BKTEngine.apply_transition(posterior, params=standard_params)
        expected_post_trans = 21.0 / 38.0
        assert math.isclose(post_trans, expected_post_trans, rel_tol=1e-9), (
            f"Post-transition mismatch: got {post_trans}, expected {expected_post_trans}"
        )

        # 3. Test composite atomic step
        new_mastery, step_result = BKTEngine.step(prior, is_correct=True, params=standard_params, step_index=1)
        assert math.isclose(new_mastery, expected_post_trans, rel_tol=1e-9)
        assert math.isclose(step_result.posterior_mastery, expected_posterior, rel_tol=1e-9)
        assert math.isclose(step_result.post_transition_mastery, expected_post_trans, rel_tol=1e-9)
        assert step_result.is_correct is True
        assert step_result.step_index == 1

        # 4. Strict separation check: posterior != post_transition
        assert step_result.posterior_mastery < step_result.post_transition_mastery

    def test_incorrect_response_hand_calculated(self, standard_params: BKTParameters):
        """Analytical verification for Incorrect Response (y = 0):

        Prior: P(L_0) = 0.20
        Numerator   = 0.20 * 0.10 = 0.02
        Denominator = 0.02 + (1 - 0.20) * (1 - 0.25) = 0.02 + 0.80 * 0.75 = 0.02 + 0.60 = 0.62
        Posterior P(L_1 | y=0) = 0.02 / 0.62 = 1 / 31 ≈ 0.03225806451612903

        Transition:
        P(L_1) = (1/31) + (1 - 1/31) * 0.15 = (1/31) + (30/31) * 0.15
               = (1 + 4.5) / 31 = 5.5 / 31 = 11 / 62 ≈ 0.1774193548387097
        """
        prior = 0.20

        # 1. Test isolated posterior update
        posterior = BKTEngine.compute_posterior(prior, is_correct=False, params=standard_params)
        expected_posterior = 1.0 / 31.0
        assert math.isclose(posterior, expected_posterior, rel_tol=1e-9), (
            f"Posterior mismatch: got {posterior}, expected {expected_posterior}"
        )

        # 2. Test isolated transition update
        post_trans = BKTEngine.apply_transition(posterior, params=standard_params)
        expected_post_trans = 11.0 / 62.0
        assert math.isclose(post_trans, expected_post_trans, rel_tol=1e-9), (
            f"Post-transition mismatch: got {post_trans}, expected {expected_post_trans}"
        )

        # 3. Test composite atomic step
        new_mastery, step_result = BKTEngine.step(prior, is_correct=False, params=standard_params, step_index=1)
        assert math.isclose(new_mastery, expected_post_trans, rel_tol=1e-9)
        assert math.isclose(step_result.posterior_mastery, expected_posterior, rel_tol=1e-9)
        assert math.isclose(step_result.post_transition_mastery, expected_post_trans, rel_tol=1e-9)

        # 4. Strict separation check: posterior dropped below prior
        assert step_result.posterior_mastery < prior

    def test_two_step_sequence_hand_calculated(self, standard_params: BKTParameters):
        """Analytical verification of a 2-step sequence: [Incorrect (0), Correct (1)]:

        Step 1 (Incorrect):
            Resulting mastery = 11 / 62 ≈ 0.1774193548387097 (from previous test)

        Step 2 (Correct, starting from prior = 11/62):
            Numerator = (11/62) * (1 - 0.10) = 9.9 / 62 = 99 / 620
            Denominator = (99 / 620) + (51 / 62) * 0.25 = 99/620 + 12.75/62 = (99 + 127.5)/620 = 226.5/620 = 453/1240
            Posterior P(L_2 | y=1) = (99/620) / (453/1240) = (99 * 2) / 453 = 198 / 453 = 66 / 151 ≈ 0.4370860927152318

            Transition:
            P(L_2) = (66/151) + (85/151) * 0.15 = (66 + 12.75) / 151 = 78.75 / 151 = 315 / 604 ≈ 0.5215231788079471
        """
        final_mastery, steps = BKTEngine.process_sequence(
            responses=[False, True],
            params=standard_params,
            initial_mastery=0.20,
        )

        assert len(steps) == 2

        # Step 1 asserts
        assert math.isclose(steps[0].post_transition_mastery, 11.0 / 62.0, rel_tol=1e-9)

        # Step 2 asserts
        expected_step2_posterior = 66.0 / 151.0
        expected_step2_post_trans = 315.0 / 604.0
        assert math.isclose(steps[1].posterior_mastery, expected_step2_posterior, rel_tol=1e-9)
        assert math.isclose(steps[1].post_transition_mastery, expected_step2_post_trans, rel_tol=1e-9)
        assert math.isclose(final_mastery, expected_step2_post_trans, rel_tol=1e-9)


class TestBKTBoundaryCases:
    """Verifies behavior at probability boundary extremes (0.0 and 1.0)."""

    def test_zero_mastery_prior(self):
        """When learner has strictly zero mastery: P(L) = 0.0"""
        params = BKTParameters(p_l0=0.0, p_transition=0.15, p_guess=0.20, p_slip=0.10)

        # Correct response:
        # P(L | 1) = 0 / G = 0.0
        # Post-transition = 0 + (1 - 0) * T = T
        post_corr = BKTEngine.compute_posterior(0.0, is_correct=True, params=params)
        trans_corr = BKTEngine.apply_transition(post_corr, params=params)
        assert post_corr == 0.0
        assert trans_corr == 0.15

        # Incorrect response:
        # P(L | 0) = 0 / (1 - G) = 0.0
        # Post-transition = 0 + (1 - 0) * T = T
        post_inc = BKTEngine.compute_posterior(0.0, is_correct=False, params=params)
        trans_inc = BKTEngine.apply_transition(post_inc, params=params)
        assert post_inc == 0.0
        assert trans_inc == 0.15

    def test_full_mastery_prior(self):
        """When learner has reached absorbing full mastery: P(L) = 1.0"""
        params = BKTParameters(p_l0=1.0, p_transition=0.15, p_guess=0.20, p_slip=0.10)

        # Correct response:
        # P(L | 1) = (1 - S) / (1 - S) = 1.0
        # Post-transition = 1.0 + 0 * T = 1.0
        post_corr = BKTEngine.compute_posterior(1.0, is_correct=True, params=params)
        trans_corr = BKTEngine.apply_transition(post_corr, params=params)
        assert post_corr == 1.0
        assert trans_corr == 1.0

        # Incorrect response (slip event):
        # P(L | 0) = S / S = 1.0
        # Post-transition = 1.0 + 0 * T = 1.0
        post_inc = BKTEngine.compute_posterior(1.0, is_correct=False, params=params)
        trans_inc = BKTEngine.apply_transition(post_inc, params=params)
        assert post_inc == 1.0
        assert trans_inc == 1.0

    def test_probability_bounds_preserved(self):
        """Ensure all updates produce probabilities strictly within [0.0, 1.0]."""
        params = BKTParameters(p_l0=0.10, p_transition=0.25, p_guess=0.30, p_slip=0.15)
        for p in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]:
            for outcome in [True, False]:
                post = BKTEngine.compute_posterior(p, outcome, params)
                trans = BKTEngine.apply_transition(post, params)
                assert 0.0 <= post <= 1.0
                assert 0.0 <= trans <= 1.0
