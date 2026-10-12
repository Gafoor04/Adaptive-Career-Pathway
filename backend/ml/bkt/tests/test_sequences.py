"""Unit tests for BKT sequence processing, deterministic replay, and state tracking."""

import random
import pytest

from ml.bkt.parameters import BKTParameters
from ml.bkt.model import BKTEngine, BKTTracker


class TestBKTSequences:
    @pytest.fixture
    def params(self) -> BKTParameters:
        return BKTParameters(
            p_l0=0.15,
            p_transition=0.12,
            p_guess=0.20,
            p_slip=0.10,
        )

    def test_deterministic_replay_guarantee(self, params: BKTParameters):
        """Replaying the exact same interaction sequence must yield bit-for-bit identical results."""
        random.seed(42)
        sequence = [bool(random.getrandbits(1)) for _ in range(50)]

        final_mastery_run1, log_run1 = BKTEngine.process_sequence(sequence, params)
        final_mastery_run2, log_run2 = BKTEngine.replay(sequence, params)

        assert final_mastery_run1 == final_mastery_run2
        assert len(log_run1) == len(log_run2) == 50

        for s1, s2 in zip(log_run1, log_run2):
            assert s1.step_index == s2.step_index
            assert s1.is_correct == s2.is_correct
            assert s1.prior_mastery == s2.prior_mastery
            assert s1.predicted_p_correct == s2.predicted_p_correct
            assert s1.posterior_mastery == s2.posterior_mastery
            assert s1.post_transition_mastery == s2.post_transition_mastery

    def test_sustained_correct_sequence_convergence(self, params: BKTParameters):
        """A learner consistently answering correctly must asymptotically approach mastery = 1.0."""
        sequence = [True] * 25
        final_mastery, steps = BKTEngine.process_sequence(sequence, params, initial_mastery=0.10)

        # Mastery must be monotonically strictly increasing at every step
        for i in range(len(steps) - 1):
            assert steps[i].post_transition_mastery < steps[i + 1].post_transition_mastery

        # After 25 consecutive correct answers, mastery must exceed 0.999
        assert final_mastery > 0.999

    def test_sustained_incorrect_sequence_behavior(self, params: BKTParameters):
        """A learner consistently answering incorrectly should decline towards an equilibrium."""
        sequence = [False] * 20
        final_mastery, steps = BKTEngine.process_sequence(sequence, params, initial_mastery=0.80)

        # Immediate decline from 0.80
        assert steps[0].post_transition_mastery < 0.80
        assert final_mastery < 0.20

    def test_bkt_tracker_stateful_lifecycle(self, params: BKTParameters):
        """Test BKTTracker state progression and reset functionality."""
        tracker = BKTTracker(concept_id="c-sql-basics", params=params)

        assert tracker.current_mastery == params.p_l0
        assert tracker.interaction_count == 0

        # Predict next before any observation
        pred_0 = tracker.predict_next()
        assert pred_0 == BKTEngine.predict_correct_probability(params.p_l0, params)

        # Update 1: Correct
        res1 = tracker.update(is_correct=True)
        assert tracker.interaction_count == 1
        assert tracker.current_mastery == res1.post_transition_mastery
        assert tracker.current_mastery > params.p_l0

        # Update 2: Incorrect
        res2 = tracker.update(is_correct=False)
        assert tracker.interaction_count == 2
        assert tracker.current_mastery == res2.post_transition_mastery

        # Update sequence
        res_batch = tracker.update_sequence([True, True])
        assert tracker.interaction_count == 4
        assert len(tracker.history) == 4

        # Reset
        tracker.reset()
        assert tracker.current_mastery == params.p_l0
        assert tracker.interaction_count == 0
        assert len(tracker.history) == 0
