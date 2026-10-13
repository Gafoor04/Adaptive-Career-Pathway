"""Unit tests for LearnerEngine integrating BKT inference and SQLite persistence."""

import pytest

from app.domain.competency_loader import CompetencyRepository
from app.domain.learner_engine import LearnerEngine
from app.domain.persistence import LearnerStore


@pytest.fixture
def memory_store() -> LearnerStore:
    return LearnerStore(db_path=":memory:")


class TestLearnerEngine:
    def test_initial_state_loads_defaults(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="alice", store=memory_store)
        assert engine.learner_id == "alice"
        assert len(engine.events) == 0

        # Check default baselines
        for concept in real_repository.list_concepts():
            assert engine.mastery[concept.id] == concept.default_bkt_params.p_l0
            assert engine.interaction_counts[concept.id] == 0

    def test_answer_correct_updates_bkt_and_persists(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="alice", store=memory_store)
        q = real_repository.questions[0]
        concept_id = q.concept_id
        initial_mastery = engine.mastery[concept_id]

        # Submit correct answer
        result = engine.answer(q.id, q.correct_option_index)

        assert result["is_correct"] is True
        assert result["updated_mastery"] > initial_mastery
        assert result["interaction_count"] == 1
        assert len(engine.events) == 1

        # Check in memory store
        saved = memory_store.get_concept_mastery("alice")
        assert concept_id in saved
        assert round(saved[concept_id].p_mastery, 4) == result["updated_mastery"]
        assert saved[concept_id].interaction_count == 1

        # Check stored event
        events = memory_store.get_interaction_events("alice")
        assert len(events) == 1
        assert events[0].question_id == q.id
        assert events[0].is_correct is True

    def test_answer_incorrect_updates_bkt_and_persists(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="bob", store=memory_store)
        q = real_repository.questions[0]
        wrong_idx = (q.correct_option_index + 1) % len(q.options)

        result = engine.answer(q.id, wrong_idx)

        assert result["is_correct"] is False
        assert result["interaction_count"] == 1
        assert len(engine.events) == 1

        # Posterior for incorrect is lower, transition applies
        saved = memory_store.get_concept_mastery("bob")
        assert q.concept_id in saved
        assert round(saved[q.concept_id].p_mastery, 4) == result["updated_mastery"]

    def test_repeated_attempts_increment_interaction_count(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="charlie", store=memory_store)
        q = real_repository.questions[0]

        res1 = engine.answer(q.id, q.correct_option_index)
        res2 = engine.answer(q.id, q.correct_option_index)

        assert res1["interaction_count"] == 1
        assert res2["interaction_count"] == 2
        assert len(engine.events) == 2
        assert engine.interaction_counts[q.concept_id] == 2

    def test_session_state_restoration_across_engine_restarts(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        # 1. First session: answer two questions
        engine1 = LearnerEngine(real_repository, learner_id="dan", store=memory_store)
        q1 = real_repository.questions[0]
        q2 = real_repository.questions[1]

        engine1.answer(q1.id, q1.correct_option_index)
        engine1.answer(q2.id, (q2.correct_option_index + 1) % len(q2.options))

        saved_mastery_1 = dict(engine1.mastery)
        saved_counts_1 = dict(engine1.interaction_counts)

        # 2. Simulate app restart: create new engine instance with the same store
        engine2 = LearnerEngine(real_repository, learner_id="dan", store=memory_store)

        assert engine2.mastery == saved_mastery_1
        assert engine2.interaction_counts == saved_counts_1
        assert len(engine2.events) == 2
        assert engine2.events[0]["question_id"] == q1.id
        assert engine2.events[1]["question_id"] == q2.id

    def test_multi_learner_isolation(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine_alice = LearnerEngine(real_repository, learner_id="alice", store=memory_store)
        engine_bob = LearnerEngine(real_repository, learner_id="bob", store=memory_store)

        q = real_repository.questions[0]
        engine_alice.answer(q.id, q.correct_option_index)

        # Alice's interaction count is 1, Bob's is still 0
        assert engine_alice.interaction_counts[q.concept_id] == 1
        assert engine_bob.interaction_counts[q.concept_id] == 0
        assert len(engine_alice.events) == 1
        assert len(engine_bob.events) == 0

    def test_explicit_reset_affects_only_target_learner(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine_alice = LearnerEngine(real_repository, learner_id="alice", store=memory_store)
        engine_bob = LearnerEngine(real_repository, learner_id="bob", store=memory_store)

        q = real_repository.questions[0]
        engine_alice.answer(q.id, q.correct_option_index)
        engine_bob.answer(q.id, q.correct_option_index)

        # Reset Alice only
        engine_alice.reset()

        assert len(engine_alice.events) == 0
        assert engine_alice.interaction_counts[q.concept_id] == 0
        assert engine_alice.mastery[q.concept_id] == real_repository.get_concept(q.concept_id).default_bkt_params.p_l0

        # Bob remains unaffected
        assert len(engine_bob.events) == 1
        assert engine_bob.interaction_counts[q.concept_id] == 1

    def test_reconstruct_state_from_history(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="reconstruct-user", store=memory_store)
        q = real_repository.questions[0]
        engine.answer(q.id, q.correct_option_index)

        reconstructed = engine.reconstruct_state_from_history()
        assert reconstructed[q.concept_id] == engine.mastery[q.concept_id]

    def test_invalid_question_id_raises_value_error(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="err-user", store=memory_store)
        with pytest.raises(ValueError, match="Unknown question ID"):
            engine.answer("nonexistent-q", 0)

    def test_invalid_option_index_raises_value_error(
        self,
        real_repository: CompetencyRepository,
        memory_store: LearnerStore,
    ):
        engine = LearnerEngine(real_repository, learner_id="err-user", store=memory_store)
        q = real_repository.questions[0]
        with pytest.raises(ValueError, match="out of range"):
            engine.answer(q.id, 999)
