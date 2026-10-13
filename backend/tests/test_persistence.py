"""Unit tests for SQLite learner persistence layer."""

from datetime import datetime, timezone
import pytest

from app.domain.models import ConceptMasterySnapshot, LearnerInteractionEvent
from app.domain.persistence import LearnerPersistenceError, LearnerStore


@pytest.fixture
def memory_store() -> LearnerStore:
    """Provides an isolated in-memory LearnerStore."""
    return LearnerStore(db_path=":memory:")


class TestLearnerStore:
    def test_create_and_get_learner(self, memory_store: LearnerStore):
        learner = memory_store.get_or_create_learner("learner-01")
        assert learner["learner_id"] == "learner-01"
        assert "created_at" in learner

        # Calling again retrieves same learner
        same_learner = memory_store.get_or_create_learner("learner-01")
        assert same_learner["created_at"] == learner["created_at"]
        assert memory_store.learner_exists("learner-01") is True

    def test_empty_learner_id_rejected(self, memory_store: LearnerStore):
        with pytest.raises(ValueError, match="cannot be empty"):
            memory_store.get_or_create_learner("   ")

    def test_list_learners_multiple(self, memory_store: LearnerStore):
        memory_store.get_or_create_learner("charlie")
        memory_store.get_or_create_learner("alice")
        memory_store.get_or_create_learner("bob")

        learners = memory_store.list_learners()
        assert learners == ["alice", "bob", "charlie"]

    def test_save_and_retrieve_concept_mastery(self, memory_store: LearnerStore):
        snapshot = ConceptMasterySnapshot(
            learner_id="learner-01",
            concept_id="c-sql-basics",
            p_mastery=0.45,
            interaction_count=2,
            last_updated=datetime.now(timezone.utc),
        )
        memory_store.save_concept_mastery(snapshot)

        retrieved = memory_store.get_concept_mastery("learner-01")
        assert "c-sql-basics" in retrieved
        s = retrieved["c-sql-basics"]
        assert s.concept_id == "c-sql-basics"
        assert s.p_mastery == 0.45
        assert s.interaction_count == 2

    def test_record_interaction_event_atomic(self, memory_store: LearnerStore):
        now = datetime.now(timezone.utc)
        event = LearnerInteractionEvent(
            id="evt-001",
            attempt_id="att-001",
            learner_id="learner-01",
            concept_id="c-python-basics",
            question_id="q-py-01",
            is_correct=True,
            prior_mastery=0.20,
            posterior_mastery=0.50,
            post_transition_mastery=0.575,
            timestamp=now,
        )
        snapshot = ConceptMasterySnapshot(
            learner_id="learner-01",
            concept_id="c-python-basics",
            p_mastery=0.575,
            interaction_count=1,
            last_updated=now,
        )

        memory_store.record_interaction_event(event, snapshot)

        # Verify event saved
        events = memory_store.get_interaction_events("learner-01")
        assert len(events) == 1
        assert events[0].id == "evt-001"
        assert events[0].is_correct is True
        assert events[0].post_transition_mastery == 0.575

        # Verify mastery snapshot saved
        mastery = memory_store.get_concept_mastery("learner-01")
        assert mastery["c-python-basics"].p_mastery == 0.575
        assert mastery["c-python-basics"].interaction_count == 1
        assert memory_store.get_interaction_count("learner-01") == 1

    def test_multiple_learners_isolation(self, memory_store: LearnerStore):
        now = datetime.now(timezone.utc)
        # Learner 1
        event1 = LearnerInteractionEvent(
            id="evt-101",
            attempt_id="att-101",
            learner_id="alice",
            concept_id="c-sql-basics",
            question_id="q-sql-01",
            is_correct=True,
            prior_mastery=0.10,
            posterior_mastery=0.45,
            post_transition_mastery=0.52,
            timestamp=now,
        )
        snap1 = ConceptMasterySnapshot(
            learner_id="alice",
            concept_id="c-sql-basics",
            p_mastery=0.52,
            interaction_count=1,
            last_updated=now,
        )
        memory_store.record_interaction_event(event1, snap1)

        # Learner 2
        event2 = LearnerInteractionEvent(
            id="evt-102",
            attempt_id="att-102",
            learner_id="bob",
            concept_id="c-sql-basics",
            question_id="q-sql-01",
            is_correct=False,
            prior_mastery=0.10,
            posterior_mastery=0.03,
            post_transition_mastery=0.15,
            timestamp=now,
        )
        snap2 = ConceptMasterySnapshot(
            learner_id="bob",
            concept_id="c-sql-basics",
            p_mastery=0.15,
            interaction_count=1,
            last_updated=now,
        )
        memory_store.record_interaction_event(event2, snap2)

        # Check isolation
        alice_mastery = memory_store.get_concept_mastery("alice")
        bob_mastery = memory_store.get_concept_mastery("bob")

        assert alice_mastery["c-sql-basics"].p_mastery == 0.52
        assert bob_mastery["c-sql-basics"].p_mastery == 0.15

        assert len(memory_store.get_interaction_events("alice")) == 1
        assert len(memory_store.get_interaction_events("bob")) == 1

        # Resetting Alice does NOT affect Bob
        memory_store.reset_learner_state("alice")
        assert len(memory_store.get_concept_mastery("alice")) == 0
        assert len(memory_store.get_interaction_events("alice")) == 0
        assert len(memory_store.get_concept_mastery("bob")) == 1
        assert len(memory_store.get_interaction_events("bob")) == 1

    def test_repeated_interactions_sequence_and_reconstruction(self, memory_store: LearnerStore):
        now = datetime.now(timezone.utc)
        concept = "c-trees"

        # Step 1: wrong
        evt1 = LearnerInteractionEvent(
            id="step-1",
            attempt_id="att-1",
            learner_id="learner-01",
            concept_id=concept,
            question_id="q-tree-1",
            is_correct=False,
            prior_mastery=0.20,
            posterior_mastery=0.04,
            post_transition_mastery=0.18,
            timestamp=now,
        )
        snap1 = ConceptMasterySnapshot(
            learner_id="learner-01",
            concept_id=concept,
            p_mastery=0.18,
            interaction_count=1,
            last_updated=now,
        )
        memory_store.record_interaction_event(evt1, snap1)

        # Step 2: correct
        evt2 = LearnerInteractionEvent(
            id="step-2",
            attempt_id="att-2",
            learner_id="learner-01",
            concept_id=concept,
            question_id="q-tree-2",
            is_correct=True,
            prior_mastery=0.18,
            posterior_mastery=0.55,
            post_transition_mastery=0.62,
            timestamp=now,
        )
        snap2 = ConceptMasterySnapshot(
            learner_id="learner-01",
            concept_id=concept,
            p_mastery=0.62,
            interaction_count=2,
            last_updated=now,
        )
        memory_store.record_interaction_event(evt2, snap2)

        # Step 3: correct
        evt3 = LearnerInteractionEvent(
            id="step-3",
            attempt_id="att-3",
            learner_id="learner-01",
            concept_id=concept,
            question_id="q-tree-3",
            is_correct=True,
            prior_mastery=0.62,
            posterior_mastery=0.88,
            post_transition_mastery=0.90,
            timestamp=now,
        )
        snap3 = ConceptMasterySnapshot(
            learner_id="learner-01",
            concept_id=concept,
            p_mastery=0.90,
            interaction_count=3,
            last_updated=now,
        )
        memory_store.record_interaction_event(evt3, snap3)

        assert memory_store.get_interaction_count("learner-01") == 3
        events = memory_store.get_interaction_events("learner-01")
        assert len(events) == 3
        assert [e.id for e in events] == ["step-1", "step-2", "step-3"]

        current_mastery = memory_store.get_concept_mastery("learner-01")[concept]
        assert current_mastery.p_mastery == 0.90
        assert current_mastery.interaction_count == 3

        # Reconstruct state from events
        reconstructed = memory_store.reconstruct_mastery_from_events(
            "learner-01",
            baseline_p_l0={"c-trees": 0.20, "c-other": 0.10},
        )
        assert reconstructed["c-trees"] == 0.90
        assert reconstructed["c-other"] == 0.10

    def test_record_interaction_event_rollback_on_failure(self, memory_store: LearnerStore):
        """Verify that if database execution fails, the transaction rolls back cleanly."""
        now = datetime.now(timezone.utc)
        # First save an initial snapshot
        initial_snap = ConceptMasterySnapshot(
            learner_id="fail-learner",
            concept_id="c-sql",
            p_mastery=0.20,
            interaction_count=0,
            last_updated=now,
        )
        memory_store.save_concept_mastery(initial_snap)

        # Break the schema temporarily inside memory connection to induce an operational error on update
        conn = memory_store._get_connection()
        conn.execute("DROP TABLE concept_mastery")

        event = LearnerInteractionEvent(
            id="evt-err-1",
            attempt_id="att-err-1",
            learner_id="fail-learner",
            concept_id="c-sql",
            question_id="q-1",
            is_correct=True,
            prior_mastery=0.20,
            posterior_mastery=0.50,
            post_transition_mastery=0.55,
            timestamp=now,
        )
        snap = ConceptMasterySnapshot(
            learner_id="fail-learner",
            concept_id="c-sql",
            p_mastery=0.55,
            interaction_count=1,
            last_updated=now,
        )

        with pytest.raises(LearnerPersistenceError):
            memory_store.record_interaction_event(event, snap)

        # The event should NOT have been recorded because transaction was rolled back
        events = memory_store.get_interaction_events("fail-learner")
        assert len(events) == 0
