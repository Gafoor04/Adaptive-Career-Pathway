"""End-to-end integration tests verifying quiz -> BKT -> SQLite -> recommendation -> roadmap pipeline across app restarts."""

from pathlib import Path
import pytest

from app.domain.competency_loader import CompetencyRepository
from app.domain.learner_engine import LearnerEngine
from app.domain.persistence import LearnerStore


@pytest.fixture
def file_store(tmp_path: Path) -> LearnerStore:
    db_file = tmp_path / "test_e2e_learners.db"
    return LearnerStore(db_path=db_file)


class TestEndToEndWorkflow:
    def test_complete_quiz_persistence_and_recommendation_flow(
        self,
        real_repository: CompetencyRepository,
        file_store: LearnerStore,
    ):
        learner_id = "researcher-01"

        # 1. Start learner session
        engine1 = LearnerEngine(real_repository, learner_id=learner_id, store=file_store)
        assert len(engine1.events) == 0

        # Baseline career ranking
        initial_rankings = engine1.answer(
            real_repository.questions[0].id,
            real_repository.questions[0].correct_option_index,
        )["career_roles"]
        assert len(initial_rankings) == len(real_repository.list_roles())

        # 2. Submit multiple answers across different concepts
        q1 = real_repository.questions[0]
        q2 = real_repository.questions[1]
        q3 = real_repository.questions[2]

        # Already answered q1 once; answer q1 again (repeated interaction)
        res1 = engine1.answer(q1.id, q1.correct_option_index)
        assert res1["interaction_count"] == 2

        # Answer q2 correctly
        res2 = engine1.answer(q2.id, q2.correct_option_index)

        # Answer q3 incorrectly
        wrong_idx3 = (q3.correct_option_index + 1) % len(q3.options)
        res3 = engine1.answer(q3.id, wrong_idx3)

        total_interactions_session1 = len(engine1.events)
        assert total_interactions_session1 == 4  # 1 initial + 3 more
        mastery_snapshot_session1 = dict(engine1.mastery)
        counts_snapshot_session1 = dict(engine1.interaction_counts)

        # Generate roadmap in session 1
        role_id = "role-backend-eng"
        roadmap_session1 = engine1.roadmap(role_id, limit=5)
        assert len(roadmap_session1) > 0

        # 3. Simulate application restart: instantiate a completely new LearnerEngine
        # using the same underlying SQLite database
        engine2 = LearnerEngine(real_repository, learner_id=learner_id, store=file_store)

        # 4. Verify all mastery states, counts, and history are faithfully restored
        assert len(engine2.events) == total_interactions_session1
        assert engine2.mastery == mastery_snapshot_session1
        assert engine2.interaction_counts == counts_snapshot_session1

        # 5. Verify reconstructed state from raw event logs matches active mastery
        reconstructed = engine2.reconstruct_state_from_history()
        for cid in [q1.concept_id, q2.concept_id, q3.concept_id]:
            assert round(reconstructed[cid], 4) == round(engine2.mastery[cid], 4)

        # 6. Verify career recommendations and roadmaps compute identically after restart
        roadmap_session2 = engine2.roadmap(role_id, limit=5)
        assert len(roadmap_session2) == len(roadmap_session1)
        for item1, item2 in zip(roadmap_session1, roadmap_session2):
            assert item1["concept_id"] == item2["concept_id"]
            assert item1["current_mastery"] == item2["current_mastery"]
            assert item1["mastery_gap"] == item2["mastery_gap"]
