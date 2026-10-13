"""Connect assessment interactions to BKT mastery, persistent storage, and career recommendations."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from app.domain.competency_loader import CompetencyRepository
from app.domain.models import ConceptMasterySnapshot, LearnerInteractionEvent
from app.domain.persistence import LearnerStore
from app.domain.recommendation import (
    calculate_skill_readiness,
    rank_career_roles,
    generate_learning_roadmap,
)
from ml.bkt.model import BKTEngine
from ml.bkt.parameters import BKTParameters as EngineBKTParameters


class LearnerEngine:
    """Stateful learner session backed by SQLite persistence and BKT inference."""

    def __init__(
        self,
        repository: CompetencyRepository,
        learner_id: str = "demo-learner",
        store: Optional[LearnerStore] = None,
    ):
        self.repository = repository
        self.learner_id = learner_id.strip() or "demo-learner"
        self.store = store or LearnerStore()

        # Ensure learner profile exists in persistence
        self.store.get_or_create_learner(self.learner_id)

        # Restore concept mastery states from persistence
        self.mastery: Dict[str, float] = {}
        self.interaction_counts: Dict[str, int] = {}

        saved_snapshots = self.store.get_concept_mastery(self.learner_id)
        for concept in repository.list_concepts():
            if concept.id in saved_snapshots:
                snapshot = saved_snapshots[concept.id]
                self.mastery[concept.id] = snapshot.p_mastery
                self.interaction_counts[concept.id] = snapshot.interaction_count
            else:
                self.mastery[concept.id] = concept.default_bkt_params.p_l0
                self.interaction_counts[concept.id] = 0

        # Load persisted event audit history
        self.events: List[dict] = [
            {
                "id": e.id,
                "attempt_id": e.attempt_id,
                "learner_id": e.learner_id,
                "question_id": e.question_id,
                "concept_id": e.concept_id,
                "is_correct": e.is_correct,
                "prior_mastery": round(e.prior_mastery, 4),
                "posterior_mastery": round(e.posterior_mastery, 4),
                "updated_mastery": round(e.post_transition_mastery, 4),
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.store.get_interaction_events(self.learner_id, ascending=True)
        ]

        self.questions = {
            question.id: question
            for question in repository.questions
        }

    def answer(self, question_id: str, selected_option_index: int) -> dict:
        """Evaluate an answer, update BKT state, persist event & snapshot atomically, and return recommendations."""
        question = self.questions.get(question_id)
        if question is None:
            raise ValueError(f"Unknown question ID: {question_id}")

        if not 0 <= selected_option_index < len(question.options):
            raise ValueError("Selected option index is out of range.")

        correct = selected_option_index == question.correct_option_index
        concept = self.repository.get_concept(question.concept_id)

        if concept is None:
            raise ValueError(f"Unknown concept: {question.concept_id}")

        source_params = concept.default_bkt_params
        params = EngineBKTParameters(
            p_l0=source_params.p_l0,
            p_transition=source_params.p_transition,
            p_guess=source_params.p_guess,
            p_slip=source_params.p_slip,
            source="concept_baseline",
        )

        prior = self.mastery[concept.id]
        step_idx = self.interaction_counts[concept.id] + 1
        new_mastery, audit = BKTEngine.step(
            current_mastery=prior,
            is_correct=correct,
            params=params,
            step_index=step_idx,
        )

        now = datetime.now(timezone.utc)
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        attempt_id = f"att_{uuid.uuid4().hex[:8]}"

        event_model = LearnerInteractionEvent(
            id=event_id,
            attempt_id=attempt_id,
            learner_id=self.learner_id,
            concept_id=concept.id,
            question_id=question.id,
            is_correct=correct,
            prior_mastery=prior,
            posterior_mastery=audit.posterior_mastery,
            post_transition_mastery=new_mastery,
            timestamp=now,
        )

        snapshot_model = ConceptMasterySnapshot(
            learner_id=self.learner_id,
            concept_id=concept.id,
            p_mastery=new_mastery,
            interaction_count=step_idx,
            last_updated=now,
        )

        # Atomic persistence write (rolls back if database fails)
        self.store.record_interaction_event(event_model, snapshot_model)

        # Commit to in-memory state only upon successful persistence
        self.mastery[concept.id] = new_mastery
        self.interaction_counts[concept.id] = step_idx

        event_dict = {
            "id": event_id,
            "attempt_id": attempt_id,
            "learner_id": self.learner_id,
            "question_id": question.id,
            "concept_id": concept.id,
            "is_correct": correct,
            "prior_mastery": round(prior, 4),
            "posterior_mastery": round(audit.posterior_mastery, 4),
            "updated_mastery": round(new_mastery, 4),
            "interaction_count": self.interaction_counts[concept.id],
            "timestamp": now.isoformat(),
        }
        self.events.append(event_dict)

        return {
            **event_dict,
            "explanation": question.explanation,
            "skill_readiness": calculate_skill_readiness(
                self.repository, self.mastery
            ),
            "career_roles": rank_career_roles(
                self.repository, self.mastery
            ),
        }

    def roadmap(self, role_id: str, limit: int = 10) -> list:
        """Build a prerequisite-aware learning roadmap for a selected career role."""
        return generate_learning_roadmap(
            self.repository,
            self.mastery,
            role_id,
            limit=limit,
        )

    def reset(self) -> None:
        """Explicitly reset this learner's saved state to initial baseline, without affecting others."""
        self.store.reset_learner_state(self.learner_id)
        for concept in self.repository.list_concepts():
            self.mastery[concept.id] = concept.default_bkt_params.p_l0
            self.interaction_counts[concept.id] = 0
        self.events.clear()

    def reconstruct_state_from_history(self) -> Dict[str, float]:
        """Verify and reconstruct mastery state directly from the event log."""
        baseline = {c.id: c.default_bkt_params.p_l0 for c in self.repository.list_concepts()}
        return self.store.reconstruct_mastery_from_events(self.learner_id, baseline_p_l0=baseline)

    def get_recent_events(self, limit: int = 10) -> List[dict]:
        """Return the most recent assessment interaction events."""
        return self.events[-limit:] if limit > 0 else []
