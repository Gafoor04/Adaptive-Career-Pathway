"""Loader and repository interface for validated competency dataset and questions."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx

from app.core.config import settings
from app.domain.models import (
    CareerRole,
    CompetencyDataset,
    Concept,
    Question,
    Skill,
)
from app.domain.validator import CompetencyValidator, ValidationReport


class CompetencyRepository:
    """In-memory loaded and validated repository for domain competency graph and question bank."""

    def __init__(
        self,
        dataset: CompetencyDataset,
        questions: Optional[List[Question]] = None,
    ):
        self._dataset = dataset
        self._questions = questions or []

        # Run validation on initialization
        report = CompetencyValidator.validate(self._dataset, self._questions)
        report.raise_if_invalid()

        # Build fast lookup indexes
        self._concepts_by_id: Dict[str, Concept] = {c.id: c for c in dataset.concepts}
        self._skills_by_id: Dict[str, Skill] = {s.id: s for s in dataset.skills}
        self._roles_by_id: Dict[str, CareerRole] = {r.id: r for r in dataset.career_roles}
        self._questions_by_concept: Dict[str, List[Question]] = {}
        for q in self._questions:
            self._questions_by_concept.setdefault(q.concept_id, []).append(q)

        # Build prerequisite DAG (edge: prerequisite -> concept)
        self._prereq_dag: nx.DiGraph = nx.DiGraph()
        for c in dataset.concepts:
            self._prereq_dag.add_node(c.id, name=c.name, category=c.category)
        for p in dataset.prerequisites:
            self._prereq_dag.add_edge(
                p.prerequisite_id,
                p.concept_id,
                relation_type=p.relation_type,
                rationale=p.rationale,
            )

    @property
    def dataset(self) -> CompetencyDataset:
        return self._dataset

    @property
    def questions(self) -> List[Question]:
        return self._questions

    @property
    def prerequisite_dag(self) -> nx.DiGraph:
        """Returns the Directed Acyclic Graph where an edge (u, v) denotes u is a prerequisite for v."""
        return self._prereq_dag

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        return self._concepts_by_id.get(concept_id)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self._skills_by_id.get(skill_id)

    def get_role(self, role_id: str) -> Optional[CareerRole]:
        return self._roles_by_id.get(role_id)

    def list_concepts(self) -> List[Concept]:
        return list(self._concepts_by_id.values())

    def list_skills(self) -> List[Skill]:
        return list(self._skills_by_id.values())

    def list_roles(self) -> List[CareerRole]:
        return list(self._roles_by_id.values())

    def get_concepts_for_skill(self, skill_id: str) -> List[Tuple[Concept, float]]:
        """Returns list of (Concept, contribution_weight) for the specified skill."""
        mappings = [m for m in self._dataset.concept_skill_mappings if m.skill_id == skill_id]
        results = []
        for m in mappings:
            concept = self.get_concept(m.concept_id)
            if concept:
                results.append((concept, m.contribution_weight))
        return results

    def get_skills_for_role(self, role_id: str) -> List[Tuple[Skill, float, float]]:
        """Returns list of (Skill, importance_weight, target_mastery) for the specified role."""
        requirements = [r for r in self._dataset.role_skill_requirements if r.role_id == role_id]
        results = []
        for req in requirements:
            skill = self.get_skill(req.skill_id)
            if skill:
                results.append((skill, req.importance_weight, req.target_mastery))
        return results

    def get_direct_prerequisites(self, concept_id: str) -> List[str]:
        """Returns IDs of concepts that must be learned directly BEFORE concept_id."""
        if not self._prereq_dag.has_node(concept_id):
            return []
        return list(self._prereq_dag.predecessors(concept_id))

    def get_all_ancestor_prerequisites(self, concept_id: str) -> List[str]:
        """Returns all transitive prerequisite concept IDs required before concept_id in topological order."""
        if not self._prereq_dag.has_node(concept_id):
            return []
        ancestors = nx.ancestors(self._prereq_dag, concept_id)
        # Induce subgraph and sort topologically
        subgraph = self._prereq_dag.subgraph(ancestors)
        return list(nx.topological_sort(subgraph))

    def get_questions_for_concept(self, concept_id: str) -> List[Question]:
        return self._questions_by_concept.get(concept_id, [])

    @classmethod
    def load_from_files(
        cls,
        competency_path: Optional[Path] = None,
        questions_path: Optional[Path] = None,
    ) -> "CompetencyRepository":
        """Load and validate dataset from json files on disk."""
        c_path = competency_path or settings.competency_data_path
        q_path = questions_path or settings.questions_data_path

        if not c_path.exists():
            raise FileNotFoundError(f"Competency dataset file not found at: {c_path}")

        with open(c_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        dataset = CompetencyDataset.model_validate(c_data)

        questions: List[Question] = []
        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                q_data = json.load(f)
            questions = [Question.model_validate(q) for q in q_data]

        return cls(dataset=dataset, questions=questions)
