"""Dataset and Competency Graph Validation Engine.

Performs static integrity checks:
1. Identifier uniqueness (IDs, slugs, composite keys).
2. Reference integrity (foreign key consistency across concepts, skills, roles, and questions).
3. Weight bounds and domain constraints.
4. Prerequisite graph acyclicity and cycle path extraction using NetworkX.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import networkx as nx

from app.domain.models import (
    CompetencyDataset,
    Question,
)


class CompetencyValidationError(ValueError):
    """Raised when competency dataset fails structural or relational integrity checks."""
    pass


@dataclass
class ValidationReport:
    """Report detailing all errors and warnings discovered during dataset validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_invalid(self) -> None:
        """Raise CompetencyValidationError if any errors are recorded."""
        if not self.is_valid:
            error_details = "\n - ".join(self.errors)
            raise CompetencyValidationError(
                f"Competency dataset validation failed with {len(self.errors)} error(s):\n - {error_details}"
            )


class CompetencyValidator:
    """Validator ensuring data integrity, acyclicity, and relational completeness."""

    @classmethod
    def validate(
        cls,
        dataset: CompetencyDataset,
        questions: Optional[List[Question]] = None,
    ) -> ValidationReport:
        """Run full validation suite on a competency dataset and optional question bank."""
        report = ValidationReport()

        cls._check_duplicate_identifiers(dataset, questions, report)
        cls._check_referential_integrity(dataset, questions, report)
        cls._check_weights_and_coverage(dataset, report)
        cls._check_prerequisite_cycles(dataset, report)

        return report

    @classmethod
    def _check_duplicate_identifiers(
        cls,
        dataset: CompetencyDataset,
        questions: Optional[List[Question]],
        report: ValidationReport,
    ) -> None:
        """Validate uniqueness of primary keys and slugs."""
        # Concepts
        seen_concept_ids: Set[str] = set()
        seen_concept_slugs: Set[str] = set()
        for c in dataset.concepts:
            if c.id in seen_concept_ids:
                report.add_error(f"Duplicate concept ID detected: '{c.id}'")
            seen_concept_ids.add(c.id)

            if c.slug in seen_concept_slugs:
                report.add_error(f"Duplicate concept slug detected: '{c.slug}'")
            seen_concept_slugs.add(c.slug)

        # Skills
        seen_skill_ids: Set[str] = set()
        for s in dataset.skills:
            if s.id in seen_skill_ids:
                report.add_error(f"Duplicate skill ID detected: '{s.id}'")
            seen_skill_ids.add(s.id)

        # Career Roles
        seen_role_ids: Set[str] = set()
        seen_role_slugs: Set[str] = set()
        for r in dataset.career_roles:
            if r.id in seen_role_ids:
                report.add_error(f"Duplicate career role ID detected: '{r.id}'")
            seen_role_ids.add(r.id)

            if r.slug in seen_role_slugs:
                report.add_error(f"Duplicate career role slug detected: '{r.slug}'")
            seen_role_slugs.add(r.slug)

        # Prerequisite pairs
        seen_prereq_pairs: Set[Tuple[str, str]] = set()
        for p in dataset.prerequisites:
            pair = (p.concept_id, p.prerequisite_id)
            if pair in seen_prereq_pairs:
                report.add_error(
                    f"Duplicate prerequisite relation detected: concept '{p.concept_id}' -> prereq '{p.prerequisite_id}'"
                )
            seen_prereq_pairs.add(pair)

        # Concept-Skill mappings
        seen_cs_pairs: Set[Tuple[str, str]] = set()
        for m in dataset.concept_skill_mappings:
            pair = (m.concept_id, m.skill_id)
            if pair in seen_cs_pairs:
                report.add_error(
                    f"Duplicate concept-skill mapping detected: concept '{m.concept_id}' -> skill '{m.skill_id}'"
                )
            seen_cs_pairs.add(pair)

        # Role-Skill requirements
        seen_rs_pairs: Set[Tuple[str, str]] = set()
        for req in dataset.role_skill_requirements:
            pair = (req.role_id, req.skill_id)
            if pair in seen_rs_pairs:
                report.add_error(
                    f"Duplicate role-skill requirement detected: role '{req.role_id}' -> skill '{req.skill_id}'"
                )
            seen_rs_pairs.add(pair)

        # Questions
        if questions:
            seen_question_ids: Set[str] = set()
            for q in questions:
                if q.id in seen_question_ids:
                    report.add_error(f"Duplicate question ID detected: '{q.id}'")
                seen_question_ids.add(q.id)

    @classmethod
    def _check_referential_integrity(
        cls,
        dataset: CompetencyDataset,
        questions: Optional[List[Question]],
        report: ValidationReport,
    ) -> None:
        """Validate all foreign key references point to existing records."""
        concept_ids = {c.id for c in dataset.concepts}
        skill_ids = {s.id for s in dataset.skills}
        role_ids = {r.id for r in dataset.career_roles}

        # Prerequisites
        for p in dataset.prerequisites:
            if p.concept_id not in concept_ids:
                report.add_error(
                    f"Missing reference: prerequisite relation references non-existent target concept '{p.concept_id}'"
                )
            if p.prerequisite_id not in concept_ids:
                report.add_error(
                    f"Missing reference: prerequisite relation references non-existent prerequisite '{p.prerequisite_id}'"
                )

        # Concept-Skill mappings
        for m in dataset.concept_skill_mappings:
            if m.concept_id not in concept_ids:
                report.add_error(
                    f"Missing reference: concept-skill mapping references non-existent concept '{m.concept_id}'"
                )
            if m.skill_id not in skill_ids:
                report.add_error(
                    f"Missing reference: concept-skill mapping references non-existent skill '{m.skill_id}'"
                )

        # Role-Skill requirements
        for req in dataset.role_skill_requirements:
            if req.role_id not in role_ids:
                report.add_error(
                    f"Missing reference: role-skill requirement references non-existent role '{req.role_id}'"
                )
            if req.skill_id not in skill_ids:
                report.add_error(
                    f"Missing reference: role-skill requirement references non-existent skill '{req.skill_id}'"
                )

        # Questions
        if questions:
            for q in questions:
                if q.concept_id not in concept_ids:
                    report.add_error(
                        f"Missing reference: question '{q.id}' references non-existent concept '{q.concept_id}'"
                    )

    @classmethod
    def _check_weights_and_coverage(
        cls,
        dataset: CompetencyDataset,
        report: ValidationReport,
    ) -> None:
        """Ensure all skills and roles have non-empty mappings with valid weights."""
        skills_with_concepts = {m.skill_id for m in dataset.concept_skill_mappings}
        roles_with_skills = {req.role_id for req in dataset.role_skill_requirements}

        for s in dataset.skills:
            if s.id not in skills_with_concepts:
                report.add_error(f"Orphan skill: skill '{s.id}' has no mapped concepts.")

        for r in dataset.career_roles:
            if r.id not in roles_with_skills:
                report.add_error(f"Orphan role: career role '{r.id}' has no required skills.")

        # Check weights are strictly positive
        for m in dataset.concept_skill_mappings:
            if m.contribution_weight <= 0:
                report.add_error(
                    f"Invalid weight: concept-skill mapping ({m.concept_id} -> {m.skill_id}) "
                    f"has non-positive weight {m.contribution_weight}."
                )

        for req in dataset.role_skill_requirements:
            if req.importance_weight <= 0:
                report.add_error(
                    f"Invalid weight: role-skill requirement ({req.role_id} -> {req.skill_id}) "
                    f"has non-positive importance weight {req.importance_weight}."
                )
            if not (0.0 <= req.target_mastery <= 1.0):
                report.add_error(
                    f"Invalid target mastery: role-skill requirement ({req.role_id} -> {req.skill_id}) "
                    f"has out-of-range target mastery {req.target_mastery}."
                )

    @classmethod
    def _check_prerequisite_cycles(
        cls,
        dataset: CompetencyDataset,
        report: ValidationReport,
    ) -> None:
        """Construct directed graph and assert acyclicity using NetworkX."""
        # Edge direction: prerequisite -> concept (directed flow of learning)
        dag = nx.DiGraph()

        # Add all concept nodes
        for c in dataset.concepts:
            dag.add_node(c.id, name=c.name)

        # Add prerequisite edges: u -> v means u is required BEFORE v
        for p in dataset.prerequisites:
            dag.add_edge(p.prerequisite_id, p.concept_id, relation_type=p.relation_type)

        if not nx.is_directed_acyclic_graph(dag):
            cycles = list(nx.simple_cycles(dag))
            formatted_cycles = []
            for cycle in cycles:
                # Cycle list representation: [A, B, C] where C -> A
                cycle_str = " -> ".join(cycle) + f" -> {cycle[0]}"
                formatted_cycles.append(cycle_str)
            cycles_detail = "; ".join(formatted_cycles)
            report.add_error(
                f"Prerequisite cycle detected: the concept prerequisite graph must be a DAG. Cycles found: {cycles_detail}"
            )
