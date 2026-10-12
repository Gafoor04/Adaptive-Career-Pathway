"""Unit tests validating the real initial competency dataset (competency_v1.json) and question bank."""

import networkx as nx
import pytest

from app.domain.competency_loader import CompetencyRepository


class TestRealCompetencyDataset:
    def test_repository_loads_without_validation_errors(self, real_repository: CompetencyRepository):
        assert real_repository is not None
        assert real_repository.dataset.version == "1.0.0"

    def test_concept_count_within_target_range(self, real_repository: CompetencyRepository):
        # Master instructions require ~25-40 technical concepts
        concepts = real_repository.list_concepts()
        assert 25 <= len(concepts) <= 40, f"Expected 25-40 concepts, got {len(concepts)}"
        assert len(concepts) == 32

    def test_career_roles_count_within_target_range(self, real_repository: CompetencyRepository):
        # Master instructions require 5 or 6 career pathways as initial planning targets
        roles = real_repository.list_roles()
        assert 5 <= len(roles) <= 6, f"Expected 5 or 6 roles, got {len(roles)}"
        assert len(roles) == 6

        expected_slugs = {
            "backend-software-engineer",
            "frontend-software-engineer",
            "full-stack-engineer",
            "data-engineer",
            "machine-learning-engineer",
            "devops-platform-engineer",
        }
        actual_slugs = {r.slug for r in roles}
        assert expected_slugs == actual_slugs

    def test_prerequisite_graph_is_acyclic_dag(self, real_repository: CompetencyRepository):
        dag = real_repository.prerequisite_dag
        assert nx.is_directed_acyclic_graph(dag) is True, "Prerequisite graph must be a strict DAG"
        assert len(dag.nodes) == 32
        assert len(dag.edges) > 0

    def test_topological_sort_produces_valid_linear_order(self, real_repository: CompetencyRepository):
        dag = real_repository.prerequisite_dag
        topo_order = list(nx.topological_sort(dag))
        assert len(topo_order) == 32

        # Verify that for every edge u -> v, u appears before v in the order
        node_indices = {node: idx for idx, node in enumerate(topo_order)}
        for u, v in dag.edges:
            assert node_indices[u] < node_indices[v], (
                f"Prerequisite violation in topological sort: '{u}' must precede '{v}'"
            )

    def test_ancestor_prerequisites_traversal(self, real_repository: CompetencyRepository):
        # 'c-dynamic-programming' transitively requires 'c-prog-fundamentals', 'c-linear-data-structures', 'c-algo-complexity'
        ancestors = real_repository.get_all_ancestor_prerequisites("c-dynamic-programming")
        assert "c-prog-fundamentals" in ancestors
        assert "c-linear-data-structures" in ancestors
        assert "c-algo-complexity" in ancestors

        # Check topological ordering of ancestors
        assert ancestors.index("c-prog-fundamentals") < ancestors.index("c-linear-data-structures")
        assert ancestors.index("c-linear-data-structures") < ancestors.index("c-algo-complexity")

    def test_root_concepts_have_no_prerequisites(self, real_repository: CompetencyRepository):
        # Programming foundations, Linux shell, Git, and HTML should be root nodes
        roots = [n for n, deg in real_repository.prerequisite_dag.in_degree() if deg == 0]
        assert "c-prog-fundamentals" in roots
        assert "c-linux-shell" in roots
        assert "c-git-version-control" in roots
        assert "c-html-css-semantics" in roots

    def test_skill_concept_mappings_coverage(self, real_repository: CompetencyRepository):
        skills = real_repository.list_skills()
        assert len(skills) == 12

        for skill in skills:
            concepts = real_repository.get_concepts_for_skill(skill.id)
            assert len(concepts) > 0, f"Skill '{skill.id}' must map to at least one concept"
            total_weight = sum(w for _, w in concepts)
            assert total_weight > 0, f"Skill '{skill.id}' has non-positive total weight"

    def test_career_role_requirements_coverage(self, real_repository: CompetencyRepository):
        roles = real_repository.list_roles()
        for role in roles:
            skills = real_repository.get_skills_for_role(role.id)
            assert len(skills) >= 4, f"Role '{role.id}' should require at least 4 skills"
            for skill, imp_weight, target_mastery in skills:
                assert imp_weight > 0
                assert 0.60 <= target_mastery <= 0.95

    def test_questions_coverage_and_validity(self, real_repository: CompetencyRepository):
        questions = real_repository.questions
        assert len(questions) >= 20, f"Expected at least 20 questions, got {len(questions)}"

        for q in questions:
            # Concept must exist
            concept = real_repository.get_concept(q.concept_id)
            assert concept is not None, f"Question '{q.id}' references unknown concept '{q.concept_id}'"
            # Options and correct index
            assert len(q.options) >= 2
            assert 0 <= q.correct_option_index < len(q.options)
            assert 0.0 <= q.difficulty <= 1.0
            assert len(q.prompt) > 10
            assert len(q.explanation) > 10
