"""Skill readiness, career-role ranking, and prerequisite-aware adaptive roadmaps.

Core analytical algorithms:
1. calculate_skill_readiness: Aggregates concept mastery into skill proficiency
   using concept-skill contribution weights.
2. rank_career_roles: Evaluates learner fit for career roles based on required skills,
   importance weights, and target mastery thresholds (competency-fit score).
3. generate_learning_roadmap: Constructs a topologically sound, priority-aware learning
   sequence that strictly respects prerequisite dependencies while prioritizing
   high-relevance career concepts.
"""

from typing import Dict, List, Optional, Set
import networkx as nx

from app.domain.competency_loader import CompetencyRepository


DEFAULT_MASTERY = 0.10
MASTERY_THRESHOLD = 0.70


def calculate_skill_readiness(
    repository: CompetencyRepository,
    concept_mastery: Dict[str, float],
) -> List[dict]:
    """Calculate weighted readiness for every skill from concept mastery.

    Uses configured concept-skill contribution weights. Falls back to concept-level
    baseline p_l0 if mastery is missing or empty.
    """
    results = []

    for skill in repository.list_skills():
        mappings = repository.get_concepts_for_skill(skill.id)

        if not mappings:
            readiness = 0.0
        else:
            total_weight = sum(weight for _, weight in mappings)
            if total_weight <= 0.0:
                readiness = 0.0
            else:
                readiness = sum(
                    concept_mastery.get(concept.id, concept.default_bkt_params.p_l0) * weight
                    for concept, weight in mappings
                ) / total_weight

        # Clamp readiness to [0.0, 1.0]
        readiness = max(0.0, min(1.0, readiness))
        rounded_readiness = round(readiness, 4)

        results.append({
            "skill_id": skill.id,
            "skill_name": skill.name,
            "readiness": rounded_readiness,
            "readiness_percent": round(rounded_readiness * 100, 1),
        })

    return sorted(results, key=lambda item: item["readiness"], reverse=True)


def rank_career_roles(
    repository: CompetencyRepository,
    concept_mastery: Dict[str, float],
) -> List[dict]:
    """Rank career pathways using weighted skill readiness against role requirements.

    Scores represent competency-fit coverage, NOT employment probabilities.
    Identifies specific skill gaps where learner readiness falls below the configured target.
    """
    skills = {
        item["skill_id"]: item
        for item in calculate_skill_readiness(repository, concept_mastery)
    }
    results = []

    for role in repository.list_roles():
        requirements = repository.get_skills_for_role(role.id)

        if not requirements:
            continue

        total_weight = sum(importance for _, importance, _ in requirements)
        weighted_fit = 0.0
        gaps = []

        for skill, importance, target in requirements:
            actual = skills.get(skill.id, {}).get("readiness", 0.0)

            # A skill meeting or exceeding target mastery receives full credit (1.0).
            fit = min(actual / target, 1.0) if target > 0 else 1.0
            weighted_fit += fit * importance

            if actual < target:
                gap_val = target - actual
                gaps.append({
                    "skill_id": skill.id,
                    "skill_name": skill.name,
                    "current_readiness": round(actual, 4),
                    "target_mastery": round(target, 4),
                    "gap": round(gap_val, 4),
                    "importance": round(importance, 4),
                })

        score = weighted_fit / total_weight if total_weight > 0 else 0.0
        score = max(0.0, min(1.0, score))
        gaps.sort(key=lambda item: item["gap"] * item["importance"], reverse=True)

        results.append({
            "role_id": role.id,
            "role_title": role.title,
            "fit_score": round(score, 4),
            "fit_percent": round(score * 100, 1),
            "skill_gaps": gaps,
            "description": "Competency-fit coverage score based on role skill requirements",
        })

    return sorted(results, key=lambda item: item["fit_score"], reverse=True)


def generate_learning_roadmap(
    repository: CompetencyRepository,
    concept_mastery: Dict[str, float],
    role_id: str,
    mastery_threshold: float = MASTERY_THRESHOLD,
    limit: int = 10,
) -> List[dict]:
    """Recommend an adaptive learning sequence for a target career role.

    Guarantees:
    1. Concepts already meeting or exceeding mastery_threshold are not recommended.
    2. Strict prerequisite dependencies are always respected: a prerequisite concept is
       always scheduled BEFORE any concept that depends on it.
    3. Weak/unmastered prerequisites of target concepts are automatically identified and included.
    4. Among concepts whose prerequisites are satisfied, those with higher career relevance
       and larger mastery gaps are prioritized.
    5. Each recommendation contains an explicit pedagogical reason.
    """
    role = repository.get_role(role_id)
    if role is None:
        raise ValueError(f"Unknown career role: '{role_id}'")

    if limit <= 0 or mastery_threshold <= 0.0:
        return []

    # Map each concept to its effective mastery
    mastery: Dict[str, float] = {
        c.id: concept_mastery.get(c.id, c.default_bkt_params.p_l0)
        for c in repository.list_concepts()
    }

    # 1. Identify target concepts contributing to the role's required skills
    target_concepts: Dict[str, float] = {}
    concept_to_skills: Dict[str, Set[str]] = {}

    for skill, importance, _ in repository.get_skills_for_role(role_id):
        for concept, contribution in repository.get_concepts_for_skill(skill.id):
            priority = importance * contribution
            target_concepts[concept.id] = max(
                target_concepts.get(concept.id, 0.0), priority
            )
            concept_to_skills.setdefault(concept.id, set()).add(skill.name)

    if not target_concepts:
        return []

    # 2. Select target concepts that are below threshold
    candidate_targets = {
        cid for cid in target_concepts
        if mastery.get(cid, DEFAULT_MASTERY) < mastery_threshold
    }

    # 3. Collect unmastered prerequisite ancestors
    weak_prerequisites: Set[str] = set()
    for cid in candidate_targets:
        ancestors = repository.get_all_ancestor_prerequisites(cid)
        for anc_id in ancestors:
            if mastery.get(anc_id, DEFAULT_MASTERY) < mastery_threshold:
                weak_prerequisites.add(anc_id)

    # All candidate concepts to schedule
    all_candidates = candidate_targets | weak_prerequisites
    if not all_candidates:
        return []

    # 4. Construct candidate dependency DAG
    # For every pair (u, v) in all_candidates, if u is an ancestor of v in the prerequisite DAG,
    # then u must strictly precede v.
    dag = repository.prerequisite_dag
    candidate_dag = nx.DiGraph()
    for cid in all_candidates:
        candidate_dag.add_node(cid)

    for u in all_candidates:
        for v in all_candidates:
            if u != v and nx.has_path(dag, u, v):
                candidate_dag.add_edge(u, v)

    # 5. Compute priority score for each candidate node
    concept_by_id = {c.id: c for c in repository.list_concepts()}
    priority_scores: Dict[str, float] = {}

    for cid in all_candidates:
        gap = max(mastery_threshold - mastery.get(cid, DEFAULT_MASTERY), 0.0)

        if cid in target_concepts:
            base_relevance = target_concepts[cid]
        else:
            # Prerequisite inherits relevance from the target concepts it enables
            enabled_targets = [
                t for t in candidate_targets
                if nx.has_path(dag, cid, t)
            ]
            base_relevance = max(
                [target_concepts[t] for t in enabled_targets] or [0.1]
            )

        # Score balances career importance and size of knowledge gap
        priority_scores[cid] = base_relevance * (1.0 + gap)

    # 6. Priority-aware topological scheduling (Kahn's Algorithm with Priority Queue)
    in_degrees = {cid: candidate_dag.in_degree(cid) for cid in all_candidates}
    ready_pool = [cid for cid, deg in in_degrees.items() if deg == 0]
    scheduled: List[str] = []

    while ready_pool:
        # Pick the ready concept with the highest priority score
        # Secondary keys: larger mastery gap, then deterministic concept ID
        ready_pool.sort(
            key=lambda cid: (
                priority_scores[cid],
                mastery_threshold - mastery.get(cid, DEFAULT_MASTERY),
                cid,
            ),
            reverse=True,
        )
        selected = ready_pool.pop(0)
        scheduled.append(selected)

        # Decrement in-degree for dependents
        for dependent in candidate_dag.successors(selected):
            in_degrees[dependent] -= 1
            if in_degrees[dependent] == 0:
                ready_pool.append(dependent)

    # 7. Build rich roadmap items with pedagogical rationale
    roadmap: List[dict] = []
    for cid in scheduled:
        concept = concept_by_id[cid]
        current = mastery.get(cid, DEFAULT_MASTERY)
        gap = max(mastery_threshold - current, 0.0)
        relevance = target_concepts.get(cid, 0.0)

        # Generate clear pedagogical explanation
        skills_str = ", ".join(sorted(concept_to_skills.get(cid, [])))
        direct_target = cid in target_concepts
        dependents = [
            concept_by_id[dep].name
            for dep in candidate_dag.successors(cid)
            if dep in candidate_targets
        ]

        if direct_target and dependents:
            reason = (
                f"Contributes to required skill(s) [{skills_str}] and provides foundation "
                f"for [{', '.join(dependents[:2])}]."
            )
        elif direct_target:
            reason = f"Directly contributes to required skill(s) [{skills_str}] for {role.title}."
        else:
            enabled_names = [concept_by_id[dep].name for dep in candidate_dag.successors(cid)]
            reason = (
                f"Foundational prerequisite required before studying [{', '.join(enabled_names[:2])}]."
            )

        roadmap.append({
            "concept_id": cid,
            "concept_name": concept.name,
            "category": concept.category,
            "current_mastery": round(current, 4),
            "mastery_percent": round(current * 100, 1),
            "mastery_gap": round(gap, 4),
            "career_relevance": round(relevance, 4),
            "is_prerequisite_only": not direct_target,
            "reason": reason,
        })

    return roadmap[:limit]
