# Adaptive Career-Pathway Recommendation via Concept-Centric Long-Term Learner Modeling

## Research & Project Overview
This repository implements an adaptive career-pathway recommendation and learning system grounded in probabilistic cognitive modeling and graph-theoretic path planning.

Instead of one-time aptitude surveys or static resume keyword matches, the system maintains a persistent, concept-level learner model updated through Bayesian Knowledge Tracing (BKT). It propagates concept mastery to high-level technical skills, calculates quantitative readiness and skill gaps against versioned industry career profiles, and generates personalized, topologically valid learning roadmaps.

### Core Processing Pipeline
$$\text{Learner Evidence } (y_t) \longrightarrow \text{Concept Mastery } P(L_c) \longrightarrow \text{Skill Readiness } (S_s) \longrightarrow \text{Career Fit } (RS_r) \longrightarrow \text{Adaptive Roadmap}$$

---

## Project Status: Milestones 1 & 2 Completed

### Completed Components:
* **Backend Foundation & Domain Models:** Validated entities for Concepts, Skills, Career Roles, Prerequisite Dependencies, Questions, and Learner Interaction Events.
* **Curated Competency Dataset (`v1.0.0`):**
  - **32 Technical Concepts** spanning core CS, databases, web architecture, frontend, cloud/systems, data engineering, and machine learning.
  - **12 Technical Skills** with explicit contribution weight matrices.
  - **6 Career Pathways**: Backend Engineer, Frontend Engineer, Full-Stack Engineer, Data Engineer, Machine Learning Engineer, and DevOps/Platform Engineer.
  - **29 Explicit Prerequisite Relations** forming a verified Directed Acyclic Graph (DAG).
  - **20 Calibrated Assessment Items** with pedagogical explanations.
* **Integrity Validation Engine:** Comprehensive static verification detecting orphan entities, invalid weights, missing foreign keys, and prerequisite cycles.
* **Bayesian Knowledge Tracing (BKT) Engine (`ml.bkt`):**
  - Closed-form Bayesian observation update strictly distinguishing observation posterior from state transition.
  - Learning transition applied exactly once per opportunity.
  - Cognitive monotonicity invariants enforcing $1 - S > G$.
  - Deterministic sequence processing and interaction history replay.
  - Cross-validated against `pyBKT` forward equations.
* **Automated Test Suite:** **51 tests** passing with 100% success rate.

---

## Quick Start & Verification

### Running All Automated Tests
```bash
cd backend
python -m pytest -v
```

For detailed mathematical derivations and step-by-step hand-worked examples, see [backend/README.md](backend/README.md).
