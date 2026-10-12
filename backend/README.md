# Backend Foundation, Competency Domain & BKT Engine

## Overview
This package implements the core analytical engine for the **Adaptive Career-Pathway Recommendation System via Concept-Centric Long-Term Learner Modeling**.

It integrates:
1. Validated Pydantic domain models for concepts, skills, career pathways, questions, and interaction events.
2. An initial curated competency graph (`competency_v1.json`) with strict DAG cycle detection.
3. An independent, pure mathematical **Bayesian Knowledge Tracing (BKT)** engine (`ml.bkt`) with exact analytical updates, next-response prediction, deterministic sequence replay, and comparative pyBKT alignment.

---

## Directory Structure
```
backend/
├── app/
│   ├── core/
│   │   ├── config.py             # Application settings & data paths
│   ├── domain/
│   │   ├── models.py             # Pydantic v2 domain schemas
│   │   ├── validator.py          # Structural, relational & DAG cycle validation engine
│   │   ├── competency_loader.py  # Repository interface & in-memory indexed graph
│   ├── data/
│   │   ├── competency_v1.json    # Versioned competency graph (32 concepts, 12 skills, 6 roles)
│   │   └── questions_v1.json     # Calibrated assessment item bank
├── ml/
│   └── bkt/
│       ├── __init__.py           # Package exports
│       ├── parameters.py         # BKT parameters, cognitive monotonicity, and baselines
│       ├── model.py              # Pure mathematical BKTEngine & stateful BKTTracker
│       └── tests/
│           ├── test_predictions.py    # Next-item prediction probability tests
│           ├── test_updates.py        # Hand-calculated analytical update tests
│           ├── test_sequences.py      # Sequence processing & deterministic replay tests
│           └── test_pybkt_comparison.py# Comparative validation against pyBKT
├── tests/
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_domain_models.py     # Pydantic model boundary & rule tests
│   ├── test_validation_engine.py # Integrity, missing foreign key & cycle detection tests
│   └── test_competency_dataset.py# End-to-end dataset validation & topological sort tests
├── pyproject.toml                # Build & pytest configuration
├── requirements.txt              # Production and test dependencies
└── README.md
```

---

## Bayesian Knowledge Tracing (BKT) Engine

### 1. Mathematical Formulation
Each atomic concept $c$ is parameterized by a tuple $\theta_c = \{P(L_0), T, G, S, F\}$:
* **$P(L_0)$**: Prior probability that the learner has mastered the concept before observation.
* **$T$**: Probability of learning transition from unmastered to mastered state ($P(L_{t+1}=1 \mid L_t=0)$).
* **$G$**: Guess probability ($P(\text{Correct} \mid \text{Unmastered})$), constrained to $G \in [0.0, 0.5)$.
* **$S$**: Slip probability ($P(\text{Incorrect} \mid \text{Mastered})$), constrained to $S \in [0.0, 0.5)$.
* **$F$**: Forget probability ($P(L_{t+1}=0 \mid L_t=1)$), $0.0$ in standard BKT (knowledge retention assumption).

#### Cognitive Monotonicity Invariant
To ensure that demonstrated mastery strictly increases the likelihood of correct performance:
$$P(\text{Correct} \mid \text{Mastered}) > P(\text{Correct} \mid \text{Unmastered}) \iff 1 - S > G \iff G + S < 1.0$$

#### Pipeline: Prediction $\to$ Posterior Update $\to$ State Transition
1. **Prediction Step (before observation $y_t$):**
   $$P(\text{Correct}_t) = P(L_{t-1}) \cdot (1 - S) + (1 - P(L_{t-1})) \cdot G$$

2. **Posterior Observation Update (Bayes' Theorem):**
   * If response is **Correct** ($y_t = 1$):
     $$P(L_t \mid y_t = 1) = \frac{P(L_{t-1}) \cdot (1 - S)}{P(L_{t-1}) \cdot (1 - S) + (1 - P(L_{t-1})) \cdot G}$$
   * If response is **Incorrect** ($y_t = 0$):
     $$P(L_t \mid y_t = 0) = \frac{P(L_{t-1}) \cdot S}{P(L_{t-1}) \cdot S + (1 - P(L_{t-1})) \cdot (1 - G)}$$

3. **Learning Transition Step (Applied Exactly Once):**
   $$P(L_t) = P(L_t \mid y_t) \cdot (1 - F) + (1 - P(L_t \mid y_t)) \cdot T$$
   In standard BKT ($F = 0$):
   $$P(L_t) = P(L_t \mid y_t) + (1 - P(L_t \mid y_t)) \cdot T$$

---

### 2. Hand-Worked Numerical Verification Examples

Given baseline parameters: $P(L_0) = 0.20$, $T = 0.15$, $G = 0.25$, $S = 0.10$.

#### Scenario A: Correct Response ($y_1 = 1$)
1. **Prediction:**
   $$P(\text{Correct}_1) = 0.20 \cdot (1 - 0.10) + 0.80 \cdot 0.25 = 0.18 + 0.20 = 0.38$$
2. **Observation Update:**
   $$\text{Numerator} = 0.20 \cdot 0.90 = 0.18$$
   $$\text{Denominator} = 0.18 + 0.80 \cdot 0.25 = 0.38$$
   $$\text{Posterior } P(L_1 \mid y=1) = \frac{0.18}{0.38} = \frac{9}{19} \approx 0.47368421$$
3. **Transition Step:**
   $$P(L_1) = \frac{9}{19} + \left(1 - \frac{9}{19}\right) \cdot 0.15 = \frac{9}{19} + \frac{10}{19} \cdot 0.15 = \frac{9 + 1.5}{19} = \frac{10.5}{19} = \frac{21}{38} \approx 0.55263158$$

#### Scenario B: Incorrect Response ($y_1 = 0$)
1. **Prediction:**
   $$P(\text{Correct}_1) = 0.38$$
2. **Observation Update:**
   $$\text{Numerator} = 0.20 \cdot 0.10 = 0.02$$
   $$\text{Denominator} = 0.02 + 0.80 \cdot (1 - 0.25) = 0.02 + 0.80 \cdot 0.75 = 0.02 + 0.60 = 0.62$$
   $$\text{Posterior } P(L_1 \mid y=0) = \frac{0.02}{0.62} = \frac{1}{31} \approx 0.03225806$$
3. **Transition Step:**
   $$P(L_1) = \frac{1}{31} + \left(1 - \frac{1}{31}\right) \cdot 0.15 = \frac{1}{31} + \frac{30}{31} \cdot 0.15 = \frac{1 + 4.5}{31} = \frac{5.5}{31} = \frac{11}{62} \approx 0.17741935$$

---

### 3. pyBKT Cross-Validation & Parameter Mapping

Parameter alignment between our `BKTEngine` and `pyBKT`:
* `BKTEngine.p_l0` $\longleftrightarrow$ `pyBKT.prior`
* `BKTEngine.p_transition` $\longleftrightarrow$ `pyBKT.learns`
* `BKTEngine.p_guess` $\longleftrightarrow$ `pyBKT.guesses`
* `BKTEngine.p_slip` $\longleftrightarrow$ `pyBKT.slips`
* `BKTEngine.p_forget` $\longleftrightarrow$ `pyBKT.forgets`

**Comparative Verification:** Automated tests in `test_pybkt_comparison.py` verify that `BKTEngine` matches pyBKT's forward state propagation and response likelihoods with floating-point tolerance $\le 10^{-5}$.

---

## Running the Automated Test Suite

```bash
# From within the backend directory:
python -m pytest -v
```

**Test Suite Coverage (51 Tests Total):**
* `tests/test_competency_dataset.py`: 10 tests (graph acyclicity, topological sort, concept counts)
* `tests/test_domain_models.py`: 16 tests (Pydantic models, boundary validation)
* `tests/test_validation_engine.py`: 8 tests (cycle detection, orphan skills, missing references)
* `ml/bkt/tests/test_predictions.py`: 5 tests (prediction equations, monotonicity, bounds)
* `ml/bkt/tests/test_updates.py`: 6 tests (hand-calculated fixtures, posterior/transition separation, boundaries)
* `ml/bkt/tests/test_sequences.py`: 4 tests (deterministic replay, convergence, tracker lifecycle)
* `ml/bkt/tests/test_pybkt_comparison.py`: 2 tests (pyBKT numerical alignment, dictionary conversion)
