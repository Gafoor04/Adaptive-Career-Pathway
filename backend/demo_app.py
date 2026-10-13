"""Streamlit demonstration interface for Adaptive Career-Pathway Recommendation.

Features:
- Concept-Centric Learner Modeling with Bayesian Knowledge Tracing (BKT)
- SQLite persistent state across restarts
- Multi-learner management and explicit isolated reset
- Role competency-fit ranking and skill-gap analysis
- Prerequisite-aware adaptive learning roadmaps
"""

import streamlit as st

from app.domain.competency_loader import CompetencyRepository
from app.domain.learner_engine import LearnerEngine
from app.domain.persistence import LearnerStore
from app.domain.recommendation import rank_career_roles


st.set_page_config(
    page_title="Adaptive Career Pathway",
    page_icon="🎯",
    layout="wide",
)

st.title("Adaptive Career Pathway Recommendation")
st.caption(
    "Concept-Centric Learner Modeling | Bayesian Knowledge Tracing "
    "| Prerequisite-Aware Roadmaps | SQLite Persistence"
)


@st.cache_resource
def get_repository() -> CompetencyRepository:
    return CompetencyRepository.load_from_files()


@st.cache_resource
def get_store() -> LearnerStore:
    return LearnerStore()


repository = get_repository()
store = get_store()

# -----------------------------------------------------------------------------
# Sidebar: Learner Profile & Multi-Learner Management
# -----------------------------------------------------------------------------
st.sidebar.header("Learner Profile")

existing_learners = store.list_learners()
if not existing_learners:
    existing_learners = ["demo-learner"]

if "active_learner_id" not in st.session_state:
    st.session_state.active_learner_id = existing_learners[0]

# Selection between existing learner and new learner
options = existing_learners + ["➕ Create New Learner"]
try:
    current_idx = existing_learners.index(st.session_state.active_learner_id)
except ValueError:
    current_idx = 0

learner_selection = st.sidebar.selectbox(
    "Active Learner",
    options,
    index=current_idx,
    key="learner_dropdown",
)

if learner_selection == "➕ Create New Learner":
    new_learner_input = st.sidebar.text_input("Enter New Learner ID", value="", key="new_learner_input").strip()
    if st.sidebar.button("Create & Switch", type="secondary"):
        if new_learner_input:
            st.session_state.active_learner_id = new_learner_input
            st.session_state.learner = LearnerEngine(repository, learner_id=new_learner_input, store=store)
            st.session_state.last_result = None
            st.rerun()
        else:
            st.sidebar.error("Learner ID cannot be empty.")
else:
    if st.session_state.active_learner_id != learner_selection:
        st.session_state.active_learner_id = learner_selection
        st.session_state.learner = LearnerEngine(repository, learner_id=learner_selection, store=store)
        st.session_state.last_result = None
        st.rerun()

# Ensure LearnerEngine is initialized for the active learner
if "learner" not in st.session_state or st.session_state.learner.learner_id != st.session_state.active_learner_id:
    st.session_state.learner = LearnerEngine(
        repository,
        learner_id=st.session_state.active_learner_id,
        store=store,
    )
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

learner = st.session_state.learner

st.sidebar.info(f"Active ID: **{learner.learner_id}**\n\nState is persisted to SQLite.")
st.sidebar.metric("Questions Answered", len(learner.events))
tracked_count = sum(1 for count in learner.interaction_counts.values() if count > 0)
st.sidebar.metric("Concepts Interacted", tracked_count)

with st.sidebar.expander("⚠️ Reset Learner State", expanded=False):
    st.write(f"Resetting will delete all interaction history and restore concept masteries to baseline for **{learner.learner_id}** only.")
    if st.button("Reset Learner Data", type="secondary", key="reset_learner_btn"):
        learner.reset()
        st.session_state.last_result = None
        st.success(f"Learner '{learner.learner_id}' successfully reset.")
        st.rerun()

# -----------------------------------------------------------------------------
# 1. Knowledge Assessment (Quiz Submission)
# -----------------------------------------------------------------------------
st.header("1. Knowledge Assessment")

questions = repository.questions
question_by_id = {q.id: q for q in questions}
question_ids = [q.id for q in questions]

selected_id = st.selectbox(
    "Choose a question",
    question_ids,
    format_func=lambda qid: f"{qid} — {question_by_id[qid].prompt[:70]}...",
    key="selected_question_dropdown",
)

question = question_by_id[selected_id]
concept = repository.get_concept(question.concept_id)

st.subheader(concept.name if concept else question.concept_id)
st.write(question.prompt)

# Use st.form to guarantee that Streamlit reruns do NOT accidentally resubmit duplicate answers
with st.form(key=f"quiz_form_{selected_id}"):
    selected_option = st.radio(
        "Select your answer",
        range(len(question.options)),
        format_func=lambda i: question.options[i],
        key=f"option_{selected_id}",
    )
    submitted = st.form_submit_button("Submit answer", type="primary")

if submitted:
    result = learner.answer(selected_id, selected_option)
    st.session_state.last_result = result
    st.rerun()

if st.session_state.last_result:
    result = st.session_state.last_result

    if result["is_correct"]:
        st.success("✅ Correct answer! Knowledge probability updated.")
    else:
        st.error("❌ Incorrect answer. Knowledge probability updated.")

    before = result["prior_mastery"] * 100
    after = result["updated_mastery"] * 100
    delta = after - before

    col1, col2, col3 = st.columns(3)
    col1.metric("Prior Mastery", f"{before:.1f}%")
    col2.metric("Posterior Mastery", f"{result['posterior_mastery'] * 100:.1f}%")
    col3.metric("Updated Mastery (Post-Transition)", f"{after:.1f}%", f"{delta:+.1f}%")

    st.write(f"**Concept:** {concept.name if concept else result['concept_id']}")
    st.write(f"**Explanation:** {result['explanation']}")

# Recent Assessment History Audit Table
if learner.events:
    with st.expander(f"📜 Recent Attempt History for {learner.learner_id} ({len(learner.events)} total)", expanded=False):
        history_rows = []
        for evt in reversed(learner.events[-10:]):
            c_name = repository.get_concept(evt["concept_id"])
            history_rows.append({
                "Timestamp": evt.get("timestamp", "")[:19].replace("T", " "),
                "Question": evt["question_id"],
                "Concept": c_name.name if c_name else evt["concept_id"],
                "Outcome": "Correct" if evt["is_correct"] else "Incorrect",
                "Prior (%)": f"{evt['prior_mastery'] * 100:.1f}%",
                "Updated (%)": f"{evt['updated_mastery'] * 100:.1f}%",
            })
        st.dataframe(history_rows, width="stretch", hide_index=True)

st.divider()

# -----------------------------------------------------------------------------
# 2. Concept Mastery Overview
# -----------------------------------------------------------------------------
st.header("2. Concept Mastery Overview")
st.info(
    "ℹ️ **Epistemic Notice**: For concepts with 0 attempts, the percentage represents a configured "
    "baseline prior assumption $P(L_0)$, **not** empirically measured student knowledge. "
    "Once quiz questions are completed, values update via analytical Bayesian Knowledge Tracing."
)

mastery_rows = []
for c in repository.list_concepts():
    value = learner.mastery[c.id]
    attempts = learner.interaction_counts[c.id]
    mastery_rows.append({
        "Concept ID": c.id,
        "Concept": c.name,
        "Category": c.category,
        "Mastery Estimate (%)": round(value * 100, 1),
        "Attempts": attempts,
        "Data Source": "Bayesian Estimate (BKT)" if attempts > 0 else "Baseline Prior Assumption",
    })

st.dataframe(
    sorted(mastery_rows, key=lambda row: row["Mastery Estimate (%)"], reverse=True),
    width="stretch",
    hide_index=True,
)

st.divider()

# -----------------------------------------------------------------------------
# 3. Career Role Recommendations
# -----------------------------------------------------------------------------
st.header("3. Career Role Recommendations")
st.caption(
    "Competency-fit scores quantify curriculum coverage against target skill thresholds. "
    "They are **not** probabilities of employment or job offer predictions."
)

roles = repository.list_roles()
ranked_roles = rank_career_roles(repository, learner.mastery)

for rank, item in enumerate(ranked_roles, start=1):
    with st.expander(
        f"{rank}. {item['role_title']} — {item['fit_percent']:.1f}% Competency Fit",
        expanded=(rank == 1),
    ):
        st.caption("Role fit indicates target skill coverage, not an employment probability.")
        st.progress(item["fit_score"])
        gaps = item["skill_gaps"]

        if gaps:
            st.write("**Priority Skill Gaps**")
            st.dataframe(
                [
                    {
                        "Skill": gap["skill_name"],
                        "Current Readiness (%)": round(gap["current_readiness"] * 100, 1),
                        "Target Mastery (%)": round(gap["target_mastery"] * 100, 1),
                        "Mastery Gap (%)": round(gap["gap"] * 100, 1),
                    }
                    for gap in gaps
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("Target skill readiness satisfied for this career role!")

st.divider()

# -----------------------------------------------------------------------------
# 4. Adaptive Learning Roadmap
# -----------------------------------------------------------------------------
st.header("4. Adaptive Learning Roadmap")

role_options = {role.title: role.id for role in roles}
selected_role_title = st.selectbox(
    "Choose your target career role",
    list(role_options.keys()),
    key="roadmap_role_select",
)

target_role_id = role_options[selected_role_title]
roadmap = learner.roadmap(target_role_id, limit=10)

if roadmap:
    st.write(f"Top Recommended Learning Sequence for **{selected_role_title}** (prerequisites scheduled first):")
    for index, item in enumerate(roadmap, start=1):
        prefix = "📌 [Prerequisite]" if item.get("is_prerequisite_only") else "🎯 [Role Core]"
        st.write(
            f"**{index}. {prefix} {item['concept_name']}** "
            f"— Current: {item['mastery_percent']:.1f}% | Gap: {item['mastery_gap']:.2f}"
        )
        st.caption(
            f"**Category:** {item['category']} | **Rationale:** {item['reason']}"
        )
else:
    st.success("No concepts currently fall below the roadmap threshold for this role!")

st.caption(
    "Research prototype: Knowledge probabilities updated via analytical BKT. "
    "Role scores represent competency-fit coverage, not employment probabilities. "
    "Learner models and interaction history are saved to local SQLite persistence."
)
