"""SQLite persistence layer for long-term learner state and interaction audit history.

Provides robust, transaction-safe storage for:
- Learner profiles
- Concept-level mastery snapshots
- Detailed assessment interaction events

Separates database access and SQL mechanics from the mathematical BKT engine.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Union

from app.core.config import settings
from app.domain.models import ConceptMasterySnapshot, LearnerInteractionEvent


class LearnerPersistenceError(Exception):
    """Raised when a database transaction or operation fails."""
    pass


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_mastery (
    learner_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    p_mastery REAL NOT NULL,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (learner_id, concept_id),
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interaction_events (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    prior_mastery REAL NOT NULL,
    posterior_mastery REAL NOT NULL,
    post_transition_mastery REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_concept_mastery_learner
    ON concept_mastery(learner_id);

CREATE INDEX IF NOT EXISTS idx_events_learner_time
    ON interaction_events(learner_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_events_learner_concept
    ON interaction_events(learner_id, concept_id);
"""


class LearnerStore:
    """Thread-safe SQLite storage for learner modeling and interaction history."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            self.db_path = settings.database_path
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        # Ensure directory exists if filesystem path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # In-memory connections must be persistent if using :memory:
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.execute("PRAGMA foreign_keys = ON;")

        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.OperationalError:
            pass
        return conn

    def init_db(self) -> None:
        """Initialize explicit tables and indexes if they do not exist."""
        conn = self._get_connection()
        try:
            with conn:
                conn.executescript(SCHEMA_SQL)
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to initialize database schema: {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def get_or_create_learner(self, learner_id: str) -> dict:
        """Retrieve existing learner or create a new learner record."""
        clean_id = learner_id.strip()
        if not clean_id:
            raise ValueError("Learner ID cannot be empty.")

        conn = self._get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT learner_id, created_at, updated_at FROM learners WHERE learner_id = ?",
                    (clean_id,),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "learner_id": row["learner_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }

                conn.execute(
                    "INSERT INTO learners (learner_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (clean_id, now_iso, now_iso),
                )
                return {
                    "learner_id": clean_id,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to get or create learner '{clean_id}': {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def list_learners(self) -> List[str]:
        """List all unique learner IDs stored in the database."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT learner_id FROM learners ORDER BY learner_id ASC")
            return [row["learner_id"] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to list learners: {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def learner_exists(self, learner_id: str) -> bool:
        """Check if a learner profile exists."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT 1 FROM learners WHERE learner_id = ?",
                (learner_id.strip(),),
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to check learner existence: {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def get_concept_mastery(self, learner_id: str) -> Dict[str, ConceptMasterySnapshot]:
        """Retrieve saved concept mastery snapshots for a specific learner."""
        clean_id = learner_id.strip()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT learner_id, concept_id, p_mastery, interaction_count, last_updated
                FROM concept_mastery
                WHERE learner_id = ?
                """,
                (clean_id,),
            )
            snapshots: Dict[str, ConceptMasterySnapshot] = {}
            for row in cursor.fetchall():
                snapshots[row["concept_id"]] = ConceptMasterySnapshot(
                    learner_id=row["learner_id"],
                    concept_id=row["concept_id"],
                    p_mastery=float(row["p_mastery"]),
                    interaction_count=int(row["interaction_count"]),
                    last_updated=datetime.fromisoformat(row["last_updated"]),
                )
            return snapshots
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to get concept mastery for '{clean_id}': {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def save_concept_mastery(self, snapshot: ConceptMasterySnapshot) -> None:
        """Upsert a single concept mastery snapshot for a learner."""
        conn = self._get_connection()
        now_iso = snapshot.last_updated.isoformat()
        try:
            with conn:
                # Ensure learner exists
                conn.execute(
                    "INSERT OR IGNORE INTO learners (learner_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (snapshot.learner_id, now_iso, now_iso),
                )
                conn.execute(
                    """
                    INSERT INTO concept_mastery (learner_id, concept_id, p_mastery, interaction_count, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                        p_mastery = excluded.p_mastery,
                        interaction_count = excluded.interaction_count,
                        last_updated = excluded.last_updated
                    """,
                    (
                        snapshot.learner_id,
                        snapshot.concept_id,
                        float(snapshot.p_mastery),
                        int(snapshot.interaction_count),
                        now_iso,
                    ),
                )
                conn.execute(
                    "UPDATE learners SET updated_at = ? WHERE learner_id = ?",
                    (now_iso, snapshot.learner_id),
                )
        except sqlite3.Error as e:
            raise LearnerPersistenceError(
                f"Failed to save concept mastery for learner '{snapshot.learner_id}', "
                f"concept '{snapshot.concept_id}': {e}"
            ) from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def save_mastery_bulk(self, snapshots: List[ConceptMasterySnapshot]) -> None:
        """Upsert a list of concept mastery snapshots in a single safe transaction."""
        if not snapshots:
            return
        conn = self._get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with conn:
                for s in snapshots:
                    conn.execute(
                        "INSERT OR IGNORE INTO learners (learner_id, created_at, updated_at) VALUES (?, ?, ?)",
                        (s.learner_id, now_iso, now_iso),
                    )
                    conn.execute(
                        """
                        INSERT INTO concept_mastery (learner_id, concept_id, p_mastery, interaction_count, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                            p_mastery = excluded.p_mastery,
                            interaction_count = excluded.interaction_count,
                            last_updated = excluded.last_updated
                        """,
                        (
                            s.learner_id,
                            s.concept_id,
                            float(s.p_mastery),
                            int(s.interaction_count),
                            s.last_updated.isoformat(),
                        ),
                    )
                # Update learner timestamp
                learner_ids = {s.learner_id for s in snapshots}
                for lid in learner_ids:
                    conn.execute(
                        "UPDATE learners SET updated_at = ? WHERE learner_id = ?",
                        (now_iso, lid),
                    )
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to bulk-save concept mastery snapshots: {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def record_interaction_event(
        self,
        event: LearnerInteractionEvent,
        updated_snapshot: ConceptMasterySnapshot,
    ) -> None:
        """Atomically persist an interaction audit event and update the concept mastery state.

        If either the event insert or mastery upsert fails, the transaction is rolled back,
        preventing inconsistent state between history and current mastery.
        """
        conn = self._get_connection()
        ts_iso = event.timestamp.isoformat()
        snapshot_ts_iso = updated_snapshot.last_updated.isoformat()
        try:
            with conn:
                # 1. Ensure learner exists
                conn.execute(
                    "INSERT OR IGNORE INTO learners (learner_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (event.learner_id, ts_iso, ts_iso),
                )

                # 2. Insert interaction audit event
                conn.execute(
                    """
                    INSERT INTO interaction_events (
                        id, learner_id, attempt_id, concept_id, question_id,
                        is_correct, prior_mastery, posterior_mastery,
                        post_transition_mastery, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.id,
                        event.learner_id,
                        event.attempt_id,
                        event.concept_id,
                        event.question_id,
                        1 if event.is_correct else 0,
                        float(event.prior_mastery),
                        float(event.posterior_mastery),
                        float(event.post_transition_mastery),
                        ts_iso,
                    ),
                )

                # 3. Update concept mastery snapshot
                conn.execute(
                    """
                    INSERT INTO concept_mastery (
                        learner_id, concept_id, p_mastery, interaction_count, last_updated
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                        p_mastery = excluded.p_mastery,
                        interaction_count = excluded.interaction_count,
                        last_updated = excluded.last_updated
                    """,
                    (
                        updated_snapshot.learner_id,
                        updated_snapshot.concept_id,
                        float(updated_snapshot.p_mastery),
                        int(updated_snapshot.interaction_count),
                        snapshot_ts_iso,
                    ),
                )

                # 4. Touch learner updated_at
                conn.execute(
                    "UPDATE learners SET updated_at = ? WHERE learner_id = ?",
                    (ts_iso, event.learner_id),
                )
        except sqlite3.Error as e:
            raise LearnerPersistenceError(
                f"Failed to record interaction event '{event.id}' for learner '{event.learner_id}': {e}"
            ) from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def get_interaction_events(
        self,
        learner_id: str,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> List[LearnerInteractionEvent]:
        """Query historical interaction audit events for a learner."""
        clean_id = learner_id.strip()
        conn = self._get_connection()
        direction = "ASC" if ascending else "DESC"
        query = f"""
            SELECT id, learner_id, attempt_id, concept_id, question_id,
                   is_correct, prior_mastery, posterior_mastery,
                   post_transition_mastery, timestamp
            FROM interaction_events
            WHERE learner_id = ?
            ORDER BY timestamp {direction}, id {direction}
        """
        params: list = [clean_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        try:
            cursor = conn.execute(query, tuple(params))
            events: List[LearnerInteractionEvent] = []
            for row in cursor.fetchall():
                events.append(
                    LearnerInteractionEvent(
                        id=row["id"],
                        learner_id=row["learner_id"],
                        attempt_id=row["attempt_id"],
                        concept_id=row["concept_id"],
                        question_id=row["question_id"],
                        is_correct=bool(row["is_correct"]),
                        prior_mastery=float(row["prior_mastery"]),
                        posterior_mastery=float(row["posterior_mastery"]),
                        post_transition_mastery=float(row["post_transition_mastery"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )
            return events
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to query events for learner '{clean_id}': {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def get_interaction_count(self, learner_id: str) -> int:
        """Count total recorded interactions for a learner."""
        clean_id = learner_id.strip()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM interaction_events WHERE learner_id = ?",
                (clean_id,),
            )
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to count interactions for '{clean_id}': {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def reset_learner_state(self, learner_id: str) -> None:
        """Reset only the specified learner's mastery snapshots and interaction events.

        Preserves other learners completely.
        """
        clean_id = learner_id.strip()
        conn = self._get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with conn:
                conn.execute("DELETE FROM interaction_events WHERE learner_id = ?", (clean_id,))
                conn.execute("DELETE FROM concept_mastery WHERE learner_id = ?", (clean_id,))
                conn.execute(
                    "UPDATE learners SET updated_at = ? WHERE learner_id = ?",
                    (now_iso, clean_id),
                )
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to reset state for learner '{clean_id}': {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def delete_learner(self, learner_id: str) -> None:
        """Completely remove a learner and all associated data."""
        clean_id = learner_id.strip()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM interaction_events WHERE learner_id = ?", (clean_id,))
                conn.execute("DELETE FROM concept_mastery WHERE learner_id = ?", (clean_id,))
                conn.execute("DELETE FROM learners WHERE learner_id = ?", (clean_id,))
        except sqlite3.Error as e:
            raise LearnerPersistenceError(f"Failed to delete learner '{clean_id}': {e}") from e
        finally:
            if self._memory_conn is None:
                conn.close()

    def reconstruct_mastery_from_events(
        self,
        learner_id: str,
        baseline_p_l0: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Reconstruct current concept mastery strictly from recorded event sequence.

        For any concept observed in events, uses the post_transition_mastery of the
        latest interaction. For unobserved concepts, falls back to baseline_p_l0 or 0.10.
        """
        events = self.get_interaction_events(learner_id, ascending=True)
        mastery: Dict[str, float] = {}
        if baseline_p_l0:
            mastery.update(baseline_p_l0)

        for event in events:
            mastery[event.concept_id] = event.post_transition_mastery

        return mastery
