import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Union

from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.flow.state_utils import to_serializable
from pydantic import BaseModel


class SQLiteFlowPersistenceJSON(SQLiteFlowPersistence):
    """SQLite persistence with robust JSON serialization for state data."""

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: Union[Dict[str, Any], BaseModel],
    ) -> None:
        """Save the current flow state to SQLite using robust serialization."""
        # Use to_serializable for robust conversion to JSON-compatible types
        serializable_state = to_serializable(state_data)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            INSERT INTO flow_states (
                flow_uuid,
                method_name,
                timestamp,
                state_json
            ) VALUES (?, ?, ?, ?)
            """,
                (
                    flow_uuid,
                    method_name,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(serializable_state),  # Dump the serializable dict
                ),
            )
