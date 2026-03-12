#!/usr/bin/env python3
"""Project-scoped Gemini session management."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATE_DIRNAME = ".gemini"
SESSION_FILENAME = "project_session.json"


@dataclass
class GeminiProjectClient:
    project_root: Path
    gemini_command: str = "gemini"
    session_filename: str = SESSION_FILENAME

    @property
    def state_dir(self) -> Path:
        return self.project_root / STATE_DIRNAME

    @property
    def session_path(self) -> Path:
        return self.state_dir / self.session_filename

    def load_session_id(self) -> str | None:
        if not self.session_path.exists():
            return None
        payload = json.loads(self.session_path.read_text())
        return payload.get("session_id")

    def save_session_id(self, session_id: str) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.session_path.write_text(json.dumps({"session_id": session_id}, indent=2) + "\n")

    def ensure_session(self, bootstrap_prompt: str) -> str:
        session_id = self.load_session_id()
        if session_id:
            return session_id
        response = self.ask(
            bootstrap_prompt,
            resume=False,
            output_format="json",
        )
        return response["session_id"]

    def ask(
        self,
        prompt: str,
        *,
        resume: bool = True,
        model: str | None = None,
        output_format: str = "json",
    ) -> dict[str, Any]:
        command = [self.gemini_command]
        if model:
            command.extend(["--model", model])

        session_id = self.load_session_id() if resume else None
        if session_id:
            command.extend(["--resume", session_id])

        command.extend(["--output-format", output_format, "--prompt", prompt])
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Gemini command failed")

        payload = json.loads(completed.stdout)
        session_id = payload.get("session_id")
        if session_id:
            self.save_session_id(session_id)
        return payload


def repo_root_from(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from current path.")
