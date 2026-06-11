"""
Simplified Wiki state management (Python class, not Zustand).
Derived from src/stores/wiki-store.ts
"""

import json
import os

from llm_wiki.wiki_models import WikiProject, LlmConfig
from llm_wiki.wiki_utils import normalize_path, read_file_utf8, write_file_utf8, file_exists


class WikiStore:
    """
    Simple in-memory + disk persistence store for a single Wiki project.
    """

    def __init__(self, project_path: str):
        self.project_path = normalize_path(project_path)
        self.config_path = f"{self.project_path}/.llm-wiki/config.json"
        self.project: WikiProject | None = None
        self.llm_config = LlmConfig()
        self.data_version = 0
        self._load()

    def _load(self) -> None:
        if file_exists(self.config_path):
            try:
                data = json.loads(read_file_utf8(self.config_path))
                self.llm_config = LlmConfig(**data.get("llm_config", {}))
                self.data_version = data.get("data_version", 0)
            except Exception:
                pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        data = {
            "llm_config": self.llm_config.model_dump(),
            "data_version": self.data_version,
        }
        write_file_utf8(self.config_path, json.dumps(data, indent=2, ensure_ascii=False))

    def bump_data_version(self) -> None:
        self.data_version += 1
        self.save()
