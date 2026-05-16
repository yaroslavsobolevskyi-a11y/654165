from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    vault_path: Path = field(default_factory=lambda: Path.cwd())
    db_path: Path = field(default_factory=lambda: Path.home() / ".axon" / "axon.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_links: int = 5
    min_similarity: float = 0.35
    batch_size: int = 64

    def __post_init__(self) -> None:
        self.vault_path = Path(self.vault_path).expanduser().resolve()
        self.db_path = Path(self.db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            vault_path=Path(os.environ.get("AXON_VAULT", Path.cwd())),
            db_path=Path(os.environ.get("AXON_DB", Path.home() / ".axon" / "axon.db")),
            embedding_model=os.environ.get("AXON_MODEL", "all-MiniLM-L6-v2"),
            top_k_links=int(os.environ.get("AXON_TOP_K", "5")),
            min_similarity=float(os.environ.get("AXON_MIN_SIM", "0.35")),
            batch_size=int(os.environ.get("AXON_BATCH_SIZE", "64")),
        )
