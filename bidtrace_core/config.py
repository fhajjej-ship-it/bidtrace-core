# SPDX-License-Identifier: MPL-2.0
"""Explicit local core configuration; never loads application dotenv files."""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class CoreSettings:
    data_dir: Path
    model_url: str
    workspace_id: str = 'local'
    storage_backend: str = 'embedded'
    inference_provider: str = 'mlx'
    standalone_core: bool = True
    mlx_assessment_strategy: str = 'condition_nli'
    mlx_memory_limit_gib: float = 10
    qdrant_url: str = 'http://127.0.0.1:16333'
    model_timeout: int = 600

    def __post_init__(self):
        url = urlsplit(self.model_url)
        if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or url.username or url.password or url.path not in {'', '/'} or url.query or url.fragment):
            raise ValueError('The core requires a credential-free loopback HTTP model service.')
        if self.inference_provider != 'mlx' or self.storage_backend != 'embedded' or not self.standalone_core:
            raise ValueError('The standalone core requires local MLX inference and embedded storage.')
        if self.mlx_assessment_strategy != 'condition_nli' or self.mlx_memory_limit_gib != 10:
            raise ValueError('This core CLI uses the benchmarked condition_nli profile and memory limit.')
