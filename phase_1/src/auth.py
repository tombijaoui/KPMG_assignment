from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class DocumentIntelligenceConfig:
    endpoint: str
    key: str


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise ValueError(f"Missing environment variable: {name}. Set it in {ENV_PATH}")

    return value


def get_document_intelligence_config() -> DocumentIntelligenceConfig:
    return DocumentIntelligenceConfig(endpoint=_require("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
                                      key=_require("AZURE_DOCUMENT_INTELLIGENCE_KEY"))


def create_document_intelligence_client(config: DocumentIntelligenceConfig | 
                                        None = None) -> DocumentIntelligenceClient:

    cfg = config or get_document_intelligence_config()

    return DocumentIntelligenceClient(endpoint=cfg.endpoint, credential=AzureKeyCredential(cfg.key))
