from __future__ import annotations

import os
from dataclasses import dataclass

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import find_dotenv, load_dotenv
from config.logger import get_logger

load_dotenv(find_dotenv())

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentIntelligenceConfig:
    endpoint: str
    key: str


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        logger.error("Missing environment variable: %s", name)
        raise ValueError(f"Missing environment variable: {name}")

    return value


def get_document_intelligence_config() -> DocumentIntelligenceConfig:
    try:
        config = DocumentIntelligenceConfig(endpoint=_require("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"), 
                                            key=_require("AZURE_DOCUMENT_INTELLIGENCE_KEY"))

        logger.info("Document Intelligence configuration loaded successfully")
        return config

    except ValueError:
        logger.error("Failed to load Document Intelligence configuration")
        raise


def create_document_intelligence_client(config: DocumentIntelligenceConfig | None = None) -> DocumentIntelligenceClient:
    try:
        cfg = config or get_document_intelligence_config()

        logger.info("Creating Document Intelligence client")

        client = DocumentIntelligenceClient(endpoint=cfg.endpoint, credential=AzureKeyCredential(cfg.key))

        logger.info("Document Intelligence client created successfully")
        return client

    except ValueError:
        logger.error("Failed to create Document Intelligence client: invalid configuration")
        raise
        
    except Exception:
        logger.exception("Failed to create Document Intelligence client")
        raise
