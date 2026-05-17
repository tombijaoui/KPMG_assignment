from __future__ import annotations

import os
from dataclasses import dataclass

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import find_dotenv, load_dotenv
from config.logger import get_logger
from openai import AzureOpenAI

load_dotenv(find_dotenv())

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentIntelligenceConfig:
    endpoint: str
    key: str

@dataclass(frozen=True)
class AzureFoundryModelConfig:
    endpoint: str
    key: str
    api_version: str


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

def get_llm_gpt_4o_config() -> AzureFoundryModelConfig:
    try:
        config = AzureFoundryModelConfig(endpoint=_require("AZURE_GPT_4O_MODEL_ENDPOINT"), 
                                        key=_require("AZURE_FOUNDRY_MODELS_KEY"),
                                        api_version=_require("AZURE_GPT_4O_MODEL_API_VERSION"))

        logger.info("GPT-4o model configuration loaded successfully")
        return config

    except ValueError:
        logger.error("Failed to load GPT-4o model configuration")
        raise

def get_llm_gpt_4o_mini_config() -> AzureFoundryModelConfig:
    try:
        config = AzureFoundryModelConfig(endpoint=_require("AZURE_GPT_4O_MINI_MODEL_ENDPOINT"), 
                                        key=_require("AZURE_FOUNDRY_MODELS_KEY"),
                                        api_version=_require("AZURE_GPT_4O_MINI_MODEL_API_VERSION"))

        logger.info("GPT-4o model configuration loaded successfully")
        return config

    except ValueError:
        logger.error("Failed to load GPT-4o model configuration")
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

def create_llm_gpt_4o_client(config: AzureFoundryModelConfig | None = None) -> AzureOpenAI:
    try:
        cfg = config or get_llm_gpt_4o_config()

        logger.info("Creating GPT-4o model client")

        client = AzureOpenAI(azure_endpoint=cfg.endpoint, api_key=cfg.key, api_version=cfg.api_version)

        logger.info("GPT-4o model client created successfully")
        return client

    except ValueError:
        logger.error("Failed to create GPT-4o model client: invalid configuration")
        raise

    except Exception:
        logger.exception("Failed to create GPT-4o model client")
        raise

def create_llm_gpt_4o_mini_client(config: AzureFoundryModelConfig | None = None) -> AzureOpenAI:
    try:
        cfg = config or get_llm_gpt_4o_mini_config()

        logger.info("Creating GPT-4o mini model client")

        client = AzureOpenAI(azure_endpoint=cfg.endpoint, api_key=cfg.key, api_version=cfg.api_version)

        logger.info("GPT-4o mini model client created successfully")
        return client
    
    except ValueError:
        logger.error("Failed to create GPT-4o mini model client: invalid configuration")
        raise

    except Exception:
        logger.exception("Failed to create GPT-4o mini model client")
        raise