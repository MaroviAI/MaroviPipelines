"""
Schemas for LLM client and providers.

This module contains Pydantic models and data structures for LLM requests,
responses, and configuration.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Type variables for generic type safety
ResponseType = TypeVar('ResponseType')


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"
    CUSTOM = "custom"


class LLMRequest(BaseModel):
    """Container for LLM request parameters."""
    prompt: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 8000
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse(Generic[ResponseType]):
    """Container for LLM response with metadata."""
    content: ResponseType
    model: str
    usage: Dict[str, int]
    latency: float
    raw_response: Any = None
    finish_reason: Optional[str] = None
