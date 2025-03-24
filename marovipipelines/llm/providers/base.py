"""
Base provider class for LLM client.

This module provides the abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, AsyncIterator

from pydantic import BaseModel

from ..schemas import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    def complete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion."""
        pass
    
    @abstractmethod
    async def acomplete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion asynchronously."""
        pass
    
    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion."""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass 