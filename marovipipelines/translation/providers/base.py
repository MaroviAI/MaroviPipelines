"""
Base provider class for translation services.

This module provides the abstract base class for translation providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class TranslationRequest:
    """Request for translation."""
    text: Union[str, List[str]]
    source_lang: str
    target_lang: str
    metadata: Optional[Dict] = None

@dataclass
class TranslationResponse:
    """Response from translation service."""
    translated_text: Union[str, List[str]]
    source_lang: str
    target_lang: str
    metadata: Optional[Dict] = None

class TranslationProvider(ABC):
    """Abstract base class for translation providers."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client."""
        pass
    
    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text from source language to target language."""
        pass
    
    @abstractmethod
    async def atranslate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text asynchronously."""
        pass
    
    @abstractmethod
    def batch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResponse]:
        """Translate multiple texts in batch."""
        pass
    
    @abstractmethod
    async def abatch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResponse]:
        """Translate multiple texts asynchronously in batch."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        pass 