"""
Translation client for interacting with various translation providers.

This module provides a unified interface for making requests to different translation
providers with support for batching, retries, and comprehensive observability.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional, Union, Type
from enum import Enum

from ..core.context import PipelineContext
from .providers.base import TranslationProvider, TranslationRequest, TranslationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Supported translation providers."""
    GOOGLE = "google"
    DEEPL = "deepl"

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    "max_retries": 3,
    "initial_backoff": 1.0,
    "max_backoff": 10.0,
    "backoff_factor": 2.0,
    "retryable_errors": [
        "rate_limit_exceeded",
        "server_error",
        "connection_error",
        "timeout"
    ]
}

class TranslationClient:
    """
    A unified client for interacting with various translation providers.
    
    Features:
    - Support for multiple providers (Google Translate, DeepL, etc.)
    - Batch translation support
    - Async and sync interfaces
    - Automatic logging to pipeline context
    - Comprehensive observability
    - Retry logic for transient failures
    """
    
    def __init__(self, 
                provider: Union[str, ProviderType] = ProviderType.GOOGLE,
                api_key: Optional[str] = None,
                custom_provider: Optional[TranslationProvider] = None,
                retry_config: Optional[Dict] = None):
        """
        Initialize the translation client.
        
        Args:
            provider: Translation provider ("google", "deepl", or ProviderType enum)
            api_key: Optional API key (if not provided, will use environment variables)
            custom_provider: Optional custom provider implementation
            retry_config: Configuration for retry logic (None to use defaults)
        """
        if isinstance(provider, str):
            try:
                self.provider_type = ProviderType(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.provider_type = provider
        
        # Initialize the appropriate provider
        if custom_provider:
            self.provider = custom_provider
            self.provider_type = ProviderType.CUSTOM
        elif self.provider_type == ProviderType.GOOGLE:
            from .providers.google import GoogleTranslateProvider
            self.provider = GoogleTranslateProvider(api_key=api_key)
        elif self.provider_type == ProviderType.DEEPL:
            from .providers.deepl import DeepLProvider
            self.provider = DeepLProvider(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")
        
        # Initialize the provider
        self.provider.initialize()
        
        # Set retry configuration
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        
        logger.info(f"Initialized TranslationClient with provider={self.provider_type.value}")
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if a request should be retried based on the error and attempt number."""
        if attempt >= self.retry_config["max_retries"]:
            return False
        
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        # Check if error matches any retryable error patterns
        for retryable_error in self.retry_config["retryable_errors"]:
            if retryable_error in error_type or retryable_error in error_msg:
                return True
        
        return False
    
    async def _backoff(self, attempt: int) -> None:
        """Implement exponential backoff for retries."""
        backoff_time = min(
            self.retry_config["initial_backoff"] * (self.retry_config["backoff_factor"] ** attempt),
            self.retry_config["max_backoff"]
        )
        await asyncio.sleep(backoff_time)
    
    def _sync_backoff(self, attempt: int) -> None:
        """Implement synchronous exponential backoff for retries."""
        backoff_time = min(
            self.retry_config["initial_backoff"] * (self.retry_config["backoff_factor"] ** attempt),
            self.retry_config["max_backoff"]
        )
        time.sleep(backoff_time)
    
    def translate(self, 
                 text: Union[str, List[str]], 
                 source_lang: str, 
                 target_lang: str,
                 context: Optional[PipelineContext] = None,
                 step_name: Optional[str] = None) -> Union[str, List[str]]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text or list of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            
        Returns:
            Translated text or list of translated texts
        """
        start_time = time.time()
        
        # Prepare request
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text_length": len(text) if isinstance(text, str) else sum(len(t) for t in text),
            "timestamp": start_time,
        }
        
        # Implement retry logic
        attempt = 0
        while True:
            try:
                # Call the provider
                response = self.provider.translate(request)
                
                # Log to context if provided
                if context and step_name:
                    # Add translation info to context
                    translation_info = {
                        "request": request_metadata,
                        "text": text,
                        "translated_text": response.translated_text,
                        "latency": time.time() - start_time,
                        "success": True,
                        "attempts": attempt + 1
                    }
                    
                    # Log metrics
                    context.log_metrics({
                        f"{step_name}_translation_latency": time.time() - start_time,
                        f"{step_name}_translation_text_length": len(text) if isinstance(text, str) else sum(len(t) for t in text),
                        f"{step_name}_translation_attempts": attempt + 1
                    })
                    
                    # Update context state
                    context.update_state(
                        f"{step_name}_translation",
                        response.translated_text,
                        translation_info
                    )
                
                logger.info(f"Translation successful: {self.provider_type.value}, "
                           f"source={source_lang}, target={target_lang}, "
                           f"latency: {time.time() - start_time:.2f}s")
                
                return response.translated_text
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying translation (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    self._sync_backoff(attempt - 1)
                    continue
                
                # If we shouldn't retry, log the error and raise
                error_info = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request": request_metadata,
                    "attempts": attempt
                }
                
                # Log error to context if provided
                if context and step_name:
                    context.update_state(
                        f"{step_name}_translation_error",
                        None,
                        error_info
                    )
                    
                    context.log_metrics({
                        f"{step_name}_translation_error_count": 1,
                        f"{step_name}_translation_attempts": attempt
                    })
                
                logger.error(f"Translation failed after {attempt} attempts: {str(e)}")
                raise
    
    async def atranslate(self, 
                        text: Union[str, List[str]], 
                        source_lang: str, 
                        target_lang: str,
                        context: Optional[PipelineContext] = None,
                        step_name: Optional[str] = None) -> Union[str, List[str]]:
        """
        Translate text asynchronously.
        
        Args:
            text: Text or list of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            
        Returns:
            Translated text or list of translated texts
        """
        start_time = time.time()
        
        # Prepare request
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text_length": len(text) if isinstance(text, str) else sum(len(t) for t in text),
            "timestamp": start_time,
        }
        
        # Implement retry logic
        attempt = 0
        while True:
            try:
                # Call the provider
                response = await self.provider.atranslate(request)
                
                # Log to context if provided
                if context and step_name:
                    # Add translation info to context
                    translation_info = {
                        "request": request_metadata,
                        "text": text,
                        "translated_text": response.translated_text,
                        "latency": time.time() - start_time,
                        "success": True,
                        "attempts": attempt + 1
                    }
                    
                    # Log metrics
                    context.log_metrics({
                        f"{step_name}_translation_latency": time.time() - start_time,
                        f"{step_name}_translation_text_length": len(text) if isinstance(text, str) else sum(len(t) for t in text),
                        f"{step_name}_translation_attempts": attempt + 1
                    })
                    
                    # Update context state
                    context.update_state(
                        f"{step_name}_translation",
                        response.translated_text,
                        translation_info
                    )
                
                logger.info(f"Async translation successful: {self.provider_type.value}, "
                           f"source={source_lang}, target={target_lang}, "
                           f"latency: {time.time() - start_time:.2f}s")
                
                return response.translated_text
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying async translation (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    await self._backoff(attempt - 1)
                    continue
                
                # If we shouldn't retry, log the error and raise
                error_info = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request": request_metadata,
                    "attempts": attempt
                }
                
                # Log error to context if provided
                if context and step_name:
                    context.update_state(
                        f"{step_name}_translation_error",
                        None,
                        error_info
                    )
                    
                    context.log_metrics({
                        f"{step_name}_translation_error_count": 1,
                        f"{step_name}_translation_attempts": attempt
                    })
                
                logger.error(f"Async translation failed after {attempt} attempts: {str(e)}")
                raise
    
    def batch_translate(self,
                       texts: List[str],
                       source_lang: str,
                       target_lang: str,
                       context: Optional[PipelineContext] = None,
                       step_name: Optional[str] = None,
                       max_concurrency: int = 5) -> List[str]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of translated texts
        """
        batch_start_time = time.time()
        
        # Prepare requests
        requests = [
            TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                metadata={"step_name": f"{step_name}_{i}"} if step_name else None
            )
            for i, text in enumerate(texts)
        ]
        
        # Process in batches to respect concurrency limits
        results = []
        for i in range(0, len(requests), max_concurrency):
            batch = requests[i:i + max_concurrency]
            batch_results = self.provider.batch_translate(batch)
            results.extend(batch_results)
        
        # Log batch metrics if context provided
        if context and step_name:
            batch_time = time.time() - batch_start_time
            
            context.log_metrics({
                f"{step_name}_batch_translation_total_time": batch_time,
                f"{step_name}_batch_translation_avg_time": batch_time / len(texts),
                f"{step_name}_batch_translation_size": len(texts)
            })
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(texts),
                "batch_size": len(texts),
                "concurrent_limit": max_concurrency
            }
            
            context.update_state(
                f"{step_name}_batch_translation_summary",
                None,
                batch_info
            )
        
        # Extract translated texts from responses
        translated_texts = [r.translated_text for r in results]
        return translated_texts
    
    async def abatch_translate(self,
                             texts: List[str],
                             source_lang: str,
                             target_lang: str,
                             context: Optional[PipelineContext] = None,
                             step_name: Optional[str] = None,
                             max_concurrency: int = 5) -> List[str]:
        """
        Translate multiple texts asynchronously in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of translated texts
        """
        batch_start_time = time.time()
        
        # Prepare requests
        requests = [
            TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                metadata={"step_name": f"{step_name}_{i}"} if step_name else None
            )
            for i, text in enumerate(texts)
        ]
        
        # Process in batches to respect concurrency limits
        results = []
        for i in range(0, len(requests), max_concurrency):
            batch = requests[i:i + max_concurrency]
            batch_results = await self.provider.abatch_translate(batch)
            results.extend(batch_results)
        
        # Log batch metrics if context provided
        if context and step_name:
            batch_time = time.time() - batch_start_time
            
            context.log_metrics({
                f"{step_name}_batch_translation_total_time": batch_time,
                f"{step_name}_batch_translation_avg_time": batch_time / len(texts),
                f"{step_name}_batch_translation_size": len(texts)
            })
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(texts),
                "batch_size": len(texts),
                "concurrent_limit": max_concurrency
            }
            
            context.update_state(
                f"{step_name}_batch_translation_summary",
                None,
                batch_info
            )
        
        # Extract translated texts from responses
        translated_texts = [r.translated_text for r in results]
        return translated_texts
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return self.provider.get_supported_languages()


def create_translation_client(provider: Union[str, ProviderType] = ProviderType.GOOGLE,
                            api_key: Optional[str] = None,
                            custom_provider: Optional[TranslationProvider] = None,
                            retry_config: Optional[Dict] = None) -> TranslationClient:
    """
    Factory function to create a translation client with optimized configuration.
    
    Args:
        provider: Translation provider ("google", "deepl", or ProviderType enum)
        api_key: Optional API key (if not provided, will use environment variables)
        custom_provider: Optional custom provider implementation
        retry_config: Configuration for retry logic
        
    Returns:
        TranslationClient instance
    """
    return TranslationClient(
        provider=provider,
        api_key=api_key,
        custom_provider=custom_provider,
        retry_config=retry_config
    )
