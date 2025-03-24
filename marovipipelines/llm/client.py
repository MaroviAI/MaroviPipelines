"""
LLM client for interacting with various language model providers.

This module provides a unified interface for making requests to different LLM
providers with support for structured outputs, streaming, batching, and
comprehensive observability.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional, Type, Any, Union, AsyncIterator, TypeVar, Callable, Tuple
from functools import lru_cache

from pydantic import BaseModel

from ..core.context import PipelineContext
from .schemas import LLMRequest, LLMResponse, ProviderType
from .providers import LLMProvider, OpenAIProvider, AnthropicProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic type safety
ResponseType = TypeVar('ResponseType')

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

class LLMClient:
    """
    A unified client for interacting with various LLM providers.
    
    Features:
    - Support for multiple providers (OpenAI, Anthropic, etc.)
    - Structured JSON responses using Pydantic models
    - Streaming support
    - Async and sync interfaces
    - Automatic logging to pipeline context
    - Comprehensive observability
    - Retry logic for transient failures
    - Optional response caching
    """

    def __init__(self, 
                provider: Union[str, ProviderType] = ProviderType.OPENAI, 
                model: Optional[str] = None,
                api_key: Optional[str] = None,
                custom_provider: Optional[LLMProvider] = None,
                retry_config: Optional[Dict[str, Any]] = None,
                enable_cache: bool = False,
                cache_size: int = 100):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "llama", or ProviderType enum)
            model: Default model to use (provider-specific)
            api_key: Optional API key (if not provided, will use environment variables)
            custom_provider: Optional custom provider implementation
            retry_config: Configuration for retry logic (None to use defaults)
            enable_cache: Whether to enable response caching
            cache_size: Maximum number of responses to cache
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
        elif self.provider_type == ProviderType.OPENAI:
            self.provider = OpenAIProvider(api_key=api_key)
        elif self.provider_type == ProviderType.ANTHROPIC:
            self.provider = AnthropicProvider(api_key=api_key)
        elif self.provider_type == ProviderType.LLAMA:
            raise NotImplementedError("Llama provider not yet implemented")
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")
        
        # Initialize the provider
        self.provider.initialize()
        
        # Set default model if not specified
        self.model = model or self.provider.get_default_model()
        
        # Set retry configuration
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        
        # Set up caching if enabled
        self.enable_cache = enable_cache
        if enable_cache:
            self._setup_cache(cache_size)
        
        logger.info(f"Initialized LLMClient with provider={self.provider_type.value}, model={self.model}")
    
    def _setup_cache(self, cache_size: int) -> None:
        """Set up LRU cache for responses."""
        @lru_cache(maxsize=cache_size)
        def cached_complete(prompt_hash: str, model: str, temp: float, 
                           max_tokens: int, system_prompt_hash: Optional[str] = None):
            # This is just a placeholder - the actual implementation will use the hash
            # to look up cached responses
            return None
        
        self._cached_complete = cached_complete
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float, 
                      max_tokens: int, system_prompt: Optional[str] = None) -> Tuple[str, str, float, int, Optional[str]]:
        """Generate a cache key for the given request parameters."""
        # Use hash of prompt and system_prompt to reduce memory usage
        prompt_hash = str(hash(prompt))
        system_prompt_hash = str(hash(system_prompt)) if system_prompt else None
        return (prompt_hash, model, temperature, max_tokens, system_prompt_hash)
    
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
    
    def complete(self, 
                prompt: str, 
                model: Optional[str] = None, 
                temperature: float = 0.1, 
                max_tokens: int = 8000, 
                response_model: Optional[Type[BaseModel]] = None,
                system_prompt: Optional[str] = None,
                stop_sequences: Optional[List[str]] = None,
                top_p: Optional[float] = None,
                frequency_penalty: Optional[float] = None,
                presence_penalty: Optional[float] = None,
                seed: Optional[int] = None,
                context: Optional[PipelineContext] = None,
                step_name: Optional[str] = None) -> Union[str, BaseModel]:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: The input prompt
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_model: Optional Pydantic model for structured output
            system_prompt: Optional system prompt
            stop_sequences: Optional list of stop sequences
            top_p: Optional top-p sampling parameter
            frequency_penalty: Optional frequency penalty
            presence_penalty: Optional presence penalty
            seed: Optional random seed for reproducibility
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            
        Returns:
            Response from the LLM (parsed as Pydantic model if specified)
        """
        model = model or self.model
        start_time = time.time()
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(prompt, model, temperature, max_tokens, system_prompt)
            cached_response = self._cached_complete(*cache_key)
            if cached_response:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "timestamp": start_time,
        }
        
        if system_prompt:
            request_metadata["system_prompt_length"] = len(system_prompt)
        
        if response_model:
            request_metadata["response_model"] = response_model.__name__
        
        # Implement retry logic
        attempt = 0
        while True:
            try:
                # Call the provider
                response = self.provider.complete(request, response_model)
                
                # Update cache if enabled
                if self.enable_cache:
                    self._cached_complete.cache_clear()  # Clear old entries if needed
                    self._cached_complete(*cache_key, _return=response.content)
                
                # Log to context if provided
                if context and step_name:
                    # Add completion info to context
                    completion_info = {
                        "request": request_metadata,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": response.content,
                        "usage": response.usage,
                        "latency": response.latency,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "success": True,
                        "attempts": attempt + 1
                    }
                    
                    # Log metrics
                    context.log_metrics({
                        f"{step_name}_llm_latency": response.latency,
                        f"{step_name}_llm_prompt_length": len(prompt),
                        f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
                        f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
                        f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0),
                        f"{step_name}_llm_attempts": attempt + 1
                    })
                    
                    # Update context state
                    context.update_state(
                        f"{step_name}_llm_call",
                        response.content,
                        completion_info
                    )
                
                logger.info(f"LLM completion successful: {self.provider_type.value}/{model}, "
                           f"tokens: {response.usage.get('total_tokens', 0)}, "
                           f"latency: {response.latency:.2f}s")
                
                return response.content
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying LLM call (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
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
                        f"{step_name}_llm_error",
                        None,
                        error_info
                    )
                    
                    context.log_metrics({
                        f"{step_name}_llm_error_count": 1,
                        f"{step_name}_llm_attempts": attempt
                    })
                
                logger.error(f"LLM call failed after {attempt} attempts: {str(e)}")
                raise
    
    async def acomplete(self, 
                       prompt: str, 
                       model: Optional[str] = None, 
                       temperature: float = 0.1, 
                       max_tokens: int = 8000, 
                       response_model: Optional[Type[BaseModel]] = None,
                       system_prompt: Optional[str] = None,
                       stop_sequences: Optional[List[str]] = None,
                       top_p: Optional[float] = None,
                       context: Optional[PipelineContext] = None,
                       step_name: Optional[str] = None) -> Union[str, BaseModel]:
        """
        Generate a completion from the LLM asynchronously.
        
        Args:
            prompt: The input prompt
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_model: Optional Pydantic model for structured output
            system_prompt: Optional system prompt
            stop_sequences: Optional list of stop sequences
            top_p: Optional top-p sampling parameter
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            
        Returns:
            Response from the LLM (parsed as Pydantic model if specified)
        """
        model = model or self.model
        start_time = time.time()
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(prompt, model, temperature, max_tokens, system_prompt)
            cached_response = self._cached_complete(*cache_key)
            if cached_response:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
            top_p=top_p,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "timestamp": start_time,
        }
        
        if system_prompt:
            request_metadata["system_prompt_length"] = len(system_prompt)
        
        if response_model:
            request_metadata["response_model"] = response_model.__name__
        
        # Implement retry logic
        attempt = 0
        while True:
            try:
                # Call the provider
                response = await self.provider.acomplete(request, response_model)
                
                # Update cache if enabled
                if self.enable_cache:
                    self._cached_complete.cache_clear()  # Clear old entries if needed
                    self._cached_complete(*cache_key, _return=response.content)
                
                # Log to context if provided
                if context and step_name:
                    # Add completion info to context
                    completion_info = {
                        "request": request_metadata,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": response.content,
                        "usage": response.usage,
                        "latency": response.latency,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "success": True,
                        "attempts": attempt + 1
                    }
                    
                    # Log metrics
                    context.log_metrics({
                        f"{step_name}_llm_latency": response.latency,
                        f"{step_name}_llm_prompt_length": len(prompt),
                        f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
                        f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
                        f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0),
                        f"{step_name}_llm_attempts": attempt + 1
                    })
                    
                    # Update context state
                    context.update_state(
                        f"{step_name}_llm_call",
                        response.content,
                        completion_info
                    )
                
                logger.info(f"Async LLM completion successful: {self.provider_type.value}/{model}, "
                           f"tokens: {response.usage.get('total_tokens', 0)}, "
                           f"latency: {response.latency:.2f}s")
                
                return response.content
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying async LLM call (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
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
                        f"{step_name}_llm_error",
                        None,
                        error_info
                    )
                    
                    context.log_metrics({
                        f"{step_name}_llm_error_count": 1,
                        f"{step_name}_llm_attempts": attempt
                    })
                
                logger.error(f"Async LLM call failed after {attempt} attempts: {str(e)}")
                raise
    
    async def stream(self, 
                    prompt: str, 
                    model: Optional[str] = None, 
                    temperature: float = 0.1, 
                    max_tokens: int = 8000,
                    system_prompt: Optional[str] = None,
                    stop_sequences: Optional[List[str]] = None,
                    context: Optional[PipelineContext] = None,
                    step_name: Optional[str] = None) -> AsyncIterator[str]:
        """
        Stream a completion from the LLM.
        
        Args:
            prompt: The input prompt
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt
            stop_sequences: Optional list of stop sequences
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            
        Yields:
            Chunks of the response as they become available
        """
        model = model or self.model
        start_time = time.time()
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "timestamp": start_time,
            "streaming": True
        }
        
        if system_prompt:
            request_metadata["system_prompt_length"] = len(system_prompt)
        
        # Log the start of streaming if context provided
        if context and step_name:
            context.update_state(
                f"{step_name}_llm_stream_start",
                None,
                {
                    "request": request_metadata,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "start_time": start_time
                }
            )
        
        full_response = []
        attempt = 0
        
        while True:
            try:
                # Stream from the provider
                async for chunk in self.provider.stream(request):
                    full_response.append(chunk)
                    yield chunk
                
                # Log completion of streaming if context provided
                if context and step_name:
                    end_time = time.time()
                    latency = end_time - start_time
                    full_text = "".join(full_response)
                    
                    # Add completion info to context
                    completion_info = {
                        "request": request_metadata,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": full_text,
                        "latency": latency,
                        "success": True,
                        "attempts": attempt + 1
                    }
                    
                    # Log metrics
                    context.log_metrics({
                        f"{step_name}_llm_stream_latency": latency,
                        f"{step_name}_llm_stream_length": len(full_text),
                        f"{step_name}_llm_attempts": attempt + 1
                    })
                    
                    # Update context state
                    context.update_state(
                        f"{step_name}_llm_stream_complete",
                        full_text,
                        completion_info
                    )
                
                logger.info(f"LLM streaming completed: {self.provider_type.value}/{model}, "
                           f"latency: {time.time() - start_time:.2f}s")
                
                break  # Successful completion
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying streaming LLM call (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    await self._backoff(attempt - 1)
                    # Reset full_response for the retry
                    full_response = []
                    continue
                
                # If we shouldn't retry, log the error and raise
                error_info = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request": request_metadata,
                    "partial_response": "".join(full_response) if full_response else None,
                    "attempts": attempt
                }
                
                # Log error to context if provided
                if context and step_name:
                    context.update_state(
                        f"{step_name}_llm_stream_error",
                        None,
                        error_info
                    )
                    
                    context.log_metrics({
                        f"{step_name}_llm_stream_error_count": 1,
                        f"{step_name}_llm_attempts": attempt
                    })
                
                logger.error(f"LLM streaming failed after {attempt} attempts: {str(e)}")
                raise
    
    def batch_complete(self,
                      prompts: List[str],
                      model: Optional[str] = None,
                      temperature: float = 0.1,
                      max_tokens: int = 8000,
                      response_model: Optional[Type[BaseModel]] = None,
                      system_prompt: Optional[str] = None,
                      stop_sequences: Optional[List[str]] = None,
                      top_p: Optional[float] = None,
                      context: Optional[PipelineContext] = None,
                      step_name: Optional[str] = None,
                      max_concurrency: int = 5) -> List[Any]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_model: Optional Pydantic model for structured output
            system_prompt: Optional system prompt
            stop_sequences: Optional list of stop sequences
            top_p: Optional top-p sampling parameter
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of responses from the LLM
        """
        results = []
        batch_start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing batch item {i+1}/{len(prompts)}")
                
                # Track attempts for each item
                attempt = 0
                while True:
                    try:
                        result = self.complete(
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_model=response_model,
                            system_prompt=system_prompt,
                            stop_sequences=stop_sequences,
                            top_p=top_p,
                            context=context,
                            step_name=f"{step_name}_{i}" if step_name else None
                        )
                        results.append(result)
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        attempt += 1
                        
                        # Check if we should retry
                        if self._should_retry(e, attempt):
                            logger.warning(f"Retrying batch item {i+1} (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                            self._sync_backoff(attempt - 1)
                            continue
                        
                        # If we shouldn't retry, log the error and append None
                        logger.error(f"Error processing batch item {i+1} after {attempt} attempts: {str(e)}")
                        results.append(None)
                        break
                        
            except Exception as e:
                logger.error(f"Unhandled error processing batch item {i+1}: {str(e)}")
                results.append(None)
        
        # Log batch metrics if context provided
        if context and step_name:
            batch_time = time.time() - batch_start_time
            success_rate = sum(1 for r in results if r is not None) / len(prompts)
            
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": success_rate,
                f"{step_name}_batch_failure_count": len(prompts) - sum(1 for r in results if r is not None)
            })
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(prompts),
                "batch_size": len(prompts),
                "success_rate": success_rate,
                "success_count": sum(1 for r in results if r is not None),
                "failure_count": len(prompts) - sum(1 for r in results if r is not None)
            }
            
            context.update_state(
                f"{step_name}_batch_summary",
                None,
                batch_info
            )
        
        return results
    
    async def abatch_complete(self,
                             prompts: List[str],
                             model: Optional[str] = None,
                             temperature: float = 0.1,
                             max_tokens: int = 8000,
                             response_model: Optional[Type[BaseModel]] = None,
                             system_prompt: Optional[str] = None,
                             stop_sequences: Optional[List[str]] = None,
                             top_p: Optional[float] = None,
                             context: Optional[PipelineContext] = None,
                             step_name: Optional[str] = None,
                             max_concurrency: int = 5) -> List[Any]:
        """
        Generate completions for a batch of prompts asynchronously.
        
        Args:
            prompts: List of input prompts
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_model: Optional Pydantic model for structured output
            system_prompt: Optional system prompt
            stop_sequences: Optional list of stop sequences
            top_p: Optional top-p sampling parameter
            context: Optional pipeline context for logging
            step_name: Name of the step (for context logging)
            max_concurrency: Maximum number of concurrent requests
            
        Returns:
            List of responses from the LLM
        """
        batch_start_time = time.time()
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_prompt(i, prompt):
            async with semaphore:
                # Track attempts for each item
                attempt = 0
                while True:
                    try:
                        result = await self.acomplete(
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_model=response_model,
                            system_prompt=system_prompt,
                            stop_sequences=stop_sequences,
                            top_p=top_p,
                            context=context,
                            step_name=f"{step_name}_{i}" if step_name else None
                        )
                        return result  # Success
                        
                    except Exception as e:
                        attempt += 1
                        
                        # Check if we should retry
                        if self._should_retry(e, attempt):
                            logger.warning(f"Retrying async batch item {i+1} (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                            await self._backoff(attempt - 1)
                            continue
                        
                        # If we shouldn't retry, log the error and return None
                        logger.error(f"Error processing async batch item {i+1} after {attempt} attempts: {str(e)}")
                        return None
        
        # Create tasks for all prompts
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Log batch metrics if context provided
        if context and step_name:
            batch_time = time.time() - batch_start_time
            success_rate = sum(1 for r in results if r is not None) / len(prompts)
            
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": success_rate,
                f"{step_name}_batch_failure_count": len(prompts) - sum(1 for r in results if r is not None)
            })
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(prompts),
                "batch_size": len(prompts),
                "success_rate": success_rate,
                "success_count": sum(1 for r in results if r is not None),
                "failure_count": len(prompts) - sum(1 for r in results if r is not None),
                "concurrent_limit": max_concurrency
            }
            
            context.update_state(
                f"{step_name}_batch_summary",
                None,
                batch_info
            )
        
        return results

class LLMClientWithResponse(LLMClient):
    """
    A variant of LLMClient that returns full LLMResponse objects instead of just the content.
    This allows access to metadata like token usage, latency, and raw responses.
    """
    
    def _log_completion_to_context(self, context, step_name, request_metadata, prompt, 
                                  system_prompt, response, attempt):
        """Helper method to log completion information to context."""
        # Add completion info to context
        completion_info = {
            "request": request_metadata,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response.content,
            "usage": response.usage,
            "latency": response.latency,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "success": True,
            "attempts": attempt + 1
        }
        
        # Log metrics
        context.log_metrics({
            f"{step_name}_llm_latency": response.latency,
            f"{step_name}_llm_prompt_length": len(prompt),
            f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
            f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
            f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0),
            f"{step_name}_llm_attempts": attempt + 1
        })
        
        # Update context state
        context.update_state(
            f"{step_name}_llm_call",
            response.content,
            completion_info
        )
    
    def _log_error_to_context(self, context, step_name, request_metadata, error, attempt):
        """Helper method to log error information to context."""
        if not context or not step_name:
            return
            
        error_info = {
            "error": str(error),
            "error_type": type(error).__name__,
            "request": request_metadata,
            "attempts": attempt
        }
        
        context.update_state(
            f"{step_name}_llm_error",
            None,
            error_info
        )
        
        context.log_metrics({
            f"{step_name}_llm_error_count": 1,
            f"{step_name}_llm_attempts": attempt
        })
    
    def complete(self, *args, **kwargs) -> LLMResponse:
        """
        Generate a completion and return the full response object.
        
        Args:
            Same as LLMClient.complete
            
        Returns:
            LLMResponse object containing content and metadata
        """
        prompt = kwargs.get('prompt')
        if not prompt and args:
            prompt = args[0]
            # Remove prompt from args to avoid duplicate args
            args = args[1:]
        
        model = kwargs.get('model') or self.model
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8000)
        response_model = kwargs.get('response_model')
        system_prompt = kwargs.get('system_prompt')
        stop_sequences = kwargs.get('stop_sequences')
        top_p = kwargs.get('top_p')
        frequency_penalty = kwargs.get('frequency_penalty')
        presence_penalty = kwargs.get('presence_penalty')
        seed = kwargs.get('seed')
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        
        start_time = time.time()
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(prompt, model, temperature, max_tokens, system_prompt)
            cached_response = self._cached_complete(*cache_key)
            if cached_response:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "timestamp": start_time,
        }
        
        if system_prompt:
            request_metadata["system_prompt_length"] = len(system_prompt)
        
        if response_model:
            request_metadata["response_model"] = response_model.__name__
        
        # Implement retry logic
        attempt = 0
        while True:
            try:
                # Call the provider
                response = self.provider.complete(request, response_model)
                
                # Update cache if enabled
                if self.enable_cache:
                    self._cached_complete.cache_clear()  # Clear old entries if needed
                    self._cached_complete(*cache_key, _return=response)
                
                # Log to context if provided
                if context and step_name:
                    self._log_completion_to_context(context, step_name, request_metadata, prompt, 
                                                 system_prompt, response, attempt)
                
                logger.info(f"LLM completion successful: {self.provider_type.value}/{model}, "
                           f"tokens: {response.usage.get('total_tokens', 0)}, "
                           f"latency: {response.latency:.2f}s")
                
                return response
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying LLM call (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    self._sync_backoff(attempt - 1)
                    continue
                
                # If we shouldn't retry, log the error and raise
                self._log_error_to_context(context, step_name, request_metadata, e, attempt)
                logger.error(f"LLM call failed after {attempt} attempts: {str(e)}")
                raise
    
    async def acomplete(self, *args, **kwargs) -> LLMResponse:
        """
        Generate a completion asynchronously and return the full response object.
        
        Args:
            Same as LLMClient.acomplete
            
        Returns:
            LLMResponse object containing content and metadata
        """
        prompt = kwargs.get('prompt')
        if not prompt and args:
            prompt = args[0]
            # Remove prompt from args to avoid duplicate args
            args = args[1:]
        
        model = kwargs.get('model') or self.model
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8000)
        response_model = kwargs.get('response_model')
        system_prompt = kwargs.get('system_prompt')
        stop_sequences = kwargs.get('stop_sequences')
        top_p = kwargs.get('top_p')
        frequency_penalty = kwargs.get('frequency_penalty')
        presence_penalty = kwargs.get('presence_penalty')
        seed = kwargs.get('seed')
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        
        start_time = time.time()
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(prompt, model, temperature, max_tokens, system_prompt)
            cached_response = self._cached_complete(*cache_key)
            if cached_response:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "timestamp": start_time,
        }
        
        if system_prompt:
            request_metadata["system_prompt_length"] = len(system_prompt)
        
        if response_model:
            request_metadata["response_model"] = response_model.__name__
        
        # Implement retry logic
        attempt = 0
        while True:
            try:
                # Call the provider
                response = await self.provider.acomplete(request, response_model)
                
                # Update cache if enabled
                if self.enable_cache:
                    self._cached_complete.cache_clear()  # Clear old entries if needed
                    self._cached_complete(*cache_key, _return=response)
                
                # Log to context if provided
                if context and step_name:
                    self._log_completion_to_context(context, step_name, request_metadata, prompt, 
                                                  system_prompt, response, attempt)
                
                logger.info(f"Async LLM completion successful: {self.provider_type.value}/{model}, "
                           f"tokens: {response.usage.get('total_tokens', 0)}, "
                           f"latency: {response.latency:.2f}s")
                
                return response
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying async LLM call (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    await self._backoff(attempt - 1)
                    continue
                
                # If we shouldn't retry, log the error and raise
                self._log_error_to_context(context, step_name, request_metadata, e, attempt)
                logger.error(f"Async LLM call failed after {attempt} attempts: {str(e)}")
                raise
    
    class StreamResult:
        """Helper class to store streaming results and final response."""
        def __init__(self):
            self.chunks = []
            self.full_response = None
            
        def add_chunk(self, chunk):
            self.chunks.append(chunk)
            
        def finalize(self, response):
            self.full_response = response
            
        @property
        def content(self):
            if self.full_response:
                return self.full_response.content
            return "".join(self.chunks)
    
    async def stream(self, prompt: str, *args, **kwargs) -> AsyncIterator[str]:
        """
        Stream a completion and collect chunks for the full response.
        
        Args:
            prompt: The input prompt
            *args, **kwargs: Same as LLMClient.stream
            
        Yields:
            Chunks of the response as they become available
        """
        model = kwargs.get('model') or self.model
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8000)
        system_prompt = kwargs.get('system_prompt')
        stop_sequences = kwargs.get('stop_sequences')
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        
        start_time = time.time()
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            stop_sequences=stop_sequences,
            metadata={"step_name": step_name} if step_name else None
        )
        
        # Prepare request metadata for logging
        request_metadata = {
            "provider": self.provider_type.value,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "timestamp": start_time,
            "streaming": True
        }
        
        if system_prompt:
            request_metadata["system_prompt_length"] = len(system_prompt)
        
        # Log the start of streaming if context provided
        if context and step_name:
            context.update_state(
                f"{step_name}_llm_stream_start",
                None,
                {
                    "request": request_metadata,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "start_time": start_time
                }
            )
        
        # Storage for collected chunks
        result = self.StreamResult()
        attempt = 0
        
        while True:
            try:
                # Stream from the provider
                async for chunk in await self.provider.stream(request):
                    result.add_chunk(chunk)
                    yield chunk
                
                # Log completion of streaming if context provided
                if context and step_name:
                    end_time = time.time()
                    latency = end_time - start_time
                    full_text = "".join(result.chunks)
                    
                    # Add completion info to context
                    completion_info = {
                        "request": request_metadata,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "response": full_text,
                        "latency": latency,
                        "success": True,
                        "attempts": attempt + 1
                    }
                    
                    # Log metrics
                    context.log_metrics({
                        f"{step_name}_llm_stream_latency": latency,
                        f"{step_name}_llm_stream_length": len(full_text),
                        f"{step_name}_llm_attempts": attempt + 1
                    })
                    
                    # Update context state
                    context.update_state(
                        f"{step_name}_llm_stream_complete",
                        full_text,
                        completion_info
                    )
                
                logger.info(f"LLM streaming completed: {self.provider_type.value}/{model}, "
                           f"latency: {time.time() - start_time:.2f}s")
                
                break  # Successful completion
                
            except Exception as e:
                attempt += 1
                
                # Check if we should retry
                if self._should_retry(e, attempt):
                    logger.warning(f"Retrying streaming LLM call (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                    await self._backoff(attempt - 1)
                    # Reset result for the retry
                    result = self.StreamResult()
                    continue
                
                # If we shouldn't retry, log the error and raise
                error_info = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request": request_metadata,
                    "partial_response": "".join(result.chunks) if result.chunks else None,
                    "attempts": attempt
                }
                
                # Log error to context if provided
                if context and step_name:
                    context.update_state(
                        f"{step_name}_llm_stream_error",
                        None,
                        error_info
                    )
                    
                    context.log_metrics({
                        f"{step_name}_llm_stream_error_count": 1,
                        f"{step_name}_llm_attempts": attempt
                    })
                
                logger.error(f"LLM streaming failed after {attempt} attempts: {str(e)}")
                raise
    
    def get_stream_result(self) -> StreamResult:
        """
        Get the collected stream result after streaming is complete.
        This allows access to the full text and metadata.
        
        Returns:
            StreamResult object with chunks and metadata
        """
        if not hasattr(self, '_current_stream_result'):
            raise ValueError("No active stream result. Call stream() first.")
        return self._current_stream_result
    
    def batch_complete(self, *args, **kwargs) -> List[LLMResponse]:
        """
        Generate completions for a batch of prompts and return full response objects.
        
        Args:
            Same as LLMClient.batch_complete
            
        Returns:
            List of LLMResponse objects
        """
        prompts = kwargs.get('prompts', [])
        if not prompts and args:
            prompts = args[0]
            
        results = []
        batch_start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing batch item {i+1}/{len(prompts)}")
                
                # Track attempts for each item
                attempt = 0
                while True:
                    try:
                        kwargs_copy = kwargs.copy()
                        kwargs_copy['prompt'] = prompt
                        kwargs_copy['step_name'] = f"{kwargs.get('step_name')}_{i}" if kwargs.get('step_name') else None
                        
                        result = self.complete(**kwargs_copy)
                        results.append(result)
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        attempt += 1
                        
                        # Check if we should retry
                        if self._should_retry(e, attempt):
                            logger.warning(f"Retrying batch item {i+1} (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                            self._sync_backoff(attempt - 1)
                            continue
                        
                        # If we shouldn't retry, log the error and append None
                        logger.error(f"Error processing batch item {i+1} after {attempt} attempts: {str(e)}")
                        results.append(None)
                        break
                        
            except Exception as e:
                logger.error(f"Unhandled error processing batch item {i+1}: {str(e)}")
                results.append(None)
        
        # Log batch metrics if context provided
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        if context and step_name:
            batch_time = time.time() - batch_start_time
            success_rate = sum(1 for r in results if r is not None) / len(prompts)
            
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": success_rate,
                f"{step_name}_batch_failure_count": len(prompts) - sum(1 for r in results if r is not None)
            })
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(prompts),
                "batch_size": len(prompts),
                "success_rate": success_rate,
                "success_count": sum(1 for r in results if r is not None),
                "failure_count": len(prompts) - sum(1 for r in results if r is not None)
            }
            
            context.update_state(
                f"{step_name}_batch_summary",
                None,
                batch_info
            )
        
        return results
    
    async def abatch_complete(self, *args, **kwargs) -> List[LLMResponse]:
        """
        Generate completions for a batch of prompts asynchronously and return full response objects.
        
        Args:
            Same as LLMClient.abatch_complete
            
        Returns:
            List of LLMResponse objects
        """
        prompts = kwargs.get('prompts', [])
        if not prompts and args:
            prompts = args[0]
            
        batch_start_time = time.time()
        
        # Create a semaphore to limit concurrency
        max_concurrency = kwargs.get('max_concurrency', 5)
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_prompt(i, prompt):
            async with semaphore:
                # Track attempts for each item
                attempt = 0
                while True:
                    try:
                        kwargs_copy = kwargs.copy()
                        kwargs_copy['prompt'] = prompt
                        kwargs_copy['step_name'] = f"{kwargs.get('step_name')}_{i}" if kwargs.get('step_name') else None
                        
                        result = await self.acomplete(**kwargs_copy)
                        return result  # Success
                        
                    except Exception as e:
                        attempt += 1
                        
                        # Check if we should retry
                        if self._should_retry(e, attempt):
                            logger.warning(f"Retrying async batch item {i+1} (attempt {attempt}/{self.retry_config['max_retries']}): {str(e)}")
                            await self._backoff(attempt - 1)
                            continue
                        
                        # If we shouldn't retry, log the error and return None
                        logger.error(f"Error processing async batch item {i+1} after {attempt} attempts: {str(e)}")
                        return None
        
        # Create tasks for all prompts
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Log batch metrics if context provided
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        if context and step_name:
            batch_time = time.time() - batch_start_time
            success_rate = sum(1 for r in results if r is not None) / len(prompts)
            
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": success_rate,
                f"{step_name}_batch_failure_count": len(prompts) - sum(1 for r in results if r is not None)
            })
            
            # Log detailed batch info
            batch_info = {
                "total_time": batch_time,
                "avg_time": batch_time / len(prompts),
                "batch_size": len(prompts),
                "success_rate": success_rate,
                "success_count": sum(1 for r in results if r is not None),
                "failure_count": len(prompts) - sum(1 for r in results if r is not None),
                "concurrent_limit": max_concurrency
            }
            
            context.update_state(
                f"{step_name}_batch_summary",
                None,
                batch_info
            )
        
        return results


def create_llm_client(provider: Union[str, ProviderType] = ProviderType.OPENAI,
                     model: Optional[str] = None,
                     api_key: Optional[str] = None,
                     return_full_response: bool = False,
                     custom_provider: Optional[LLMProvider] = None,
                     retry_config: Optional[Dict[str, Any]] = None,
                     enable_cache: bool = False,
                     cache_size: int = 100) -> Union[LLMClient, LLMClientWithResponse]:
    """
    Factory function to create an LLM client with optimized configuration.
    
    Args:
        provider: LLM provider ("openai", "anthropic", "llama", or ProviderType enum)
        model: Default model to use (provider-specific)
        api_key: Optional API key (if not provided, will use environment variables)
        return_full_response: Whether to return full LLMResponse objects
        custom_provider: Optional custom provider implementation
        retry_config: Configuration for retry logic
        enable_cache: Whether to enable response caching
        cache_size: Maximum number of responses to cache
        
    Returns:
        LLMClient or LLMClientWithResponse instance
    """
    if return_full_response:
        return LLMClientWithResponse(
            provider=provider,
            model=model,
            api_key=api_key,
            custom_provider=custom_provider,
            retry_config=retry_config,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
    else:
        return LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            custom_provider=custom_provider,
            retry_config=retry_config,
            enable_cache=enable_cache,
            cache_size=cache_size
        )