"""
LLM client for interacting with various language model providers.

This module provides a unified interface for making requests to different LLM
providers with support for structured outputs, streaming, batching, and
comprehensive observability.
"""

import time
import logging
import asyncio
from typing import List, Dict, Optional, Type, Any, Union, AsyncIterator, TypeVar

from pydantic import BaseModel

from ..core.context import PipelineContext
from .schemas import LLMRequest, LLMResponse, ProviderType
from .providers import LLMProvider, OpenAIProvider, AnthropicProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic type safety
ResponseType = TypeVar('ResponseType')


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
    """

    def __init__(self, 
                provider: Union[str, ProviderType] = ProviderType.OPENAI, 
                model: Optional[str] = None,
                api_key: Optional[str] = None,
                custom_provider: Optional[LLMProvider] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "llama", or ProviderType enum)
            model: Default model to use (provider-specific)
            api_key: Optional API key (if not provided, will use environment variables)
            custom_provider: Optional custom provider implementation
        """
        if isinstance(provider, str):
            try:
                self.provider_type = ProviderType(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.provider_type = provider
        
        # Initialize the appropriate provider
        if self.provider_type == ProviderType.OPENAI:
            self.provider = OpenAIProvider(api_key=api_key)
        elif self.provider_type == ProviderType.ANTHROPIC:
            self.provider = AnthropicProvider(api_key=api_key)
        elif self.provider_type == ProviderType.LLAMA:
            raise NotImplementedError("Llama provider not yet implemented")
        elif self.provider_type == ProviderType.CUSTOM:
            if not custom_provider:
                raise ValueError("Custom provider type specified but no custom_provider provided")
            self.provider = custom_provider
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")
        
        # Initialize the provider
        self.provider.initialize()
        
        # Set default model if not specified
        self.model = model or self.provider.get_default_model()
        
        logger.info(f"Initialized LLMClient with provider={self.provider_type.value}, model={self.model}")
    
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
        
        try:
            # Call the provider
            response = self.provider.complete(request, response_model)
            
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
                    "success": True
                }
                
                # Log metrics
                context.log_metrics({
                    f"{step_name}_llm_latency": response.latency,
                    f"{step_name}_llm_prompt_length": len(prompt),
                    f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
                    f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
                    f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0)
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
            
            # Return just the content by default
            return response.content
            
        except Exception as e:
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "request": request_metadata
            }
            
            # Log error to context if provided
            if context and step_name:
                context.update_state(
                    f"{step_name}_llm_error",
                    None,
                    error_info
                )
            
            logger.error(f"LLM call failed: {str(e)}")
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
        
        try:
            # Call the provider
            response = await self.provider.acomplete(request, response_model)
            
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
                    "success": True
                }
                
                # Log metrics
                context.log_metrics({
                    f"{step_name}_llm_latency": response.latency,
                    f"{step_name}_llm_prompt_length": len(prompt),
                    f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
                    f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
                    f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0)
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
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "request": request_metadata
            }
            
            # Log error to context if provided
            if context and step_name:
                context.update_state(
                    f"{step_name}_llm_error",
                    None,
                    error_info
                )
            
            logger.error(f"Async LLM call failed: {str(e)}")
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
                    "success": True
                }
                
                # Log metrics
                context.log_metrics({
                    f"{step_name}_llm_stream_latency": latency,
                    f"{step_name}_llm_stream_length": len(full_text),
                })
                
                # Update context state
                context.update_state(
                    f"{step_name}_llm_stream_complete",
                    full_text,
                    completion_info
                )
            
            logger.info(f"LLM streaming completed: {self.provider_type.value}/{model}, "
                       f"latency: {time.time() - start_time:.2f}s")
            
        except Exception as e:
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "request": request_metadata,
                "partial_response": "".join(full_response) if full_response else None
            }
            
            # Log error to context if provided
            if context and step_name:
                context.update_state(
                    f"{step_name}_llm_stream_error",
                    None,
                    error_info
                )
            
            logger.error(f"LLM streaming failed: {str(e)}")
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
            except Exception as e:
                logger.error(f"Error processing batch item {i+1}: {str(e)}")
                results.append(None)
        
        # Log batch metrics if context provided
        if context and step_name:
            batch_time = time.time() - batch_start_time
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": sum(1 for r in results if r is not None) / len(prompts)
            })
        
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
                try:
                    return await self.acomplete(
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
                except Exception as e:
                    logger.error(f"Error processing batch item {i+1}: {str(e)}")
                    return None
        
        # Create tasks for all prompts
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Log batch metrics if context provided
        if context and step_name:
            batch_time = time.time() - batch_start_time
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": sum(1 for r in results if r is not None) / len(prompts)
            })
        
        return results
    
    def with_response(self) -> 'LLMClientWithResponse':
        """
        Return a client that returns full LLMResponse objects instead of just content.
        
        Returns:
            LLMClientWithResponse instance
        """
        return LLMClientWithResponse(
            provider=self.provider_type,
            model=self.model,
            custom_provider=self.provider
        )


class LLMClientWithResponse(LLMClient):
    """
    A variant of LLMClient that returns full LLMResponse objects.
    
    This is useful when you need access to metadata like token usage,
    latency, and finish reason in addition to the response content.
    """
    
    def complete(self, *args, **kwargs) -> LLMResponse:
        """
        Generate a completion and return the full response object.
        
        Args:
            Same as LLMClient.complete
            
        Returns:
            LLMResponse object with content and metadata
        """
        # Call the provider directly to get the full response
        request = LLMRequest(
            prompt=kwargs.get('prompt'),
            model=kwargs.get('model') or self.model,
            temperature=kwargs.get('temperature', 0.1),
            max_tokens=kwargs.get('max_tokens', 8000),
            system_prompt=kwargs.get('system_prompt'),
            stop_sequences=kwargs.get('stop_sequences'),
            top_p=kwargs.get('top_p'),
            frequency_penalty=kwargs.get('frequency_penalty'),
            presence_penalty=kwargs.get('presence_penalty'),
            seed=kwargs.get('seed'),
            metadata={"step_name": kwargs.get('step_name')} if kwargs.get('step_name') else None
        )
        
        response = self.provider.complete(request, kwargs.get('response_model'))
        
        # Log to context if provided
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        if context and step_name:
            # Add completion info to context
            completion_info = {
                "request": {
                    "provider": self.provider_type.value,
                    "model": response.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "prompt_length": len(request.prompt),
                    "timestamp": time.time() - response.latency,
                },
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "response": response.content,
                "usage": response.usage,
                "latency": response.latency,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "success": True
            }
            
            # Log metrics
            context.log_metrics({
                f"{step_name}_llm_latency": response.latency,
                f"{step_name}_llm_prompt_length": len(request.prompt),
                f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
                f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
                f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0)
            })
            
            # Update context state
            context.update_state(
                f"{step_name}_llm_call",
                response.content,
                completion_info
            )
        
        return response
    
    async def acomplete(self, *args, **kwargs) -> LLMResponse:
        """
        Generate a completion asynchronously and return the full response object.
        
        Args:
            Same as LLMClient.acomplete
            
        Returns:
            LLMResponse object with content and metadata
        """
        # Call the provider directly to get the full response
        request = LLMRequest(
            prompt=kwargs.get('prompt'),
            model=kwargs.get('model') or self.model,
            temperature=kwargs.get('temperature', 0.1),
            max_tokens=kwargs.get('max_tokens', 8000),
            system_prompt=kwargs.get('system_prompt'),
            stop_sequences=kwargs.get('stop_sequences'),
            top_p=kwargs.get('top_p'),
            metadata={"step_name": kwargs.get('step_name')} if kwargs.get('step_name') else None
        )
        
        response = await self.provider.acomplete(request, kwargs.get('response_model'))
        
        # Log to context if provided
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        if context and step_name:
            # Add completion info to context
            completion_info = {
                "request": {
                    "provider": self.provider_type.value,
                    "model": response.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "prompt_length": len(request.prompt),
                    "timestamp": time.time() - response.latency,
                },
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "response": response.content,
                "usage": response.usage,
                "latency": response.latency,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "success": True
            }
            
            # Log metrics
            context.log_metrics({
                f"{step_name}_llm_latency": response.latency,
                f"{step_name}_llm_prompt_length": len(request.prompt),
                f"{step_name}_llm_prompt_tokens": response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0),
                f"{step_name}_llm_completion_tokens": response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0),
                f"{step_name}_llm_total_tokens": response.usage.get("total_tokens", 0)
            })
            
            # Update context state
            context.update_state(
                f"{step_name}_llm_call",
                response.content,
                completion_info
            )
        
        return response
    
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
        start_time = time.time()
        
        # Prepare request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=kwargs.get('temperature', 0.1),
            max_tokens=kwargs.get('max_tokens', 8000),
            system_prompt=kwargs.get('system_prompt'),
            stop_sequences=kwargs.get('stop_sequences'),
            metadata={"step_name": kwargs.get('step_name')} if kwargs.get('step_name') else None
        )
        
        # Stream from the provider and collect chunks
        full_response = []
        try:
            async for chunk in await self.provider.stream(request):
                full_response.append(chunk)
                yield chunk
                
            # Note: We can't return the full response in an async generator,
            # but the client can access the collected chunks through a separate method
            # or by collecting them as they stream
            
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise
    
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
                kwargs['prompt'] = prompt
                kwargs['step_name'] = f"{kwargs.get('step_name')}_{i}" if kwargs.get('step_name') else None
                result = self.complete(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch item {i+1}: {str(e)}")
                results.append(None)
        
        # Log batch metrics if context provided
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        if context and step_name:
            batch_time = time.time() - batch_start_time
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": sum(1 for r in results if r is not None) / len(prompts)
            })
        
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
                try:
                    kwargs_copy = kwargs.copy()
                    kwargs_copy['prompt'] = prompt
                    kwargs_copy['step_name'] = f"{kwargs.get('step_name')}_{i}" if kwargs.get('step_name') else None
                    return await self.acomplete(**kwargs_copy)
                except Exception as e:
                    logger.error(f"Error processing batch item {i+1}: {str(e)}")
                    return None
        
        # Create tasks for all prompts
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Log batch metrics if context provided
        context = kwargs.get('context')
        step_name = kwargs.get('step_name')
        if context and step_name:
            batch_time = time.time() - batch_start_time
            context.log_metrics({
                f"{step_name}_batch_total_time": batch_time,
                f"{step_name}_batch_avg_time": batch_time / len(prompts),
                f"{step_name}_batch_size": len(prompts),
                f"{step_name}_batch_success_rate": sum(1 for r in results if r is not None) / len(prompts)
            })
        
        return results