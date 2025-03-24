"""
Anthropic provider implementation for the LLM client.

This module provides the AnthropicProvider class for interacting with Anthropic's API.
"""

import os
import time
import logging
import json
from typing import Dict, Optional, Type, Any, AsyncIterator

import anthropic
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from ..schemas import LLMRequest, LLMResponse
from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic provider."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")
        self.client = None
        self.async_client = None
    
    def initialize(self) -> None:
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
    
    def get_default_model(self) -> str:
        """Get the default model for Anthropic."""
        return "claude-3-sonnet-20240229"
    
    def _prepare_params(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        """Prepare parameters for Anthropic API call."""
        params = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}]
        }
        
        # Add system prompt if provided
        if request.system_prompt:
            params["system"] = request.system_prompt
        
        # Add optional parameters if provided
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop_sequences"] = request.stop_sequences
        
        # Add schema guidance for structured output
        if response_model:
            schema = response_model.model_json_schema()
            schema_prompt = f"You must respond with valid JSON that conforms to this schema: {json.dumps(schema)}"
            
            if "system" in params:
                params["system"] += f"\n\n{schema_prompt}"
            else:
                params["system"] = schema_prompt
        
        return params
    
    def complete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion using Anthropic API."""
        if not self.client:
            self.initialize()
        
        start_time = time.time()
        params = self._prepare_params(request, response_model)
        
        try:
            response = self.client.messages.create(**params)
            raw_content = response.content[0].text
            
            # Parse structured output if model provided
            if response_model:
                try:
                    content = response_model.model_validate_json(raw_content)
                except Exception as e:
                    logger.error(f"Failed to parse response as {response_model.__name__}: {str(e)}")
                    logger.debug(f"Raw response: {raw_content}")
                    raise
            else:
                content = raw_content
            
            # Extract usage statistics (Anthropic provides different metrics)
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency=time.time() - start_time,
                raw_response=response,
                finish_reason=response.stop_reason
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    async def acomplete(self, request: LLMRequest, response_model: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate a completion asynchronously using Anthropic API."""
        if not self.async_client:
            self.initialize()
        
        start_time = time.time()
        params = self._prepare_params(request, response_model)
        
        try:
            response = await self.async_client.messages.create(**params)
            raw_content = response.content[0].text
            
            # Parse structured output if model provided
            if response_model:
                try:
                    content = response_model.model_validate_json(raw_content)
                except Exception as e:
                    logger.error(f"Failed to parse response as {response_model.__name__}: {str(e)}")
                    logger.debug(f"Raw response: {raw_content}")
                    raise
            else:
                content = raw_content
            
            # Extract usage statistics
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency=time.time() - start_time,
                raw_response=response,
                finish_reason=response.stop_reason
            )
            
        except Exception as e:
            logger.error(f"Async Anthropic API call failed: {str(e)}")
            raise
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion from Anthropic API."""
        if not self.async_client:
            self.initialize()
        
        params = self._prepare_params(request)
        params["stream"] = True
        
        try:
            stream = await self.async_client.messages.create(**params)
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    yield chunk.delta.text
        except Exception as e:
            logger.error(f"Anthropic streaming API call failed: {str(e)}")
            raise
