"""
Translation pipeline steps.

This module provides pipeline steps for translation tasks, supporting various
translation approaches including API-based, LLM-based, and hybrid workflows.
"""

import logging
from typing import List, Dict, Any, Optional, TypeVar, Generic, Union, Callable
from abc import ABC, abstractmethod

from ..core.pipeline import PipelineStep
from ..core.context import PipelineContext
from .client import TranslationClient, ProviderType
from ..llm.client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic type safety
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

class TranslationStep(PipelineStep[InputType, OutputType], ABC):
    """
    Base class for translation pipeline steps.
    
    This abstract class defines the common interface for all translation steps.
    Subclasses should implement specific translation approaches (API, LLM, hybrid).
    """
    
    def __init__(self,
                 source_lang: str,
                 target_lang: str,
                 batch_size: int = 1,
                 step_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a translation step.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of items to process in a batch
            step_id: Optional unique identifier for this step
            metadata: Optional metadata for translation context
        """
        super().__init__(batch_size=batch_size, step_id=step_id or f"translate_{source_lang}_to_{target_lang}")
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.metadata = metadata or {}
        
        logger.info(f"Initialized {self.step_id} with source={source_lang}, target={target_lang}")
    
    def _extract_text(self, inputs: List[InputType]) -> List[str]:
        """Extract text from input items."""
        texts = []
        for item in inputs:
            if isinstance(item, str):
                texts.append(item)
            elif hasattr(item, "text"):
                texts.append(item.text)
            else:
                raise ValueError(f"Input item must be a string or have a text attribute: {item}")
        return texts
    
    def _reconstruct_outputs(self, inputs: List[InputType], translated_texts: List[str]) -> List[OutputType]:
        """Reconstruct output items with translated text."""
        outputs = []
        for input_item, translated_text in zip(inputs, translated_texts):
            if isinstance(input_item, str):
                outputs.append(translated_text)
            else:
                # Create a copy of the input item with translated text
                output = input_item.copy()
                output.text = translated_text
                outputs.append(output)
        return outputs


class APITranslationStep(TranslationStep[InputType, OutputType]):
    """
    Translation step that uses translation APIs (Google Translate, DeepL, etc.).
    """
    
    def __init__(self,
                 source_lang: str,
                 target_lang: str,
                 provider: Union[str, ProviderType] = ProviderType.GOOGLE,
                 api_key: Optional[str] = None,
                 batch_size: int = 1,
                 max_concurrency: int = 5,
                 step_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an API-based translation step.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider ("google", "deepl", or ProviderType enum)
            api_key: Optional API key (if not provided, will use environment variables)
            batch_size: Number of items to process in a batch
            max_concurrency: Maximum number of concurrent translation requests
            step_id: Optional unique identifier for this step
            metadata: Optional metadata for translation context
        """
        super().__init__(source_lang, target_lang, batch_size, step_id, metadata)
        self.max_concurrency = max_concurrency
        
        # Initialize translation client
        self.client = TranslationClient(
            provider=provider,
            api_key=api_key
        )
        
        logger.info(f"Initialized API translation step with provider={provider}")
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """Process inputs using translation API."""
        # Extract texts from inputs
        texts = self._extract_text(inputs)
        
        # Add metadata to context
        if self.metadata:
            for key, value in self.metadata.items():
                context.add_metadata(key, value)
        
        # Log step start
        logger.info(f"{self.step_id}: Processing {len(texts)} items")
        context.log_step(self.step_id, texts, None, {"status": "started"})
        
        try:
            # Translate texts
            translated_texts = self.client.batch_translate(
                texts=texts,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                context=context,
                step_name=self.step_id,
                max_concurrency=self.max_concurrency
            )
            
            # Log success
            logger.info(f"{self.step_id}: Successfully translated {len(texts)} items")
            context.log_step(self.step_id, texts, translated_texts, {"status": "completed"})
            
            # Update state
            context.update_state(
                self.step_id,
                translated_texts,
                {
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "provider": self.client.provider_type.value,
                    "item_count": len(texts)
                }
            )
            
            return self._reconstruct_outputs(inputs, translated_texts)
            
        except Exception as e:
            # Log error
            logger.error(f"{self.step_id}: Translation failed: {str(e)}")
            context.log_step(
                self.step_id,
                texts,
                None,
                {
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise


class LLMTranslationStep(TranslationStep[InputType, OutputType]):
    """
    Translation step that uses LLMs for translation.
    """
    
    def __init__(self,
                 source_lang: str,
                 target_lang: str,
                 llm_client: Optional[LLMClient] = None,
                 model: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 batch_size: int = 1,
                 step_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an LLM-based translation step.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            llm_client: Optional LLM client (if not provided, will create one)
            model: Optional model name (if not provided, will use default)
            system_prompt: Optional system prompt for translation
            batch_size: Number of items to process in a batch
            step_id: Optional unique identifier for this step
            metadata: Optional metadata for translation context
        """
        super().__init__(source_lang, target_lang, batch_size, step_id, metadata)
        
        # Initialize LLM client
        self.client = llm_client or LLMClient()
        self.model = model
        self.system_prompt = system_prompt or (
            f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. "
            "Maintain the original meaning, tone, and style. Do not add or remove content."
        )
        
        logger.info(f"Initialized LLM translation step with model={model or 'default'}")
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """Process inputs using LLM for translation."""
        # Extract texts from inputs
        texts = self._extract_text(inputs)
        
        # Add metadata to context
        if self.metadata:
            for key, value in self.metadata.items():
                context.add_metadata(key, value)
        
        # Log step start
        logger.info(f"{self.step_id}: Processing {len(texts)} items")
        context.log_step(self.step_id, texts, None, {"status": "started"})
        
        try:
            # Translate texts using LLM
            translated_texts = []
            for text in texts:
                prompt = f"Translate the following text to {self.target_lang}:\n\n{text}"
                response = self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    system_prompt=self.system_prompt,
                    context=context,
                    step_name=self.step_id,
                    metadata=self.metadata
                )
                translated_texts.append(response)
            
            # Log success
            logger.info(f"{self.step_id}: Successfully translated {len(texts)} items")
            context.log_step(self.step_id, texts, translated_texts, {"status": "completed"})
            
            # Update state
            context.update_state(
                self.step_id,
                translated_texts,
                {
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "model": self.model or "default",
                    "item_count": len(texts)
                }
            )
            
            return self._reconstruct_outputs(inputs, translated_texts)
            
        except Exception as e:
            # Log error
            logger.error(f"{self.step_id}: Translation failed: {str(e)}")
            context.log_step(
                self.step_id,
                texts,
                None,
                {
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise


class HybridTranslationStep(TranslationStep[InputType, OutputType]):
    """
    Translation step that combines multiple translation approaches.
    
    For example, could use API translation followed by LLM refinement,
    or parallel translation with multiple methods and voting.
    """
    
    def __init__(self,
                 source_lang: str,
                 target_lang: str,
                 steps: List[TranslationStep],
                 combiner: Callable[[List[List[str]]], List[str]],
                 batch_size: int = 1,
                 step_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a hybrid translation step.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            steps: List of translation steps to combine
            combiner: Function to combine results from multiple steps
            batch_size: Number of items to process in a batch
            step_id: Optional unique identifier for this step
            metadata: Optional metadata for translation context
        """
        super().__init__(source_lang, target_lang, batch_size, step_id, metadata)
        self.steps = steps
        self.combiner = combiner
        
        # Propagate metadata to sub-steps if they don't have their own
        for step in self.steps:
            if not step.metadata and self.metadata:
                step.metadata = self.metadata
        
        logger.info(f"Initialized hybrid translation step with {len(steps)} sub-steps")
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """Process inputs using multiple translation approaches."""
        # Extract texts from inputs
        texts = self._extract_text(inputs)
        
        # Add metadata to context
        if self.metadata:
            for key, value in self.metadata.items():
                context.add_metadata(key, value)
        
        # Log step start
        logger.info(f"{self.step_id}: Processing {len(texts)} items with {len(self.steps)} sub-steps")
        context.log_step(self.step_id, texts, None, {"status": "started"})
        
        try:
            # Get translations from each step
            all_translations = []
            for step in self.steps:
                step_translations = step.process(inputs, context)
                all_translations.append([t if isinstance(t, str) else t.text for t in step_translations])
            
            # Combine translations
            combined_translations = self.combiner(all_translations)
            
            # Log success
            logger.info(f"{self.step_id}: Successfully combined translations from {len(self.steps)} sub-steps")
            context.log_step(self.step_id, texts, combined_translations, {"status": "completed"})
            
            # Update state
            context.update_state(
                self.step_id,
                combined_translations,
                {
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                    "sub_steps": [step.step_id for step in self.steps],
                    "item_count": len(texts)
                }
            )
            
            return self._reconstruct_outputs(inputs, combined_translations)
            
        except Exception as e:
            # Log error
            logger.error(f"{self.step_id}: Translation failed: {str(e)}")
            context.log_step(
                self.step_id,
                texts,
                None,
                {
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise 