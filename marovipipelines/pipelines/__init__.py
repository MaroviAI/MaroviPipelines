"""
Marovi Pipelines Framework

A production-grade, modular AI/ML pipeline framework designed for document processing,
translation, correction, summarization, glossary-tagging, and human-in-the-loop workflows.

Features:
- Batch-aware typing
- Pre/Post hooks for inline tasks like glossary tagging
- Intermediate checkpointing and resumability
- Full observability (prompts, latency, LLM token usage)
- Human review and conditional routing
- Future integration with Spark/Ray/AWS
"""

# Core pipeline components
from .base import (
    PipelineContext,
    PipelineStep,
    ProcessingPipeline,
    ConditionalStep,
    HumanReviewStep,
    InputType,
    OutputType
)

# Example step implementations
from .steps import (
    UrlDownloadStep,
    PdfExtractionStep,
    GlossaryTaggingStep,
    LlmSummarizationStep,
    JsonOutputStep
)

# Example pipeline
from .example import create_sample_pipeline, run_example

__all__ = [
    # Core components
    'PipelineContext',
    'PipelineStep',
    'ProcessingPipeline',
    'ConditionalStep',
    'HumanReviewStep',
    'InputType',
    'OutputType',
    
    # Step implementations
    'UrlDownloadStep',
    'PdfExtractionStep',
    'GlossaryTaggingStep',
    'LlmSummarizationStep',
    'JsonOutputStep',
    
    # Example pipeline
    'create_sample_pipeline',
    'run_example'
]
