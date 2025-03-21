"""
Example usage of the Marovi pipeline framework.

This module demonstrates how to build and run a document processing pipeline
with the following steps:
- URL download
- PDF extraction 
- LLM summarization
- Human review
- JSON output
"""

import logging
from typing import Dict, Any, List
import sys

from .base import PipelineContext, ProcessingPipeline, HumanReviewStep
from .steps import (
    UrlDownloadStep, 
    PdfExtractionStep, 
    GlossaryTaggingStep,
    LlmSummarizationStep,
    JsonOutputStep
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_sample_pipeline() -> ProcessingPipeline:
    """
    Create an example document processing pipeline.
    
    Returns:
        Configured ProcessingPipeline instance
    """
    # Sample glossary terms
    glossary_terms = {
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning",
        "NLP": "Natural Language Processing",
        "LLM": "Large Language Model"
    }
    
    # Create pipeline steps
    url_downloader = UrlDownloadStep(timeout=30, batch_size=5)
    pdf_extractor = PdfExtractionStep(extraction_mode="text")
    glossary_tagger = GlossaryTaggingStep(glossary_terms=glossary_terms)
    summarizer = LlmSummarizationStep(max_summary_length=200, batch_size=4)
    human_reviewer = HumanReviewStep(review_prompt="Please review this summary")
    json_outputter = JsonOutputStep(output_dir="./output")
    
    # Create pipeline
    pipeline = ProcessingPipeline(
        steps=[
            url_downloader,
            pdf_extractor,
            glossary_tagger,  # Applies tagging as preprocessing step
            summarizer,
            human_reviewer,
            json_outputter
        ],
        name="document_processing",
        checkpoint_dir="./checkpoints"
    )
    
    return pipeline


def run_example(urls: List[str], resume_from: str = None) -> List[Dict[str, Any]]:
    """
    Run the example pipeline on a list of URLs.
    
    Args:
        urls: List of URLs to process
        resume_from: Optional step name to resume from
        
    Returns:
        List of processed documents
    """
    # Create pipeline and context
    pipeline = create_sample_pipeline()
    context = PipelineContext(
        metadata={
            "pipeline_name": "document_processing",
            "source_type": "url",
            "source_count": len(urls)
        },
        checkpoint_dir="./checkpoints"
    )
    
    # Run pipeline
    logger.info(f"Starting pipeline with {len(urls)} URLs")
    
    if resume_from:
        logger.info(f"Resuming from step: {resume_from}")
        # In a real implementation, we would load state from the checkpoint
        # For demo purposes, we'll just start from the specified step
    
    results = pipeline.run(urls, context, resume_from=resume_from)
    
    logger.info(f"Pipeline completed with {len(results)} documents")
    logger.info(f"Pipeline metadata: {context.metadata}")
    
    return results


if __name__ == "__main__":
    # Sample URLs
    sample_urls = [
        "https://example.com/document1.pdf",
        "https://example.com/document2.pdf",
        "https://example.com/document3.pdf"
    ]
    
    # Run the pipeline
    results = run_example(sample_urls)
    
    # Alternatively, resume from a specific step
    # results = run_example(sample_urls, resume_from="LlmSummarizationStep") 