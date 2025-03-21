"""
Example pipeline step implementations for Marovi document processing.

This module contains concrete implementations of PipelineStep for various
document processing tasks such as downloading, extraction, translation, etc.
"""

import time
import logging
import random
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

from .base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class UrlDownloadStep(PipelineStep[str, Dict[str, Any]]):
    """
    Pipeline step for downloading content from URLs.
    
    Inputs: URLs as strings
    Outputs: Dictionaries containing downloaded content and metadata
    """
    
    def __init__(self, timeout: int = 30, batch_size: int = 5):
        """
        Initialize URL download step.
        
        Args:
            timeout: Request timeout in seconds
            batch_size: Number of URLs to download in parallel
        """
        super().__init__(batch_size=batch_size, batch_handling='wrap')
        self.timeout = timeout
    
    def process(self, inputs: List[str], context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Download content from URLs.
        
        Args:
            inputs: List of URLs to download
            context: Pipeline context
            
        Returns:
            List of dictionaries with downloaded content and metadata
        """
        results = []
        
        for url in inputs:
            try:
                logger.info(f"Downloading {url}")
                start_time = time.time()
                
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                download_time = time.time() - start_time
                
                result = {
                    "url": url,
                    "content": response.text,
                    "content_type": response.headers.get("Content-Type"),
                    "status_code": response.status_code,
                    "download_time": download_time,
                    "content_length": len(response.content)
                }
                
                results.append(result)
                logger.info(f"Downloaded {url} ({result['content_length']} bytes) in {download_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to download {url}: {str(e)}")
                results.append({
                    "url": url,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results


class PdfExtractionStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline step for extracting text content from PDF documents.
    
    Inputs: Dictionaries with PDF content
    Outputs: Dictionaries with extracted text and metadata
    """
    
    def __init__(self, extraction_mode: str = "text"):
        """
        Initialize PDF extraction step.
        
        Args:
            extraction_mode: Mode of extraction ('text', 'layout', 'full')
        """
        super().__init__(batch_size=1)
        self.extraction_mode = extraction_mode
    
    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Extract text from PDF content.
        
        Note: This is a stub implementation. In production, use a real PDF library.
        
        Args:
            inputs: List of dictionaries with PDF content
            context: Pipeline context
            
        Returns:
            List of dictionaries with extracted text and metadata
        """
        results = []
        
        for doc in inputs:
            # In a real implementation, this would use a PDF library
            # For demo purposes, we'll just simulate extraction
            logger.info(f"Extracting text from document: {doc.get('url', 'unknown')}")
            
            # Simulate processing delay
            time.sleep(0.5)
            
            # Simulate extraction
            extracted_text = f"Extracted text from {doc.get('url', 'unknown')}.\n"
            extracted_text += "This is simulated extracted content for demonstration purposes."
            
            result = {
                "original_url": doc.get("url"),
                "extracted_text": extracted_text,
                "num_pages": random.randint(1, 20),
                "extraction_mode": self.extraction_mode,
                "document_id": f"doc_{int(time.time())}_{random.randint(1000, 9999)}"
            }
            
            results.append(result)
            logger.info(f"Extracted {result['num_pages']} pages from document")
        
        return results


class GlossaryTaggingStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline step for tagging text with glossary terms.
    
    This is designed to be used as a preprocessing or postprocessing step.
    """
    
    def __init__(self, glossary_terms: Dict[str, str]):
        """
        Initialize glossary tagging step.
        
        Args:
            glossary_terms: Dictionary mapping terms to their definitions
        """
        super().__init__()
        self.glossary_terms = glossary_terms
        logger.info(f"Initialized glossary tagger with {len(glossary_terms)} terms")
    
    def preprocess(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Apply glossary tags to input documents.
        
        Args:
            inputs: List of document dictionaries
            context: Pipeline context
            
        Returns:
            Documents with glossary tags applied
        """
        tagged_docs = []
        
        for doc in inputs:
            if "extracted_text" in doc:
                text = doc["extracted_text"]
                tags = []
                
                # Simple term detection (in production, use more sophisticated NLP)
                for term, definition in self.glossary_terms.items():
                    if term.lower() in text.lower():
                        tags.append({
                            "term": term,
                            "definition": definition,
                            "count": text.lower().count(term.lower())
                        })
                
                # Add tags to document
                tagged_doc = doc.copy()
                tagged_doc["glossary_tags"] = tags
                tagged_docs.append(tagged_doc)
                
                logger.info(f"Tagged document with {len(tags)} glossary terms")
            else:
                tagged_docs.append(doc)
        
        return tagged_docs
    
    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Process method (required but unused since we use preprocess/postprocess).
        """
        return inputs


class LlmSummarizationStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline step for summarizing text using a Language Model.
    
    Inputs: Dictionaries with text content
    Outputs: Dictionaries with text summaries
    """
    
    def __init__(self, max_summary_length: int = 200, 
                batch_size: int = 4, api_key: Optional[str] = None):
        """
        Initialize LLM summarization step.
        
        Args:
            max_summary_length: Maximum length of summary in words
            batch_size: Number of documents to summarize in one batch
            api_key: API key for LLM service (if applicable)
        """
        super().__init__(batch_size=batch_size, batch_handling='inherent')
        self.max_summary_length = max_summary_length
        self.api_key = api_key
    
    def batch_process(self, inputs: List[Dict[str, Any]], 
                     context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Process a batch of documents for summarization.
        
        Note: This is a stub implementation. In production, use a real LLM API.
        
        Args:
            inputs: List of document dictionaries
            context: Pipeline context
            
        Returns:
            List of dictionaries with summaries
        """
        logger.info(f"Summarizing batch of {len(inputs)} documents")
        
        # Track token usage for observability
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        results = []
        
        # In production, this would make a batch API call to an LLM
        for doc in inputs:
            # Simulate API call
            time.sleep(1)
            
            text = doc.get("extracted_text", "")
            
            # Simulate LLM summary
            summary = f"Summary of document {doc.get('document_id', 'unknown')}. "
            summary += "This is a placeholder for a real LLM-generated summary."
            
            # Track simulated token usage
            prompt_tokens = len(text.split()) // 4
            completion_tokens = len(summary.split())
            
            token_usage["prompt_tokens"] += prompt_tokens
            token_usage["completion_tokens"] += completion_tokens
            token_usage["total_tokens"] += prompt_tokens + completion_tokens
            
            # Add summary to result
            result = doc.copy()
            result["summary"] = summary
            result["token_usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            results.append(result)
        
        # Add batch token usage to context metadata
        context.add_metadata("llm_token_usage", token_usage)
        logger.info(f"Completed summarization with {token_usage['total_tokens']} total tokens used")
        
        return results
    
    def process(self, inputs: List[Dict[str, Any]], 
               context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Process method (required but unused since we use batch_process).
        """
        return self.batch_process(inputs, context)


class JsonOutputStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline step for saving results to JSON files.
    
    Inputs: Dictionaries with processed data
    Outputs: Same dictionaries with file paths added
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize JSON output step.
        
        Args:
            output_dir: Directory to store output files
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def process(self, inputs: List[Dict[str, Any]], 
               context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Save documents to JSON files.
        
        Args:
            inputs: List of document dictionaries
            context: Pipeline context
            
        Returns:
            Same documents with output file paths added
        """
        results = []
        
        for doc in inputs:
            doc_id = doc.get("document_id", f"doc_{int(time.time())}")
            output_file = self.output_dir / f"{doc_id}.json"
            
            # Save to file
            with open(output_file, "w") as f:
                json.dump(doc, f, indent=2)
            
            # Add output path to result
            result = doc.copy()
            result["output_file"] = str(output_file)
            results.append(result)
            
            logger.info(f"Saved output to {output_file}")
        
        return results 