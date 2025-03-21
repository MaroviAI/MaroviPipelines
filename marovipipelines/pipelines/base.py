"""
Core pipeline components for Marovi document processing.

This module provides the base classes for building modular, type-safe processing
pipelines with support for batching, checkpointing, and observability.
"""

import json
import time
import logging
from pathlib import Path
from typing import Generic, TypeVar, List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic type safety
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class PipelineContext:
    """
    Captures execution metadata and per-step input/output for a pipeline run.
    
    Stores:
    - Global metadata (doc_id, language, glossary terms)
    - Step logs (inputs, outputs, prompts, latencies, retries, errors)
    - Intermediate states for checkpointing
    
    This class is serializable to JSON for persistence.
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize a new pipeline context.
        
        Args:
            metadata: Optional dictionary of global metadata for the pipeline run
            checkpoint_dir: Directory to store checkpoint files
        """
        self.metadata = metadata or {}
        self.step_logs = []
        self.state = {}
        self.history = []  # Optional correction/versioning trail
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized PipelineContext with metadata: {metadata}")
    
    def log_step(self, step_name: str, input_data: Any, output_data: Any, 
                 extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the execution of a pipeline step.
        
        Args:
            step_name: Name of the step
            input_data: Input data to the step
            output_data: Output data from the step
            extra: Additional metadata about the step execution
        """
        log_entry = {
            "step": step_name,
            "input": input_data,
            "output": output_data,
            "extra": extra or {},
            "timestamp": time.time()
        }
        self.step_logs.append(log_entry)
        logger.debug(f"Logged step {step_name} with {len(input_data) if isinstance(input_data, list) else 1} items")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata field."""
        self.metadata[key] = value
    
    def to_json(self) -> str:
        """Serialize the context to JSON."""
        return json.dumps({
            "metadata": self.metadata,
            "step_logs": self.step_logs,
            "state": self.state
        }, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineContext':
        """Reconstruct a context from JSON."""
        data = json.loads(json_str)
        context = cls(metadata=data.get("metadata", {}))
        context.step_logs = data.get("step_logs", [])
        context.state = data.get("state", {})
        return context


class PipelineStep(Generic[InputType, OutputType], ABC):
    """
    Base class for pipeline processing steps with type safety and batch awareness.
    
    Features:
    - Type-safe (InputType → OutputType)
    - Batch processing with configurable strategies
    - Preprocessing and postprocessing hooks
    - Error handling with retries
    - Execution logging
    """
    
    def __init__(self, 
                 batch_size: int = 1, 
                 batch_handling: str = 'wrap', 
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize a pipeline step.
        
        Args:
            batch_size: Number of items to process in a batch
            batch_handling: Strategy for batch processing ('wrap', 'inherent', 'stream')
            max_retries: Maximum number of retry attempts 
            retry_delay: Delay between retry attempts in seconds
        """
        self.batch_size = batch_size
        self.batch_handling = batch_handling
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if batch_handling not in ['wrap', 'inherent', 'stream']:
            raise ValueError("batch_handling must be one of: 'wrap', 'inherent', 'stream'")
        
        logger.debug(f"Initialized {self.__class__.__name__} with batch_size={batch_size}, "
                    f"batch_handling='{batch_handling}'")
    
    def preprocess(self, inputs: List[InputType], context: PipelineContext) -> List[InputType]:
        """
        Preprocess inputs before the main processing step.
        
        Override this method to implement custom preprocessing logic.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Preprocessed inputs
        """
        return inputs
    
    def postprocess(self, outputs: List[OutputType], context: PipelineContext) -> List[OutputType]:
        """
        Postprocess outputs after the main processing step.
        
        Override this method to implement custom postprocessing logic.
        
        Args:
            outputs: List of output items
            context: Pipeline context
            
        Returns:
            Postprocessed outputs
        """
        return outputs
    
    def batch_process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process a batch of inputs.
        
        Default implementation processes items sequentially.
        Override this method for true batch processing (e.g., for LLM APIs, Spark).
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List of output items
        """
        outputs = []
        for item in inputs:
            outputs.extend(self.process([item], context))
        return outputs
    
    @abstractmethod
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process a list of inputs.
        
        All subclasses must implement this method.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List of output items
        """
        raise NotImplementedError
    
    def run_with_retries(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Run the processing step with retry logic.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List of output items
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        step_name = self.__class__.__name__
        attempts = 0
        start_time = time.time()
        
        while attempts < self.max_retries:
            try:
                processed_inputs = self.preprocess(inputs, context)
                logger.debug(f"{step_name}: Running batch_process on {len(processed_inputs)} items")
                outputs = self.batch_process(processed_inputs, context)
                processed_outputs = self.postprocess(outputs, context)
                
                # Log success with timing information
                execution_time = time.time() - start_time
                context.log_step(
                    step_name, 
                    inputs, 
                    processed_outputs, 
                    extra={
                        "execution_time": execution_time,
                        "attempt": attempts + 1
                    }
                )
                logger.info(f"{step_name}: Successfully processed {len(inputs)} items in {execution_time:.2f}s")
                return processed_outputs
                
            except Exception as e:
                attempts += 1
                logger.warning(f"{step_name}: Attempt {attempts} failed with error: {str(e)}")
                context.log_step(
                    step_name, 
                    inputs, 
                    None, 
                    extra={
                        "error": str(e),
                        "attempt": attempts,
                        "exception_type": type(e).__name__
                    }
                )
                
                if attempts < self.max_retries:
                    logger.info(f"{step_name}: Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"{step_name}: All {self.max_retries} retry attempts failed")
        
        raise RuntimeError(f"{step_name} failed after {self.max_retries} retries")


class ProcessingPipeline:
    """
    Main pipeline executor that runs a sequence of processing steps.
    
    Features:
    - Sequential step execution
    - Batching and optional parallelism
    - Checkpointing after each step
    - Resumable from any step
    """
    
    def __init__(self, steps: List[PipelineStep], name: str = "default_pipeline", 
                 checkpoint_dir: str = "./checkpoints", parallelism: int = 4):
        """
        Initialize a processing pipeline.
        
        Args:
            steps: List of PipelineStep instances to execute in sequence
            name: Name of the pipeline for checkpointing
            checkpoint_dir: Directory to store checkpoint files
            parallelism: Level of parallelism for step execution (for future use)
        """
        self.steps = steps
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.parallelism = parallelism
        
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Initialized {name} pipeline with {len(steps)} steps")
    
    def run(self, inputs: List[Any], context: PipelineContext, 
            resume_from: Optional[str] = None) -> List[Any]:
        """
        Run the pipeline on the provided inputs.
        
        Args:
            inputs: Initial input data
            context: Pipeline context
            resume_from: Optional step name to resume from
            
        Returns:
            Final output of the pipeline
        """
        skip = resume_from is not None
        if skip:
            logger.info(f"Pipeline will resume from step '{resume_from}'")
        
        for step in self.steps:
            step_name = step.__class__.__name__
            
            if skip and step_name != resume_from:
                logger.info(f"Skipping step '{step_name}' (resuming from '{resume_from}')")
                continue
            
            skip = False
            logger.info(f"Running step '{step_name}' with {len(inputs)} inputs")
            
            # Process the current step
            inputs = self._process_batches(step, inputs, context)
            
            # Store state for checkpointing
            context.state[step_name] = inputs
            self._checkpoint(step_name, context)
            
            logger.info(f"Completed step '{step_name}' with {len(inputs)} outputs")
        
        logger.info(f"Pipeline '{self.name}' completed successfully")
        return inputs
    
    def _process_batches(self, step: PipelineStep, inputs: List[Any], 
                        context: PipelineContext) -> List[Any]:
        """
        Process inputs through a step with appropriate batching.
        
        Args:
            step: The pipeline step to execute
            inputs: Input data
            context: Pipeline context
            
        Returns:
            Processed outputs
        """
        if step.batch_handling == 'inherent':
            # Step handles batching internally
            return step.run_with_retries(inputs, context)
        else:
            # We handle batching
            outputs = []
            batches = self._batch(inputs, step.batch_size)
            total_batches = len(batches)
            
            for i, batch in enumerate(batches):
                logger.debug(f"Processing batch {i+1}/{total_batches} with {len(batch)} items")
                batch_outputs = step.run_with_retries(batch, context)
                outputs.extend(batch_outputs)
            
            return outputs
    
    def _batch(self, inputs: List[Any], batch_size: int) -> List[List[Any]]:
        """Split inputs into batches of specified size."""
        return [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    
    def _checkpoint(self, step_name: str, context: PipelineContext) -> None:
        """
        Save checkpoint after a step completes.
        
        Args:
            step_name: Name of the completed step
            context: Pipeline context with current state
        """
        checkpoint_file = self.checkpoint_dir / f"{self.name}_{step_name}_checkpoint.json"
        
        try:
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "step": step_name,
                    "state": context.state[step_name],
                    "metadata": context.metadata,
                    "timestamp": time.time()
                }, f, default=str)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, step_name: str) -> Dict[str, Any]:
        """
        Load data from a checkpoint file.
        
        Args:
            step_name: Name of the step to load checkpoint for
            
        Returns:
            Dictionary with checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_file = self.checkpoint_dir / f"{self.name}_{step_name}_checkpoint.json"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_file, "r") as f:
            data = json.loads(f.read())
            logger.info(f"Loaded checkpoint for step '{step_name}'")
            return data


class ConditionalStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that conditionally routes inputs to different branches.
    """
    
    def __init__(self, condition_fn: Callable[[InputType], bool], 
                true_branch: PipelineStep, false_branch: PipelineStep):
        """
        Initialize a conditional branching step.
        
        Args:
            condition_fn: Function that evaluates each input item to True or False
            true_branch: Step to process items when condition is True
            false_branch: Step to process items when condition is False
        """
        super().__init__()
        self.condition_fn = condition_fn
        self.true_branch = true_branch
        self.false_branch = false_branch
        
        logger.info(f"Initialized conditional step with {true_branch.__class__.__name__} and "
                   f"{false_branch.__class__.__name__} branches")
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs by routing to the appropriate branch based on condition.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Combined outputs from both branches
        """
        true_items = []
        false_items = []
        
        # Split inputs based on condition
        for item in inputs:
            if self.condition_fn(item):
                true_items.append(item)
            else:
                false_items.append(item)
        
        outputs = []
        
        # Process true branch if there are items
        if true_items:
            logger.info(f"Routing {len(true_items)} items to true branch ({self.true_branch.__class__.__name__})")
            true_outputs = self.true_branch.run_with_retries(true_items, context)
            outputs.extend(true_outputs)
        
        # Process false branch if there are items
        if false_items:
            logger.info(f"Routing {len(false_items)} items to false branch ({self.false_branch.__class__.__name__})")
            false_outputs = self.false_branch.run_with_retries(false_items, context)
            outputs.extend(false_outputs)
        
        return outputs


class HumanReviewStep(PipelineStep[InputType, InputType]):
    """
    A pipeline step that pauses for human review.
    
    In a production environment, this would interface with a review queue or dashboard.
    """
    
    def __init__(self, review_prompt: str = "HUMAN REVIEW REQUIRED"):
        """
        Initialize a human review step.
        
        Args:
            review_prompt: Message to display for the reviewer
        """
        super().__init__()
        self.review_prompt = review_prompt
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[InputType]:
        """
        Process inputs by presenting them for human review.
        
        In this implementation, it simply logs the inputs and prompts.
        In production, it would integrate with a review system.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            The input items (possibly modified by review)
        """
        logger.info(f"{self.review_prompt}: {len(inputs)} items ready for review")
        
        # In production, this would:
        # 1. Send items to a review queue or dashboard
        # 2. Wait for review completion or timeout
        # 3. Retrieve reviewed/modified items
        
        # For demo purposes, just simulate a review delay
        for i, item in enumerate(inputs):
            logger.info(f"Item {i+1}: {item}")
        
        # Add review metadata to context
        context.add_metadata("human_review", {
            "timestamp": time.time(),
            "items_reviewed": len(inputs)
        })
        
        return inputs 