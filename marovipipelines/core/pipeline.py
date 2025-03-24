"""
Core pipeline components for Marovi document processing.

This module provides the base classes for building modular, type-safe processing
pipelines with support for batching, checkpointing, and observability.
"""

import json
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Generic, TypeVar, List, Dict, Any, Optional, Union, Callable, Literal
from abc import ABC, abstractmethod

from .context import PipelineContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic type safety
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
IntermediateType = TypeVar("IntermediateType")

# Valid processing modes
ProcessingMode = Literal["sequential", "parallel", "custom"]


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
                 retry_delay: float = 2.0,
                 process_mode: ProcessingMode = "sequential",
                 preprocess_mode: ProcessingMode = "sequential",
                 postprocess_mode: ProcessingMode = "sequential",
                 step_id: Optional[str] = None):
        """
        Initialize a pipeline step.
        
        Args:
            batch_size: Number of items to process in a batch
            batch_handling: Strategy for batch processing ('wrap', 'inherent', 'stream')
            max_retries: Maximum number of retry attempts 
            retry_delay: Delay between retry attempts in seconds
            process_mode: Mode for main processing ('sequential', 'parallel', 'custom')
            preprocess_mode: Mode for preprocessing ('sequential', 'parallel', 'custom')
            postprocess_mode: Mode for postprocessing ('sequential', 'parallel', 'custom')
            step_id: Optional unique identifier for this step
        """
        self.batch_size = batch_size
        self.batch_handling = batch_handling
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.process_mode = process_mode
        self.preprocess_mode = preprocess_mode
        self.postprocess_mode = postprocess_mode
        self.step_id = step_id or self.__class__.__name__
        
        if batch_handling not in ['wrap', 'inherent', 'stream']:
            raise ValueError("batch_handling must be one of: 'wrap', 'inherent', 'stream'")
        
        for mode_name, mode in [("process_mode", process_mode), 
                               ("preprocess_mode", preprocess_mode), 
                               ("postprocess_mode", postprocess_mode)]:
            if mode not in ["sequential", "parallel", "custom"]:
                raise ValueError(f"{mode_name} must be one of: 'sequential', 'parallel', 'custom'")
        
        logger.debug(f"Initialized {self.step_id} with batch_size={batch_size}, "
                    f"batch_handling='{batch_handling}', process_mode='{process_mode}'")
    
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
    
    def _execute_with_mode(self, 
                          mode: ProcessingMode, 
                          func: Callable[[Any], Any], 
                          items: List[Any], 
                          context: PipelineContext) -> List[Any]:
        """
        Execute a function on items using the specified processing mode.
        
        Args:
            mode: Processing mode ('sequential', 'parallel', 'custom')
            func: Function to execute on each item
            items: List of items to process
            context: Pipeline context
            
        Returns:
            List of processed items
        """
        if mode == "sequential":
            return [func(item) for item in items]
        
        elif mode == "parallel":
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return list(executor.map(func, items))
        
        elif mode == "custom":
            # Custom mode requires the subclass to override the appropriate method
            # This is just a fallback to sequential mode
            logger.warning(f"Custom mode specified but not implemented, falling back to sequential")
            return [func(item) for item in items]
        
        else:
            raise ValueError(f"Unknown processing mode: {mode}")
    
    def batch_process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process a batch of inputs.
        
        Default implementation processes items based on the process_mode setting.
        Override this method for custom batch processing (e.g., for LLM APIs, Spark).
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List of output items
        """
        if self.process_mode == "custom":
            # For custom mode, subclass should override this method
            return self.process(inputs, context)
        
        # For sequential or parallel modes, we use the appropriate execution strategy
        def process_single_item(item):
            return self.process([item], context)[0]
        
        return self._execute_with_mode(self.process_mode, process_single_item, inputs, context)
    
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
        attempts = 0
        start_time = time.time()
        
        while attempts < self.max_retries:
            try:
                # Apply preprocessing with the specified mode
                if self.preprocess_mode == "custom":
                    processed_inputs = self.preprocess(inputs, context)
                else:
                    def preprocess_single_item(item):
                        return self.preprocess([item], context)[0]
                    processed_inputs = self._execute_with_mode(
                        self.preprocess_mode, preprocess_single_item, inputs, context
                    )
                
                logger.debug(f"{self.step_id}: Running batch_process on {len(processed_inputs)} items")
                outputs = self.batch_process(processed_inputs, context)
                
                # Apply postprocessing with the specified mode
                if self.postprocess_mode == "custom":
                    processed_outputs = self.postprocess(outputs, context)
                else:
                    def postprocess_single_item(item):
                        return self.postprocess([item], context)[0]
                    processed_outputs = self._execute_with_mode(
                        self.postprocess_mode, postprocess_single_item, outputs, context
                    )
                
                # Log success with timing information
                execution_time = time.time() - start_time
                context.log_step(
                    self.step_id, 
                    inputs, 
                    processed_outputs, 
                    extra={
                        "execution_time": execution_time,
                        "attempt": attempts + 1
                    }
                )
                
                # Update state for checkpointing
                context.update_state(
                    self.step_id,
                    processed_outputs,
                    {
                        "execution_time": execution_time,
                        "attempt": attempts + 1,
                        "input_count": len(inputs),
                        "output_count": len(processed_outputs)
                    }
                )
                
                logger.info(f"{self.step_id}: Successfully processed {len(inputs)} items in {execution_time:.2f}s")
                return processed_outputs
                
            except Exception as e:
                attempts += 1
                logger.warning(f"{self.step_id}: Attempt {attempts} failed with error: {str(e)}")
                context.log_step(
                    self.step_id, 
                    inputs, 
                    None, 
                    extra={
                        "error": str(e),
                        "attempt": attempts,
                        "exception_type": type(e).__name__
                    }
                )
                
                if attempts < self.max_retries:
                    logger.info(f"{self.step_id}: Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"{self.step_id}: All {self.max_retries} retry attempts failed")
        
        raise RuntimeError(f"{self.step_id} failed after {self.max_retries} retries")


class Pipeline:
    """
    Main pipeline executor that runs a sequence of processing steps.
    
    Features:
    - Sequential step execution
    - Batching and optional parallelism
    - Checkpointing after each step
    - Resumable from any step
    - Metrics collection
    """
    
    def __init__(self, steps: List[PipelineStep], name: str = "default_pipeline", 
                 checkpoint_dir: str = "./checkpoints", parallelism: int = 4,
                 auto_checkpoint: bool = True):
        """
        Initialize a processing pipeline.
        
        Args:
            steps: List of PipelineStep instances to execute in sequence
            name: Name of the pipeline for checkpointing
            checkpoint_dir: Directory to store checkpoint files
            parallelism: Level of parallelism for step execution
            auto_checkpoint: Whether to automatically create checkpoints after each step
        """
        self.steps = steps
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.parallelism = parallelism
        self.auto_checkpoint = auto_checkpoint
        
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Initialized {name} pipeline with {len(steps)} steps")
    
    def run(self, inputs: List[Any], context: PipelineContext, 
            resume_from: Optional[str] = None,
            reuse_outputs: bool = True) -> List[Any]:
        """
        Run the pipeline on the provided inputs.
        
        Args:
            inputs: Initial input data
            context: Pipeline context
            resume_from: Optional step ID to resume from
            reuse_outputs: Whether to reuse cached outputs when resuming
            
        Returns:
            Final output of the pipeline
        """
        pipeline_start = time.time()
        context.add_metadata("pipeline_name", self.name)
        context.add_metadata("pipeline_start_time", pipeline_start)
        
        # Find the index of the resume step if specified
        start_idx = 0
        if resume_from:
            for i, step in enumerate(self.steps):
                if step.step_id == resume_from:
                    start_idx = i
                    logger.info(f"Pipeline will resume from step '{resume_from}' (index {start_idx})")
                    
                    # If reusing outputs and we have cached results, use them
                    if reuse_outputs:
                        cached_outputs = context.get_outputs(resume_from)
                        if cached_outputs is not None:
                            logger.info(f"Reusing cached outputs from step '{resume_from}'")
                            inputs = cached_outputs
                    break
            else:
                logger.warning(f"Resume step '{resume_from}' not found, starting from beginning")
        
        # Run the pipeline from the start_idx
        for i in range(start_idx, len(self.steps)):
            step = self.steps[i]
            step_start = time.time()
            
            logger.info(f"Running step '{step.step_id}' ({i+1}/{len(self.steps)}) with {len(inputs)} inputs")
            
            # Process the current step
            try:
                inputs = self._process_batches(step, inputs, context)
                
                # Log step metrics
                step_metrics = {
                    f"step_{step.step_id}_execution_time": time.time() - step_start,
                    f"step_{step.step_id}_output_count": len(inputs)
                }
                context.log_metrics(step_metrics)
                
                # Create checkpoint if auto_checkpoint is enabled
                if self.auto_checkpoint:
                    self._checkpoint(step.step_id, context)
                
                logger.info(f"Completed step '{step.step_id}' with {len(inputs)} outputs")
                
            except Exception as e:
                logger.error(f"Pipeline execution failed at step '{step.step_id}': {str(e)}")
                context.add_metadata("pipeline_error", {
                    "step": step.step_id,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                    "timestamp": time.time()
                })
                raise
        
        # Log pipeline completion metrics
        pipeline_metrics = {
            "pipeline_total_execution_time": time.time() - pipeline_start,
            "pipeline_steps_executed": len(self.steps) - start_idx,
            "pipeline_final_output_count": len(inputs)
        }
        context.log_metrics(pipeline_metrics)
        context.add_metadata("pipeline_end_time", time.time())
        
        logger.info(f"Pipeline '{self.name}' completed successfully in {time.time() - pipeline_start:.2f}s")
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
    
    def _checkpoint(self, step_id: str, context: PipelineContext) -> None:
        """
        Save checkpoint after a step completes.
        
        Args:
            step_id: ID of the completed step
            context: Pipeline context with current state
        """
        checkpoint_name = f"{self.name}_after_{step_id}"
        try:
            checkpoint_path = context.save_checkpoint(checkpoint_name)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, context: PipelineContext, step_id: str) -> None:
        """
        Load a specific checkpoint into the context.
        
        Args:
            context: Pipeline context to update
            step_id: ID of the step to load checkpoint for
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = self.checkpoint_dir / f"{context.context_id}_{self.name}_after_{step_id}.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        context.load_checkpoint(str(checkpoint_path))
        logger.info(f"Loaded checkpoint for step '{step_id}'")


class BranchingPipeline(Pipeline):
    """
    Enhanced pipeline that supports conditional branching.
    
    This allows for complex workflows with multiple execution paths.
    """
    
    def __init__(self, branches: Dict[str, List[PipelineStep]], 
                entry_point: str = "main", **kwargs):
        """
        Initialize a branching pipeline.
        
        Args:
            branches: Dictionary mapping branch names to lists of steps
            entry_point: Name of the default branch to execute
            **kwargs: Additional arguments for the Pipeline constructor
        """
        self.branches = branches
        self.entry_point = entry_point
        
        if entry_point not in branches:
            raise ValueError(f"Entry point '{entry_point}' not found in branches")
        
        super().__init__(branches[entry_point], **kwargs)
    
    def run_branch(self, branch_name: str, inputs: List[Any], 
                  context: PipelineContext, **kwargs) -> List[Any]:
        """
        Run a specific branch of the pipeline.
        
        Args:
            branch_name: Name of the branch to execute
            inputs: Input data for the branch
            context: Pipeline context
            **kwargs: Additional arguments for the run method
            
        Returns:
            Output from the branch execution
        """
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' not found")
        
        # Temporarily swap the steps
        original_steps = self.steps
        self.steps = self.branches[branch_name]
        
        try:
            # Run the branch
            context.add_metadata("current_branch", branch_name)
            outputs = super().run(inputs, context, **kwargs)
            return outputs
        finally:
            # Restore the original steps
            self.steps = original_steps