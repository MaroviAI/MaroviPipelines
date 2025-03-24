"""
Specialized pipeline steps for Marovi pipelines.

This module provides specialized pipeline step implementations like
ConditionalStep, HumanReviewStep, CheckpointStep, and more.
"""

import time
import logging
import uuid
from typing import Generic, TypeVar, List, Dict, Any, Optional, Callable, Union

from .pipeline import PipelineStep
from .context import PipelineContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic type safety
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
IntermediateType = TypeVar("IntermediateType")


class ConditionalStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that conditionally routes inputs to different branches.
    """
    
    def __init__(self, condition_fn: Callable[[InputType], bool], 
                true_branch: PipelineStep, false_branch: PipelineStep,
                step_id: Optional[str] = None):
        """
        Initialize a conditional branching step.
        
        Args:
            condition_fn: Function that evaluates each input item to True or False
            true_branch: Step to process items when condition is True
            false_branch: Step to process items when condition is False
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"conditional_{id(self):x}")
        self.condition_fn = condition_fn
        self.true_branch = true_branch
        self.false_branch = false_branch
        
        logger.info(f"Initialized {self.step_id} with {true_branch.step_id} and "
                   f"{false_branch.step_id} branches")
    
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
            logger.info(f"{self.step_id}: Routing {len(true_items)} items to true branch ({self.true_branch.step_id})")
            true_outputs = self.true_branch.run_with_retries(true_items, context)
            outputs.extend(true_outputs)
        
        # Process false branch if there are items
        if false_items:
            logger.info(f"{self.step_id}: Routing {len(false_items)} items to false branch ({self.false_branch.step_id})")
            false_outputs = self.false_branch.run_with_retries(false_items, context)
            outputs.extend(false_outputs)
        
        return outputs


class HumanReviewStep(PipelineStep[InputType, InputType]):
    """
    A pipeline step that pauses for human review.
    
    In a production environment, this would interface with a review queue or dashboard.
    """
    
    def __init__(self, review_prompt: str = "HUMAN REVIEW REQUIRED", step_id: Optional[str] = None):
        """
        Initialize a human review step.
        
        Args:
            review_prompt: Message to display for the reviewer
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"human_review_{id(self):x}")
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
        logger.info(f"{self.step_id}: {self.review_prompt}: {len(inputs)} items ready for review")
        
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
            "items_reviewed": len(inputs),
            "step_id": self.step_id
        })
        
        return inputs


class CheckpointStep(PipelineStep[InputType, InputType]):
    """
    A pipeline step that creates an explicit checkpoint in the pipeline execution.
    
    This allows resuming the pipeline from this point later.
    """
    
    def __init__(self, checkpoint_name: Optional[str] = None, step_id: Optional[str] = None):
        """
        Initialize a checkpoint step.
        
        Args:
            checkpoint_name: Optional name for the checkpoint. If not provided,
                            a unique name will be generated.
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"checkpoint_{id(self):x}")
        self.checkpoint_name = checkpoint_name or f"checkpoint_{uuid.uuid4().hex[:8]}"
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[InputType]:
        """
        Process inputs by creating a checkpoint and passing through the inputs.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            The input items unchanged
        """
        # Store the current inputs in the context state
        context.update_state(self.checkpoint_name, inputs, {
            "timestamp": time.time(),
            "explicit_checkpoint": True,
            "item_count": len(inputs)
        })
        
        # Save the checkpoint
        checkpoint_path = context.save_checkpoint(self.checkpoint_name)
        logger.info(f"{self.step_id}: Created explicit checkpoint '{self.checkpoint_name}' at {checkpoint_path}")
        
        return inputs


class ResumableStep(PipelineStep[InputType, OutputType]):
    """
    A wrapper that makes any pipeline step resumable from a checkpoint.
    
    This integrates with the Pipeline's checkpoint mechanism to allow
    resuming from specific steps without reprocessing earlier steps.
    """
    
    def __init__(self, step: PipelineStep[InputType, OutputType], step_id: Optional[str] = None):
        """
        Initialize a resumable step wrapper.
        
        Args:
            step: The pipeline step to make resumable
            step_id: Optional unique identifier for this step (defaults to wrapped step's ID)
        """
        # Copy configuration from wrapped step
        super().__init__(
            batch_size=step.batch_size,
            batch_handling=step.batch_handling,
            max_retries=step.max_retries,
            retry_delay=step.retry_delay,
            process_mode=step.process_mode,
            preprocess_mode=step.preprocess_mode,
            postprocess_mode=step.postprocess_mode,
            step_id=step_id or step.step_id
        )
        self.step = step
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs with the wrapped step, with checkpoint/resume capability.
        
        This method checks if outputs are already in the context state before
        processing, allowing for resumption from checkpoints.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Outputs from the wrapped step
        """
        # Check if we have cached outputs for this step in the context state
        state = context.get_state(self.step_id)
        if state and "outputs" in state:
            logger.info(f"{self.step_id}: Resuming from cached outputs")
            return state["outputs"]
        
        # Process inputs with the wrapped step
        outputs = self.step.process(inputs, context)
        
        # We don't need to cache outputs here as the Pipeline will do it
        # after each step using update_state
        
        return outputs
    
    def preprocess(self, inputs: List[InputType], context: PipelineContext) -> List[InputType]:
        """Delegate preprocessing to the wrapped step."""
        return self.step.preprocess(inputs, context)
    
    def postprocess(self, outputs: List[OutputType], context: PipelineContext) -> List[OutputType]:
        """Delegate postprocessing to the wrapped step."""
        return self.step.postprocess(outputs, context)
    
    def batch_process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """Delegate batch processing to the wrapped step."""
        return self.step.batch_process(inputs, context)


class ParallelStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that processes inputs through multiple parallel steps and combines the results.
    """
    
    def __init__(self, steps: List[PipelineStep[InputType, OutputType]], 
                combiner: Callable[[List[List[OutputType]]], List[OutputType]],
                step_id: Optional[str] = None):
        """
        Initialize a parallel step.
        
        Args:
            steps: List of steps to execute in parallel
            combiner: Function to combine outputs from parallel steps
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"parallel_{id(self):x}")
        self.steps = steps
        self.combiner = combiner
        
        logger.info(f"Initialized {self.step_id} with {len(steps)} parallel steps")
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs through multiple parallel steps and combine the results.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Combined outputs from parallel steps
        """
        all_outputs = []
        
        # Process inputs through each step
        for step in self.steps:
            logger.info(f"{self.step_id}: Running parallel step {step.step_id}")
            step_outputs = step.run_with_retries(inputs, context)
            all_outputs.append(step_outputs)
        
        # Combine outputs using the provided combiner function
        combined_outputs = self.combiner(all_outputs)
        logger.info(f"{self.step_id}: Combined {len(all_outputs)} parallel outputs into {len(combined_outputs)} items")
        
        return combined_outputs


class SequentialStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that processes inputs through a sequence of steps.
    
    This is similar to a mini-pipeline within a step.
    """
    
    def __init__(self, steps: List[PipelineStep], step_id: Optional[str] = None):
        """
        Initialize a sequential step.
        
        Args:
            steps: List of steps to execute in sequence
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"sequential_{id(self):x}")
        self.steps = steps
        
        logger.info(f"Initialized {self.step_id} with {len(steps)} sequential steps")
    
    def process(self, inputs: List[Any], context: PipelineContext) -> List[Any]:
        """
        Process inputs through a sequence of steps.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Outputs from the final step
        """
        current_inputs = inputs
        
        # Process inputs through each step in sequence
        for i, step in enumerate(self.steps):
            logger.info(f"{self.step_id}: Running step {i+1}/{len(self.steps)} ({step.step_id})")
            current_inputs = step.run_with_retries(current_inputs, context)
        
        return current_inputs


class FilterStep(PipelineStep[InputType, InputType]):
    """
    A pipeline step that filters inputs based on a predicate function.
    """
    
    def __init__(self, predicate: Callable[[InputType], bool], step_id: Optional[str] = None):
        """
        Initialize a filter step.
        
        Args:
            predicate: Function that returns True for items to keep
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"filter_{id(self):x}")
        self.predicate = predicate
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[InputType]:
        """
        Process inputs by filtering based on the predicate.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Filtered list of input items
        """
        filtered = [item for item in inputs if self.predicate(item)]
        
        # Log filtering results
        filtered_out = len(inputs) - len(filtered)
        logger.info(f"{self.step_id}: Filtered {filtered_out} of {len(inputs)} items ({filtered_out/len(inputs)*100:.1f}%)")
        
        return filtered


class MapStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that applies a transformation function to each input item.
    """
    
    def __init__(self, transform_fn: Callable[[InputType], OutputType], step_id: Optional[str] = None):
        """
        Initialize a map step.
        
        Args:
            transform_fn: Function to apply to each input item
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"map_{id(self):x}")
        self.transform_fn = transform_fn
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs by applying the transform function to each item.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List of transformed items
        """
        outputs = [self.transform_fn(item) for item in inputs]
        logger.info(f"{self.step_id}: Mapped {len(inputs)} items")
        
        return outputs


class FlatMapStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that applies a one-to-many transformation to each input item.
    """
    
    def __init__(self, transform_fn: Callable[[InputType], List[OutputType]], step_id: Optional[str] = None):
        """
        Initialize a flatmap step.
        
        Args:
            transform_fn: Function that returns a list of outputs for each input
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"flatmap_{id(self):x}")
        self.transform_fn = transform_fn
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs by applying the transform function and flattening results.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            Flattened list of transformed items
        """
        outputs = []
        for item in inputs:
            item_outputs = self.transform_fn(item)
            outputs.extend(item_outputs)
        
        logger.info(f"{self.step_id}: FlatMapped {len(inputs)} items into {len(outputs)} outputs")
        
        return outputs


class AggregateStep(PipelineStep[InputType, OutputType]):
    """
    A pipeline step that aggregates multiple inputs into a single output.
    """
    
    def __init__(self, aggregate_fn: Callable[[List[InputType]], OutputType], step_id: Optional[str] = None):
        """
        Initialize an aggregate step.
        
        Args:
            aggregate_fn: Function that combines inputs into a single output
            step_id: Optional unique identifier for this step
        """
        super().__init__(step_id=step_id or f"aggregate_{id(self):x}")
        self.aggregate_fn = aggregate_fn
    
    def process(self, inputs: List[InputType], context: PipelineContext) -> List[OutputType]:
        """
        Process inputs by aggregating them into a single output.
        
        Args:
            inputs: List of input items
            context: Pipeline context
            
        Returns:
            List containing the single aggregated output
        """
        output = self.aggregate_fn(inputs)
        logger.info(f"{self.step_id}: Aggregated {len(inputs)} items into a single output")
        
        return [output]