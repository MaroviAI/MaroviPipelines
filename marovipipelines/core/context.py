"""
Pipeline context for tracking execution state and metadata.

This module provides the PipelineContext class which serves as a central
repository for pipeline state, metrics, artifacts, and execution history.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineContext:
    """
    Captures execution metadata and per-step input/output for a pipeline run.
    
    Stores:
    - Global metadata (doc_id, language, glossary terms, model parameters, etc.)
    - Step logs (inputs, outputs, prompts, latencies, retries, errors)
    - Intermediate states for checkpointing
    - Metrics tracking (loss, accuracy, etc. for ML workflows)
    - Artifacts (model weights, embeddings, etc.)
    
    This class is serializable to JSON for persistence and supports ML workflows
    as well as text processing pipelines.
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None, 
                 checkpoint_dir: str = "./checkpoints",
                 context_id: Optional[str] = None):
        """
        Initialize a new pipeline context.
        
        Args:
            metadata: Optional dictionary of global metadata for the pipeline run
            checkpoint_dir: Directory to store checkpoint files
            context_id: Optional unique identifier for this context
        """
        self.context_id = context_id or f"ctx_{int(time.time())}_{id(self):x}"
        self.metadata = metadata or {}
        self.step_logs = []
        self.state = {}
        self.history = []  # Optional correction/versioning trail
        self.metrics = {}  # For tracking ML metrics like loss, accuracy, etc.
        self.artifacts = {}  # For storing references to model weights, embeddings, etc.
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.creation_time = time.time()
        self.last_updated = self.creation_time
        
        logger.info(f"Initialized PipelineContext {self.context_id} with metadata: {metadata}")
    
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
        self.last_updated = time.time()
        logger.debug(f"Logged step {step_name} with {len(input_data) if isinstance(input_data, list) else 1} items")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata field."""
        self.metadata[key] = value
        self.last_updated = time.time()
    
    def update_state(self, step_name: str, outputs: Any, 
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the state with step outputs and metadata.
        
        Args:
            step_name: Name of the step
            outputs: Output data from the step
            metadata: Additional metadata about the state update
        """
        self.state[step_name] = {
            "outputs": outputs,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.last_updated = time.time()
        logger.debug(f"Updated state for step {step_name}")
    
    def get_state(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the state for a specific step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            State dictionary or None if not found
        """
        return self.state.get(step_name)
    
    def get_outputs(self, step_name: str) -> Optional[Any]:
        """
        Get the outputs for a specific step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step outputs or None if not found
        """
        state = self.get_state(step_name)
        return state["outputs"] if state else None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics for ML workflows (loss, accuracy, etc.)
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        timestamp = time.time()
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            
            entry = {"value": value, "timestamp": timestamp}
            if step is not None:
                entry["step"] = step
                
            self.metrics[name].append(entry)
        
        self.last_updated = timestamp
        logger.debug(f"Logged metrics: {metrics}")
    
    def register_artifact(self, name: str, artifact_path: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an ML artifact (model weights, embeddings, etc.)
        
        Args:
            name: Name of the artifact
            artifact_path: Path to the artifact file
            metadata: Optional metadata about the artifact
        """
        self.artifacts[name] = {
            "path": artifact_path,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.last_updated = time.time()
        logger.debug(f"Registered artifact: {name} at {artifact_path}")
    
    def save_checkpoint(self, name: str) -> str:
        """
        Save the current context state as a checkpoint.
        
        Args:
            name: Name of the checkpoint
            
        Returns:
            Path to the saved checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"{self.context_id}_{name}.json"
        with open(checkpoint_path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load context state from a checkpoint file.
        
        Args:
            path: Path to the checkpoint file
        """
        with open(path, 'r') as f:
            checkpoint_data = json.loads(f.read())
        
        self.metadata = checkpoint_data.get("metadata", {})
        self.step_logs = checkpoint_data.get("step_logs", [])
        self.state = checkpoint_data.get("state", {})
        self.metrics = checkpoint_data.get("metrics", {})
        self.artifacts = checkpoint_data.get("artifacts", {})
        self.context_id = checkpoint_data.get("context_id", self.context_id)
        self.creation_time = checkpoint_data.get("creation_time", self.creation_time)
        self.last_updated = time.time()
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def to_json(self) -> str:
        """Serialize the context to JSON."""
        return json.dumps({
            "context_id": self.context_id,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
            "step_logs": self.step_logs,
            "state": self.state,
            "metrics": self.metrics,
            "artifacts": self.artifacts
        }, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineContext':
        """Reconstruct a context from JSON."""
        data = json.loads(json_str)
        context = cls(
            metadata=data.get("metadata", {}),
            context_id=data.get("context_id")
        )
        context.step_logs = data.get("step_logs", [])
        context.state = data.get("state", {})
        context.metrics = data.get("metrics", {})
        context.artifacts = data.get("artifacts", {})
        context.creation_time = data.get("creation_time", context.creation_time)
        context.last_updated = data.get("last_updated", context.last_updated)
        return context
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the context state."""
        return {
            "context_id": self.context_id,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated,
            "steps_executed": len(self.step_logs),
            "state_entries": len(self.state),
            "metrics_tracked": list(self.metrics.keys()),
            "artifacts_registered": list(self.artifacts.keys()),
            "metadata": self.metadata
        }

    def get_metric(self, metric_name: str) -> Optional[float]:
        """
        Get the latest value for a specific metric.
        
        Args:
            metric_name: Name of the metric to retrieve
            
        Returns:
            The latest value of the metric, or None if not found
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        # Get the latest value (last entry in the list)
        return self.metrics[metric_name][-1]["value"]