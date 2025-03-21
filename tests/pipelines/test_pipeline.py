"""
Unit tests for the Marovi pipeline framework.

Tests pipeline components, steps, and end-to-end execution.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to sys.path to import from marovipipelines
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from marovipipelines.pipelines.base import (
    PipelineContext, 
    PipelineStep, 
    ProcessingPipeline,
    ConditionalStep,
    HumanReviewStep
)


class MockStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """Mock step for testing that adds a field to each input."""
    
    def __init__(self, field_name: str, field_value: Any):
        super().__init__()
        self.field_name = field_name
        self.field_value = field_value
    
    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        results = []
        for item in inputs:
            result = item.copy()
            result[self.field_name] = self.field_value
            results.append(result)
        return results


class FailingStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """Step that fails a certain number of times then succeeds."""
    
    def __init__(self, fail_count: int = 2):
        super().__init__(max_retries=3)
        self.fail_count = fail_count
        self.attempts = 0
    
    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ValueError(f"Simulated failure (attempt {self.attempts})")
        return inputs


class TestPipelineContext(unittest.TestCase):
    """Tests for the PipelineContext class."""
    
    def test_init(self):
        """Test context initialization."""
        context = PipelineContext(metadata={"test": "value"})
        self.assertEqual(context.metadata, {"test": "value"})
        self.assertEqual(context.step_logs, [])
        self.assertEqual(context.state, {})
    
    def test_log_step(self):
        """Test step logging."""
        context = PipelineContext()
        context.log_step("TestStep", ["input"], ["output"], {"extra": "data"})
        
        self.assertEqual(len(context.step_logs), 1)
        log = context.step_logs[0]
        self.assertEqual(log["step"], "TestStep")
        self.assertEqual(log["input"], ["input"])
        self.assertEqual(log["output"], ["output"])
        self.assertEqual(log["extra"], {"extra": "data"})
    
    def test_serialization(self):
        """Test context serialization and deserialization."""
        context = PipelineContext(metadata={"test": "value"})
        context.log_step("TestStep", ["input"], ["output"])
        
        # Serialize
        json_str = context.to_json()
        
        # Deserialize
        new_context = PipelineContext.from_json(json_str)
        
        self.assertEqual(new_context.metadata, context.metadata)
        self.assertEqual(len(new_context.step_logs), len(context.step_logs))
        self.assertEqual(new_context.step_logs[0]["step"], "TestStep")


class TestPipelineStep(unittest.TestCase):
    """Tests for the PipelineStep class."""
    
    def test_preprocess_postprocess(self):
        """Test that preprocess and postprocess are called."""
        
        class TestStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
            def preprocess(self, inputs, context):
                for item in inputs:
                    item["preprocessed"] = True
                return inputs
                
            def postprocess(self, outputs, context):
                for item in outputs:
                    item["postprocessed"] = True
                return outputs
                
            def process(self, inputs, context):
                return inputs
        
        step = TestStep()
        context = PipelineContext()
        inputs = [{"data": 1}]
        
        outputs = step.run_with_retries(inputs, context)
        
        self.assertTrue(outputs[0]["preprocessed"])
        self.assertTrue(outputs[0]["postprocessed"])
    
    def test_retry_logic(self):
        """Test retry logic when a step fails."""
        step = FailingStep(fail_count=2)
        context = PipelineContext()
        inputs = [{"data": 1}]
        
        # Should succeed on the third attempt
        outputs = step.run_with_retries(inputs, context)
        
        self.assertEqual(outputs, inputs)
        self.assertEqual(step.attempts, 3)
        self.assertEqual(len(context.step_logs), 3)
        
        # First two attempts should have error info
        self.assertIn("error", context.step_logs[0]["extra"])
        self.assertIn("error", context.step_logs[1]["extra"])
        
        # Last attempt should have succeeded
        self.assertEqual(context.step_logs[2]["output"], inputs)
    
    def test_max_retries_exhausted(self):
        """Test that exception is raised when max retries are exhausted."""
        step = FailingStep(fail_count=4)  # More failures than allowed retries
        context = PipelineContext()
        inputs = [{"data": 1}]
        
        with self.assertRaises(RuntimeError):
            step.run_with_retries(inputs, context)
        
        self.assertEqual(step.attempts, 3)  # Only tried 3 times (max_retries)
        self.assertEqual(len(context.step_logs), 3)


class TestProcessingPipeline(unittest.TestCase):
    """Tests for the ProcessingPipeline class."""
    
    def test_pipeline_execution(self):
        """Test basic pipeline execution."""
        # Create steps
        step1 = MockStep("step1", "value1")
        step2 = MockStep("step2", "value2")
        
        # Create pipeline
        pipeline = ProcessingPipeline(steps=[step1, step2], name="test_pipeline")
        
        # Create context
        context = PipelineContext()
        
        # Run pipeline
        inputs = [{"initial": "data"}]
        outputs = pipeline.run(inputs, context)
        
        # Check outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["step1"], "value1")
        self.assertEqual(outputs[0]["step2"], "value2")
        
        # Check logs
        self.assertEqual(len(context.step_logs), 2)
        self.assertEqual(context.step_logs[0]["step"], "MockStep")
        self.assertEqual(context.step_logs[1]["step"], "MockStep")
    
    def test_checkpointing(self):
        """Test checkpoint creation and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create steps
            step1 = MockStep("step1", "value1")
            step2 = MockStep("step2", "value2")
            
            # Create pipeline with checkpoint dir
            pipeline = ProcessingPipeline(
                steps=[step1, step2], 
                name="test_pipeline",
                checkpoint_dir=temp_dir
            )
            
            # Create context
            context = PipelineContext(checkpoint_dir=temp_dir)
            
            # Run pipeline
            inputs = [{"initial": "data"}]
            pipeline.run(inputs, context)
            
            # Check checkpoint files were created
            checkpoint_files = list(Path(temp_dir).glob("*.json"))
            self.assertEqual(len(checkpoint_files), 2)
            
            # Try loading a checkpoint
            checkpoint_data = pipeline.load_checkpoint("MockStep")
            self.assertEqual(checkpoint_data["step"], "MockStep")
            self.assertIn("state", checkpoint_data)
    
    def test_resume_from_step(self):
        """Test resuming pipeline from a specific step."""
        # Create steps
        step1 = MockStep("step1", "value1")
        step2 = MockStep("step2", "value2")
        step3 = MockStep("step3", "value3")
        
        # Give steps distinct class names for resume to work
        step1.__class__.__name__ = "Step1"
        step2.__class__.__name__ = "Step2"
        step3.__class__.__name__ = "Step3"
        
        # Create pipeline
        pipeline = ProcessingPipeline(steps=[step1, step2, step3], name="test_pipeline")
        
        # Create context
        context = PipelineContext()
        
        # Run pipeline with resume from step2
        inputs = [{"initial": "data"}]
        outputs = pipeline.run(inputs, context, resume_from="Step2")
        
        # Check step1 was skipped
        self.assertEqual(len(outputs), 1)
        self.assertNotIn("step1", outputs[0])
        self.assertEqual(outputs[0]["step2"], "value2")
        self.assertEqual(outputs[0]["step3"], "value3")
        
        # Check logs
        self.assertEqual(len(context.step_logs), 2)  # Only 2 steps executed


class TestConditionalStep(unittest.TestCase):
    """Tests for the ConditionalStep class."""
    
    def test_conditional_branching(self):
        """Test conditional routing of items."""
        # Create branches
        true_branch = MockStep("branch", "true")
        false_branch = MockStep("branch", "false")
        
        # Create condition function
        def condition(item):
            return item.get("value", 0) > 5
        
        # Create conditional step
        conditional = ConditionalStep(condition, true_branch, false_branch)
        
        # Create context
        context = PipelineContext()
        
        # Run with mixed inputs
        inputs = [
            {"id": 1, "value": 10},  # Should go to true branch
            {"id": 2, "value": 3}    # Should go to false branch
        ]
        
        outputs = conditional.run_with_retries(inputs, context)
        
        # Check routing
        self.assertEqual(len(outputs), 2)
        
        # Find outputs by id
        output1 = next(o for o in outputs if o["id"] == 1)
        output2 = next(o for o in outputs if o["id"] == 2)
        
        self.assertEqual(output1["branch"], "true")
        self.assertEqual(output2["branch"], "false")


class TestHumanReviewStep(unittest.TestCase):
    """Tests for the HumanReviewStep class."""
    
    def test_human_review(self):
        """Test human review step."""
        # Create review step
        review_step = HumanReviewStep(review_prompt="Test review")
        
        # Create context
        context = PipelineContext()
        
        # Run review
        inputs = [{"id": 1, "data": "review this"}]
        outputs = review_step.run_with_retries(inputs, context)
        
        # Check outputs (should be same as inputs)
        self.assertEqual(outputs, inputs)
        
        # Check metadata was added
        self.assertIn("human_review", context.metadata)
        self.assertEqual(context.metadata["human_review"]["items_reviewed"], 1)


if __name__ == "__main__":
    unittest.main() 