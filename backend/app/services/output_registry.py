"""Output processor registry."""

from pydantic import BaseModel
from typing import Any
from app.domain.enums import ModelTask
from app.services.output_processors.base import OutputProcessor

class OutputProcessorDefinition(BaseModel):
    """Defines a supported output processor."""
    name: str
    supported_tasks: list[ModelTask]
    required_configuration: list[str]

class OutputProcessorRegistry:
    """Registry for available output post-processors."""
    def __init__(self) -> None:
        self._processors: dict[str, OutputProcessorDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(OutputProcessorDefinition(
            name="UNKNOWN",
            supported_tasks=[],
            required_configuration=[]
        ))
        self.register(OutputProcessorDefinition(
            name="YOLO",
            supported_tasks=[ModelTask.OBJECT_DETECTION],
            required_configuration=["bbox_format", "confidence_threshold"]
        ))
        self.register(OutputProcessorDefinition(
            name="SSD",
            supported_tasks=[ModelTask.OBJECT_DETECTION],
            required_configuration=["bbox_format"]
        ))
        self.register(OutputProcessorDefinition(
            name="CLASSIFICATION_LOGITS",
            supported_tasks=[ModelTask.CLASSIFICATION],
            required_configuration=["confidence_interpretation"]
        ))
        self.register(OutputProcessorDefinition(
            name="CLASSIFICATION_PROBABILITIES",
            supported_tasks=[ModelTask.CLASSIFICATION],
            required_configuration=[]
        ))

    def register(self, processor: OutputProcessorDefinition) -> None:
        """Register a new output processor definition."""
        self._processors[processor.name] = processor

    def get(self, name: str) -> OutputProcessorDefinition | None:
        """Retrieve a processor definition by name."""
        return self._processors.get(name)

    def create_processor(self, name: str) -> OutputProcessor | None:
        """Create a concrete instance of the processor."""
        if name == "YOLO":
            from app.services.output_processors.yolo import YOLOOutputProcessor
            return YOLOOutputProcessor()
        # Fallback for others unimplemented
        return None

    def list_available(self) -> list[str]:
        """List all available processor names."""
        return list(self._processors.keys())
        
    def validate_profile(self, processor_name: str, task: ModelTask) -> list[str]:
        """Validate if the given processor supports the selected task.
        
        Returns a list of error strings. Returns empty list if valid.
        """
        proc = self.get(processor_name)
        if not proc:
            return [f"Unknown output processor: {processor_name}"]
        if processor_name == "UNKNOWN":
            return ["Output processor is UNKNOWN."]
        if task not in proc.supported_tasks:
            return [f"Processor {processor_name} does not support task {task.value}"]
        return []

# Singleton instance
output_registry = OutputProcessorRegistry()
