"""Domain enums for model configuration."""
from enum import Enum

class ModelFormat(str, Enum):
    ONNX = "ONNX"
    UNKNOWN = "UNKNOWN"

class ModelTask(str, Enum):
    OBJECT_DETECTION = "OBJECT_DETECTION"
    CLASSIFICATION = "CLASSIFICATION"
    SEGMENTATION = "SEGMENTATION"
    POSE_ESTIMATION = "POSE_ESTIMATION"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    CUSTOM = "CUSTOM"
    UNKNOWN = "UNKNOWN"

class ModelStatus(str, Enum):
    UPLOADED = "UPLOADED"
    INSPECTING = "INSPECTING"
    CONFIGURATION_REQUIRED = "CONFIGURATION_REQUIRED"
    VALIDATING = "VALIDATING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"

class InputLayout(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    CHW = "CHW"
    HWC = "HWC"
    UNKNOWN = "UNKNOWN"

class ColorFormat(str, Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAYSCALE = "GRAYSCALE"
    RGBA = "RGBA"
    UNKNOWN = "UNKNOWN"

class ResizeMethod(str, Enum):
    LETTERBOX = "LETTERBOX"
    STRETCH = "STRETCH"
    CENTER_CROP = "CENTER_CROP"
    NONE = "NONE"
    CUSTOM = "CUSTOM"

class NormalizationType(str, Enum):
    NONE = "NONE"
    SCALE_0_1 = "SCALE_0_1"
    MEAN_STD = "MEAN_STD"
    IMAGENET = "IMAGENET"
    CUSTOM = "CUSTOM"
    UNKNOWN = "UNKNOWN"

class BoundingBoxFormat(str, Enum):
    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"
    UNKNOWN = "UNKNOWN"

class ConfidenceInterpretation(str, Enum):
    SIGMOID = "SIGMOID"
    SOFTMAX = "SOFTMAX"
    DIRECT = "DIRECT"
    UNKNOWN = "UNKNOWN"
