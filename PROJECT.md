# UNO Vision Runtime

## Project Purpose

UNO Vision Runtime is an industrial edge-AI vision runtime designed to run on Arduino UNO Q hardware.

The system allows a customer or developer to:

1. Connect a USB camera.
2. Upload a custom AI model package.
3. Activate the model.
4. Trigger object detection.
5. Receive a stable JSON response.
6. Send PASS/FAIL results to industrial systems.

The system must be modular, reliable, and suitable for industrial edge deployment.

---

# Target Hardware

## Primary Hardware

* Arduino UNO Q
* Linux / Qualcomm QRB2210 compute side
* STM32 real-time MCU side
* USB camera
* Network connection

## Hardware Responsibilities

### Linux / QRB2210 Side

Responsible for:

* FastAPI backend
* Camera handling
* ONNX Runtime inference
* Model management
* REST API
* MQTT
* Database
* Logging
* System health
* Communication with STM32

### STM32 Side

Responsible for deterministic real-time I/O.

Inputs:

* Trigger
* Reset
* Enable

Outputs:

* Busy
* Pass
* Fail
* Error

The Linux application sends high-level commands and results to the STM32.

Do not place non-deterministic AI inference logic on the STM32.

---

# V1 Scope

V1 supports only:

* USB cameras
* Object detection
* ONNX models
* Custom model upload
* Model validation
* Model activation
* Camera inference
* Uploaded image inference
* REST API
* Standard JSON responses
* SQLite logging
* MQTT output
* STM32 integration
* GPIO trigger workflow
* PASS/FAIL decisions
* System health monitoring

---

# Explicitly Out of Scope for V1

Do not implement these unless specifically requested:

* Model training
* Segmentation
* OCR
* Face recognition
* Multiple AI frameworks
* TFLite
* TensorFlow
* PyTorch runtime
* Cloud synchronization
* Complex user management
* Kubernetes
* Multiple inference workers

Do not introduce unnecessary features.

---

# Technology Stack

Backend:

* Python 3.13
* FastAPI
* Uvicorn
* Pydantic
* OpenCV
* ONNX Runtime
* SQLite
* SQLAlchemy or equivalent lightweight ORM
* MQTT client

Frontend will be added later.

Do not add unnecessary dependencies.

---

# Core Architecture

The system architecture is:

Client
→ REST API
→ Vision Controller
→ Camera Service
→ Inference Service
→ Decision Engine
→ Result/Event Layer

The result/event layer may send results to:

* REST response
* MQTT
* SQLite
* STM32

---

# Architecture Rules

Use service-based architecture.

Recommended structure:

backend/

app/
main.py
config.py

```
api/
    health.py
    camera.py
    detect.py
    models.py
    system.py

services/
    camera_manager.py
    inference_engine.py
    model_manager.py
    decision_engine.py
    mqtt_service.py
    stm32_service.py

schemas/
    common.py
    detection.py
    model.py
    health.py

core/
    logging.py
    exceptions.py
    lifecycle.py

tests/
```

Do not put all application logic inside main.py.

Do not create circular imports.

Do not mix API route logic with AI inference logic.

---

# API Design

All APIs must use:

/api/v1/

Example endpoints:

GET /api/v1/health

GET /api/v1/camera/status
POST /api/v1/camera/start
POST /api/v1/camera/stop
GET /api/v1/camera/frame

POST /api/v1/detect
POST /api/v1/detect/image

POST /api/v1/models
GET /api/v1/models
GET /api/v1/models/{id}
POST /api/v1/models/{id}/activate
DELETE /api/v1/models/{id}

Use consistent HTTP status codes.

Use Pydantic response models.

Never return Python exceptions directly to API clients.

---

# Stable Detection Response

All object detection models must eventually produce a normalized internal detection format.

External model-specific output must never be exposed directly to clients.

Target structure:

{
"request_id": "string",
"success": true,
"timestamp": "ISO-8601 timestamp",

"model": {
"id": "string",
"version": "string"
},

"performance": {
"capture_ms": 0,
"preprocess_ms": 0,
"inference_ms": 0,
"postprocess_ms": 0,
"total_ms": 0
},

"inspection": {
"result": "PASS",
"reason": []
},

"objects": []
}

Do not change this contract without explicit approval.

---

# Model Package Standard

Custom models use:

.unomodel

Internally this is a ZIP package.

Expected structure:

model.unomodel

manifest.json
model.onnx
labels.json
rules.json
checksum.sha256

The package must contain data only.

Never execute scripts contained inside a model package.

Model validation must eventually include:

1. Safe ZIP extraction.
2. Path traversal protection.
3. Manifest validation.
4. File validation.
5. Checksum validation.
6. ONNX model validation.
7. ONNX Runtime session creation.
8. Dummy inference.
9. Input/output compatibility validation.

---

# ONNX Model Rules

V1 supports ONNX only.

Do not assume all YOLO ONNX models have identical output formats.

Inference must use model adapters.

Examples:

* yolo_v5_raw
* yolo_v8_raw
* yolo_v11_raw
* yolo_nms

The manifest determines which adapter is used.

Do not use large chains of model-specific if/else logic inside the main inference engine.

---

# Camera Rules

The Camera Manager must:

* Detect camera availability.
* Start the camera.
* Stop the camera.
* Capture frames.
* Detect disconnection.
* Support reconnection.
* Be thread-safe.
* Avoid opening multiple camera handles unnecessarily.

The API must not directly control OpenCV internals.

The API calls the Camera Manager.

---

# Inference Rules

V1 uses exactly one inference worker.

Inference execution must be serialized.

Use a queue or lock where necessary.

Do not allow uncontrolled parallel ONNX inference requests.

Model switching must be safe.

Preferred activation flow:

Validate new model
→ Load model
→ Warm up
→ Verify
→ Atomically switch active model

If activation fails:

Keep the previous model active.

---

# Decision Engine

The Decision Engine converts object detection into industrial inspection results.

Supported rules may include:

* Required objects.
* Minimum confidence.
* Expected object counts.
* Unexpected objects.

The inference engine must not contain industrial PASS/FAIL business logic.

Keep AI inference and inspection logic separate.

---

# Error Handling

Use structured application exceptions.

Each API error should return a consistent structure.

Example:

{
"success": false,
"error": {
"code": "CAMERA_NOT_AVAILABLE",
"message": "No active camera is available."
}
}

Never expose:

* Python stack traces
* Internal filesystem paths
* Secrets
* Raw system exceptions

to external clients.

---

# Logging

Use structured logging where practical.

Log important events:

* Application startup
* Camera start/stop
* Camera failure
* Model upload
* Model validation
* Model activation
* Inference errors
* STM32 communication errors

Do not log:

* Passwords
* Tokens
* Secrets

---

# Database

Use SQLite initially.

Potential entities:

* models
* model_activations
* detections
* inspection_results
* system_events
* errors
* performance_metrics

Do not store every captured image indefinitely.

Eventually implement image retention policies.

---

# Development Rules

Before implementing a task:

1. Inspect the current repository.
2. Understand existing architecture.
3. Reuse existing patterns.
4. Do not unnecessarily rewrite working files.
5. Keep changes limited to the requested scope.

After implementation:

1. Run formatting.
2. Run linting if configured.
3. Run tests.
4. Report modified files.
5. Report test results.
6. Report assumptions.
7. Report any remaining limitations.

---

# Code Quality

Code must be:

* Readable.
* Modular.
* Type hinted.
* Testable.
* Production-oriented.
* Conservative with dependencies.

Avoid:

* Giant functions.
* Giant classes.
* Hidden global state.
* Hardcoded paths.
* Hardcoded camera indexes without configuration.
* Silent exception handling.
* Broad exception catches unless re-raised appropriately.

---

# Configuration

Use environment-based configuration.

Provide:

.env.example

Do not commit secrets.

Configuration should eventually support:

* API host
* API port
* Camera index
* Camera resolution
* Model storage path
* Database path
* MQTT configuration
* Logging level
* STM32 communication port

---

# Testing Philosophy

Hardware-dependent components must be abstracted where possible.

Unit tests must not require:

* A real camera.
* A real STM32.
* A real ONNX model.

Hardware integration tests may be separate.

Use mocks for unit tests.

---

# Important Development Principle

Do not build the complete platform at once.

Implement and validate one vertical slice at a time.

Current development sequence:

1. Project foundation.
2. Health API.
3. Camera service.
4. Local ONNX inference.
5. Detection API.
6. Model package system.
7. Decision engine.
8. STM32 integration.
9. MQTT.
10. Dashboard.
11. Reliability and deployment.

The current task must be completed and tested before expanding scope.

When requirements are ambiguous, preserve the existing architecture and choose the simplest reliable solution.
