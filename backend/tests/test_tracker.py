import pytest
from app.services.tracker import SimpleTracker
from app.schemas.detection import DetectedObject, BoundingBox

def test_tracker_stationary():
    tracker = SimpleTracker()
    obj = DetectedObject(id=1, class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(x=10, y=10, width=50, height=50))
    tracker.update([obj], 1.0)
    preds = tracker.predict(1.5)
    assert len(preds) == 1
    assert preds[0].velocity_x == 0.0
    assert preds[0].bbox.x == 10

def test_tracker_moving():
    tracker = SimpleTracker()
    obj1 = DetectedObject(id=1, class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(x=10, y=10, width=50, height=50))
    tracker.update([obj1], 1.0)
    
    obj2 = DetectedObject(id=1, class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(x=20, y=10, width=50, height=50))
    tracker.update([obj2], 2.0)
    
    preds = tracker.predict(2.5)
    assert len(preds) == 1
    # dt is 1.0s, dx is 10. velocity = 10 * 0.7 = 7
    # pred is 20 + 7 * 0.5 = 23.5
    assert preds[0].velocity_x > 0.0
    assert preds[0].bbox.x == 23.5
