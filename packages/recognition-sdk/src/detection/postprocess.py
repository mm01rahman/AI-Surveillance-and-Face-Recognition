import numpy as np
from  models import BoundingBox, FaceLandmarks
from .preprocess import LetterboxMetadata

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Standard Non-Maximum Suppression to filter overlapping bounding boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def map_and_clip_coordinates(
    box: np.ndarray, 
    kpss: np.ndarray, 
    score: float, 
    meta: LetterboxMetadata, 
    img_w: int, 
    img_h: int
) -> tuple[BoundingBox, FaceLandmarks]:
    """Maps coordinates back to the original image space and strictly clips them to boundaries."""
    
    def clip(val: float, maximum: float) -> float:
        return max(0.0, min(float(val), float(maximum)))

    # Un-pad, un-scale, and clip bounding box
    x1 = clip((box[0] - meta.pad_x) / meta.scale, img_w)
    y1 = clip((box[1] - meta.pad_y) / meta.scale, img_h)
    x2 = clip((box[2] - meta.pad_x) / meta.scale, img_w)
    y2 = clip((box[3] - meta.pad_y) / meta.scale, img_h)
    
    # Un-pad, un-scale, and clip landmarks
    mapped_kpss = [
        (clip((p[0] - meta.pad_x) / meta.scale, img_w), 
         clip((p[1] - meta.pad_y) / meta.scale, img_h)) 
        for p in kpss
    ]

    return (
        BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(score)),
        FaceLandmarks(
            left_eye=mapped_kpss[0],
            right_eye=mapped_kpss[1],
            nose=mapped_kpss[2],
            left_mouth=mapped_kpss[3],
            right_mouth=mapped_kpss[4]
        )
    )