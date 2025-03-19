import re
import ast


def grounding_iou_score(predict_str: str, ground_truth: tuple[float, float]) -> float:
    print(ground_truth)
    def calculate_iou(start1, end1, start2, end2):
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        iou = intersection / union if union != 0 else 0
        return iou
    
    def extract_time_range(sentence):
        match = re.search(r"(\d+\.\d+)\s*-\s*(\d+\.\d+)\s*", sentence)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            return start_time, end_time
        else:
            return None, None
    
    pred_start, pred_end = extract_time_range(predict_str)
    gt_start, gt_end = ground_truth
    reward = 0.0
    if pred_start is not None and pred_end is not None:
        reward = calculate_iou(gt_start, gt_end, pred_start, pred_end) * 10 + 1
    
    return reward
    



