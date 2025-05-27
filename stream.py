import cv2
import torch
import time
import os
from rfdetr import RFDETRBase
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2

from transformer import FireDetectionTransformer
from extract_features import FeatureExtractor

# Fixed resolution (Full HD)
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080  # Fixed typo from TARGET_HEIGHT

def crop_bbox(frame, bbox):
    """Crop a region from frame defined by bbox coordinates with boundary checks"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    return frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def resize_with_aspect(frame, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """Resize frame with aspect ratio preservation and padding"""
    h, w = frame.shape[:2]
    scale = min(target_width/w, target_height/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Add padding if needed
    delta_w = target_width - new_w
    delta_h = target_height - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))

def process_video(video_path, device, detection_threshold=0.3):
    detector = RFDETRBase(pretrain_weights="models/best_total_rf-detr50.pth", device=device)
    transformer = FireDetectionTransformer(model_filename="models/transformer_model_10.pth", device=device)
    extractor = FeatureExtractor(device=device)
    transformer.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Extract filename without extension
    video_name = Path(video_path).stem
    
    # Create output path
    output_path = f'output/output_with_probability_{video_name}.mp4'

    # Use fixed resolution for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (TARGET_WIDTH, TARGET_HEIGHT))

    frame_idx = 0
    collecting = False
    buffer = []
    target_bbox = None
    frames_to_collect = 30
    frame_step = 5
    current_step = 0
    last_prob = None
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS with zero division protection
        current_time = time.time()
        time_diff = current_time - prev_time
        if time_diff > 0:
            fps = 1 / time_diff
        prev_time = current_time

        if not collecting:
            detections = detector.predict(frame[:, :, ::-1].copy(), threshold=detection_threshold)
            fire_detected = any(
                class_id in [0, 1] and confidence >= detection_threshold
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            )

            if fire_detected:
                valid_indices = [i for i, (class_id, confidence) in 
                                enumerate(zip(detections.class_id, detections.confidence))
                                if class_id in [0, 1] and confidence >= detection_threshold]

                if valid_indices:
                    max_conf_idx = max(valid_indices, key=lambda i: detections.confidence[i])
                    collecting = True
                    target_bbox = detections.xyxy[max_conf_idx]
                    buffer = []
                    current_step = 0

        else:
            if current_step % frame_step == 0:
                cropped = crop_bbox(frame, target_bbox)
                if cropped is not None:
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    transformed = transform(image=cropped)
                    cropped_tensor = transformed['image'].to(device)
                    buffer.append(cropped_tensor)
            
            current_step += 1
            
            if len(buffer) >= frames_to_collect:
                frames_tensor = torch.stack(buffer[:frames_to_collect]).unsqueeze(0)
                features = extractor.extract_features_batch(frames_tensor)
                prob = transformer(features).item()
                last_prob = prob
                collecting = False
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # Resize frame to fixed resolution
        resized_frame = resize_with_aspect(frame)

        # Visualization on resized frame
        if collecting:
            # Scale bbox coordinates for new size
            h, w = frame.shape[:2]
            scale = min(TARGET_WIDTH/w, TARGET_HEIGHT/h)
            x1, y1, x2, y2 = [int(coord * scale) for coord in target_bbox]
            x1 += (TARGET_WIDTH - int(w * scale)) // 2
            x2 += (TARGET_WIDTH - int(w * scale)) // 2
            y1 += (TARGET_HEIGHT - int(h * scale)) // 2
            y2 += (TARGET_HEIGHT - int(h * scale)) // 2
            
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(resized_frame, "ANALYZING...", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display FPS (top-left)
        cv2.putText(resized_frame, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display probability (bottom-left)
        if last_prob is not None:
            cv2.putText(resized_frame, f"Fire: {last_prob*100:.1f}%", (20, TARGET_HEIGHT-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        out.write(resized_frame)
        cv2.imshow("Fire Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_video("videos/smoke1.avi", device)