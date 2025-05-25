import cv2
import torch
from rfdetr import RFDETRBase
import albumentations as A
from albumentations.pytorch import ToTensorV2

from training_transformer.transformer import FireDetectionTransformer
from training_transformer.extract_features import FeatureExtractor

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

def process_video(video_path, device, detection_threshold=0.3):
    detector = RFDETRBase(pretrain_weights="models/best_total_rf-detr50.pth", device=device)
    transformer = FireDetectionTransformer(model_filename="models/transformer_model_10.pth", device=device)
    extractor = FeatureExtractor(device=device)
    transformer.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    frame_idx = 0
    collecting = False
    buffer = []
    target_bbox = None
    frames_to_collect = 30
    frame_step = 5
    current_step = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not collecting:
            # Detection mode: look for any fire/smoke bounding box
            detections = detector.predict(frame[:, :, ::-1].copy(), threshold=detection_threshold)
            fire_detected = any(
                class_id in [0, 1] and confidence >= detection_threshold
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            )

            if fire_detected:
                # Find the detection with highest confidence among fire/smoke detections
                valid_indices = [i for i, (class_id, confidence) in 
                                enumerate(zip(detections.class_id, detections.confidence))
                                if class_id in [0, 1] and confidence >= detection_threshold]

                if valid_indices:  # If we have any valid detections
                    # Get index of detection with maximum confidence
                    max_conf_idx = max(valid_indices, key=lambda i: detections.confidence[i])

                    collecting = True
                    target_bbox = detections.xyxy[max_conf_idx]  # Use bbox with highest confidence
                    buffer = []
                    current_step = 0
                    print(f"Object detected! Starting frame collection from frame {frame_idx}")

        else:
            # Frame collection mode
            if current_step % frame_step == 0:
                cropped = crop_bbox(frame, target_bbox)
                if cropped is not None:
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    # Apply preprocessing transformations
                    transformed = transform(image=cropped)
                    cropped_tensor = transformed['image'].to(device)  # [3, 224, 224]
                    buffer.append(cropped_tensor)
            
            current_step += 1
            
            # When enough frames collected, process them
            if len(buffer) >= frames_to_collect:
                frames_tensor = torch.stack(buffer[:frames_to_collect]).unsqueeze(0)
                
                features = extractor.extract_features_batch(frames_tensor)  # [1, 30, 1280]
                prob = transformer(features).item()
                
                print(f"Result: fire probability = {prob:.4f}")
                collecting = False
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # Visualization (red rectangle around analyzed bbox)
        if collecting:
            x1, y1, x2, y2 = map(int, target_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "ANALYZING...", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Fire Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    process_video("videos/boom.mp4", device)