import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load the tuktuk detection model
model = YOLO('best.pt')

# Load image
img_path = 'img/input.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

# Run inference
results = model(img)

# Process results
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = box.cls[0].cpu().numpy()
            
            # Only show detections with confidence > 50%
            if confidence > 0.5:
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f'tuktuk: {confidence:.2f}'
                cv2.putText(img, label, (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save result
output_path = 'img/output.jpg'
cv2.imwrite(output_path, img)
print(f"Detection completed. Result saved to {output_path}")

# Display result (optional)
cv2.imshow('Tuktuk Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()