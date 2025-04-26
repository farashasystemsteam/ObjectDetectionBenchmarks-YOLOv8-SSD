import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

class SSDObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Load pre-trained SSD model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ssd300_vgg16(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Set confidence threshold
        self.confidence_threshold = confidence_threshold

        # COCO dataset class labels
        self.CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect(self, image):
        # Convert image to RGB if it's BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        image_pil = Image.fromarray(image)

        # Transform image
        image_tensor = self.transform(image_pil)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Get detections above confidence threshold
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Filter detections based on confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        return boxes, labels, scores

    def draw_detections(self, image, boxes, labels, scores):
        # Make a copy of the image
        image_copy = image.copy()

        # Draw each detection
        for box, label, score in zip(boxes, labels, scores):
            # Convert box coordinates to integers
            box = box.astype(int)

            # Draw rectangle
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Create label text
            label_text = f'{self.CLASSES[label]}: {score:.2f}'

            # Draw label
            cv2.putText(image_copy, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image_copy


class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image):
        # Run inference
        results = self.model(image)[0]

        # Extract boxes, confidence scores and class ids
        boxes = []
        scores = []
        class_ids = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.confidence_threshold:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(int(class_id))

        return np.array(boxes), np.array(class_ids), np.array(scores)

    def draw_detections(self, image, boxes, labels, scores):
        image_copy = image.copy()

        for box, label, score in zip(boxes, labels, scores):
            box = [int(x) for x in box]

            # Get class name from YOLO model's names dictionary
            class_name = self.model.names[label]

            # Draw rectangle
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Draw label and score
            label_text = f'{class_name}: {score:.2f}'
            cv2.putText(image_copy, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image_copy

def process_video(self, video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, scores = self.detect(frame)
        result_frame = self.draw_detections(frame, boxes, labels, scores)

        cv2.imshow('Object Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def webcam_detection(self):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, labels, scores = self.detect(frame)
        result_frame = self.draw_detections(frame, boxes, labels, scores)

        cv2.imshow('Webcam Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("\nChoose detector:")
    print("1. SSD Detector")
    print("2. YOLOv8 Detector")
    detector_choice = input("Enter your choice (1-2): ")

    # Initialize detector based on choice
    detector_name = ""
    if detector_choice == '1':
        detector_name = "SSD"
        detector = SSDObjectDetector(confidence_threshold=0.5)
    elif detector_choice == '2':
        detector_name = "YOLOv8"
        detector = YOLOv8Detector(confidence_threshold=0.5)
    else:
        print("Invalid choice. Exiting...")
        return

    while True:
        print("\nChoose detection mode:")
        print("1. Image detection")
        print("2. Video detection")
        print("3. Webcam detection")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")
        if choice == '1':
            # Image detection
            image_path = input("Enter image path (or press enter for default 'cars.png'): ").strip()
            if not image_path:
                image_path = "cars.png"

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image at {image_path}")
                continue

            # Perform detection
            boxes, labels, scores = detector.detect(image)

            # Draw detections
            result_image = detector.draw_detections(image, boxes, labels, scores)

            # Display results
            cv2.imshow('Object Detection Result of '+detector_name, result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif choice == '2':
            # Video detection
            video_path = input("Enter video path: ").strip()
            if not video_path:
                video_path = "cars.mp4"
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Could not open video at {video_path}")
                continue

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            output_path = 'output_video.mp4'
            out = cv2.VideoWriter(output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                boxes, labels, scores = detector.detect(frame)
                result_frame = detector.draw_detections(frame, boxes, labels, scores)

                out.write(result_frame)
                cv2.imshow('Video Detection', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Processed video saved as {output_path}")

        elif choice == '3':
            # Webcam detection
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Error: Could not open webcam")
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                boxes, labels, scores = detector.detect(frame)
                result_frame = detector.draw_detections(frame, boxes, labels, scores)

                cv2.imshow('Webcam Detection', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif choice == '4':
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
