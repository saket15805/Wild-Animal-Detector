import cv2
import time
import requests
import torch
from ultralytics import YOLO
from pathlib import Path
import os

# === CONFIG ===
TELEGRAM_BOT_TOKEN = '7731179635:AAFCfaA-ELFhzUToOV8tgtp_-YwCjDzA084'
TELEGRAM_CHAT_ID = '1315050812'
WILD_ANIMALS = ['elephant', 'deer', 'boar', 'cow', 'bear', 'horse']
ALERT_DELAY = 5  # Minimum time between alerts (in seconds)
MODEL_PATH = Path('yolov8s.pt')
VIDEO_PATH = r'C:\Users\saket\OneDrive\Documents\DL(cl)\animal-detc\WhatsApp Video 2025-05-14 at 12.21.03_bcc212d9.mp4'

# === LOAD YOLO MODEL ===
print('[INFO] Loading YOLO model...')
model = YOLO(str(MODEL_PATH))
print('[INFO] YOLO model loaded successfully.')

# === SEND ALERT ===
def send_telegram_alert(animal_class: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('[ERROR] Telegram credentials are missing!')
        return
    message = f'🚨 Alert! A wild *{animal_class}* has been detected on the farm!'
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload)
        response_data = response.json()
        if response.status_code == 200:
            print(f'[{time.ctime()}] ✅ Alert sent: {animal_class}')
        else:
            print(f'[{time.ctime()}] ❌ Telegram error: {response_data.get('description', 'Unknown error')}')
    except Exception as e:
        print(f'[{time.ctime()}] ⚠️ Telegram request failed: {e}')

# === ACCURACY METRICS ===
true_positives = 0
false_positives = 0
false_negatives = 0
total_frames = 0

# === MAIN ===
def main():
    global true_positives, false_positives, false_negatives, total_frames
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'[ERROR] Could not open video file: {VIDEO_PATH}')
        return

    print(f'[INFO] Processing video: {VIDEO_PATH}')
    print('[INFO] Press q to quit.')

    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[INFO] End of video reached.')
            break

        results = model.predict(frame[..., ::-1])
        annotated = results[0].plot()
        cv2.imshow('Wild Animal Detection', annotated)

        detected_this_frame = set()
        for box in results[0].boxes.data:
            cls_id = int(box[5])
            cls_name = model.names[cls_id]
            if cls_name in WILD_ANIMALS:
                detected_this_frame.add(cls_name)
                print(f'[{time.ctime()}] Detected: {cls_name}')

        # Placeholder for ground truth (replace with actual labels if available)
        ground_truth = set(["horse"])  # Replace with actual ground truth for each frame

        # Calculate accuracy
        true_positives += len(detected_this_frame & ground_truth)
        false_positives += len(detected_this_frame - ground_truth)
        false_negatives += len(ground_truth - detected_this_frame)
        total_frames += 1

        if detected_this_frame and (time.time() - last_alert_time > ALERT_DELAY):
            print(f'[DEBUG] Sending alert for: {list(detected_this_frame)[0]}')
            send_telegram_alert(list(detected_this_frame)[0])
            last_alert_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('[INFO] Quitting.')
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final accuracy metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print('\n[INFO] Final Accuracy Metrics:')
    print(f'Total Frames: {total_frames}')
    print(f'True Positives: {true_positives}')
    print(f'False Positives: {false_positives}')
    print(f'False Negatives: {false_negatives}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')

if __name__ == '__main__':
    main()
