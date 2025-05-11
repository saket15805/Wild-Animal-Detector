
import cv2
import time
import requests
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from collections import deque
import os

# === CONFIG ===
TELEGRAM_BOT_TOKEN = '7731179635:AAFCfaA-ELFhzUToOV8tgtp_-YwCjDzA084'
TELEGRAM_CHAT_ID = '1315050812'
WILD_ANIMALS = ['elephant', 'deer', 'boar', 'cow', 'bear', 'horse']
ALERT_DELAY = 5  # Minimum time between alerts (in seconds)
SEQUENCE_LENGTH = 10
USE_RNN = False  # Set to True after training your RNN
MODEL_PATH = Path('yolov8s.pt')
VIDEO_PATH = r'C:\Users\saket\OneDrive\Documents\DL(cl)\Horse.mp4'

# === LOAD MODEL ===
print("[INFO] Loading YOLO model...")
model = YOLO(str(MODEL_PATH))

# === RNN CLASS ===
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# === INIT RNN ===
input_size = len(WILD_ANIMALS)
rnn = SimpleGRU(input_size, hidden_size=16, output_size=1)
rnn.eval()  # RNN is not trained yet

# === SEND ALERT ===
def send_telegram_alert(animal_class: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram credentials are missing!")
        return
    message = f"🚨 Alert! A wild *{animal_class}* has been detected on the farm!"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload)
        response_data = response.json()
        if response.status_code == 200:
            print(f"[{time.ctime()}] ✅ Alert sent: {animal_class}")
        else:
            print(f"[{time.ctime()}] ❌ Telegram error: {response_data.get('description', 'Unknown error')}")
    except Exception as e:
        print(f"[{time.ctime()}] ⚠️ Telegram request failed: {e}")

# === HELPERS ===
history = deque(maxlen=SEQUENCE_LENGTH)
last_alert_time = 0

def one_hot_vector(animal):
    vec = [0] * len(WILD_ANIMALS)
    if animal in WILD_ANIMALS:
        vec[WILD_ANIMALS.index(animal)] = 1
    return vec

# === MAIN ===
def main():
    global last_alert_time
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
        return

    print(f"[INFO] Processing video: {VIDEO_PATH}")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video reached.")
            break

        results = model(frame[..., ::-1])
        annotated = results[0].plot()
        cv2.imshow("Wild Animal Detection", annotated)

        detected_this_frame = []
        for result in results:
            for box in result.boxes.data:
                cls_id = int(box[5])
                cls_name = model.names[cls_id]
                if cls_name in WILD_ANIMALS:
                    detected_this_frame.append(cls_name)
                    print(f"[{time.ctime()}] Detected: {cls_name}")

        if detected_this_frame:
            history.append(one_hot_vector(detected_this_frame[0]))
        else:
            history.append([0] * len(WILD_ANIMALS))

        if len(history) == SEQUENCE_LENGTH:
            if USE_RNN:
                seq_tensor = torch.tensor([history], dtype=torch.float32)
                with torch.no_grad():
                    prediction = torch.sigmoid(rnn(seq_tensor)).item()
                print(f"[DEBUG] RNN prediction score: {prediction:.2f}")
                if prediction > 0.7 and detected_this_frame:
                    if time.time() - last_alert_time > ALERT_DELAY:
                        send_telegram_alert(detected_this_frame[0])
                        last_alert_time = time.time()
            else:
                if detected_this_frame and (time.time() - last_alert_time > ALERT_DELAY):
                    print(f"[DEBUG] Sending alert without RNN: {detected_this_frame[0]}")
                    send_telegram_alert(detected_this_frame[0])
                    last_alert_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
