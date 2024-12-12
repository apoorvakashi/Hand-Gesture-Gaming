import time
import csv
from datetime import datetime
from time import sleep
import cv2
import mediapipe as mp
import pyautogui
import argparse
import numpy as np
from src.config import RED, WHITE, BLUE, GREEN

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hand gesture control using TensorFlow and OpenCV.")
    parser.add_argument("--width", type=int, default=640, help="Screen width")
    parser.add_argument("--height", type=int, default=480, help="Screen height")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for score")
    parser.add_argument("--alpha", type=float, default=0.3, help="Transparent level")
    parser.add_argument("-eval", action="store_true", help="Enable latency evaluation")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    latency_data = [] if args.eval else None

    while cap.isOpened():
        start_time = time.time()  # Start time for frame capture

        ret, frame = cap.read()
        if not ret:
            break

        capture_time = time.time() - start_time if args.eval else None

        # Resize the frame to reduce resolution
        frame = cv2.resize(frame, (640, 480))

        # Flip and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create transparent overlay for division
        overlay = frame.copy()

        # Apply transparency (alpha blending)
        cv2.rectangle(overlay, (0, 0), (int(args.width / 3), args.height), RED, -1)
        cv2.rectangle(overlay, (int(args.width / 3), 0), (int(2 * args.width / 3), args.height), GREEN, -1)
        cv2.rectangle(overlay, (int(2 * args.width / 3), 0), (args.width, args.height), BLUE, -1)

        frame = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)

        processing_start_time = time.time()
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                def is_open():
                    return all(
                        abs(wrist.y - landmark.y) > 0.2
                        for landmark in [index_tip, middle_tip, ring_tip, pinky_tip]
                    )

                x_pos = int((thumb_tip.x + pinky_tip.x) / 2 * args.width)

                if is_open():
                    if x_pos <= args.width / 3:
                        pyautogui.keyDown('left')
                        pyautogui.keyDown('up')
                        # sleep(0.3)
                        pyautogui.keyUp('left')
                        pyautogui.keyUp('up')
                        text = "Jump Left"
                    elif x_pos > args.width / 3 and x_pos <= 2 * args.width / 3:
                        pyautogui.keyDown('up')
                        # sleep(0.2)
                        pyautogui.keyUp('up')
                        text = "Jump"
                    else:
                        pyautogui.keyDown('right')
                        pyautogui.keyDown('up')
                        # sleep(0.3)
                        pyautogui.keyUp('right')
                        pyautogui.keyUp('up')
                        text = "Jump Right"
                else:
                    if x_pos <= args.width / 3:
                        pyautogui.keyDown('left')
                        # sleep(0.2)
                        pyautogui.keyUp('left')
                        text = "Run Left"
                    elif x_pos > args.width / 3 and x_pos <= 2 * args.width / 3:
                        text = "Stay"
                    else:
                        pyautogui.keyDown('right')
                        # sleep(0.2)
                        pyautogui.keyUp('right')
                        text = "Run Right"

                cv2.putText(frame, text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        processing_time = time.time() - processing_start_time if args.eval else None

        render_start_time = time.time()
        cv2.imshow('Hand Gesture Recognition', frame)
        render_time = time.time() - render_start_time if args.eval else None

        total_time = time.time() - start_time if args.eval else None

        if args.eval:
            latency_data.append({
                "capture_time": capture_time,
                "processing_time": processing_time,
                "render_time": render_time,
                "total_time": total_time
            })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.eval and latency_data:
        filename = f"latency_logs/Mediapipe/latency_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["capture_time", "processing_time", "render_time", "total_time"])
            writer.writeheader()
            writer.writerows(latency_data)
        print(f"Latency metrics saved to {filename}")

if __name__ == '__main__':
    main()
