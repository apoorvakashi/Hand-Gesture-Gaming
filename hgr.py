# import cv2
# import mediapipe as mp
# import pyautogui
# import argparse


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Hand gesture control using TensorFlow and OpenCV.")
#     parser.add_argument("--width", type=int, default=640, help="Screen width")
#     parser.add_argument("--height", type=int, default=480, help="Screen height")
#     parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for score")
#     parser.add_argument("--alpha", type=float, default=0.3, help="Transparent level")
#     return parser.parse_args()


# def main():
#     args = parse_arguments()
#     # Initialize MediaPipe Hands
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
#     mp_drawing = mp.solutions.drawing_utils

#     # Open webcam
#     cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Resize the frame to reduce resolution
#         frame = cv2.resize(frame, (640, 480))
        
#         # Flip and convert to RGB
#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


#         # Process the frame
#         result = hands.process(rgb_frame)

#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 # Draw landmarks
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Calculate landmarks
#                 thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#                 index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#                 ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
#                 pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
#                 wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

#                 # Define conditions for open and closed gestures
#                 # Example: Measure vertical distance from fingertips to wrist
#                 def is_closed():
#                     return all(
#                         abs(wrist.y - landmark.y) < 0.3
#                         for landmark in [index_tip, middle_tip, ring_tip, pinky_tip]
#                     )

#                 if is_closed():
#                     pyautogui.press('down')  # Simulate "down arrow"
#                     cv2.putText(frame, "CLOSED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 else:
#                     pyautogui.press('up')  # Simulate "up arrow"
#                     cv2.putText(frame, "OPEN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Display the frame
#         cv2.imshow('Hand Gesture Recognition', frame)

#         # Exit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import pyautogui
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hand gesture control using TensorFlow and OpenCV.")
    parser.add_argument("--width", type=int, default=640, help="Screen width")
    parser.add_argument("--height", type=int, default=480, help="Screen height")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for score")
    parser.add_argument("--alpha", type=float, default=0.3, help="Transparent level")
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to reduce resolution
        frame = cv2.resize(frame, (640, 480))
        
        # Flip and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate landmarks
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Define conditions for open and closed gestures
                # Open hand is when the distance between wrist and fingertips is large
                def is_open():
                    return all(
                        abs(wrist.y - landmark.y) > 0.2
                        for landmark in [index_tip, middle_tip, ring_tip, pinky_tip]
                    )

                # Get horizontal position of the hand (x-coordinate)
                x_pos = int((thumb_tip.x + pinky_tip.x) / 2 * args.width)

                # Gesture detection logic
                if is_open():
                    if x_pos <= args.width / 3:
                        pyautogui.keyDown('left')
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('left')
                        pyautogui.keyUp('up')
                        text = "Jump Left"
                    elif x_pos > args.width / 3 and x_pos <= 2 * args.width / 3:
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('up')
                        text = "Jump"
                    else:
                        pyautogui.keyDown('right')
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('right')
                        pyautogui.keyUp('up')
                        text = "Jump Right"
                else:
                    if x_pos <= args.width / 3:
                        pyautogui.keyDown('left')
                        pyautogui.keyUp('left')
                        text = "Run Left"
                    elif x_pos > args.width / 3 and x_pos <= 2 * args.width / 3:
                        text = "Stay"
                    else:
                        pyautogui.keyDown('right')
                        pyautogui.keyUp('right')
                        text = "Run Right"

                # Display gesture action on the frame
                cv2.putText(frame, "{}".format(text), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
