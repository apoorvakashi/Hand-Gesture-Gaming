import tensorflow as tf
import cv2
import pyautogui
import argparse
from src.utils import load_graph, detect_hands, predict
from src.config import ORANGE, RED, GREEN
from time import sleep

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hand gesture control using TensorFlow and OpenCV.")
    parser.add_argument("--width", type=int, default=640, help="Screen width")
    parser.add_argument("--height", type=int, default=480, help="Screen height")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for score")
    parser.add_argument("--alpha", type=float, default=0.3, help="Transparent level")
    parser.add_argument("--pre_trained_model_path", type=str, default="src/pretrained_model.pb", help="Path to pre-trained model")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    graph, sess = load_graph(args.pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    while True:
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, args.threshold, args.width, args.height)

        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)

            text = ""
            if category == "Open" and x <= args.width / 3:
                pyautogui.keyDown('left')  
                pyautogui.keyDown('up')      
                sleep(0.4)            
                pyautogui.keyUp('left')
                pyautogui.keyUp('up') 
                text = "Jump left"
            elif category == "Closed" and x <= args.width / 3:
                pyautogui.keyDown('left') 
                sleep(0.1) 
                pyautogui.keyUp('left')
                text = "Run left"
                
            elif category == "Open" and args.width / 3 < x <= 2 * args.width / 3:
                pyautogui.keyDown('up') 
                sleep(0.4) 
                pyautogui.keyUp('up')  
                text = "Jump"
            elif category == "Closed" and args.width / 3 < x <= 2 * args.width / 3:
                text = "Stay" 
            elif category == "Open" and x > 2 * args.width / 3:
                pyautogui.keyDown('right')  
                pyautogui.keyDown('up')      
                sleep(0.4)           
                pyautogui.keyUp('right')
                pyautogui.keyUp('up')
                text = "Jump right"
            elif category == "Closed" and x > 2 * args.width / 3:
                pyautogui.keyDown('right')
                sleep(0.1) 
                pyautogui.keyUp('right')
                text = "Run right" 
            else:
                text = "Stay"

            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(args.width / 3), args.height), ORANGE, -1)
        cv2.rectangle(overlay, (int(2 * args.width / 3), 0), (args.width, args.height), ORANGE, -1)
        cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
