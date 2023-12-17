import cv2
import numpy as np

class BackgroundSubtraction:
    def __init__(self):
        pass

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=3000.0, detectShadows=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            foreground_mask = subtractor.apply(frame)
            # show only after 4 frames
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 4:
                #horizontal stack
                stacked = np.hstack((frame, cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR), cv2.bitwise_and(frame, frame, mask=foreground_mask)))
                
                cv2.imshow("Frame", stacked)

            frame = cv2.GaussianBlur(cap.read()[1], (5, 5), 0)
            fps = cap.get(cv2.CAP_PROP_FPS)

            pressed_key = cv2.waitKey(int(1000/fps)) & 0xFF
            if pressed_key == ord('q') or pressed_key == 27:
                break
            
        cv2.destroyAllWindows()
                

def main():
    video_path = './dataset/Q1/traffic.mp4'
    BackgroundSubtraction().process(video_path)