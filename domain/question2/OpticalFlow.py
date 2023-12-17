import cv2
import numpy as np

class OpticalFlow:
    def __init__(self):
        pass
    
    def preprocessing(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, 1, 0.3, 7, 7)
        for i in p0:
            x, y = i.ravel()
            x, y = int(x), int(y)
            # line length = 20, thickness = 4
            cv2.line(frame, (x-10, y), (x+10, y), (0, 0, 255), 4)
            cv2.line(frame, (x, y-10), (x, y+10), (0, 0, 255), 4)
            
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def video_tracking(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, 1, 0.3, 7, 7)
        for i in p0:
            x, y = i.ravel()
            x, y = int(x), int(y)
            # line length = 20, thickness = 4
            cv2.line(frame, (x-10, y), (x+10, y), (0, 0, 255), 4)
            cv2.line(frame, (x, y-10), (x, y+10), (0, 0, 255), 4)
        
        mask = np.zeros_like(frame)
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None)
            
            if p1 is None:
                continue
            
            good_cur = p1[status == 1]
            good_prev = p0[status == 1]
            
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_cur, good_prev)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b = int(a), int(b)
                c, d = int(c), int(d)
                cv2.line(frame, (a-10, b), (a+10, b), (0, 0, 255), 4)
                cv2.line(frame, (a, b-10), (a, b+10), (0, 0, 255), 4)
                # draw the tracks and save the mask
                mask = cv2.line(mask, (a, b), (c, d), (0, 100, 255), 4)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            
            frame = cv2.add(frame, mask)
            cv2.imshow("Frame", frame)
            fps = cap.get(cv2.CAP_PROP_FPS)
            wait_time = max(int(1000/fps) - 15, 1)
            pressed_key = cv2.waitKey(wait_time) & 0xFF
            if pressed_key == ord('q') or pressed_key == 27:
                break
            
            prev_gray = cur_gray.copy()
            p0 = good_cur.reshape(-1, 1, 2)
        
        cv2.destroyAllWindows()


    
    
    
def main():
    video_path = './dataset/Q2/optical_flow.mp4'
    OpticalFlow().process(video_path)