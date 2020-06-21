import cv2
import numpy as np

GF_parameters = dict(maxCorners = 30, qualityLevel = 0.1, minDistance = 7, blockSize = 7)
LK_parameters = dict(winSize = (11,11), maxLevel= 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture("sample.mp4")

ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

features_prev = cv2.goodFeaturesToTrack(prev_frame_gray, **GF_parameters)

mask = np.zeros_like(prev_frame)

while (cap.isOpened()):

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    features_next, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray,frame_gray,features_prev,None, ** LK_parameters)

    good_features_prev = features_prev[status==1]
    good_features_next = features_next[status==1]
    print(good_features_next)

    for i, (new,old) in enumerate(zip(good_features_next,good_features_prev)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d),(0,255,0), 2)
        frame = cv2.circle(frame, (a, b), 5,(0,255,0), -1)
    output = cv2.add(mask,frame)
    prev_frame_gray = frame_gray.copy()
    features_prev = good_features_next.reshape(-1, 1, 2)

    cv2.imshow("flow",output)
    if (cv2.waitKey(300) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
