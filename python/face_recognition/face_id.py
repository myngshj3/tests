# ======================================================================
# Project Name    : Face Identify 
# File Name       : face_id.py
# Encoding        : utf-8
# Creation Date   : 2021/02/22
# ======================================================================

import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cascade_path_human = 'haarcascade_frontalface_default.xml'
    cascade_path = 'cascade.xml'

    cascade_human = cv2.CascadeClassifier(cascade_path_human)
    cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()

        facerect_human = cascade_human.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))
        facerect = cascade.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=4, minSize=(100,100))

        if len(facerect_human) > 0:
            for rect in facerect_human:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 255, 255), thickness=2)


        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=2)
                cv2.putText(frame, 'name', tuple(rect[0:2]), font, 2,(0,0,0),2,cv2.LINE_AA)


        cv2.imshow("frame", frame)

        # quit with q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

