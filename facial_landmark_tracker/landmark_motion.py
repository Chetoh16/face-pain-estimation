import cv2
import mediapipe as mp

# capture webcam (0 for internal more for external cameras)
cap = cv2.VideoCapture(0)

while cap.isOpened():

    # cap.Read reads images (30 images per second) from the camera (which are stored in image)
    # if unsuccessful, success variabe will be False
    success, image = cap.read()

    if not success:
        break

    # window for displaying video
    cv2.imshow("My video capture", image)

    # if q is pressed for 100 milliseconds, escape
    if cv2.waitKey(100) == ord('q'):
        break

# release the camera
cap.release()

cv2.destroyAllWindows()

