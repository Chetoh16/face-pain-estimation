import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# get google's model
model_path = 'facial_landmark_tracker/face_landmarker.task'

# configure landmarker for VIDEO mode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces = 1
)

# capture webcam (0 for internal more for external cameras)
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        # cap.Read reads images (30 images per second) from the camera (which are stored in image)
        # if unsuccessful, success variabe will be False
        success, image = cap.read()


        if not success:
            break

        # convert OpenCV BGR frame to MediaPipe RGB Image format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load input frames as numpy arrays
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # calculate timestamp in milliseconds required for VIDEO mode
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # perform detection on the frame
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # print landmarks when detected
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                print(f"Detected {len(face_landmarks)} landmarks")

        # window for displaying video
        cv2.imshow("My video capture", cv2.flip(image,1))

        # 0xFF extracts the pressed key ('q') across different operating systems
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # release the camera
    cap.release()
    cv2.destroyAllWindows()

