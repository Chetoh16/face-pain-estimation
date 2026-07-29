import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# get google's model
model_path = 'facial_landmark_tracker/face_landmarker.task'


# the blendshapes/expressions related to pain, grouped by which AU (Action Unit, NOT Alternative Universe) they're close to
PAIN_RELEVANT_BLENDSHAPES = [
    
    # ~AU4  brow lowerer
    "browDownLeft", "browDownRight",

    # ~AU6/7 cheek raiser / lid tightener
    "eyeSquintLeft", "eyeSquintRight",

    # ~AU9  nose wrinkler
    "noseSneerLeft", "noseSneerRight",

    # ~AU10 upper lip raiser
    "mouthUpperUpLeft", "mouthUpperUpRight",

    # ~AU43 eye closure
    "eyeBlinkLeft", "eyeBlinkRight",
]



# configure landmarker for VIDEO mode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces = 1,
    output_face_blendshapes = True
)

# capture webcam (0 for internal more for external cameras)
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        # cap.Read reads images (30 images per second) from the camera (which are stored in image)
        # if unsuccessful, success variabe will be False
        success, frame = cap.read()


        if not success:
            break

        # height and with
        h, w, _ = frame.shape
        
        # convert OpenCV BGR frame to MediaPipe RGB Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # load input frames as numpy arrays
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # calculate timestamp in milliseconds required for VIDEO mode
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # perform detection on the frame
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # print landmarks when detected
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:

                    # draw circles for all landmarks
                    cx, cy = int(landmark.x * w), int(landmark.y * h)

                    # OpenCV uses BGR instead of RGB (sure go be unique)
                    cv2.circle(frame, (cx,cy), 1, (0, 250, 0), -1)

        # check that blendshapes come through
        # result.face_blendshapes is a list (one entry per detected face)
        # each entry is a list of Category objects with .category_name and .score
        if result.face_blendshapes:
            first_blendshape = result.face_blendshapes[0][0]
            print(f"{first_blendshape.category_name}: {first_blendshape.score:.3f}")


        # window for displaying video
        cv2.imshow("Facial Landmarks", frame)

        # 0xFF extracts the pressed key ('q') across different operating systems
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # release the camera
    cap.release()
    cv2.destroyAllWindows()

