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

def calculate_pspi(blendshape_scores):
    """
    Prkachin & Solomon Pain Intensity (PSPI) is used to classify four levels of pain intensity (none, trace, weak, and strong).

    Approximate the PSPI score:
        PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

    blendshape_scores: dict of {category_name: score} for ONE frame.
    Returns a single float - higher means MORE PAIN

    The real formula uses actual AU intensities (like from py-feat/OpenFace)
    not blendshapes (which are not true AUs) so this is just a demo / proof of concept. 
    """

    # helper function to fetch scores with a default fallback of 0.0
    def get_score(name):
        return blendshape_scores.get(name, 0.0)

    au4 = max(get_score("browDownLeft"), get_score("browDownRight"))
    au6_7 = max(get_score("eyeSquintLeft"), get_score("eyeSquintRight"))  #6 7
    au9_10 = max(
        get_score("noseSneerLeft"), 
        get_score("noseSneerRight"),
        get_score("mouthUpperUpLeft"), 
        get_score("mouthUpperUpRight")
    )
    au43 = max(get_score("eyeBlinkLeft"), get_score("eyeBlinkRight"))

    pspi_score = au4 + au6_7 + au9_10 + au43

    return pspi_score

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


        # {name: score} dict is used to get pspi score
        if result.face_blendshapes:

            # bs for blendshape not the other word
            scores = {bs.category_name: bs.score for bs in result.face_blendshapes[0]}

            pspi_score = calculate_pspi(scores)

            print(f"PSPI score: {pspi_score:.3f}")

        # window for displaying video
        cv2.imshow("Facial Landmarks", frame)

        # 0xFF extracts the pressed key ('q') across different operating systems
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # release the camera
    cap.release()
    cv2.destroyAllWindows()

