import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# get google's model
MODEL_PATH = 'facial_landmark_tracker/face_landmarker.task'


PAIN_RELEVANT_LANDMARKS = {
    "left_brow_inner": 55,
    "right_brow_inner": 285,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
    "upper_lip_center": 13,
    "lower_lip_center": 14,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
}

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

# threshold above which is considered "painful"
# this is just a guess for now
PAIN_THRESHOLD = 1.5

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


# Draw Landmark Plot (how the face moves up and down)
def plot_landmark_motion(landmark_history, out_path="landmark_motion.png"):
    """Plot vertical (y) motion of each tracked landmark over time."""

    # create a figure (canvas) and axes (drawing area) of size 10x5 inches
    fig, ax = plt.subplots(figsize=(10, 5))

    # loop through the landmark history dictionary
    for name, points in landmark_history.items():

        # take just the y-coordinate of each (x,y) pair
        ys = [p[1] for p in points]

        # plot the Y-coordinates on the graph and assign a label for the legend
        ax.plot(ys, label=name)

    # add titles and axis labels
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vertical pixel position (y)")
    ax.set_title("Facial landmark motion over time")

    # place the legend box in the top-right corner with a small font size
    ax.legend(loc="upper right", fontsize=8)

    # in OpenCV, y=0 is the TOP of the screen, so smaller Y means HIGHER on the face
    # invert y-axis so it's more intuitive
    ax.invert_yaxis()

    # adjust padding so text labels don't get cut off, then save as an image
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_blendshape_history(blendshape_history, out_path="blendshape_motion.png"):
    """Plot each pain-relevant blendshape's activation over time."""

    fig, ax = plt.subplots(figsize=(11, 5))

    for name, scores in blendshape_history.items():
        ax.plot(scores, label=name)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Blendshape activation (0-1)")
    ax.set_title("Pain-relevant blendshape activation over time")

    ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_pspi_history(pspi_history, pain_threshold, out_path="pspi_score.png"):
    """Plot the combined PSPI score over time, with the PAIN threshold marked."""

    fig, ax = plt.subplots(figsize=(10, 4))

    # plot the PSPI score line in dark red
    ax.plot(pspi_history, label="PSPI score", color="darkred")

    # draw a horizontal dashed line across the plot at Y = pain_threshold
    ax.axhline(pain_threshold, color="gray", linestyle="--", label=f"threshold ({pain_threshold})")

    ax.set_xlabel("Frame")
    ax.set_ylabel("PSPI score")
    ax.set_title("PSPI pain score over time")

    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    

# configure landmarker for VIDEO mode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces = 1,
    output_face_blendshapes = True
)

# capture webcam (0 for internal more for external cameras)
cap = cv2.VideoCapture(0)

# containers that persist across the whole session, one entry appended per frame.
# this is what makes "sequence-level" analysis possible afterward
# -- without this, each frame's data is thrown away the instant we move on.


# Session Time-Series Data (Persists across all frames)
# these containers/dictionaries persist across the whole session, one entry appended per frame.
# this is what makes "sequence-level" analysis possible afterward. 
# helps with:
# temporal smoothing (removing single-frame noise like random blinks).
# baseline calibration (adjusting for a user's natural neutral expression).
# plotting trend lines and identifying pain duration over time.

# tracks (x, y) pixel coordinates for key facial features across every frame
landmark_history = {name: [] for name in PAIN_RELEVANT_LANDMARKS}

# tracks intensity scores (0.0 to 1.0) for pain-related blendshapes
blendshape_history = {name: [] for name in PAIN_RELEVANT_BLENDSHAPES}

# tracks the calculated PSPI pain score for each frame
pspi_history = []

frame_count = 0

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        # cap.Read reads images (30 images per second) from the camera (which are stored in image)
        # if unsuccessful, success variabe will be False
        success, frame = cap.read()


        if not success:
            break

        frame_count+= 1

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

        # if landmarks detected
        if result.face_landmarks:

            # [0] = a list containing 478 individual landmark points for the first face
            # result.face_landmarks[0][55] = a specific landmark object (e.g. index 55 for left_brow_inner)
            face_landmarks = result.face_landmarks[0]

            # record (x, y) coordinates for key tracked landmarks
            # loop through the PAIN_RELEVANT_LANDMARKS dictionary
            for name, index in PAIN_RELEVANT_LANDMARKS.items():

                # get landmark
                landmark = face_landmarks[index]

                # mediapipe returns normalised coordinates between 0.0 and 1.0. 
                # multiply `lm.x` by frame width (`w`) and `lm.y` by frame height (`h`) to convert them to pixel coordinates, 
                # then append the (x, y) tuple to the history list for this specific landmark.
                landmark_history[name].append((landmark.x * w, landmark.y * h))

            # iterate through ALL 478 detected facial landmarks to draw them
            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:

                    # draw circles for all landmarks
                    cx, cy = int(landmark.x * w), int(landmark.y * h)

                    # OpenCV uses BGR instead of RGB (sure go be unique)
                    cv2.circle(frame, (cx,cy), 1, (0, 250, 0), -1)
            

        else:
            # if no face detected, repeat the last known value so every list stays the same length (frame count)
            # keeps things aligned for plotting later, instead of lists silently drifting out of sync.
            for name in PAIN_RELEVANT_LANDMARKS:
                landmark_history[name].append(landmark_history[name][-1] if landmark_history[name] else (np.nan, np.nan))


        if result.face_blendshapes:

            # bs for blendshape not the other word
            scores = {bs.category_name: bs.score for bs in result.face_blendshapes[0]}

            # record every pain-relevant blendshape's score this frame
            for name in PAIN_RELEVANT_BLENDSHAPES:
                blendshape_history[name].append(scores.get(name, np.nan))
            
            pspi_score = calculate_pspi(scores)

            # record the score itself to track its history/trajectory
            pspi_history.append(pspi_score)

            # draw the pspi score on top left corner
            cv2.putText(frame, f"PSPI: {pspi_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 250, 0), 2)

            # if pspi_score is above pain threshold, write PAIN on screen
            if pspi_score >= PAIN_THRESHOLD:
                cv2.putText(frame, "PAIN", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

            print(f"PSPI score: {pspi_score:.3f}")

        # do the same frame align method as when there's no landmarks
        else:
            for name in PAIN_RELEVANT_BLENDSHAPES:
                blendshape_history[name].append(np.nan)
            pspi_history.append(np.nan)

        # window for displaying video
        cv2.imshow("Facial Landmarks", frame)

        # 0xFF extracts the pressed key ('q') across different operating systems
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break

# release the camera
cap.release()
cv2.destroyAllWindows()

plot_landmark_motion(landmark_history)
plot_blendshape_history(blendshape_history)
plot_pspi_history(pspi_history, PAIN_THRESHOLD)