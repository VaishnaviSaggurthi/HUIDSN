import cv2
import mediapipe as mp
import pystray
from PIL import Image
import threading
import time
import pyautogui
import wmi
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from pynput.mouse import Button, Controller
from pystray import MenuItem as item
import datetime
import speech_recognition as sr
import math
import google.generativeai as genai
import pyttsx3

# Initialize MediaPipe and other global variables
mp_hands = mp.solutions.hands
# Reduce detection/tracking confidence to increase speed
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils


# Initialize mouse control
mouse = Controller()
mark = 0
pyautogui.FAILSAFE = False
clear_annotations = False
# Initialize global variables
global curr_selection
curr_selection = "None"  # Set initial value for current selection
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower height
cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS (depends on camera capabilities)


temp = 0

screen_width, screen_height = pyautogui.size()

# Parameters
movement_threshold = 25  # Radius of the dead zone in the center
outer_speed_zone_radius = 100  # Radius for the maximum speed zone
base_speed = 200  # Base speed for mouse movement
max_speed_multiplier = 100 # Maximum speed multiplier at outer radius


# Get the frame size (for positioning center and speed zones)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = frame_width // 2, frame_height // 2

# Icon and taskbar setup
icon_path = "C:/Users/vaish/OneDrive/Desktop/AAC/peace_symbol.jpg"

c = wmi.WMI(namespace='wmi')

# Setup logging to log to a file or console
#logging.basicConfig(filename='metrics.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Example of logging a starting message
#logging.info("Program started.")


# Set up the Google API Key for the Generative AI service
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
print(os.getenv("GOOGLE_API_KEY"))
# Initialize the model and chat session
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])





# Initialize volume control
def get_audio_interface():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return volume
    except Exception as e:
        print(f"Error getting audio interface: {e}")
        return None

volume = get_audio_interface()


def record_and_convert_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please say something")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=3)

        print("Recording done. Converting to text...")

    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def read_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# Functions to modify global state
def brightness_modification(icon, item):
    global curr_selection
    curr_selection = "Brightness"

def volume_modification(icon, item):
    global curr_selection
    curr_selection = "Volume"


def mute_unMute(icon, item):
    global curr_selection
    curr_selection = "Mute/UnMute"

def ppt_left_right(icon, item):
    global curr_selection
    curr_selection = "PPT Left/Right"
    powerpoint_path = r"C:/Users/vaish/Downloads/DOC-20240504-WA0007..pptx"
    os.startfile(powerpoint_path)

def annotate(icon, item):
    global curr_selection
    curr_selection = "Annotation"

def exit_application(icon, item):
    icon.stop()
    global running
    running = False

# Brightness control
def brightness_increment():
    methods = c.WmiMonitorBrightnessMethods()[0]
    brightness = c.WmiMonitorBrightness()[0].CurrentBrightness
    try:
        methods.WmiSetBrightness(brightness + 2, 0)
        print("Brightness increased.")
    except Exception as e:
        print("Error:", e)

def brightness_decrement():
    methods = c.WmiMonitorBrightnessMethods()[0]
    brightness = c.WmiMonitorBrightness()[0].CurrentBrightness
    try:
        methods.WmiSetBrightness(brightness - 2, 0)
        print("Brightness decreased.")
    except Exception as e:
        print("Error:", e)

# Volume control
def volume_increment():
    if volume is None:
        return
    try:
        current_volume = volume.GetMasterVolumeLevelScalar()
        new_volume = min(1.0, current_volume + 0.03)
        volume.SetMasterVolumeLevelScalar(new_volume, None)
        print("Volume increased.")
    except Exception as e:
        print(f"Error adjusting volume: {e}")

def volume_decrement():
    if volume is None:
        return
    try:
        current_volume = volume.GetMasterVolumeLevelScalar()
        new_volume = max(0.0, current_volume - 0.05)
        volume.SetMasterVolumeLevelScalar(new_volume, None)
        print("Volume decreased.")
    except Exception as e:
        print(f"Error adjusting volume: {e}")

# Mute/Unmute control
def volume_mute():
    if volume is None:
        return
    try:
        volume.SetMute(1, None)
        print("Volume muted.")
    except Exception as e:
        print(f"Error muting volume: {e}")

def volume_unmute():
    if volume is None:
        return
    try:
        volume.SetMute(0, None)
        print("Volume unmuted.")
    except Exception as e:
        print(f"Error unmuting volume: {e}")

# Functions to control PowerPoint presentation
def ppt_left():
    pyautogui.press('left')
    print("Left arrow key pressed")

def ppt_right():
    pyautogui.press('right')
    print("Right arrow key pressed")

# Taskbar icon setup
menu = [
    item("Brightness", brightness_modification),
    item("Volume", volume_modification),
    item("Mute/UnMute", mute_unMute),
    item("PPT Left/Right", ppt_left_right),
    item("Annotation", annotate),
    item("Quit", exit_application)
]
icon_image = Image.open(icon_path)
taskbar_icon = pystray.Icon("name", icon_image, "Taskbar Icon", menu)

def run_icon():
    taskbar_icon.run()

icon_thread = threading.Thread(target=run_icon)
icon_thread.start()

# Gesture detection
# Gesture detection with image downscaling
def hand_state(image, scale_factor=0.5):
    # Downscale the image
    small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert the downscaled image to RGB for MediaPipe processing
    small_image_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    # Process the downscaled image
    results = hands.process(small_image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tips_indices = [4, 8, 12, 16, 20]
            finger_states = []
            for index in tips_indices:
                # Upscale the coordinates back to the original resolution
                tip_landmark = hand_landmarks.landmark[index]
                knuckle_landmark = hand_landmarks.landmark[index - 2]

                is_open = tip_landmark.y < knuckle_landmark.y if index != 4 else tip_landmark.x > \
                                                                                 hand_landmarks.landmark[8].x
                finger_states.append(1 if is_open else 0)

            if finger_states == [1, 1, 1, 1, 1] or finger_states == [0, 1, 1, 1, 1]:
                return "Open"
            if finger_states == [0, 0, 0, 0, 0] or finger_states == [1, 0, 0, 0, 0]:
                return "Close"
    return None


# Initialize a variable to track the previous click time
last_click_time = 0

def driver(curr_selection, sign):
    global click_executed, last_click_time
    current_time = time.time()

    if curr_selection == "Volume":
        if sign == "Open":
            volume_increment()
        if sign == "Close":
            volume_decrement()
    elif curr_selection == "Brightness":
        if sign == "Open":
            brightness_increment()
        if sign == "Close":
            brightness_decrement()
    elif curr_selection == "Mute/UnMute":
        if sign == "Open":
            volume_mute()
        if sign == "Close":
            volume_unmute()
    elif curr_selection == "PPT Left/Right":
        if sign == "Open":
            ppt_left()
        if sign == "Close":
            ppt_right()
    elif curr_selection == "Annotation":
        global mark
        mark = 1
    elif curr_selection == "Click":
        # Only execute click if sufficient time has passed (e.g., 0.5 seconds)
        if current_time - last_click_time > 0.5:  # 0.5 seconds delay between clicks
            mouse.press(Button.left)
            mouse.release(Button.left)
            last_click_time = current_time  # Update the last click time

# Function to check if a finger is open
def is_finger_open(hand_landmarks, finger_tip, finger_mcp):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_mcp].y

# Function to check the combination of finger states
def check_finger_combination(hand_landmarks):
    index_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
    middle_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    ring_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP)
    pinky_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    thumb_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_MCP)

    return index_open, middle_open, ring_open, pinky_open, thumb_open

def move_cursor(index_finger_tip, alpha, smooth_x, smooth_y):
    scale_factor = 1.5
    x = int(index_finger_tip.x * screen_width * scale_factor)
    y = int(index_finger_tip.y * screen_height * scale_factor)

    # Ensure the cursor moves smoothly within the screen bounds
    x = max(1, min(x, screen_width - 1))
    y = max(1, min(y, screen_height - 1))

    # Simple smoothing for the cursor movement
    smooth_x = alpha * smooth_x + (1 - alpha) * x
    smooth_y = alpha * smooth_y + (1 - alpha) * y

    pyautogui.moveTo(smooth_x, smooth_y, duration=0)

    return smooth_x, smooth_y




# Function to get bounding box around the hand


def save_annotated_image(frame, white_img):
    save_dir = "annotated_images"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Directory created: {save_dir}")  # Debug statement

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"annotated_{timestamp}.png")

    combined_img = cv2.addWeighted(frame, 0.5, white_img, 0.5, 0)

    success = cv2.imwrite(filename, combined_img)
    if success:
        print(f"Annotated image saved as: {filename}")  # Debug statement
    else:
        print("Failed to save the annotated image.")  # Debug statements=



# Initialize drawing variables
drawing = False
white_img = np.ones((720, 1280, 3), np.uint8) * 255  # White canvas
brush_radius = 10
brush_color = (0, 0, 0)  # Black

# Main loop to read from the webcam and process gestures
smooth_x, smooth_y = 0, 0
alpha = 0.7  # Smoothing factor

# Frame skipping to reduce lag
frame_skip = 2
frame_count = 0

# Previous fingertip positions
prev_index_x, prev_index_y = None, None
prev_thumb_x, prev_thumb_y = None, None
try:
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break



        if frame is None:
            print("No frame captured. Exiting...")
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Resize white_img to match the dimensions of frame
        white_img = cv2.resize(white_img, (frame.shape[1], frame.shape[0]))  # Create a white image with the same dimensions as frame

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = hands.process(frame_rgb)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue  # Skip this iteration

        # Initialize `is_annotating` for the current frame
        is_annotating = False

        if results.multi_hand_landmarks:


            if curr_selection == "Annotation":
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if all required landmarks are detected
                    if thumb_open and pinky_open and not index_open and not middle_open and not ring_open:
                        clear_annotations = True
                        # Clear the white canvas if the gesture is detected
                    if clear_annotations:
                        white_img = np.ones((frame.shape[0], frame.shape[1], 3), np.uint8) * 255  # Reset to a white canvas
                        clear_annotations = False  # Reset the flag after clearing
                        print("Annotations cleared.")  # Debug statement

                    if len(hand_landmarks.landmark) >= 21:
                        # Get landmarks for the thumb tip, index tip, middle tip, ring tip, and little tip
                        index_tip = hand_landmarks.landmark[8]
                        middle_tip = hand_landmarks.landmark[12]
                        thumb_tip = hand_landmarks.landmark[4]
                        index_pip = hand_landmarks.landmark[6]
                        middle_pip = hand_landmarks.landmark[10]
                        thumb_base = hand_landmarks.landmark[3]  # Thumb knuckle

                        # Convert landmarks to pixel coordinates
                        index_tip_x = int(index_tip.x * frame.shape[1])
                        index_tip_y = int(index_tip.y * frame.shape[0])
                        middle_tip_x = int(middle_tip.x * frame.shape[1])
                        middle_tip_y = int(middle_tip.y * frame.shape[0])
                        thumb_tip_x = int(thumb_tip.x * frame.shape[1])
                        thumb_tip_y = int(thumb_tip.y * frame.shape[0])
                        index_pip_y = int(index_pip.y * frame.shape[0])
                        middle_pip_y = int(middle_pip.y * frame.shape[0])

                        # Determine if the index and middle fingers are open
                        index_is_open = index_tip_y < index_pip_y
                        middle_is_open = middle_tip_y < middle_pip_y

                        # Start annotating if both fingers are open
                        if index_is_open and middle_is_open:
                            is_annotating = True

                        # Draw on the white image if annotating
                        if is_annotating:
                            if prev_index_x is not None and prev_index_y is not None:
                                cv2.line(white_img, (prev_index_x, prev_index_y), (index_tip_x, index_tip_y), brush_color, brush_radius)
                                cv2.line(frame, (prev_index_x, prev_index_y), (index_tip_x, index_tip_y), brush_color, brush_radius)
                            prev_index_x, prev_index_y = index_tip_x, index_tip_y
                        else:
                            prev_index_x, prev_index_y = None, None

                        # Check if the thumb is open (above the base)
                        if thumb_tip.y < thumb_base.y:
                            # Erase on the white image
                            if prev_thumb_x is not None and prev_thumb_y is not None:
                                cv2.line(white_img, (prev_thumb_x, prev_thumb_y), (thumb_tip_x, thumb_tip_y),
                                         (255, 255, 255), brush_radius * 2)  # Make the eraser larger
                            prev_thumb_x, prev_thumb_y = thumb_tip_x, thumb_tip_y
                        else:
                            prev_thumb_x, prev_thumb_y = None, None

            # Process hand gestures
            sign = hand_state(frame)

            if sign:
                driver(curr_selection, sign)

            for hand_landmarks in results.multi_hand_landmarks:
                index_open, middle_open, ring_open, pinky_open, thumb_open = check_finger_combination(hand_landmarks)

                # Cursor movement
                if mark == 0:  # Only move the cursor if not in annotation mode
                    # Get index finger tip coordinates
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # Get middle finger tip and MCP (metacarpal) coordinates
                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_finger_end = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

                    h, w, _ = frame.shape
                    # Get the x, y coordinates of index and middle fingers
                    index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                    middle_end_x, middle_end_y = int(middle_finger_end.x * w), int(middle_finger_end.y * h)

                    # Check if both middle and ring fingers are open
                    if pinky_open and not index_open and not middle_open and not ring_open:  # This condition checks if both are open
                        # Update the center position to the middle finger position
                        center_x, center_y = index_x, index_y

                    # Calculate offset from center (joystick position)
                    dx = index_x - center_x
                    dy = index_y - center_y
                    distance_from_center = math.hypot(dx, dy)

                    # Set the initial movement speed
                    movement_speed = base_speed

                    # Check if the index finger is within the outer speed zone but outside the dead zone
                    if movement_threshold < distance_from_center < outer_speed_zone_radius:
                        movement_speed *= max_speed_multiplier  # Increase speed in the outer speed zone

                    # Check if the movement is above the center dead zone threshold
                    if distance_from_center > movement_threshold:
                        # Scale movement by the current speed
                        move_x = int((dx / distance_from_center) * min(movement_speed, distance_from_center / 10))
                        move_y = int((dy / distance_from_center) * min(movement_speed, distance_from_center / 10))

                        # Move the mouse relatively
                        pyautogui.moveRel(move_x, move_y)



                    # Left click gesture (index and middle fingers open)
                    elif index_open and middle_open and not ring_open and not pinky_open:
                        #mouse.press(Button.left)
                        #mouse.release(Button.left)
                        pyautogui.click();




                    # Right click gesture (index and pinky fingers open)
                    elif index_open and pinky_open:
                        #mouse.press(Button.right)
                        #mouse.release(Button.right)
                        pyautogui.click(button="right")
            # Draw hand landmarks and center point on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw the center dead zone
            cv2.circle(frame, (center_x, center_y), movement_threshold, (0, 255, 0), 2)

            # Draw the outer maximum speed zone
            cv2.circle(frame, (center_x, center_y), outer_speed_zone_radius, (0, 0, 255), 2)

            # Combine the camera image with the white canvas


            # Reset selection if specific gestures are made

            if pinky_open and not index_open and middle_open and ring_open:
                a = record_and_convert_speech()

                # Get user input from the request JSON
                user_input = f"You are a smart assistant. Respond briefly to simple questions (under 10 words), and provide slightly longer responses for complex ones—up to 3 sentences max. Keep your tone friendly, clear, and easy to understand, suitable for TTS. Input: {a}"


                try:
                    # Send message to the bot
                    response_raw = chat.send_message(user_input)

                    # Debugging print to check the full response from the model
                    print("Full response:", response_raw.text)
                    read_text(response_raw.text)
                    # Ensure response_raw is valid and contains text
                    if response_raw and response_raw.text:
                        # Strip whitespace and return the full response text
                        response = response_raw.text.strip()
                except:
                    print("error")



                # save_annotated_image(frame, white_img)
            if middle_open and index_open and ring_open and not pinky_open:
                mark = 0
                # curr_selection = "None"frame_skip = 3  # Skip more frames when hands are not detected
            combined_img = cv2.addWeighted(frame, 0.5, white_img, 0.5, 0)
            # Display the resulting frameframe_skip = 3  # Skip more frames when hands are not detected
            cv2.imshow('Drawing Canvas', combined_img)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except:
    print("error")

# Clean up
#video_stream.stop()
cv2.destroyAllWindows()
icon_thread.join()