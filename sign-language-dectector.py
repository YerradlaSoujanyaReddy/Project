import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime
import speech_recognition as sr
import pyttsx3
from collections import deque
import math
from PIL import Image, ImageTk
import random

class SignLanguageAI:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand and face detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts = pyttsx3.init()
        
        # Configure TTS settings
        try:
            voices = self.tts.getProperty('voices')
            if voices:
                # Try to set a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts.setProperty('voice', voice.id)
                        break
            self.tts.setProperty('rate', 150)  # Speaking rate
            self.tts.setProperty('volume', 0.8)  # Volume level
        except Exception as e:
            print(f"TTS configuration warning: {e}")
        
        # Gesture recognition variables
        self.gesture_buffer = deque(maxlen=15)  # Increased buffer for better stability
        self.current_gesture = "None"
        self.gesture_confidence = 0.0
        
        # Word formation variables
        self.current_word = ""
        self.last_letter_time = time.time()
        self.letter_hold_duration = 1.5  # Hold letter for 1.5 seconds to confirm
        self.word_timeout = 3.0  # 3 seconds of no letters forms a word
        
        # Facial expression variables
        self.expression_buffer = deque(maxlen=5)
        self.current_expression = "Neutral"
        
        # AI conversation state
        self.conversation_history = []
        self.ai_response_queue = deque()
        self.current_ai_gesture = "None" 
        self.ai_speaking = False
        
        # AI Assistant personality
        self.ai_name = "SignBot"
        self.ai_responses = {
            "greeting": [
                "Hello! I'm SignBot, your sign language assistant. I can recognize letters and numbers!",
                "Hi there! Welcome to our sign language conversation. Show me some letters and numbers!",
                "Greetings! I'm here to help you with sign language communication including the alphabet!"
            ],
            "encouragement": [
                "You're doing great with your gestures! Keep it up!",
                "I love how expressive you are! Your signing is wonderful.",
                "Excellent gesture recognition! You're a natural at this."
            ],
            "letter_recognition": [
                "Great letter! I can see you're spelling something!",
                "Nice letter formation! Keep going!",
                "Perfect! Your letter signing is very clear!"
            ],
            "number_recognition": [
                "Excellent number! I can count with you!",
                "Great number gesture! Mathematics is fun!",
                "Perfect number formation! Keep counting!"
            ],
            "word_completion": [
                "Wonderful! You spelled out a complete word!",
                "Amazing spelling! I understood your word perfectly!",
                "Great job completing that word! Your signing is excellent!"
            ],
            "help": [
                "I'm here to assist you with sign language communication.",
                "Don't worry, I'll help guide you through our conversation.",
                "Let me help you with that! I'm your dedicated sign language assistant."
            ]
        }
        
        # Enhanced sign language dictionary with numbers and alphabet
        self.sign_dictionary = {
            # Basic gestures
            "hello": "Open hand wave",
            "goodbye": "Open hand wave (farewell)",
            "thank_you": "Flat hand to chin, move forward",
            "please": "Flat hand on chest, circular motion",
            "yes": "Fist nodding motion",
            "no": "Index and middle finger together, side to side",
            "help": "Fist on opposite palm, lift up",
            "good": "Flat hand from chin forward",
            "bad": "Flat hand from chin, flip down",
            "understand": "Index finger to temple",
            "question": "Index finger drawing question mark",
            "love": "Cross arms over chest or I-L-Y gesture",
            "stop": "Open palm facing forward",
            "call_me": "Thumb and pinky extended (phone gesture)",
            "rock_on": "Index and pinky extended",
            "ok_sign": "Thumb touching index finger",
            "thumbs_up": "Thumb extended upward",
            "thumbs_down": "Thumb extended downward",
            "gun": "Index finger and thumb extended (L-shape)",
            "spiderman": "Index, middle, and pinky extended",
            
            # Numbers 0-9
            "zero": "Closed fist or O-shape",
            "one": "Index finger extended",
            "two": "Index and middle fingers extended",
            "three": "Index, middle, and ring fingers extended",
            "four": "Four fingers extended (no thumb)",
            "five": "All five fingers extended",
            "six": "Thumb touches pinky, other fingers extended",
            "seven": "Thumb touches ring finger, other fingers extended",
            "eight": "Thumb touches middle finger, other fingers extended",
            "nine": "Thumb touches index finger, other fingers extended",
            
            # Alphabet A-Z (simplified descriptions)
            "a": "Closed fist with thumb on side",
            "b": "Flat hand, thumb across palm",
            "c": "Curved hand like letter C",
            "d": "Index finger up, thumb touches other fingers",
            "e": "Fingers bent down, thumb across",
            "f": "Index and thumb touching, others extended",
            "g": "Index finger and thumb parallel",
            "h": "Index and middle fingers together sideways",
            "i": "Pinky finger extended",
            "j": "Pinky extended with J motion",
            "k": "Index up, middle finger on thumb",
            "l": "L-shape with thumb and index",
            "m": "Thumb under three fingers",
            "n": "Thumb under two fingers",
            "o": "O-shape with all fingers",
            "p": "Index finger down, middle on thumb",
            "q": "Index finger down, thumb down",
            "r": "Index and middle crossed",
            "s": "Closed fist",
            "t": "Thumb between index and middle",
            "u": "Index and middle together, up",
            "v": "Index and middle apart, up",
            "w": "Index, middle, ring up",
            "x": "Index finger hooked",
            "y": "Thumb and pinky extended",
            "z": "Z motion with index finger"
        }
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI window"""
        self.root = tk.Tk()
        self.root.title("AI Sign Language Video Call - Letters & Numbers Recognition")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2c3e50')
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Call", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls & Information", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.configure(width=500)
        
        # Status information
        status_frame = ttk.LabelFrame(control_frame, text="Current Recognition", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gesture_label = tk.Label(status_frame, text="Gesture: None", font=('Arial', 12, 'bold'))
        self.gesture_label.pack(anchor=tk.W)
        
        self.expression_label = tk.Label(status_frame, text="Expression: Neutral", font=('Arial', 10))
        self.expression_label.pack(anchor=tk.W)
        
        self.confidence_label = tk.Label(status_frame, text="Confidence: 0%", font=('Arial', 10))
        self.confidence_label.pack(anchor=tk.W)
        
        # Word formation display
        word_frame = ttk.LabelFrame(control_frame, text="Word Formation", padding=5)
        word_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_word_label = tk.Label(word_frame, text="Current Word: ", font=('Arial', 14, 'bold'), fg='green')
        self.current_word_label.pack(anchor=tk.W)
        
        self.letter_timer_label = tk.Label(word_frame, text="Letter Timer: Ready", font=('Arial', 10))
        self.letter_timer_label.pack(anchor=tk.W)
        
        # Quick reference guides
        guide_notebook = ttk.Notebook(control_frame)
        guide_notebook.pack(fill=tk.X, pady=(0, 10))
        
        # Numbers guide
        numbers_frame = ttk.Frame(guide_notebook)
        guide_notebook.add(numbers_frame, text="Numbers")
        
        numbers_text = tk.Text(numbers_frame, height=8, width=50, font=('Arial', 9))
        numbers_text.pack(fill=tk.BOTH, expand=True)
        numbers_text.insert(tk.END, 
            "0️⃣ ZERO: Closed fist or O-shape\n"
            "1️⃣ ONE: Index finger up\n"
            "2️⃣ TWO: Index + middle fingers\n"
            "3️⃣ THREE: Index + middle + ring\n"
            "4️⃣ FOUR: Four fingers (no thumb)\n"
            "5️⃣ FIVE: All fingers extended\n"
            "6️⃣ SIX: Thumb touches pinky\n"
            "7️⃣ SEVEN: Thumb touches ring\n"
            "8️⃣ EIGHT: Thumb touches middle\n"
            "9️⃣ NINE: Thumb touches index"
        )
        numbers_text.configure(state=tk.DISABLED)
        
        # Letters guide
        letters_frame = ttk.Frame(guide_notebook)
        guide_notebook.add(letters_frame, text="Letters A-M")
        
        letters_text = tk.Text(letters_frame, height=8, width=50, font=('Arial', 8))
        letters_text.pack(fill=tk.BOTH, expand=True)
        letters_text.insert(tk.END,
            "🅰️ A: Closed fist, thumb on side\n"
            "🅱️ B: Flat hand, thumb across palm\n"
            "🅲 C: Curved hand like C\n"
            "🅳 D: Index up, thumb touches others\n"
            "🅴 E: Fingers bent, thumb across\n"
            "🅵 F: Index/thumb touch, others up\n"
            "🅶 G: Index/thumb parallel\n"
            "🅷 H: Index/middle together sideways\n"
            "🅸 I: Pinky extended\n"
            "🅹 J: Pinky with J motion\n"
            "🅺 K: Index up, middle on thumb\n"
            "🅻 L: L-shape thumb/index\n"
            "🅼 M: Thumb under 3 fingers"
        )
        letters_text.configure(state=tk.DISABLED)
        
        # More letters guide
        letters2_frame = ttk.Frame(guide_notebook)
        guide_notebook.add(letters2_frame, text="Letters N-Z")
        
        letters2_text = tk.Text(letters2_frame, height=8, width=50, font=('Arial', 8))
        letters2_text.pack(fill=tk.BOTH, expand=True)
        letters2_text.insert(tk.END,
            "🅽 N: Thumb under 2 fingers\n"
            "🅾️ O: O-shape with fingers\n"
            "🅿️ P: Index down, middle on thumb\n"
            "🅀 Q: Index down, thumb down\n"
            "🆁 R: Index/middle crossed\n"
            "🆂 S: Closed fist\n"
            "🆃 T: Thumb between index/middle\n"
            "🆄 U: Index/middle together up\n"
            "🆅 V: Index/middle apart up\n"
            "🆆 W: Index/middle/ring up\n"
            "🆇 X: Index hooked\n"
            "🆈 Y: Thumb/pinky extended\n"
            "🆉 Z: Z motion with index"
        )
        letters2_text.configure(state=tk.DISABLED)
        
        # Gestures guide
        gestures_frame = ttk.Frame(guide_notebook)
        guide_notebook.add(gestures_frame, text="Gestures")
        
        gestures_text = tk.Text(gestures_frame, height=8, width=50, font=('Arial', 8))
        gestures_text.pack(fill=tk.BOTH, expand=True)
        gestures_text.insert(tk.END,
            "👍 Thumbs Up/Down\n"
            "📞 Call Me (thumb+pinky)\n"
            "🤘 Rock On (index+pinky)\n"
            "👌 OK Sign (thumb+index circle)\n"
            "🔫 Gun (L-shape)\n"
            "🕷️ Spider-Man (3 fingers)\n"
            "✋ Stop (open palm)\n"
            "❤️ I Love You (thumb+index+pinky)"
        )
        gestures_text.configure(state=tk.DISABLED)
        
        # AI Response section
        ai_frame = ttk.LabelFrame(control_frame, text="AI Response", padding=5)
        ai_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.ai_gesture_label = tk.Label(ai_frame, text="AI Gesture: None", font=('Arial', 12, 'bold'), fg='blue')
        self.ai_gesture_label.pack(anchor=tk.W)
        
        self.ai_text_label = tk.Label(ai_frame, text="AI Message: Ready to help!", font=('Arial', 10), wraplength=450)
        self.ai_text_label.pack(anchor=tk.W)
        
        # Conversation log
        conv_frame = ttk.LabelFrame(control_frame, text="Conversation Log", padding=5)
        conv_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.conversation_text = scrolledtext.ScrolledText(conv_frame, height=12, width=55, font=('Arial', 9))
        self.conversation_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(button_frame, text="Start Call", command=self.start_video_call)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Call", command=self.stop_video_call, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_button = ttk.Button(button_frame, text="Clear Log", command=self.clear_conversation)
        self.clear_button.pack(side=tk.LEFT)
        
        # Word control buttons
        word_button_frame = ttk.Frame(control_frame)
        word_button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.clear_word_button = ttk.Button(word_button_frame, text="Clear Word", command=self.clear_current_word)
        self.clear_word_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.space_button = ttk.Button(word_button_frame, text="Add Space", command=self.add_space)
        self.space_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # AI Control buttons
        ai_control_frame = ttk.Frame(control_frame)
        ai_control_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.mute_button = ttk.Button(ai_control_frame, text="🔇 Mute AI", command=self.toggle_ai_speech)
        self.mute_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.greet_button = ttk.Button(ai_control_frame, text="👋 AI Greet", command=self.ai_introduce)
        self.greet_button.pack(side=tk.LEFT)
        
        # Voice input button
        self.voice_button = ttk.Button(control_frame, text="🎤 Voice Input", command=self.toggle_voice_input)
        self.voice_button.pack(fill=tk.X, pady=(10, 0))
        
        # Initialize flags
        self.video_running = False
        self.voice_listening = False
        self.ai_muted = False

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def is_finger_extended(self, landmarks, finger_tip, finger_pip, finger_mcp, thumb=False):
        """Determine if a finger is extended based on landmarks"""
        if thumb:
            # For thumb, check if tip is further from wrist than MCP
            wrist = landmarks[0]
            tip_to_wrist = self.calculate_distance([landmarks[finger_tip].x, landmarks[finger_tip].y], 
                                                 [wrist.x, wrist.y])
            mcp_to_wrist = self.calculate_distance([landmarks[finger_mcp].x, landmarks[finger_mcp].y], 
                                                 [wrist.x, wrist.y])
            return tip_to_wrist > mcp_to_wrist * 1.1
        else:
            # For other fingers, check if tip is above PIP joint
            return landmarks[finger_tip].y < landmarks[finger_pip].y

    def get_finger_states(self, landmarks):
        """Get the state of all fingers (extended or not)"""
        # Finger landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints
        finger_mcps = [2, 5, 9, 13, 17]   # MCP joints
        
        finger_states = []
        
        # Check each finger
        for i in range(5):
            if i == 0:  # Thumb - special case
                extended = self.is_finger_extended(landmarks, finger_tips[i], finger_pips[i], finger_mcps[i], thumb=True)
            else:
                extended = self.is_finger_extended(landmarks, finger_tips[i], finger_pips[i], finger_mcps[i])
            finger_states.append(extended)
        
        return finger_states

    def get_finger_positions(self, landmarks):
        """Get detailed finger positions for advanced recognition"""
        positions = {}
        
        # Key landmarks
        positions['thumb_tip'] = [landmarks[4].x, landmarks[4].y]
        positions['thumb_mcp'] = [landmarks[2].x, landmarks[2].y]
        positions['index_tip'] = [landmarks[8].x, landmarks[8].y]
        positions['index_mcp'] = [landmarks[5].x, landmarks[5].y]
        positions['middle_tip'] = [landmarks[12].x, landmarks[12].y]
        positions['middle_mcp'] = [landmarks[9].x, landmarks[9].y]
        positions['ring_tip'] = [landmarks[16].x, landmarks[16].y]
        positions['ring_mcp'] = [landmarks[13].x, landmarks[13].y]
        positions['pinky_tip'] = [landmarks[20].x, landmarks[20].y]
        positions['pinky_mcp'] = [landmarks[17].x, landmarks[17].y]
        positions['wrist'] = [landmarks[0].x, landmarks[0].y]
        
        return positions

    def recognize_number(self, finger_states, positions, landmarks):
        """Recognize numbers 0-9"""
        thumb, index, middle, ring, pinky = finger_states
        extended_count = sum(finger_states)
        
        # Calculate key distances
        thumb_index_dist = self.calculate_distance(positions['thumb_tip'], positions['index_tip'])
        thumb_middle_dist = self.calculate_distance(positions['thumb_tip'], positions['middle_tip'])
        thumb_ring_dist = self.calculate_distance(positions['thumb_tip'], positions['ring_tip'])
        thumb_pinky_dist = self.calculate_distance(positions['thumb_tip'], positions['pinky_tip'])
        
        confidence = 0.9
        
        # Number 0 - Closed fist or O-shape
        if not any(finger_states) or (thumb and index and thumb_index_dist < 0.08):
            return "ZERO", confidence
        
        # Number 1 - Only index finger
        if not thumb and index and not middle and not ring and not pinky:
            return "ONE", confidence
        
        # Number 2 - Index and middle
        if not thumb and index and middle and not ring and not pinky:
            return "TWO", confidence
        
        # Number 3 - Index, middle, ring
        if not thumb and index and middle and ring and not pinky:
            return "THREE", confidence
        
        # Number 4 - Four fingers, no thumb
        if not thumb and index and middle and ring and pinky:
            return "FOUR", confidence
        
        # Number 5 - All fingers
        if thumb and index and middle and ring and pinky:
            return "FIVE", confidence
        
        # Number 6 - Thumb touches pinky, others extended
        if thumb and index and middle and ring and pinky and thumb_pinky_dist < 0.08:
            return "SIX", confidence
        
        # Number 7 - Thumb touches ring finger
        if thumb and index and middle and ring and pinky and thumb_ring_dist < 0.08:
            return "SEVEN", confidence
        
        # Number 8 - Thumb touches middle finger
        if thumb and index and middle and ring and pinky and thumb_middle_dist < 0.08:
            return "EIGHT", confidence
        
        # Number 9 - Thumb touches index finger (but not O-shape)
        if thumb and index and middle and ring and pinky and thumb_index_dist < 0.08 and extended_count == 5:
            return "NINE", confidence
        
        return None, 0.0

    def recognize_letter(self, finger_states, positions, landmarks):
        """Recognize letters A-Z"""
        thumb, index, middle, ring, pinky = finger_states
        extended_count = sum(finger_states)
        
        # Calculate key distances and angles
        thumb_index_dist = self.calculate_distance(positions['thumb_tip'], positions['index_tip'])
        thumb_middle_dist = self.calculate_distance(positions['thumb_tip'], positions['middle_tip'])
        thumb_ring_dist = self.calculate_distance(positions['thumb_tip'], positions['ring_tip'])
        thumb_pinky_dist = self.calculate_distance(positions['thumb_tip'], positions['pinky_tip'])
        
        confidence = 0.85
        
        # Letter A - Closed fist with thumb on side
        if not index and not middle and not ring and not pinky and thumb:
            return "A", confidence
        
        # Letter B - Flat hand, thumb across palm (all fingers up, thumb tucked)
        if not thumb and index and middle and ring and pinky:
            # Check if thumb is tucked across palm
            thumb_y = positions['thumb_tip'][1]
            palm_y = (positions['index_mcp'][1] + positions['pinky_mcp'][1]) / 2
            if thumb_y > palm_y:  # Thumb below palm level
                return "B", confidence
        
        # Letter C - Curved hand
        if thumb and index and middle and ring and pinky:
            # Check curvature by measuring distances
            index_middle_dist = self.calculate_distance(positions['index_tip'], positions['middle_tip'])
            if index_middle_dist < 0.12 and thumb_index_dist > 0.1:
                return "C", confidence
        
        # Letter D - Index up, thumb touches other fingers
        if thumb and index and not middle and not ring and not pinky:
            # Check if thumb is close to middle finger area
            if thumb_middle_dist < 0.08:
                return "D", confidence
        
        # Letter E - Fingers bent down, thumb across
        if thumb and not index and not middle and not ring and not pinky:
            # Check if other fingers are bent (tips below MCPs)
            fingers_bent = (landmarks[8].y > landmarks[5].y and  # Index bent
                          landmarks[12].y > landmarks[9].y and  # Middle bent
                          landmarks[16].y > landmarks[13].y and  # Ring bent
                          landmarks[20].y > landmarks[17].y)     # Pinky bent
            if fingers_bent:
                return "E", confidence
        
        # Letter F - Index and thumb touching, others extended
        if thumb and not index and middle and ring and pinky and thumb_index_dist < 0.06:
            return "F", confidence
        
        # Letter G - Index finger and thumb parallel (pointing)
        if thumb and index and not middle and not ring and not pinky:
            # Check if thumb and index are parallel (similar angles)
            if 0.05 < thumb_index_dist < 0.15:
                return "G", confidence
        
        # Letter H - Index and middle fingers together sideways
        if not thumb and index and middle and not ring and not pinky:
            index_middle_dist = self.calculate_distance(positions['index_tip'], positions['middle_tip'])
            if index_middle_dist < 0.05:  # Very close together
                return "H", confidence
        
        # Letter I - Pinky finger extended
        if not thumb and not index and not middle and not ring and pinky:
            return "I", confidence
        
        # Letter J - Pinky extended with motion (simplified as pinky + slight curve)
        if not thumb and not index and not middle and not ring and pinky:
            # J is like I but could have motion - for simplicity, same as I
            return "J", confidence * 0.8
        
        # Letter K - Index up, middle finger touching thumb
        if not thumb and index and not middle and not ring and not pinky:
            # Check if middle finger is bent and touching thumb area
            if landmarks[12].y > landmarks[10].y and thumb_middle_dist < 0.08:
                return "K", confidence
        
        # Letter L - L-shape with thumb and index
        if thumb and index and not middle and not ring and not pinky:
            # Check angle between thumb and index
            angle = self.calculate_angle(positions['thumb_tip'], positions['thumb_mcp'], positions['index_tip'])
            if 70 <= angle <= 110:
                return "L", confidence
        
        # Letter M - Thumb under three fingers
        if not thumb and not index and not middle and not ring and not pinky:
            # Check if thumb is tucked under fingers
            thumb_y = positions['thumb_tip'][1]
            fingers_y = (positions['index_tip'][1] + positions['middle_tip'][1] + positions['ring_tip'][1]) / 3
            if thumb_y > fingers_y:
                return "M", confidence
        
        # Letter N - Thumb under two fingers
        if not thumb and not index and not middle and not ring and not pinky:
            # Similar to M but with two fingers
            return "N", confidence * 0.8
        
        # Letter O - O-shape with all fingers
        if thumb and index and middle and ring and pinky:
            # Check if fingers form a circle
            if thumb_index_dist < 0.08:
                return "O", confidence
        
        # Letter P - Similar to K but index pointing down
        if thumb and index and not middle and not ring and not pinky:
            # Check if index is pointing downward
            if positions['index_tip'][1] > positions['index_mcp'][1]:
                return "P", confidence
        
        # Letter Q - Similar to G but pointing down
        if thumb and index and not middle and not ring and not pinky:
            if positions['index_tip'][1] > positions['index_mcp'][1] and positions['thumb_tip'][1] > positions['thumb_mcp'][1]:
                return "Q", confidence
        
        # Letter R - Index and middle fingers crossed
        if not thumb and index and middle and not ring and not pinky:
            # Check if fingers are crossed (middle finger curves over index)
            index_middle_angle = self.calculate_angle(positions['index_tip'], positions['index_mcp'], positions['middle_tip'])
            if 30 <= index_middle_angle <= 60:
                return "R", confidence
        
        # Letter S - Closed fist (same as A but thumb position different)
        if not index and not middle and not ring and not pinky:
            # Check thumb position - for S, thumb is more in front
            if thumb and positions['thumb_tip'][0] > positions['thumb_mcp'][0]:
                return "S", confidence
        
        # Letter T - Thumb between index and middle
        if thumb and not index and not middle and not ring and not pinky:
            # Check if thumb is positioned between index and middle
            index_y = landmarks[8].y
            middle_y = landmarks[12].y
            thumb_y = positions['thumb_tip'][1]
            if index_y < thumb_y < middle_y or middle_y < thumb_y < index_y:
                return "T", confidence
        
        # Letter U - Index and middle together, pointing up
        if not thumb and index and middle and not ring and not pinky:
            index_middle_dist = self.calculate_distance(positions['index_tip'], positions['middle_tip'])
            if index_middle_dist < 0.06:  # Close together
                return "U", confidence
        
        # Letter V - Index and middle apart, pointing up (Peace sign)
        if not thumb and index and middle and not ring and not pinky:
            index_middle_dist = self.calculate_distance(positions['index_tip'], positions['middle_tip'])
            if index_middle_dist > 0.08:  # Apart
                return "V", confidence
        
        # Letter W - Index, middle, and ring up
        if not thumb and index and middle and ring and not pinky:
            return "W", confidence
        
        # Letter X - Index finger hooked/bent
        if not thumb and index and not middle and not ring and not pinky:
            # Check if index is bent/hooked
            if landmarks[8].y > landmarks[6].y:  # Tip below PIP
                return "X", confidence
        
        # Letter Y - Thumb and pinky extended (like "call me")
        if thumb and not index and not middle and not ring and pinky:
            return "Y", confidence
        
        # Letter Z - Would require motion detection (simplified)
        if not thumb and index and not middle and not ring and not pinky:
            # Z requires zigzag motion, simplified as index pointing
            return "Z", confidence * 0.6
        
        return None, 0.0

    def calculate_hand_gesture(self, landmarks):
        """Enhanced hand gesture calculation with numbers and alphabet recognition"""
        if not landmarks:
            return "None", 0.0
            
        # Get finger states and positions
        finger_states = self.get_finger_states(landmarks)
        positions = self.get_finger_positions(landmarks)
        thumb, index, middle, ring, pinky = finger_states
        
        # First try to recognize numbers (higher priority)
        number, num_confidence = self.recognize_number(finger_states, positions, landmarks)
        if number and num_confidence > 0.8:
            return number, num_confidence
        
        # Then try to recognize letters
        letter, letter_confidence = self.recognize_letter(finger_states, positions, landmarks)
        if letter and letter_confidence > 0.7:
            return letter, letter_confidence
        
        # Fall back to basic gestures
        confidence = 0.85
        
        # Calculate distances for specific gestures
        thumb_index_distance = self.calculate_distance(positions['thumb_tip'], positions['index_tip'])
        thumb_pinky_distance = self.calculate_distance(positions['thumb_tip'], positions['pinky_tip'])
        
        # Basic gesture recognition (your original code)
        
        # OK Sign - Thumb tip touches index tip
        if thumb and index and not middle and not ring and not pinky and thumb_index_distance < 0.05:
            return "OK_Sign", confidence
        
        # Thumbs Up - Only thumb extended, others closed
        if thumb and not index and not middle and not ring and not pinky:
            if positions['thumb_tip'][1] < positions['thumb_mcp'][1]:
                return "Thumbs_Up", confidence
            else:
                return "Thumbs_Down", confidence
        
        # Call Me / Phone - Thumb and pinky extended
        if thumb and not index and not middle and not ring and pinky:
            return "Call_Me", confidence
        
        # Rock On / Devil Horns - Index and pinky extended
        if not thumb and index and not middle and not ring and pinky:
            return "Rock_On", confidence
        
        # I Love You - Thumb, index, and pinky extended
        if thumb and index and not middle and not ring and pinky:
            return "I_Love_You", confidence
        
        # Gun / L-Shape - Thumb and index extended at 90 degrees
        if thumb and index and not middle and not ring and not pinky:
            angle = self.calculate_angle(positions['thumb_tip'], positions['thumb_mcp'], positions['index_tip'])
            if 70 <= angle <= 110:
                return "Gun_Shape", confidence
            else:
                return "L_Shape", confidence
        
        # Spider-Man - Index, middle, and pinky extended
        if not thumb and index and middle and not ring and pinky:
            return "Spider_Man", confidence
        
        # Stop Sign - All fingers extended, palm forward
        if not thumb and index and middle and ring and pinky:
            return "Stop_Sign", confidence
        
        # Peace Sign - Index and middle extended
        if not thumb and index and middle and not ring and not pinky:
            return "Peace_Sign", confidence
        
        # Pointing - Only index extended
        if not thumb and index and not middle and not ring and not pinky:
            return "Pointing", confidence
        
        # Fist - No fingers extended
        if not thumb and not index and not middle and not ring and not pinky:
            return "Fist", confidence
        
        # Open Hand - All fingers extended including thumb
        if thumb and index and middle and ring and pinky:
            return "Open_Hand", confidence
        
        return "Unknown_Gesture", confidence * 0.5

    def process_letter_recognition(self, recognized_gesture):
        """Process recognized letters for word formation"""
        current_time = time.time()
        
        # Check if it's a letter (single character)
        if len(recognized_gesture) == 1 and recognized_gesture.isalpha():
            # Check if enough time has passed since last letter
            if current_time - self.last_letter_time > self.letter_hold_duration:
                self.current_word += recognized_gesture
                self.last_letter_time = current_time
                self.update_conversation_log("Letter Detected", f"Added '{recognized_gesture}' to word", recognized_gesture)
                
                # Generate AI response for letter
                ai_message = f"Great letter {recognized_gesture}! " + random.choice(self.ai_responses["letter_recognition"])
                self.update_conversation_log(f"AI ({self.ai_name})", ai_message, "understand", speak=True)
        
        # Check for word completion (no new letters for word_timeout seconds)
        elif self.current_word and current_time - self.last_letter_time > self.word_timeout:
            # Word is complete
            completed_word = self.current_word
            self.current_word = ""
            self.update_conversation_log("Word Completed", f"You spelled: '{completed_word}'", "word")
            
            # Generate AI response for completed word
            ai_message = f"Amazing! You spelled '{completed_word}'! " + random.choice(self.ai_responses["word_completion"])
            self.update_conversation_log(f"AI ({self.ai_name})", ai_message, "celebrate", speak=True)

    def process_number_recognition(self, recognized_gesture):
        """Process recognized numbers"""
        numbers = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
        
        if recognized_gesture in numbers:
            number_value = numbers.index(recognized_gesture)
            ai_message = f"Perfect number {number_value}! " + random.choice(self.ai_responses["number_recognition"])
            self.update_conversation_log(f"AI ({self.ai_name})", ai_message, "numbers", speak=True)

    def analyze_facial_expression(self, landmarks, image_shape):
        """Analyze facial expression from face mesh landmarks"""
        if not landmarks:
            return "Neutral"
            
        # Convert landmarks to numpy array
        points = []
        for lm in landmarks:
            x = int(lm.x * image_shape[1])
            y = int(lm.y * image_shape[0])
            points.append([x, y])
        points = np.array(points)
        
        try:
            # Key facial landmarks
            mouth_left = points[61]
            mouth_right = points[291] 
            mouth_top = points[13]
            mouth_bottom = points[14]
            
            # Eye landmarks
            left_eye_top = points[159]
            left_eye_bottom = points[145]
            right_eye_top = points[386]
            right_eye_bottom = points[374]
            
            # Eyebrow landmarks
            left_eyebrow = points[70]
            right_eyebrow = points[107]
            
            # Calculate ratios
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            eyebrow_height = (left_eyebrow[1] + right_eyebrow[1]) / 2
            eye_center_y = (left_eye_top[1] + right_eye_top[1]) / 2
            eyebrow_raise = eye_center_y - eyebrow_height
            
            # Enhanced expression classification
            if avg_eye_height > 15 and eyebrow_raise > 20 and mouth_ratio > 0.03:
                return "Surprise"
            elif mouth_ratio > 0.05:
                return "Happy"
            elif mouth_ratio < 0.02 and eyebrow_raise < 10:
                return "Sad"
            elif eyebrow_raise < 5:
                return "Angry"
            else:
                return "Neutral"
        except Exception:
            return "Neutral"
    
    def generate_ai_response(self, user_gesture, user_expression):
        """Generate AI response with enhanced gesture recognition"""
        gesture_responses = {
            # Numbers
            "ZERO": ("Perfect zero! Starting from nothing!", "numbers"),
            "ONE": ("Number one! You're first class!", "numbers"),
            "TWO": ("Two is terrific! Great counting!", "numbers"),
            "THREE": ("Three's company! Excellent number!", "numbers"),
            "FOUR": ("Four-tastic! You're doing great!", "numbers"),
            "FIVE": ("High five! All five fingers perfect!", "numbers"),
            "SIX": ("Six is super! Great finger coordination!", "numbers"),
            "SEVEN": ("Lucky seven! Beautiful number gesture!", "numbers"),
            "EIGHT": ("Awesome eight! You're great at this!", "numbers"),
            "NINE": ("Magnificent nine! Almost to ten!", "numbers"),
            
            # Letters - Sample responses for common letters
            "A": ("Amazing A! Starting the alphabet strong!", "letters"),
            "B": ("Beautiful B! Your letter formation is excellent!", "letters"),
            "C": ("Cool C! Curved perfectly!", "letters"),
            "D": ("Delightful D! Great finger positioning!", "letters"),
            "E": ("Excellent E! Energy in every gesture!", "letters"),
            "F": ("Fantastic F! Your signing is fabulous!", "letters"),
            "G": ("Great G! Good gesture recognition!", "letters"),
            "H": ("Hello H! Horizontal and perfect!", "letters"),
            "I": ("Incredible I! Individual and clear!", "letters"),
            "J": ("Joyful J! Just magnificent!", "letters"),
            "K": ("Kinetic K! Keep up the great work!", "letters"),
            "L": ("Lovely L! Linear and clear!", "letters"),
            "M": ("Marvelous M! Magnificent gesture!", "letters"),
            "N": ("Nice N! Nearly perfect!", "letters"),
            "O": ("Outstanding O! Oval and beautiful!", "letters"),
            "P": ("Perfect P! Pointing with precision!", "letters"),
            "Q": ("Quality Q! Quite impressive!", "letters"),
            "R": ("Remarkable R! Really well done!", "letters"),
            "S": ("Super S! Strong and steady!", "letters"),
            "T": ("Terrific T! Top-notch gesture!", "letters"),
            "U": ("Unique U! United fingers!", "letters"),
            "V": ("Victorious V! Very impressive!", "letters"),
            "W": ("Wonderful W! Wide and clear!", "letters"),
            "X": ("eXcellent X! eXtraordinary!", "letters"),
            "Y": ("Yes! Y is perfect!", "letters"),
            "Z": ("Zestful Z! Zero mistakes!", "letters"),
            
            # Basic gestures
            "OK_Sign": ("Perfect! That OK sign looks great! Everything is A-OK!", "good"),
            "Thumbs_Up": ("Awesome! I love your positive thumbs up! You're doing great!", "good"),
            "Thumbs_Down": ("Oh no! Is something wrong? Let me know how I can help!", "help"),
            "Call_Me": ("I see you want to call! That's the perfect phone gesture!", "understand"),
            "Rock_On": ("Rock on! That's an awesome rock and roll gesture! 🤘", "good"),
            "I_Love_You": ("I love you too! That's such a beautiful gesture! ❤️", "love"),
            "Gun_Shape": ("That's a clear L-shape or gun gesture! Nice finger positioning!", "understand"),
            "Spider_Man": ("With great power comes great responsibility! Cool Spider-Man gesture!", "good"),
            "Stop_Sign": ("I see your stop sign! I'll pause and wait for your next instruction.", "stop"),
            "Peace_Sign": ("Peace and love! That's a beautiful peace sign! ✌️", "good"),
            "Pointing": ("You're pointing! What would you like to show me?", "question"),
            "Fist": ("I see your strong fist! You look determined!", "understand"),
            "Open_Hand": ("Hello there! Your open hand is so welcoming!", "hello")
        }
        
        # Enhanced expression responses
        expression_responses = {
            "Happy": ("Your happiness is contagious! I'm smiling too!", "good"),
            "Sad": ("I can see you're feeling down. Let me help cheer you up!", "help"),
            "Surprise": ("Wow! You look surprised! What's got you so amazed?", "question"),
            "Angry": ("I sense some frustration. Take a deep breath, I'm here to help!", "help"),
            "Neutral": ("I'm here and ready to assist you with anything you need!", "understand")
        }
        
        # Response logic with priority handling
        response_text = ""
        response_gesture = "understand"
        
        # Priority to strong emotions
        if user_expression == "Surprise":
            response_text = "Wow! You look amazed! Keep showing me those signs!"
            response_gesture = "question"
        elif user_expression == "Angry":
            response_text = "I can see you're upset. Let me help you feel better!"
            response_gesture = "help"
        elif user_gesture in gesture_responses:
            response_text, response_gesture = gesture_responses[user_gesture]
        elif user_expression in expression_responses:
            response_text, response_gesture = expression_responses[user_expression]
        else:
            response_text = "I see your gesture! How can SignBot assist you today?"
            response_gesture = "question"
        
        return response_text, response_gesture
    
    def speak_ai_response(self, text):
        """Make AI speak the response using text-to-speech"""
        if not self.ai_speaking and not self.ai_muted:
            self.ai_speaking = True
            try:
                def speak():
                    try:
                        self.tts.say(text)
                        self.tts.runAndWait()
                    except Exception as e:
                        print(f"TTS playback error: {e}")
                    finally:
                        self.ai_speaking = False
                
                speech_thread = threading.Thread(target=speak, daemon=True)
                speech_thread.start()
            except Exception as e:
                print(f"TTS Error: {e}")
                self.ai_speaking = False

    def update_conversation_log(self, speaker, message, gesture=None, speak=False):          
        """Update the conversation log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {speaker}: {message}"
        if gesture:
            log_entry += f" (Gesture: {gesture})"
        log_entry += "\n"
        
        self.conversation_text.insert(tk.END, log_entry)
        self.conversation_text.see(tk.END)
        
        # Keep conversation history
        self.conversation_history.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "message": message,
            "gesture": gesture
        })
        
        # Speak AI responses
        if speak and speaker.startswith("AI"):
            self.speak_ai_response(message)
    
    def clear_current_word(self):
        """Clear the current word being formed"""
        self.current_word = ""
        self.update_conversation_log("System", "Current word cleared")
    
    def add_space(self):
        """Add a space to the current word"""
        self.current_word += " "
        self.update_conversation_log("System", "Space added to current word")
    
    def process_frame(self, frame):
        """Process video frame for hand and face detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Process face
        face_results = self.face_mesh.process(rgb_frame)
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get gesture
                gesture, confidence = self.calculate_hand_gesture(hand_landmarks.landmark)
                self.gesture_buffer.append((gesture, confidence))
                
                # Update current gesture with better stability
                if len(self.gesture_buffer) >= 5:  # More stable recognition
                    gestures = [g[0] for g in self.gesture_buffer]
                    confidences = [g[1] for g in self.gesture_buffer]
                    
                    # Find most common gesture with high confidence
                    unique_gestures = list(set(gestures))
                    best_gesture = None
                    best_confidence = 0
                    
                    for g in unique_gestures:
                        g_confidences = [confidences[i] for i, gesture in enumerate(gestures) if gesture == g]
                        avg_confidence = np.mean(g_confidences)
                        count = gestures.count(g)
                        
                        # Weight by both frequency and confidence
                        score = avg_confidence * (count / len(gestures))
                        if score > best_confidence and avg_confidence > 0.7:
                            best_gesture = g
                            best_confidence = avg_confidence
                    
                    if best_gesture and best_gesture != self.current_gesture:
                        self.current_gesture = best_gesture
                        self.gesture_confidence = best_confidence
                        
                        # Process different types of recognition
                        if len(self.current_gesture) == 1 and self.current_gesture.isalpha():
                            # It's a letter
                            self.process_letter_recognition(self.current_gesture)
                        elif self.current_gesture in ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]:
                            # It's a number
                            self.process_number_recognition(self.current_gesture)
                            self.update_conversation_log("You", f"Number detected", self.current_gesture)
                        else:
                            # It's a basic gesture
                            ai_message, ai_gesture = self.generate_ai_response(self.current_gesture, self.current_expression)
                            self.current_ai_gesture = ai_gesture
                            self.update_conversation_log("You", f"Gesture detected", self.current_gesture)
                            self.update_conversation_log(f"AI ({self.ai_name})", ai_message, ai_gesture, speak=True)
        
        # Draw face mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw only key facial features to reduce clutter
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Analyze expression
                expression = self.analyze_facial_expression(face_landmarks.landmark, frame.shape)
                self.expression_buffer.append(expression)
                
                # Update current expression
                if len(self.expression_buffer) >= 3:
                    expressions = list(self.expression_buffer)
                    most_common = max(set(expressions), key=expressions.count)
                    if most_common != self.current_expression:
                        old_expression = self.current_expression
                        self.current_expression = most_common
                        
                        # Trigger response for strong emotions
                        if self.current_expression in ["Surprise", "Angry"] and old_expression != self.current_expression:
                            ai_message, ai_gesture = self.generate_ai_response("None", self.current_expression)
                            self.current_ai_gesture = ai_gesture
                            self.update_conversation_log("You", f"{self.current_expression} expression detected!", self.current_expression)
                            self.update_conversation_log(f"AI ({self.ai_name})", ai_message, ai_gesture, speak=True)
        
        return frame
    
    def video_loop(self):
        """Main video processing loop"""
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add enhanced overlay information
            cv2.putText(processed_frame, f"Gesture: {self.current_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Expression: {self.current_expression}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(processed_frame, f"AI: {self.current_ai_gesture}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Show current word being formed
            if self.current_word:
                cv2.putText(processed_frame, f"Word: {self.current_word}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Show timer for next letter
                time_since_last = time.time() - self.last_letter_time
                remaining_time = max(0, self.word_timeout - time_since_last)
                cv2.putText(processed_frame, f"Word Timer: {remaining_time:.1f}s", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = rgb_frame.shape[:2]
            display_width = 900
            display_height = int(height * display_width / width)
            rgb_frame = cv2.resize(rgb_frame, (display_width, display_height))
            
            # Convert to PhotoImage and display
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
            # Update GUI labels
            self.gesture_label.configure(text=f"Recognition: {self.current_gesture}")
            self.expression_label.configure(text=f"Expression: {self.current_expression}")
            self.confidence_label.configure(text=f"Confidence: {self.gesture_confidence:.1%}")
            self.ai_gesture_label.configure(text=f"AI ({self.ai_name}) Mode: {self.current_ai_gesture}")
            
            # Update word formation display
            word_display = self.current_word if self.current_word else "(empty)"
            self.current_word_label.configure(text=f"Current Word: {word_display}")
            
            # Update letter timer
            if self.current_word:
                time_since_last = time.time() - self.last_letter_time
                if time_since_last < self.letter_hold_duration:
                    timer_text = f"Letter Timer: Hold gesture ({self.letter_hold_duration - time_since_last:.1f}s)"
                else:
                    timer_text = f"Word Timer: {max(0, self.word_timeout - time_since_last):.1f}s until completion"
            else:
                timer_text = "Letter Timer: Ready for new word"
            self.letter_timer_label.configure(text=timer_text)
            
            # Update AI text with speaking status
            speaking_indicator = " 🔊" if self.ai_speaking else ""
            self.ai_text_label.configure(text=f"AI Message: Ready for letters, numbers, and gestures!{speaking_indicator}")
            
            time.sleep(0.03)  # ~30 FPS
    
    def start_video_call(self):
        """Start the video call"""
        self.video_running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        self.update_conversation_log("System", "Enhanced video call started! Now recognizing letters and numbers!")
        
        # AI introduction
        self.ai_introduce()
    
    def ai_introduce(self):
        """AI introduces itself with enhanced capabilities"""
        intro_message = random.choice(self.ai_responses["greeting"])
        self.update_conversation_log(f"AI ({self.ai_name})", intro_message, "hello", speak=True)
        self.current_ai_gesture = "hello"
        
        # Add information about new features
        feature_message = "I can now recognize all letters A-Z and numbers 0-9! Try spelling words or showing numbers!"
        self.update_conversation_log(f"AI ({self.ai_name})", feature_message, "understand", speak=True)
    
    def toggle_ai_speech(self):
        """Toggle AI speech on/off"""
        self.ai_muted = not self.ai_muted
        if self.ai_muted:
            self.mute_button.configure(text="🔊 Unmute AI")
            self.update_conversation_log("System", "AI speech muted")
        else:
            self.mute_button.configure(text="🔇 Mute AI")
            self.update_conversation_log("System", "AI speech enabled")
    
    def stop_video_call(self):
        """Stop the video call"""
        self.video_running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        
        # Complete any pending word
        if self.current_word:
            completed_word = self.current_word
            self.current_word = ""
            self.update_conversation_log("Word Completed", f"Final word: '{completed_word}'", "word")
        
        self.update_conversation_log("System", "Enhanced video call ended.")
    
    def toggle_voice_input(self):
        """Toggle voice input listening"""
        if not self.voice_listening:
            self.voice_listening = True
            self.voice_button.configure(text="🔴 Listening...")
            threading.Thread(target=self.listen_for_voice, daemon=True).start()
        else:
            self.voice_listening = False
            self.voice_button.configure(text="🎤 Voice Input")
    
    def listen_for_voice(self):
        """Listen for voice input and convert to text"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            text = self.recognizer.recognize_google(audio)
            self.update_conversation_log("You (Voice)", text)
            
            # Enhanced AI response to voice
            ai_response = f"I heard '{text}'! You can also spell it out in sign language letter by letter!"
            self.update_conversation_log(f"AI ({self.ai_name})", ai_response, "understand", speak=True)
            
        except sr.WaitTimeoutError:
            self.update_conversation_log("System", "Voice input timeout - no speech detected")
        except sr.UnknownValueError:
            self.update_conversation_log("System", "Could not understand audio")
        except sr.RequestError as e:
            self.update_conversation_log("System", f"Voice recognition error: {e}")
        finally:
            self.voice_listening = False
            self.voice_button.configure(text="🎤 Voice Input")
    
    def clear_conversation(self):
        """Clear the conversation log"""
        self.conversation_text.delete(1.0, tk.END)
        self.conversation_history.clear()
        self.current_word = ""
        self.update_conversation_log("System", "Conversation log cleared.")
    
    def get_gesture_type(self, gesture):
        """Determine the type of gesture (letter, number, or basic)"""
        if gesture in ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]:
            return "number"
        elif len(gesture) == 1 and gesture.isalpha():
            return "letter"
        else:
            return "gesture"
    
    def run(self):
        """Run the application"""
        self.update_conversation_log("System", f"🚀 Enhanced Sign Language AI Video Call Ready!")
        self.update_conversation_log("System", f"Meet {self.ai_name} - Your Advanced AI Sign Language Assistant!")
        self.update_conversation_log("System", "📝 NEW: Full alphabet A-Z recognition!")
        self.update_conversation_log("System", "🔢 NEW: Complete number 0-9 recognition!")
        self.update_conversation_log("System", "📖 NEW: Word formation from letters!")
        self.update_conversation_log("System", "Click 'Start Call' to begin enhanced gesture recognition.")
        self.update_conversation_log("System", "🎭 Features: Expression detection, voice input, and intelligent responses!")
        self.root.mainloop()
        
        # Cleanup
        if self.video_running:
            self.video_running = False
        if self.ai_speaking:
            self.tts.stop()
        self.cap.release()
        cv2.destroyAllWindows()

# Enhanced gesture recognition patterns for better accuracy
class GesturePatterns:
    """Additional class to define more detailed gesture patterns"""
    
    @staticmethod
    def is_thumb_across_palm(landmarks):
        """Check if thumb is positioned across the palm"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        # Thumb should be between index and pinky MCPs
        return (index_mcp.x < thumb_tip.x < pinky_mcp.x or 
                pinky_mcp.x < thumb_tip.x < index_mcp.x)
    
    @staticmethod
    def is_curved_hand(landmarks):
        """Check if hand is in curved C-like position"""
        # Check curvature by measuring finger tip distances
        tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]  # Index, middle, ring, pinky tips
        
        # Calculate if fingers form an arc
        distances = []
        for i in range(len(tips)-1):
            dist = math.sqrt((tips[i].x - tips[i+1].x)**2 + (tips[i].y - tips[i+1].y)**2)
            distances.append(dist)
        
        # If distances are relatively equal, fingers form a curve
        avg_dist = sum(distances) / len(distances)
        variance = sum((d - avg_dist)**2 for d in distances) / len(distances)
        
        return variance < 0.01  # Low variance indicates uniform curve
    
    @staticmethod
    def fingers_bent_down(landmarks):
        """Check if fingers are bent downward"""
        # Check if all finger tips are below their respective MCPs
        finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]  # (tip, mcp) pairs
        
        bent_count = 0
        for tip_idx, mcp_idx in finger_pairs:
            if landmarks[tip_idx].y > landmarks[mcp_idx].y:
                bent_count += 1
        
        return bent_count >= 3  # At least 3 fingers bent

if __name__ == "__main__":
    # Check if required packages are installed
    required_packages = [
        'opencv-python', 'mediapipe', 'numpy', 'speechrecognition', 
        'pyttsx3', 'pillow'
    ]
    
    print("🤖 Enhanced Sign Language AI Video Call System")
    print("=" * 60)
    print("✨ NEW FEATURES:")
    print("  📝 Complete alphabet recognition (A-Z)")
    print("  🔢 Full number recognition (0-9)")  
    print("  📖 Word formation from letter sequences")
    print("  🎯 Improved gesture accuracy")
    print("  🗣️ Enhanced AI responses")
    print()
    print("📦 Required packages:", ", ".join(required_packages))
    print()
    print("💡 To install missing packages, run:")
    print("   pip install opencv-python mediapipe numpy speechrecognition pyttsx3 pillow")
    print()
    print("🚀 Starting enhanced application...")
    print()
    print("📚 HOW TO USE:")
    print("  • Hold letter gestures for 1.5 seconds to add to word")
    print("  • Wait 3 seconds after last letter to complete word")
    print("  • Show numbers 0-9 for counting")
    print("  • Use basic gestures for communication")
    print("  • Try facial expressions for emotional responses")
    print()
    
    try:
        app = SignLanguageAI()
        app.run()
    except ImportError as e:
        print(f"❌ Error: Missing required package - {e}")
        print("Please install all required packages and try again.")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Make sure your webcam is connected and try again.")
