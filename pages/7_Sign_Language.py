import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import tensorflow as tf
import numpy as np
import mediapipe as mp
import pickle
from collections import deque
import av
import cv2
import threading
import pyttsx3
from pathlib import Path

# Load model và encoder
SL_MODEL_DIR = Path(__file__).parent / 'Source' / 'SignLanguage'
SVM_MODEL_PATH = SL_MODEL_DIR / 'label_encoder.pkl'
MODEL_PATH = SL_MODEL_DIR / 'ASL_model.h5'

# Patch model config for Keras 3 to Keras 2 backward compatibility
try:
    import h5py
    import json
    with h5py.File(MODEL_PATH, 'r+') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
            def fix_config(c):
                if isinstance(c, dict):
                    if 'batch_shape' in c:
                        c['batch_input_shape'] = c.pop('batch_shape')
                    if 'dtype' in c and isinstance(c['dtype'], dict) and c['dtype'].get('class_name') == 'DTypePolicy':
                        c['dtype'] = c['dtype'].get('config', {}).get('name', 'float32')
                    for k, v in list(c.items()):
                        fix_config(v)
                elif isinstance(c, list):
                    for item in c:
                        fix_config(item)
                        
            fix_config(config)
            f.attrs['model_config'] = json.dumps(config).encode('utf-8')
except Exception as e:
    print("Error patching model:", e)

model = tf.keras.models.load_model(MODEL_PATH)
with open(SVM_MODEL_PATH, 'rb') as f:
    le = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hàm đọc từ sử dụng pyttsx3
def speak(text):
    if text:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Tốc độ nói
        engine.setProperty('volume', 1.0)  # Âm lượng (0.0 đến 1.0)

        # Thiết lập ngôn ngữ tiếng Việt nếu có cài đặt giọng phù hợp
        voices = engine.getProperty('voices')
        for voice in voices:
            if "vi" in voice.id.lower() or "vietnam" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break

        engine.say(text)
        engine.runAndWait()

# Video Processor
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.prediction_history = deque(maxlen=10)
        self.current_prediction = ""
        self.confidence = 0.0
        self.predicted_word = ""
        self.check = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        self.current_prediction = ""
        self.confidence = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                keypoints_np = np.array(keypoints).reshape(1, -1)
                prediction = model.predict(keypoints_np, verbose=0)
                self.confidence = np.max(prediction)
                predicted_letter = le.inverse_transform([np.argmax(prediction)])[0]
                self.current_prediction = predicted_letter

                if predicted_letter == "next":
                    self.check = True
                    self.prediction_history.clear()
                elif self.confidence > 0.7:
                    self.prediction_history.append(predicted_letter)

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Logic xử lý từ
        if (len(set(self.prediction_history)) == 1
                and len(self.prediction_history) == self.prediction_history.maxlen
                and self.current_prediction not in ["next"]):
            if self.current_prediction == "del":
                self.predicted_word = self.predicted_word[:-1]
                self.prediction_history.clear()
            elif self.check:
                if self.current_prediction == "space":
                    self.predicted_word += " "
                else:
                    self.predicted_word += self.current_prediction
                self.check = False
                self.prediction_history.clear()

        # Ghi thông tin lên ảnh
        text = f"{self.current_prediction} ({self.confidence:.2f})"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Word: {self.predicted_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Check: {self.check}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
# --- Khởi tạo session_state ---
if "show_guide" not in st.session_state:
    st.session_state["show_guide"] = False

# --- Hiển thị phần hướng dẫn ---
if st.session_state["show_guide"]:
    with st.expander("📘 Hướng dẫn sử dụng", expanded=True):
        col1, col2 = st.columns([2, 1.5])

        with col1:
            st.markdown("""
            **Cách sử dụng ứng dụng:**
            - Lưu ý: sử dụng tay phải
            - Nhấn nút Start để bắt đầu nhận diện ngôn ngữ ký hiệu.
            - Đưa tay vào camera theo ký hiệu ASL.
            - Khi ký hiệu ổn định, từ sẽ được thêm vào kết quả.
            - Các ký hiệu đặc biệt:
                - `space`: tạo khoảng trắng.
                - `del`: xóa ký tự cuối.
                - `next`: cho phép thêm ký tự tiếp theo (→ check = True).
            - Chỉ khi `check = True`, từ mới được thêm vào.
            - Nhấn nút 🔊 để nghe từ được nhận diện.
            - Nhấn 🔁 để reset lại từ hiện tại.
            """)

        with col2:
            st.image("images/ImageProcessingAdvanced/Sample_ASL.jpg", caption="Minh họa ký hiệu tay", use_container_width=True)
        if st.button("🔽 Ẩn hướng dẫn"):
            st.session_state["show_guide"] = False
            st.rerun()
    col1, col2, col3 = st.columns([3,3,2])
    with col1:
        st.image("images/ImageProcessingAdvanced/del_test.jpg", caption="Minh họa tính năng delete chữ")
    with col2:
        st.image("images/ImageProcessingAdvanced/space_test.jpg", caption= "Minh họa space - khoảng cách")
    with col3:
        st.image("images/ImageProcessingAdvanced/next_test2.png", caption= "Minh họa check")
# --- Nút hiển thị nếu đang ẩn ---
else:
    if st.button("❓ Hiển thị hướng dẫn"):
        st.session_state["show_guide"] = True
        st.rerun()
# --- Thêm bố cục mới ---
st.title("🧠 Nhận diện ngôn ngữ ký hiệu (ASL)")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="asl-realtime",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("## 📌 Từ đã nhận diện:")
    if webrtc_ctx.video_processor:
        word = webrtc_ctx.video_processor.predicted_word
        st.markdown(f"<h3 style='color:blue; font-size: 20px'>{word}</h3>", unsafe_allow_html=True)

        if st.button("🔊 Đọc từ"):
            threading.Thread(target=speak, args=(word,)).start()

        if st.button("🔁 Reset từ"):
            webrtc_ctx.video_processor.predicted_word = ""
            webrtc_ctx.video_processor.prediction_history.clear()
            webrtc_ctx.video_processor.check = True
