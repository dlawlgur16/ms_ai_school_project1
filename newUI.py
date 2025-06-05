import time
import numpy as np
import gradio as gr
from io import BytesIO
from PIL import Image
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# === Detection 모델 설정 ===
DETECTION_PREDICTION_ENDPOINT = "YOUR_DETECTION_ENDPOINT"  # Azure Custom Vision Detection 엔드포인트 입력
DETECTION_PREDICTION_KEY = "YOUR_DETECTION_KEY"  # Azure Custom Vision Detection 키 입력
DETECTION_PROJECT_ID = "YOUR_DETECTION_PROJECT_ID"  # Detection 프로젝트 ID 입력
DETECTION_ITERATION_NAME = "YOUR_DETECTION_ITERATION_NAME"  # Detection 모델 iteration 이름 입력


detection_credentials = ApiKeyCredentials(in_headers={"Prediction-key": DETECTION_PREDICTION_KEY})
detection_predictor = CustomVisionPredictionClient(
    endpoint=DETECTION_PREDICTION_ENDPOINT,
    credentials=detection_credentials
)

# === Classification 모델 설정 ===
classification_prediction_endpoint = "YOUR_CLASSIFICATION_ENDPOINT"  # Azure Custom Vision Classification 엔드포인트 입력
classification_prediction_key = "YOUR_CLASSIFICATION_KEY"  # Azure Custom Vision Classification 키 입력
classification_project_id = "YOUR_CLASSIFICATION_PROJECT_ID"  # Classification 프로젝트 ID 입력
classification_publish_iteration_name = "YOUR_CLASSIFICATION_ITERATION_NAME"  # Classification 모델 iteration 이름 입력

classification_credentials = ApiKeyCredentials(in_headers={"Prediction-key": classification_prediction_key})
classification_predictor = CustomVisionPredictionClient(
    endpoint=classification_prediction_endpoint, 
    credentials=classification_credentials
)

# === 설정 값 ===
FRAME_INTERVAL = 0.2   # 0.2초 간격 (초당 5프레임)
WINDOW_SIZE = 5        # 분석 윈도우 크기 (5프레임 기준)
global_time_points = []   # 각 프레임의 시간(초)
global_kss_values = []    # 각 프레임의 KSS 점수
global_frame_count = 0
last_processed_time = 0

kss_mapping = {
    "운전하다": 1,
    "눈비비기": 21,
    "어깨를두드리다": 25,
    "목을만지다": 25,
    "하품": 36,
    "뺨을때리다": 64,
    "꾸벅꾸벅졸다": 81,
    "몸못가누기": 100
}

risk_images = {
    "매우 안전": "image1.png",
    "안전": "image2.png",
    "주의": "image3.png",
    "위험": "image4.png",
    "매우 위험": "image5.png"
}

def get_risk_status(avg_kss):
    if avg_kss < 9:
        return "매우 안전", "image1.png", None
    elif avg_kss < 20:
        return "안전", "image2.png", None
    elif avg_kss < 40:
        return "주의", "image3.png", "약간 피로해보여요.mp3"
    elif avg_kss < 75:
        return "위험", "image4.png", "미치셨습니까 휴먼.mp3"
    else:
        return "매우 위험", "image5.png", "사이렌.mp3"

def analyze_frame(image):
    try:
        global global_frame_count
        byte_io = BytesIO()
        image.save(byte_io, 'png')
        byte_io.seek(0)

        # 1️⃣ Detection (몸체 검출 후 크롭)
        detection_results = detection_predictor.detect_image(DETECTION_PROJECT_ID, DETECTION_ITERATION_NAME, byte_io)
        cropped_image = image  # 기본: 원본 이미지

        for prediction in detection_results.predictions:
            if prediction.probability >= 0.9 and prediction.tag_name == "Body":
                img_width, img_height = image.size
                left = int(prediction.bounding_box.left * img_width)
                top = int(prediction.bounding_box.top * img_height)
                width = int(prediction.bounding_box.width * img_width)
                height = int(prediction.bounding_box.height * img_height)
                cropped_image = image.crop((left, top, left + width, top + height))
                break

        # 2️⃣ Classification (운전자 행동 인식)
        byte_io = BytesIO()
        cropped_image.save(byte_io, 'png')
        byte_io.seek(0)

        classification_results = classification_predictor.classify_image(
            classification_project_id, classification_publish_iteration_name, byte_io
        )

        if not classification_results.predictions:
            return "결과 없음", "image1.png", None, 0

        top_prediction = max(classification_results.predictions, key=lambda x: x.probability)
        action_name = top_prediction.tag_name
        kss_score = kss_mapping.get(action_name, 0)

        # 3️⃣ 전역 변수 업데이트
        global_frame_count += 1
        current_frame_time = global_frame_count * FRAME_INTERVAL
        global_time_points.append(current_frame_time)
        global_kss_values.append(kss_score)

        # 4️⃣ 평균 KSS 계산 (WINDOW_SIZE 기준)
        if len(global_kss_values) < WINDOW_SIZE:
            return "부팅중...", "image1.png", None, 0

        avg_kss = np.mean(global_kss_values[-WINDOW_SIZE:])
        risk_state, risk_image, audio_file = get_risk_status(avg_kss)

        # 5️⃣ 결과 반환
        results_text = f"🎬 실시간 행동 분석\n{action_name}: {top_prediction.probability * 100:.2f}%\n"
        results_text += f"🔹 KSS 점수: {kss_score} | 평균 KSS: {avg_kss:.2f}\n📊 위험 수준: {risk_state}\n"

        # 자동 재생을 위한 Audio 컴포넌트
        if audio_file:
            audio_output = gr.Audio(value=audio_file, autoplay=True)
        else:
            audio_output = None

        audio_output = gr.Audio(value=audio_file, autoplay=True) if audio_file else None
        return results_text, risk_image, audio_output, avg_kss

    except Exception as e:
        return f"오류 발생: {str(e)}", "image1.png", None, 0

# === Gradio UI 구성 ===
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], type="pil")
        with gr.Column():
            output_label = gr.Textbox(label="🔍 분석 결과")
            output_image = gr.Image(label="📊 위험 수준", height=200, width=200)
            output_audio = gr.Audio(label="🔊 음성 경고")
            output_slider = gr.Slider(minimum=0, maximum=100, step=1, label="⚠️ 위험 수준")

    input_img.stream(analyze_frame, [input_img], [output_label, output_image, output_audio, output_slider])

if __name__ == "__main__":
    demo.launch(debug=True)