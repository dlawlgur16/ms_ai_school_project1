import gradio as gr
from io import BytesIO
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import random

# Azure Custom Vision 설정
classification_prediction_endpoint = "https://6b003cv20250210-prediction.cognitiveservices.azure.com"
classification_prediction_key = "8ypy2B3ZECnRG0PaYKzpSNvOz8yAhfF7MY2z2wQxSzkweNlhgI4SJQQJ99BBACYeBjFXJ3w3AAAIACOG0WmE"
classification_project_id = "03fa2862-cb54-4344-b484-630379edffaa"
classification_publish_iteration_name = "Iteration4"

# 인증 객체 생성
credentials = ApiKeyCredentials(in_headers={"Prediction-key": classification_prediction_key})
predictor = CustomVisionPredictionClient(endpoint=classification_prediction_endpoint, credentials=credentials)

# 행동별 KSS 점수 설정 (제곱 값 적용)
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

def get_risk_status(avg_kss, slope):
    """
    윈도우 내 평균 KSS에 따라 5단계 위험 상태를 반환.
    """
    if avg_kss < 9:
        return "매우 안전", "image1.png", None
    elif avg_kss < 20:
        return "안전", "image2.png", None
    elif avg_kss < 40:
        return "주의", "image3.png", "약간 피로해보여요.잠시 쉬었다 가시는걸 추천드려요.mp3"
    elif avg_kss < 75:
        return "위험", "image4.png", "미치셨습니까 휴먼.mp3"
    else:
        return "매우 위험", "image5.png", "사이렌.mp3"

def analyze_frame(image):
    try:
        byte_io = BytesIO()
        image.save(byte_io, 'png')
        byte_io.seek(0)

        results = predictor.classify_image(
            classification_project_id,
            classification_publish_iteration_name,
            byte_io
        )

        if not results.predictions:
            return "결과 없음", "image1.png", None

        top_predictions_3 = sorted(results.predictions, key=lambda x: x.probability, reverse=True)[:3]
        top_prediction_1 = top_predictions_3[0]
        action_name = top_prediction_1.tag_name
        kss_score = kss_mapping.get(action_name, 0)
        avg_kss = kss_score

        # 위험 상태 및 이미지/음성 정보 가져오기
        risk_state, risk_image, audio_file = get_risk_status(avg_kss, None)

        results_text = f"🎬 실시간 행동 분석\n\n"
        for prediction in top_predictions_3:
            results_text += f"{prediction.tag_name}: {prediction.probability * 100:.2f}%\n"
        results_text += f"🔹 가장 유사한 행동: {action_name} | KSS 점수: {kss_score}\n"
        results_text += f"📊 현재 위험 수준: {risk_state}\n"

        # 자동 재생을 위한 Audio 컴포넌트
        if audio_file:
            audio_output = gr.Audio(value=audio_file, autoplay=True)
        else:
            audio_output = None

        return results_text, risk_image, audio_output, avg_kss

    except Exception as e:
        return f"오류 발생: {str(e)}", "image1.png", None

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