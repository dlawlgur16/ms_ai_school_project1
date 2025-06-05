import gradio as gr
from io import BytesIO
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Azure Custom Vision 설정
classification_prediction_endpoint = "https://6b034cv20250210-prediction.cognitiveservices.azure.com/"
classification_prediction_key = "EAqbiKHt9sYSwFiO7U1NUBngdvnBoJRmd3ZLl0V0XzKkBbzibzPoJQQJ99BBACYeBjFXJ3w3AAAIACOGYNGQ"
classification_project_id = "b632b81b-f8d4-48ed-a74d-022aab130c5d"
classification_publish_iteration_name = "Iteration4"

# 인증 객체 생성
credentials = ApiKeyCredentials(in_headers={"Prediction-key": classification_prediction_key})
predictor = CustomVisionPredictionClient(endpoint=classification_prediction_endpoint, credentials=credentials)

# 행동별 위험 점수 설정
risk_scores = {
    "운전하다": 0, "중앙으로손을뻗다": 0, "손을뻗다": 0, "하품": 1, "눈비비기": 1,
    "몸못가누기": 3, "꾸벅꾸벅졸다": 3, "몸을돌리다": 2, "뺨때리기": 2, "목을 만지다": 1, "어깨를두드리다": 1
}

# 누적 점수 변수
cumulative_score = 0
warning_thresholds = [3, 6]
warnings_triggered = set()

def analyze_frame(image):
    """실시간 웹캠 프레임을 분석하여 행동 예측 및 점수 누적"""
    global cumulative_score, warnings_triggered

    try:
        # 이미지를 바이트 형태로 변환
        byte_io = BytesIO()
        image.save(byte_io, 'png')
        byte_io.seek(0)

        # Azure Custom Vision API 요청
        results = predictor.classify_image(
            classification_project_id,
            classification_publish_iteration_name,
            byte_io
        )

        if not results.predictions:
            return "결과 없음"

        # 상위 3개 행동 예측 가져오기
        top_predictions_3 = sorted(results.predictions, key=lambda x: x.probability, reverse=True)[:3]
        top_prediction_1 = top_predictions_3[0]  # 확률이 가장 높은 행동

        # 가장 높은 확률의 행동에 점수 부여
        action_name = top_prediction_1.tag_name
        action_score = risk_scores.get(action_name, 0)
        cumulative_score += action_score

        # 경고 메시지 확인
        warning_message = ""
        for threshold in warning_thresholds:
            if cumulative_score >= threshold and threshold not in warnings_triggered:
                warning_message = f"🚨 경고! 누적 점수 {threshold}점 초과 🚨"
                warnings_triggered.add(threshold)
                break

        # 결과 출력 형식
        results_text = f"🎬 실시간 행동 분석\n\n"
        for prediction in top_predictions_3:
            results_text += f"{prediction.tag_name}: {prediction.probability * 100:.2f}% | 점수: {risk_scores.get(prediction.tag_name, 0)}\n"
        results_text += f"🔹 상위 1개 행동: {action_name} | 점수: {action_score}\n"
        results_text += f"📊 현재 누적 점수: {cumulative_score}\n"

        if warning_message:
            results_text += f"\n{warning_message}\n"

        return results_text

    except Exception as e:
        return f"오류 발생: {str(e)}"

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], type="pil")  # 실시간 웹캠 입력
        with gr.Column():
            output_label = gr.Textbox(label="🔍 분석 결과")  # 행동 분석 결과 표시

    # 실시간 프레임 분석
    input_img.stream(analyze_frame, [input_img], [output_label])

# 앱 실행
if __name__ == "__main__":
    demo.launch(debug=True)