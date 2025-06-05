import os
import gradio as gr
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

# 여러 이미지 분석 함수
def get_image_predictions(image_files):
    global cumulative_score, warnings_triggered

    results_text = ""
    
    for image_file in image_files:
        with open(image_file.name, "rb") as image_data:
            results = predictor.classify_image(
                classification_project_id,
                classification_publish_iteration_name,
                image_data.read()
            )

        # 상위 3개 예측 가져오기
        top_predictions_3 = sorted(results.predictions, key=lambda x: x.probability, reverse=True)[:3]
        top_prediction_1 = max(results.predictions, key=lambda x: x.probability)

        # 가장 확률 높은 행동의 점수 누적
        action_score = risk_scores.get(top_prediction_1.tag_name, 0)
        cumulative_score += action_score

        # 경고 메시지 확인
        warning_message = ""
        for threshold in warning_thresholds:
            if cumulative_score >= threshold and threshold not in warnings_triggered:
                warning_message = f"🚨 경고! 누적 점수 {threshold}점 초과 🚨"
                warnings_triggered.add(threshold)
                break

        # 결과 문자열 추가
        results_text += f"\n=== 이미지 ===\n"
        for prediction in top_predictions_3:
            results_text += f"{prediction.tag_name}: {prediction.probability * 100:.2f}% | 점수: {risk_scores.get(prediction.tag_name, 0)}\n"
        results_text += f"🔹 상위 1개 행동 점수: {action_score}\n"
        results_text += f"📊 현재 누적 점수: {cumulative_score}\n"

        if warning_message:
            results_text += f"\n{warning_message}\n"

    return results_text

# Gradio UI 설정
interface = gr.Interface(
    fn=get_image_predictions,
    inputs=gr.Files(label="🚗 여러 개의 이미지를 업로드하세요", file_types=["image"]),
    outputs="text",
    title="🔍 운전자 이상 행동 감지 시스템",
    description="🚦 여러 장의 이미지를 업로드하면 Azure AI로 분석하여 위험 행동 점수를 누적 계산합니다."
)

# 실행
interface.launch(debug=True)
