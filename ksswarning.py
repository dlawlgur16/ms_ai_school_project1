import gradio as gr
from io import BytesIO
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Azure Custom Vision 설정
classification_prediction_endpoint = "https://6b003cv20250210-prediction.cognitiveservices.azure.com"
classification_prediction_key = "8ypy2B3ZECnRG0PaYKzpSNvOz8yAhfF7MY2z2wQxSzkweNlhgI4SJQQJ99BBACYeBjFXJ3w3AAAIACOG0WmE"
classification_project_id = "03fa2862-cb54-4344-b484-630379edffaa"
classification_publish_iteration_name = "Iteration4"

# 인증 객체 생성
credentials = ApiKeyCredentials(in_headers={"Prediction-key": classification_prediction_key})
predictor = CustomVisionPredictionClient(endpoint=classification_prediction_endpoint, credentials=credentials)

# KSS 위험 수준 매핑
kss_mapping = {
    "운전하다": 1, "하품": 3, "눈비비기": 4, "어깨를두드리다": 5, "목을만지다": 6, "뺨을때리다": 7, "꾸벅꾸벅졸다": 9, "몸못가누기": 10
}

risk_images = {
    "매우 안전": "image1.png",
    "안전": "image2.png",
    "주의": "image3.png",
    "위험": "image4.png",
    "매우 위험": "image5.png"
}

def get_risk_status(avg_kss):
    if avg_kss < 1:
        return "매우 안전"
    elif avg_kss < 3:
        return "안전"
    elif avg_kss < 5:
        return "주의"
    elif avg_kss < 9:
        return "위험"
    else:
        return "매우 위험"

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
            return "결과 없음", "image1.png"

        top_predictions_3 = sorted(results.predictions, key=lambda x: x.probability, reverse=True)[:3]
        top_prediction_1 = top_predictions_3[0]
        action_name = top_prediction_1.tag_name
        kss_score = kss_mapping.get(action_name, 0)
        avg_kss = kss_score
        risk_state = get_risk_status(avg_kss)

        results_text = f"🎬 실시간 행동 분석\n\n"
        for prediction in top_predictions_3:
            results_text += f"{prediction.tag_name}: {prediction.probability * 100:.2f}% | KSS 점수: {kss_mapping.get(prediction.tag_name, 0)}\n"
        results_text += f"🔹 상위 1개 행동: {action_name} | KSS 점수: {kss_score}\n"
        results_text += f"📊 현재 위험 수준: {risk_state}\n"

        return results_text, risk_images[risk_state]
    
    except Exception as e:
        return f"오류 발생: {str(e)}", "image1.png"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], type="pil")
        with gr.Column():
            output_label = gr.Textbox(label="🔍 분석 결과")
            output_image = gr.Image(label="📊 위험 수준", height=200, width=200)

    input_img.stream(analyze_frame, [input_img], [output_label, output_image])

if __name__ == "__main__":
    demo.launch(debug=True)