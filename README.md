# 졸음운전 감지 시스템 🚗

## ⚙️ 기술 스택

- **AI/ML**

  - Azure Custom Vision (Object Detection & Classification)
  - OpenCV
  - NumPy
  - Pandas

- **프론트엔드**

  - Gradio (웹 인터페이스)
  - Python

- **데이터 처리**
  - Jupyter Notebook
  - Python Data Analysis Libraries

## 🚫 주의사항

현재는 Azure 리소스 연결이 끊겨 있어, 일부 기능은 실행되지 않을 수 있습니다.

## ✅ 당시 구현 내용

1. **실시간 졸음 감지**

   - 웹캠을 통한 실시간 얼굴 인식
   - Azure Custom Vision을 활용한 객체 감지
   - 운전자 행동 분류 및 분석

2. **위험도 평가**

   - KSS(Karolinska Sleepiness Scale) 기반 졸음 위험도 측정
   - 실시간 위험도 점수 계산
   - 단계별 경고 시스템 구현

3. **사용자 인터페이스**

   - Gradio를 활용한 직관적인 웹 인터페이스
   - 실시간 분석 결과 시각화
   - 경고 알림 시스템

4. **데이터 분석**
   - 운전자 행동 패턴 분석
   - 졸음 위험도 통계 분석
   - 모델 성능 평가 및 개선

## 🔧 설치 및 실행

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. Azure Custom Vision 설정:

   - `newUI.py` 파일의 API 키와 엔드포인트 설정
   - 프로젝트 ID와 iteration 이름 설정

3. 실행:

```bash
python newUI.py
```
