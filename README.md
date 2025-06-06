# 졸음운전 감지 시스템

## 구현 내용

### 1. 실시간 졸음 감지
- 웹캠을 활용한 실시간 얼굴 인식
- Azure Custom Vision을 이용한 객체 감지
- 운전자 행동 분류 및 분석 기능 구현

### 2. 위험도 평가
- KSS(Karolinska Sleepiness Scale)를 기반으로 한 졸음 위험도 측정
- 실시간 점수 계산을 통한 위험도 판단
- 위험 단계별 경고 시스템 구현

### 3. 사용자 인터페이스
- Gradio를 활용한 직관적인 웹 인터페이스 제공
- 실시간 분석 결과의 시각화 기능
- 경고 알림 시스템 연동

### 4. 데이터 분석
- 운전자 행동 패턴 및 졸음 위험도에 대한 통계 분석
- 수집된 데이터를 통한 모델 성능 평가 및 개선

---

## 기술 스택

### AI/ML
- Azure Custom Vision (Object Detection & Classification)
- OpenCV
- NumPy
- Pandas

### 프론트엔드
- Gradio (웹 인터페이스)
- Python

### 데이터 처리
- Jupyter Notebook
- Python 데이터 분석 라이브러리

---

## 주의사항

- 현재는 Azure 리소스 연결이 끊겨 있어 일부 기능이 정상적으로 작동하지 않을 수 있습니다.

---

## 설치 및 실행

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
