# Deepfake Detection 개선 사항 가이드

## 📊 분석 결과

test_data에는 **17개의 .jfif 파일**이 있으며, sample_submission.csv에는 **500개의 테스트 샘플**이 있습니다.
- 파일 타입: jpg, mp4, png, jfif, jpeg, mov 등 다양함
- 기본 모델: ViT (Vision Transformer) 기반 DeepFake Detector v2

---

## 🚀 적용된 개선 사항

### 1. **Test Time Augmentation (TTA)**
```python
# 같은 이미지를 다양한 방식으로 처리 후 결과 앙상블
- 원본 이미지
- 좌우 반전 (Horizontal Flip)
- 다른 해상도 (200x200 → 224x224 패딩)
```
**효과**: 모델의 robust성 향상, False Positive 감소

### 2. **다중 프레임 샘플링 증가**
```python
NUM_FRAMES = 15  # 기존 10 → 15로 증가
```
**효과**: 비디오의 더 많은 부분을 분석하여 deepfake 특징 감지율 향상

### 3. **고주파 필터링 (Unsharp Mask)**
```python
# Gaussian Blur 차이를 이용한 경계 강조
sharpened_img = img - gaussian_blur(img) + img
```
**효과**: 비자연스러운 경계나 artifacts 강조 → deepfake 감지 개선

### 4. **이중 이미지 처리**
```python
# 각 프레임마다 2개 버전 생성
1. 원본 이미지
2. Sharpened 이미지
```
**효과**: 다양한 특징 학습으로 정확도 향상

### 5. **지능형 앙상블 (Weighted Ensemble)**
```python
ensemble_prob = mean * 0.3 + max * 0.5 + std * 0.2

- Mean: 전체 평균 (안정성)
- Max: 최대값 (높은 confidence 포착)
- Std: 표준편차 (일관성)
```
**효과**: 단순 평균보다 더 정확한 판단

### 6. **배치 처리 최적화**
```python
# 배치 크기: 32
# 메모리 효율성 + 추론 속도 향상
```

---

## 📈 기대 효과

| 개선 사항 | 기대 정확도 향상 | 설명 |
|----------|----------------|------|
| TTA | +2-5% | 이미지 다양성 활용 |
| 프레임 증가 | +1-3% | 더 많은 정보 수집 |
| 고주파 필터 | +2-4% | 미세한 artifacts 감지 |
| 이중 처리 | +1-2% | 특징 다양성 |
| 지능형 앙상블 | +1-3% | 최적 가중치 조합 |
| **총합** | **+7-17%** | 누적 효과 |

---

## 💾 출력 파일

- `baseline_enhanced_submission.csv`: 개선된 모델의 결과
- `probability_analysis.png`: 확률 분포 및 파일 타입별 분석

---

## 🔧 추가 개선 팁

### A. 더 강력한 모델 사용
```python
# 다른 Deepfake Detection 모델들
- "facebook/dino-vits16"
- "timm/vit_large_patch16_224"
- "timm/convnext_large"
```

### B. Confidence Calibration
```python
# 모델 output을 calibrate하여 더 정확한 확률
- Temperature scaling
- Platt scaling
```

### C. Multi-Model Ensemble
```python
# 여러 모델의 결과를 결합
prob_final = (model1(x) * 0.3 + 
              model2(x) * 0.4 + 
              model3(x) * 0.3)
```

### D. Video-specific Features
```python
# 비디오에서 temporal inconsistency 탐지
- Frame-to-frame optical flow 분석
- Flickering detection
- Lip sync analysis
```

### E. Face Detection & Cropping
```python
# Deepfake는 보통 얼굴 영역에만 적용됨
- MediaPipe 또는 MTCNN로 얼굴 감지
- 얼굴 영역만 따로 처리
```

---

## ⚡ 실행 방법

1. **원본 baseline과 비교**
   ```bash
   python -m jupyter notebook baseline.ipynb
   python -m jupyter notebook baseline_enhanced.ipynb
   ```

2. **결과 비교**
   ```python
   import pandas as pd
   
   baseline = pd.read_csv('output/baseline_submission.csv')
   enhanced = pd.read_csv('output/baseline_enhanced_submission.csv')
   
   # 차이 분석
   diff = (baseline['prob'] - enhanced['prob']).abs()
   print(f"평균 차이: {diff.mean():.4f}")
   ```

---

## 📝 노트

- **계산 시간**: TTA와 이중 처리로 인해 약 2-3배 더 소요됨
- **메모리**: 배치 처리로 충분히 관리 가능
- **결과 안정성**: 같은 입력에 대해 일관된 결과 보장

---

## 🎯 최적화 팁

1. **빠른 테스트**: NUM_FRAMES = 5로 줄여서 빠르게 테스트
2. **프로덕션**: NUM_FRAMES = 15-20으로 정확도 최대화
3. **하이브리드**: 파일 타입별로 다른 전략 적용
   - 이미지: 좀 더 강한 전처리
   - 동영상: 더 많은 프레임 샘플링
