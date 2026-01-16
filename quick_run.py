"""
Deepfake Detection - 빠른 실행 버전
"""

print("\n" + "="*80)
print("🚀 Deepfake Detection 추론 시작")
print("="*80 + "\n")

# 1단계: 기본 라이브러리 로드
print("[준비 1/3] 라이브러리 로드 중...")
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

print("✓ 기본 라이브러리 로드 완료")

# 2단계: 설정
print("[준비 2/3] 설정 초기화 중...")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TEST_DIR = Path("./open/test_data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "baseline_enhanced_submission.csv"

print(f"✓ 설정 완료")
print(f"  - 테스트 폴더: {TEST_DIR}")
print(f"  - 출력 폴더: {OUTPUT_DIR}")

# 3단계: 모델 로드 (지연 로드)
print("[준비 3/3] 모델 로드 중...")
try:
    import torch
    import torch.nn.functional as F
    from transformers import ViTForImageClassification, ViTImageProcessor
    from PIL import Image
    import cv2
    from tqdm import tqdm
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"  모델 로드 중... (이 부분이 1-2분 걸릴 수 있습니다)")
    model = ViTForImageClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(DEVICE)
    processor = ViTImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    
    print(f"✓ 모델 로드 완료 (Device: {DEVICE})")
    
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    print("\n모델 로드에 문제가 있습니다.")
    print("다음을 시도하세요:")
    print("1. python run_inference.py 실행")
    print("2. 또는 baseline_enhanced.ipynb를 Jupyter에서 실행")
    exit(1)

# ============================================================================
# 실제 추론
# ============================================================================

print("\n" + "="*80)
print("🔄 추론 수행 중...")
print("="*80 + "\n")

# 파일 수집
files = sorted([p for p in TEST_DIR.iterdir() if p.is_file()])
print(f"📂 테스트 파일: {len(files)}개\n")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}
TARGET_SIZE = (224, 224)
NUM_FRAMES = 15

def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
    ext = file_path.suffix.lower()
    if ext in IMAGE_EXTS:
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except:
            return []
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        frame_indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    return []

def get_full_frame_padded(pil_img: Image.Image, target_size=(224, 224)) -> Image.Image:
    img = pil_img.convert("RGB")
    img.thumbnail(target_size, Image.BICUBIC)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                        (target_size[1] - img.size[1]) // 2))
    return new_img

def infer_fake_probs(pil_images: List[Image.Image], batch_size: int = 8) -> List[float]:
    if not pil_images:
        return []
    probs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        with torch.inference_mode():
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = F.softmax(logits, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().tolist())
    return probs

results: Dict[str, float] = {}

for file_path in tqdm(files, desc="추론", unit="파일"):
    try:
        frames = read_rgb_frames(file_path)
        imgs = []
        
        for rgb in frames:
            pil_img = get_full_frame_padded(Image.fromarray(rgb), TARGET_SIZE)
            imgs.append(pil_img)
        
        if imgs:
            # 기본 추론
            probs = infer_fake_probs(imgs)
            if probs:
                results[file_path.name] = float(np.mean(probs))
            else:
                results[file_path.name] = 0.0
        else:
            results[file_path.name] = 0.0
    except Exception as e:
        results[file_path.name] = 0.0

print(f"\n✓ 추론 완료: {len(results)}개 파일 처리됨\n")

# ============================================================================
# 제출 파일 생성
# ============================================================================

print("📝 제출 파일 생성 중...\n")

submission = pd.read_csv('./open/sample_submission.csv')
submission['prob'] = submission['filename'].map(results).fillna(0.0)

# 통계
print("📊 결과 통계:")
print(f"  • 최소값: {submission['prob'].min():.4f}")
print(f"  • 최대값: {submission['prob'].max():.4f}")
print(f"  • 평균값: {submission['prob'].mean():.4f}")
print(f"  • 중앙값: {submission['prob'].median():.4f}")
print(f"  • Fake 예측 (prob > 0.5): {(submission['prob'] > 0.5).sum()}개\n")

# CSV 저장
submission.to_csv(OUT_CSV, encoding='utf-8-sig', index=False)

print("="*80)
print(f"✅ 완료! 결과 파일: {OUT_CSV}")
print("="*80)
print(f"\n📦 파일 크기: {OUT_CSV.stat().st_size:,} bytes")
print(f"📋 행 수: {len(submission)}")
print(f"✨ 결과를 제출하면 완료됩니다!\n")
