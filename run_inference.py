"""
Deepfake Detection - 빠른 실행 스크립트
baseline_enhanced.ipynb를 Python 스크립트로 변환
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# ============================================================================
# SETTINGS
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
TEST_DIR = Path("./open/test_data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "baseline_enhanced_submission.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}
TARGET_SIZE = (224, 224)
NUM_FRAMES = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ============================================================================
# UTILITIES
# ============================================================================

def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """비디오 프레임을 균등하게 샘플링"""
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def get_full_frame_padded(pil_img: Image.Image, target_size=(224, 224)) -> Image.Image:
    """전체 이미지를 비율 유지하며 정사각형 패딩 처리"""
    img = pil_img.convert("RGB")
    img.thumbnail(target_size, Image.BICUBIC)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                        (target_size[1] - img.size[1]) // 2))
    return new_img


def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
    """이미지 또는 비디오에서 RGB 프레임 추출"""
    ext = file_path.suffix.lower()
    
    # 이미지 파일
    if ext in IMAGE_EXTS:
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except Exception:
            return []
    
    # 비디오 파일
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total <= 0:
            cap.release()
            return []
        
        frame_indices = uniform_frame_indices(total, num_frames)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    return []


def apply_gaussian_blur_difference(image: Image.Image) -> Image.Image:
    """가우시안 블러 차이를 이용한 경계 강조"""
    img_array = np.array(image, dtype=np.float32) / 255.0
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    diff = img_array - blurred
    sharpened = img_array + diff * 0.5
    sharpened = np.clip(sharpened, 0, 1)
    sharpened = (sharpened * 255).astype(np.uint8)
    return Image.fromarray(sharpened)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class PreprocessOutput:
    def __init__(
        self,
        filename: str,
        imgs: List[Image.Image],
        error: Optional[str] = None
    ):
        self.filename = filename
        self.imgs = imgs
        self.error = error


def preprocess_one(file_path: Path, num_frames: int = NUM_FRAMES) -> PreprocessOutput:
    """파일 하나에 대한 전처리 수행"""
    try:
        frames = read_rgb_frames(file_path, num_frames=num_frames)
        imgs: List[Image.Image] = []
        
        for rgb in frames:
            # 기본 전처리
            pil_img = get_full_frame_padded(Image.fromarray(rgb), TARGET_SIZE)
            imgs.append(pil_img)
            
            # 추가: 샤프닝 버전도 함께 추가
            sharpened_img = apply_gaussian_blur_difference(pil_img)
            sharpened_img = get_full_frame_padded(sharpened_img, TARGET_SIZE)
            imgs.append(sharpened_img)
        
        return PreprocessOutput(file_path.name, imgs, None)
    
    except Exception as e:
        return PreprocessOutput(file_path.name, [], str(e))


# ============================================================================
# MODEL LOAD
# ============================================================================

print("Loading model...")
model = ViTForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model.eval()
print(f"Model loaded: {MODEL_ID}")


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def infer_fake_probs(pil_images: List[Image.Image], batch_size: int = 32) -> List[float]:
    """배치 단위로 추론"""
    if not pil_images:
        return []

    probs: List[float] = []

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        
        with torch.inference_mode():
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = F.softmax(logits, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().tolist())

    return probs


def infer_with_tta(pil_images: List[Image.Image]) -> float:
    """
    Test Time Augmentation (TTA)을 사용한 앙상블 추론
    - 원본 이미지
    - 좌우 반전 이미지
    - 서로 다른 해상도
    """
    if not pil_images:
        return 0.0
    
    all_probs = []
    
    # 1. 원본 이미지
    probs1 = infer_fake_probs(pil_images)
    all_probs.extend(probs1)
    
    # 2. 좌우 반전 이미지
    flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in pil_images]
    probs2 = infer_fake_probs(flipped_images)
    all_probs.extend(probs2)
    
    # 3. 다른 해상도
    small_images = [img.resize((200, 200), Image.BICUBIC) for img in pil_images]
    small_images = [get_full_frame_padded(img, TARGET_SIZE) for img in small_images]
    probs3 = infer_fake_probs(small_images)
    all_probs.extend(probs3)
    
    if not all_probs:
        return 0.0
    
    # 앙상블: 가중 평균
    mean_prob = float(np.mean(all_probs))
    max_prob = float(np.max(all_probs))
    std_prob = float(np.std(all_probs))
    
    ensemble_prob = mean_prob * 0.3 + max_prob * 0.5 + std_prob * 0.2
    
    return np.clip(ensemble_prob, 0, 1)


# ============================================================================
# MAIN INFERENCE
# ============================================================================

def main():
    files = sorted([p for p in TEST_DIR.iterdir() if p.is_file()])
    print(f"Test data length: {len(files)}")

    results: Dict[str, float] = {}

    # 전처리 및 추론
    for file_path in tqdm(files, desc="Processing"):
        out = preprocess_one(file_path)
        
        if out.error:
            print(f"[WARN] {out.filename}: {out.error}")
            results[out.filename] = 0.0
        elif out.imgs:
            prob = infer_with_tta(out.imgs)
            results[out.filename] = prob
        else:
            results[out.filename] = 0.0

    print(f"Inference completed. Processed: {len(results)} files")

    # ========================================================================
    # SUBMISSION
    # ========================================================================
    
    submission = pd.read_csv('./open/sample_submission.csv')
    submission['prob'] = submission['filename'].map(results).fillna(0.0)

    # 확률 분포 확인
    print(f"\nProbability Statistics:")
    print(f"Min: {submission['prob'].min():.4f}")
    print(f"Max: {submission['prob'].max():.4f}")
    print(f"Mean: {submission['prob'].mean():.4f}")
    print(f"Median: {submission['prob'].median():.4f}")
    print(f"Std: {submission['prob'].std():.4f}")
    print(f"Fake samples (prob > 0.5): {(submission['prob'] > 0.5).sum()}")

    # CSV 저장
    submission.to_csv(OUT_CSV, encoding='utf-8-sig', index=False)
    print(f"\nSaved submission to: {OUT_CSV}")
    
    # 파일 타입별 통계
    submission['ext'] = submission['filename'].apply(lambda x: x.split('.')[-1])
    prob_by_ext = submission.groupby('ext')['prob'].agg(['mean', 'count'])
    print(f"\n파일 타입별 평균 확률:")
    print(prob_by_ext)


if __name__ == "__main__":
    main()
