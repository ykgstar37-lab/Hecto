"""
고급 Deepfake Detection - Multi-Model Ensemble
여러 모델을 결합하여 최고 성능 달성
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor,
    AutoImageProcessor,
    AutoModelForImageClassification
)

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

# 여러 모델 사용
MODELS = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    # 필요시 추가 모델 가능
    # "facebook/dino-vits16",
]

TEST_DIR = Path("./open/test_data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "advanced_ensemble_submission.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}
TARGET_SIZE = (224, 224)
NUM_FRAMES = 20  # 고급 버전은 더 많은 프레임 사용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Using {len(MODELS)} model(s): {MODELS}")


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
    
    if ext in IMAGE_EXTS:
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except Exception:
            return []
    
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


def apply_clahe_enhancement(image: Image.Image) -> Image.Image:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 그레이스케일을 RGB로 변환
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


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


def preprocess_one_advanced(file_path: Path, num_frames: int = NUM_FRAMES) -> PreprocessOutput:
    """고급 전처리: 다양한 버전의 이미지 생성"""
    try:
        frames = read_rgb_frames(file_path, num_frames=num_frames)
        imgs: List[Image.Image] = []
        
        for rgb in frames:
            pil_img = get_full_frame_padded(Image.fromarray(rgb), TARGET_SIZE)
            
            # 1. 원본
            imgs.append(pil_img)
            
            # 2. 샤프닝
            sharpened = apply_gaussian_blur_difference(pil_img)
            imgs.append(get_full_frame_padded(sharpened, TARGET_SIZE))
            
            # 3. CLAHE (명암 강화)
            clahe_img = apply_clahe_enhancement(pil_img)
            imgs.append(get_full_frame_padded(clahe_img, TARGET_SIZE))
        
        return PreprocessOutput(file_path.name, imgs, None)
    
    except Exception as e:
        return PreprocessOutput(file_path.name, [], str(e))


# ============================================================================
# MODELS
# ============================================================================

class MultiModelEnsemble:
    def __init__(self, model_ids: List[str], device: str = "cpu"):
        self.models = []
        self.processors = []
        self.model_ids = model_ids
        self.device = device
        
        print("Loading models...")
        for model_id in model_ids:
            try:
                print(f"  Loading {model_id}...")
                model = ViTForImageClassification.from_pretrained(model_id).to(device)
                processor = ViTImageProcessor.from_pretrained(model_id)
                model.eval()
                self.models.append(model)
                self.processors.append(processor)
                print(f"  ✓ {model_id} loaded")
            except Exception as e:
                print(f"  ✗ Failed to load {model_id}: {e}")
        
        print(f"Total models loaded: {len(self.models)}")
    
    def infer(self, pil_images: List[Image.Image], batch_size: int = 32) -> Dict[str, List[float]]:
        """모든 모델로 추론 및 각 모델의 결과 반환"""
        results = defaultdict(list)
        
        for model_idx, (model, processor) in enumerate(zip(self.models, self.processors)):
            probs: List[float] = []
            
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i+batch_size]
                
                with torch.inference_mode():
                    inputs = processor(images=batch, return_tensors="pt")
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    batch_probs = F.softmax(logits, dim=1)[:, 1]
                    probs.extend(batch_probs.cpu().tolist())
            
            results[f"model_{model_idx}"] = probs
        
        return results
    
    def ensemble_with_tta(self, pil_images: List[Image.Image]) -> float:
        """TTA와 다중 모델 앙상블"""
        all_probs = []
        
        # 1. 원본
        results = self.infer(pil_images)
        for model_probs in results.values():
            all_probs.extend(model_probs)
        
        # 2. 좌우 반전
        flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in pil_images]
        results = self.infer(flipped_images)
        for model_probs in results.values():
            all_probs.extend(model_probs)
        
        # 3. 상하 반전
        vflipped_images = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in pil_images]
        results = self.infer(vflipped_images)
        for model_probs in results.values():
            all_probs.extend(model_probs)
        
        if not all_probs:
            return 0.0
        
        # 지능형 앙상블
        mean_prob = float(np.mean(all_probs))
        median_prob = float(np.median(all_probs))
        max_prob = float(np.max(all_probs))
        percentile_75 = float(np.percentile(all_probs, 75))
        
        # 가중치 최적화
        ensemble_prob = (
            mean_prob * 0.25 +
            median_prob * 0.25 +
            percentile_75 * 0.35 +
            max_prob * 0.15
        )
        
        return np.clip(ensemble_prob, 0, 1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # 모델 로드
    ensemble = MultiModelEnsemble(MODELS, device=DEVICE)
    
    # 파일 처리
    files = sorted([p for p in TEST_DIR.iterdir() if p.is_file()])
    print(f"\nTest data length: {len(files)}")

    results: Dict[str, float] = {}

    for file_path in tqdm(files, desc="Processing"):
        out = preprocess_one_advanced(file_path)
        
        if out.error:
            print(f"[WARN] {out.filename}: {out.error}")
            results[out.filename] = 0.0
        elif out.imgs:
            prob = ensemble.ensemble_with_tta(out.imgs)
            results[out.filename] = prob
        else:
            results[out.filename] = 0.0

    print(f"Inference completed. Processed: {len(results)} files")

    # ========================================================================
    # SUBMISSION
    # ========================================================================
    
    submission = pd.read_csv('./open/sample_submission.csv')
    submission['prob'] = submission['filename'].map(results).fillna(0.0)

    print(f"\nProbability Statistics:")
    print(f"Min: {submission['prob'].min():.4f}")
    print(f"Max: {submission['prob'].max():.4f}")
    print(f"Mean: {submission['prob'].mean():.4f}")
    print(f"Median: {submission['prob'].median():.4f}")
    print(f"Std: {submission['prob'].std():.4f}")
    print(f"Fake samples (prob > 0.5): {(submission['prob'] > 0.5).sum()}")

    submission.to_csv(OUT_CSV, encoding='utf-8-sig', index=False)
    print(f"\nSaved submission to: {OUT_CSV}")
    
    # 파일 타입별 통계
    submission['ext'] = submission['filename'].apply(lambda x: x.split('.')[-1])
    prob_by_ext = submission.groupby('ext')['prob'].agg(['mean', 'count'])
    print(f"\n파일 타입별 평균 확률:")
    print(prob_by_ext)


if __name__ == "__main__":
    main()
