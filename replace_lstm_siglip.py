#!/usr/bin/env python
import json
from pathlib import Path

# 노트북 로드
nb_path = Path('baseline_enhanced.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# LSTM 섹션 찾기
lstm_start_idx = None
lstm_end_idx = None

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if 'ResNext50 + LSTM 모델' in source:
            lstm_start_idx = i
    
    # 다음 섹션(##) 또는 디버깅 섹션까지
    if lstm_start_idx is not None and i > lstm_start_idx:
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if source.startswith('##'):
                if 'LSTM' not in source and 'ResNext50' not in source:
                    lstm_end_idx = i
                    break

print(f"LSTM 섹션: {lstm_start_idx} ~ {lstm_end_idx}")
print(f"총 {lstm_end_idx - lstm_start_idx} 개 셀 제거")

# LSTM 섹션 셀 ID
if lstm_start_idx is not None and lstm_end_idx is not None:
    for i in range(lstm_start_idx, lstm_end_idx):
        print(f"  [{i}] {nb['cells'][i].get('id', 'no-id')}")

# 실제 제거 (백업 생성)
import shutil
shutil.copy(nb_path, nb_path.with_stem(nb_path.stem + '_backup'))

# LSTM 셀 제거
if lstm_start_idx is not None and lstm_end_idx is not None:
    del nb['cells'][lstm_start_idx:lstm_end_idx]
    
# 수정된 노트북 저장
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ LSTM 섹션 제거 완료")
print(f"📁 백업: {nb_path.with_stem(nb_path.stem + '_backup')}")
