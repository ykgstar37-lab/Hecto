import json

with open('baseline_enhanced.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# 모든 셀 인덱스와 내용 출력
for i, cell in enumerate(nb['cells'][25:40]):
    src = ''.join(cell.get('source', []))[:80]
    cid = cell.get('id', 'no-id')
    print(f"{i+25}: [{cid}] {src}")
