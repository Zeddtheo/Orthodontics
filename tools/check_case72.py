import json
from pathlib import Path
p = Path(__file__).resolve().parents[1] / 'datasets' / 'landmarks_dataset' / 'raw' / '72'
L = json.load(open(p / '72_L.json', 'r', encoding='utf-8'))
U = json.load(open(p / '72_U.json', 'r', encoding='utf-8'))
import re
JAW = re.compile(r'^(upper|lower)[_\-]?(.*)$', flags=re.IGNORECASE)

def norm_label(s):
    if not isinstance(s, str):
        return None
    cleaned = s.strip()
    m = JAW.match(cleaned)
    if m:
        return m.group(2)
    return cleaned


def ingest(data):
    out = {}
    if isinstance(data, dict) and 'markups' in data:
        for mk in data.get('markups', []):
            for cp in mk.get('controlPoints', []):
                lab = norm_label(cp.get('label'))
                pos = cp.get('position')
                if lab and isinstance(pos, (list,tuple)) and len(pos)==3:
                    out[lab] = pos
    return out

lm = {}
lm.update(ingest(L))
lm.update(ingest(U))
interesting = ['16mb','46mb','26mb','36mb']
print('presence:')
for k in interesting:
    print(k, k in lm)
print('total keys:', len(lm))
print('sample keys:', sorted(list(lm.keys()))[:40])
print('\nkeys starting with "36":')
for k in sorted(lm.keys()):
    if k.startswith('36'):
        print(k)
