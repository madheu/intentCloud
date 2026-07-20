#!/usr/bin/env python3
# H16d: 暴力测试
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.dialog_context import DialogContext, CorrectionSignal

def mkcloud():
    class N:
        def __init__(s, i, t): s.id, s.text, s.trust = i, t, 0.5
    class C:
        def __init__(s):
            s._shell = {k: N(k,v) for k,v in [
                ('emotion_fear','恐惧'),('emotion_calm','平静'),
                ('emotion_joy','喜悦'),('emotion_awe','敬畏'),
                ('emotion_sadness','悲伤'),('nature_ocean','大海'),
            ]}
    return C()

cloud = mkcloud()
p=f=t=0
def chk(name, ok, dt=''):
    global p,f,t; t+=1
    if ok: p+=1; print(f'  OK [{name}] {dt}')
    else: f+=1; print(f'  FAIL [{name}] {dt}')

print('=== H16d: Brute Force Test ===')

# Scene 1: High-freq corrections
print('\n--- Scene 1: Repeated corrections ---')
ctx = DialogContext(max_pending=5)
for _ in range(10): ctx.detect_correction('太悲伤了', cloud)
chk('queue limit', ctx.pending_count == 5)
ctx.internalize_pending(cloud)
chk('cleared after internalize', ctx.pending_count == 0)

# Scene 2: Contradictory corrections
print('\n--- Scene 2: Contradictory ---')
ctx2 = DialogContext()
sig = ctx2.detect_correction('太悲伤了，但是再悲伤一点', cloud)
chk('contradiction detected', len(sig) >= 2)
dirs = [(s.target_node, s.direction) for s in sig]
has_inc = any(n=='emotion_sadness' and d=='increase' for n,d in dirs)
has_dec = any(n=='emotion_sadness' and d=='reduce' for n,d in dirs)
chk('both directions', has_inc and has_dec)

# Scene 3: Long text
print('\n--- Scene 3: Long text ---')
ctx3 = DialogContext()
msg = '今天去了海边。海浪拍打沙滩。但是太悲伤了。不只是悲伤，还有恐惧。'
sig3 = ctx3.detect_correction(msg, cloud)
chk('long text', len(sig3) >= 2)
print(f'  extracted {len(sig3)} signals: {[(s.type,s.target_node,s.direction) for s in sig3]}')

# Scene 4: Interrupt/resume
print('\n--- Scene 4: Interrupt/Resume ---')
ctx4 = DialogContext()
ctx4.update_topic('nature_ocean')
ctx4.detect_correction('太悲伤了', cloud)
chk('has topic before reset', ctx4.current_topic == 'nature_ocean')
chk('has pending before reset', ctx4.pending_count > 0)
ctx4.reset()
chk('topic cleared', ctx4.current_topic is None)
chk('pending cleared', ctx4.pending_count == 0)
ctx4.update_topic('nature_ocean')
chk('can resume', ctx4.current_topic == 'nature_ocean')
chk('anaphora works', ctx4.resolve_anaphora('它什么意思'))

# Scene 5: Mixed multi-intent
print('\n--- Scene 5: Mixed intent ---')
ctx5 = DialogContext()
sig5 = ctx5.detect_correction('写诗，不要太长，要敬畏，别夸张，押韵，不提海鸥', cloud)
chk('multi-intent', len(sig5) >= 1)
print(f'  {len(sig5)} signals: {[(s.type,s.target_node,s.direction) for s in sig5]}')

print(f'\nResults: {p}/{t} pass, {f} fail')
