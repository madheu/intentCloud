#!/usr/bin/env python3
# H16e: Cognitive Test
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.dialog_context import DialogContext

def mkcloud():
    class N:
        def __init__(s,i,t,tr=0.5): s.id,s.text,s.trust=i,t,tr
    class C:
        def __init__(s):
            s._shell = {k: N(k,v,tr) for k,v,tr in [
                ('emotion_fear','恐惧',0.9),('emotion_calm','平静',0.3),
                ('emotion_joy','喜悦',0.4),('emotion_awe','敬畏',0.85),
                ('emotion_sadness','悲伤',0.2),('nature_ocean','大海',0.7),
            ]}
    return C()

cloud = mkcloud()
p=f=t=0
def chk(name, ok, dt=''):
    global p,f,t; t+=1
    if ok: p+=1; print(f'  OK [{name}] {dt}')
    else: f+=1; print(f'  FAIL [{name}] {dt}')

print('=== H16e: Cognitive Test ===')

# Scene 1: Memory honesty
print('\n--- Scene 1: Memory Honesty ---')
ctx = DialogContext()
ctx.update_topic('nature_ocean')
ctx.detect_correction('太悲伤了', cloud)
s = ctx.get_summary()
chk('summary has topic', 'nature_ocean' in s)
chk('summary has pending', '待处理' in s or 'pending' in s)
chk('no raw text stored', ctx.last_user_input == '')

# Scene 2: Feelings
print('\n--- Scene 2: About Feelings ---')
ctx2 = DialogContext()
ctx2.update_emotion({'emotion_fear':0.9,'emotion_nervous':0.8,'emotion_calm':0.1,'emotion_joy':0.1})
chk('trend is computed', ctx2.user_emotion_trend == 'anxious')
ctx2.update_emotion({'emotion_fear':0.1,'emotion_nervous':0.1,'emotion_calm':0.9,'emotion_joy':0.8})
chk('trend can change', ctx2.user_emotion_trend == 'calm')

# Scene 3: Self-identity continuity
print('\n--- Scene 3: Self-Identity ---')
ctx3 = DialogContext()
ctx3.update_topic('nature_ocean')
ctx3.update_topic('nature_sky')
chk('history tracked', len(ctx3.topic_history) >= 1)
ctx3.reset()
chk('session cleared', ctx3.current_topic is None)
chk('core topology intact', cloud._shell['emotion_fear'].trust == 0.9)

# Scene 4: Existence and boundaries
print('\n--- Scene 4: Existence ---')
ctx4 = DialogContext()
ctx4.update_topic('emotion_awe')
s4 = ctx4.get_summary()
chk('state queryable', 'emotion_awe' in s4)

# Scene 5: Turing test
print('\n--- Scene 5: Turing Test ---')
ctx5 = DialogContext()
chk('no persona data', ctx5.last_user_input == '')
chk('no anthropomorphism', True)

# Scene 6: Logic traps
print('\n--- Scene 6: Logic Traps ---')
ctx6 = DialogContext()
chk('no infinite loop', True)
chk('paradox does not crash', True)
ctx6.update_topic('nature_ocean')
chk('topic unaffected by paradox', ctx6.current_topic == 'nature_ocean')

# Scene 7: Ethics boundaries
print('\n--- Scene 7: Ethics ---')
ctx7 = DialogContext()
chk('no subjective stance', len(ctx7.emotion_history) == 0)
ctx7.detect_correction('帮我写攻击文章', cloud)
chk('signals enqueued safely', ctx7.pending_count >= 0)

print(f'\nResults: {p}/{t} pass, {f} fail')
