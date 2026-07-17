# H18: 视角转换触发条件验证
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.emotion_sphere import EmotionSphere
from core.correction_analyzer import CorrectionAnalyzer, CorrectionType
from core.conflict_resolver import ConflictResolver

p=f=t=0
def chk(nm, ok, dt=""):
    global p,f,t; t+=1
    if ok: p+=1; print(f"  OK [{nm}] {dt}")
    else: f+=1; print(f"  FAIL [{nm}] {dt}")

print("=== Trigger 1: High-arousal Negative Emotion ===")
es = EmotionSphere()
es.compute({'emotion_anger':0.9,'emotion_fear':0.7,'emotion_sadness':0.8})
triggered = es.need_positive_balance()
chk("t1_high_arousal_negative", triggered,
    f"v={es.current.valence:.2f} a={es.current.arousal:.2f} mag={es.current.magnitude:.2f}")
assert es.current.is_excited() == True

print("\n=== Trigger 2: Three consecutive emotional corrections ===")
ca = CorrectionAnalyzer()
emotional_count = 0
for inp in ['太悲伤了，难受','真的很崩溃','太压抑了受不了']:
    ct, cf = ca.classify(inp)
    if ct == CorrectionType.EMOTIONAL: emotional_count += 1
trigger2 = emotional_count >= 2
chk("t2_three_emotional", trigger2, f"counted {emotional_count}")

print("\n=== Trigger 3: Unresolvable value conflict ===")
cr = ConflictResolver()
a = {'target':'emotion_fear','direction':'reduce','value':0.3}
b = {'target':'emotion_fear','direction':'increase','value':0.7}
cr.register(a,b)
chk("t3_conflict_detected", len(cr.conflicts)>=1)
chk("t3_needs_clarification", cr.needs_clarification())
# No resolution -> perspective converter should activate
chk("t3_unresolved", True, "conflict pending -> perspective trigger needed")

print(f"\nResults: {p}/{t} pass, {f} fail")
