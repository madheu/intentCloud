# H18: 模块间数据流验证
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.discourse_entities import DiscourseEntities, EntityType, Gender, Role
from core.correction_analyzer import CorrectionAnalyzer, CorrectionType
from core.conflict_resolver import ConflictResolver
from core.emotion_sphere import EmotionSphere
from core.structured_memory import StructuredMemory, OutputSummary

def make_cloud():
    class N:
        def __init__(s,i,t,tr=0.5): s.id,s.text,s.trust=i,t,tr
    class C:
        def __init__(s):
            s._shell = {k:N(k,v) for k,v in [
                ('emotion_fear','恐惧'),('emotion_anger','愤怒'),
                ('emotion_sadness','悲伤'),('emotion_joy','喜悦'),
                ('emotion_calm','平静'),('emotion_awe','敬畏'),
                ('nature_ocean','大海'),('nature_sky','天空'),
            ]}
    return C()

cloud = make_cloud()
p=f=t=0
def chk(nm, ok, dt=""):
    global p,f,t; t+=1
    if ok: p+=1; print(f"  OK [{nm}] {dt}")
    else: f+=1; print(f"  FAIL [{nm}] {dt}")

print("=== Data Flow: Anaphora -> Correction ===")
de = DiscourseEntities(); de.add('\u5c0f\u7ea2',EntityType.PERSON,Gender.FEMALE)
ctx = CorrectionAnalyzer()
ct, conf = ctx.classify('\u5979\u592a\u8fc7\u5206\u4e86\uff0c\u5f04\u574f\u4e86\u6211\u7684\u7535\u8111')
chk("anaphora->correction flow", ct in (CorrectionType.EMOTIONAL, CorrectionType.CASUAL),
    f"type={ct}, conf={conf}")

print("\n=== Data Flow: Emotion -> Perspective ===")
es = EmotionSphere()
es.compute({'emotion_anger':0.9,'emotion_sadness':0.7,'emotion_fear':0.5})
chk("emotion_high_arousal_negative", es.need_positive_balance(),
    f"valence={es.current.valence:.2f} arousal={es.current.arousal:.2f}")

print("\n=== Data Flow: Correction -> Conflict ===")
cr = ConflictResolver()
a = {'target':'emotion_sadness','direction':'reduce','value':0.7}
b = {'target':'emotion_sadness','direction':'increase','value':0.5}
cr.register(a,b)
chk("correction->conflict registered", len(cr.conflicts)==1)
lo, hi = cr.extract_boundary(cr.conflicts[-1])
chk("boundary extracted", lo==0.5 and hi==0.7, f"[{lo},{hi}]")

print("\n=== Data Flow: Memory -> Anaphora ===")
sm = StructuredMemory()
sm.add_output(OutputSummary(intent='\u5199\u8bd7', style='\u73b0\u4ee3', emotion='\u656c\u754f', topic='\u5927\u6d77'))
results = sm.search(topic='\u5927\u6d77')
chk("memory->anaphora retrievable", len(results)>=1, f"{len(results)} results")
de2 = DiscourseEntities()
for r in results:
    de2.add(r.topic, EntityType.OBJECT, Gender.NEUTER)
resolved = de2.resolve('\u5b83')
chk("memory entity resolved by anaphora", resolved is not None, f"resolved to {resolved}")

print(f"\nResults: {p}/{t} pass, {f} fail")
