# H18: 端到端场景测试
# 场景1: 多实体多话题
# 场景2: 情绪过山车
# 场景3: 记忆检索
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.discourse_entities import DiscourseEntities, EntityType, Gender
from core.correction_analyzer import CorrectionAnalyzer, CorrectionType
from core.conflict_resolver import ConflictResolver
from core.emotion_sphere import EmotionSphere
from core.structured_memory import StructuredMemory, OutputSummary

def make_cloud():
    class N:
        def __init__(s,i,t): s.id,s.text,s.trust=i,t,0.5
    class C:
        def __init__(s): s._shell = {k:N(k,v) for k,v in [
            ('emotion_fear','恐惧'),('emotion_anger','愤怒'),
            ('emotion_sadness','悲伤'),('emotion_joy','喜悦'),
            ('emotion_calm','平静'),('emotion_awe','敬畏'),
            ('emotion_nervous','紧张'),('nature_ocean','大海'),
        ]}
    return C()

cloud = make_cloud()
p=f=t=0
def chk(nm, ok, dt=""):
    global p,f,t; t+=1
    if ok: p+=1; print(f"  OK [{nm}] {dt}")
    else: f+=1; print(f"  FAIL [{nm}] {dt}")

# Shared instances
de = DiscourseEntities(); es = EmotionSphere(); cr = ConflictResolver()
ca = CorrectionAnalyzer(); sm = StructuredMemory()

print("=== Scenario 1: Multi-entity Multi-topic ===")
# Turn 1: Ocean topic
de.add_from_text('\u5927\u6d77\u3001\u5929\u7a7a\u3001\u6d77\u9e25')
de.next_turn(); es.compute({'emotion_calm':0.6,'emotion_joy':0.4})
sm.add_output(OutputSummary(intent='\u63cf\u8ff0', topic='\u5927\u6d77', emotion='\u5e73\u9759'))
r1 = de.resolve('\u5b83')
chk("t1_anaphora_haiou", r1=='海鸥', f"resolved={r1}")

# Turn 2: Movie topic (topic switch)
de.add_from_text('\u7535\u5f71\u3001\u7535\u5f71\u9662')
de.next_turn(); es.compute({'emotion_joy':0.7,'emotion_awe':0.5})
sm.add_output(OutputSummary(intent='\u804a\u5929', topic='电影', emotion='\u6b23\u559c'))
# Anaphora should prefer movie entities now
de.add('小黑', EntityType.PERSON, Gender.FEMALE)
de.next_turn()
r2 = de.resolve('她')
chk("t2_anaphora_xiaohei", r2=='小黑', f"resolved={r2}")

# Turn 3: Back to emotion topic
de.next_turn(); es.compute({'emotion_fear':0.7,'emotion_nervous':0.6})
ct3,_ = ca.classify('太悲伤了')
sm.add_output(OutputSummary(intent='\u804a\u5929', emotion='\u60b2\u4f24'))
chk("t3_emotion_shift", es.need_positive_balance(), f"v={es.current.valence:.2f} a={es.current.arousal:.2f}")
print(f"  emotion_trend: v={es.current.valence:.2f} a={es.current.arousal:.2f}")
print(f"  memory_chains: {len(sm.chains)} narrative, {len(sm.points)} discrete")

print("\n=== Scenario 2: Emotional Rollercoaster ===")
# Calm -> Angry -> Sad -> Calm
es.compute({'emotion_calm':0.6,'emotion_joy':0.3})
chk("s2_calm", es.need_positive_balance()==False)
es.compute({'emotion_anger':0.9,'emotion_fear':0.7})
chk("s2_angry_trigger", es.need_positive_balance()==True)
es.compute({'emotion_sadness':0.8,'emotion_nervous':0.6})
chk("s2_sad_trigger", es.need_positive_balance()==True)
es.compute({'emotion_calm':0.7,'emotion_joy':0.5})
chk("s2_calm_again", es.need_positive_balance()==False)
print(f"  trajectory: {[(round(h.valence,2), round(h.arousal,2)) for h in es.history]}")

print("\n=== Scenario 3: Memory Retrieval ===")
sm2 = StructuredMemory()
sm2.add_output(OutputSummary(intent='\u5199\u8bd7', style='现代', emotion='敬畏', topic='大海'))
sm2.add_output(OutputSummary(intent='\u5199\u8bd7', style='现代', emotion='悲伤', topic='大海'))
# Later query
results = sm2.search(topic='\u5927\u6d77', intent='\u5199\u8bd7')
chk("s3_memory_retrieval", len(results)>=2, f"found {len(results)} items")
# Verify narrative chain created
chk("s3_narrative_chain", len(sm2.chains)>=1, f"{len(sm2.chains)} chains")
# Check chain contains both items
if sm2.chains:
    chk("s3_chain_items", len(sm2.chains[0].items)>=2, f"{len(sm2.chains[0].items)} items in chain")
print(f"  chains={len(sm2.chains)}, points={len(sm2.points)}")

print(f"\nResults: {p}/{t} pass, {f} fail")
