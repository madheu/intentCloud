"""快速验证: get_sememes 在 P0-R1 上下文是否工作"""
import sys, json
sys.path = [p for p in sys.path if 'Diviner' not in p]
import jieba
import OpenHowNet

hownet = OpenHowNet.HowNetDict()

def get_sememes(word):
    sems = set()
    try:
        data = hownet.get_sememes_by_word(word)
        if data:
            for entry in data:
                for s in entry.get("sememes", []):
                    if isinstance(s, str):
                        sems.add(s.split("|")[0].strip())
    except:
        pass
    return sems

# 模拟层5: 对第一个盲测句
sent = "专业能力提升后，进阶门径需要专门指导。"
words = jieba.lcut(sent)
context = [w for w in words if w != "门" and len(w) >= 2]
print(f"句: {sent}")
print(f"语境词: {context}")
for w in context:
    sems = get_sememes(w)
    print(f"  {w}: {len(sems)} 义原: {list(sems)[:5]}")

ctx_sems = set()
for w in context:
    ctx_sems.update(get_sememes(w))
print(f"\n语境义原总数: {len(ctx_sems)}")

# 歧义词
word_sems = get_sememes("门")
print(f"\n'门' 的义原: {len(word_sems)}: {list(word_sems)[:10]}")
overlap = len(word_sems & ctx_sems) / max(len(word_sems), 1)
print(f"重叠: {overlap:.3f}")
