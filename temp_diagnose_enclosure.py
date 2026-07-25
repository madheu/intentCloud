"""诊断：围合特征到底长什么样，为什么不能区分义项。"""
import sys, json, os
from collections import Counter
sys.path.insert(0, r"E:\intentCloud\Haibo-Research\06_Experiment")

from hsg1_enclosure_features import HowNetInterface, EnclosureFeatureExtractor

P0_DATA_DIR = r"E:\intentCloud\Haibo-Research\06_Experiment\data"
LLM_DATA = r"E:\intentCloud\Haibo-Research\06_Experiment\p0_b2_llm_contexts.json"

def load_p0(split="dev"):
    input_path = os.path.join(P0_DATA_DIR, f"{split}_input.json")
    gold_path = os.path.join(P0_DATA_DIR, f"{split}_gold.json")
    with open(input_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        golds = json.load(f)
    gold_map = {g['id']: g['_gold_label'] for g in golds}
    result = []
    for item in inputs:
        result.append({
            'id': item['id'],
            'sentence': item['sentence'],
            'target_word': item['target_word'],
            'gold_sense': gold_map.get(item['id'], 'UNKNOWN'),
        })
    return result

extractor = EnclosureFeatureExtractor()

# 分析每个词
samples = load_p0("dev")
by_word = {}
for s in samples:
    by_word.setdefault(s['target_word'], []).append(s)

print("=" * 80)
print("围合特征诊断：每个句子提取到了什么义原")
print("=" * 80)

for word, word_samples in sorted(by_word.items()):
    print(f"\n{'─'*80}")
    print(f"【{word}】{len(word_samples)} 句")
    print(f"{'─'*80}")
    
    all_features = []
    for s in word_samples:
        feat = extractor.extract_features(
            sentence=s['sentence'],
            target_word=s['target_word'],
            occurrence=1,
        )
        all_features.append((s['sentence'], s['gold_sense'], feat))
    
    # 按义项分组看
    by_sense = {}
    for sent, gold, feat in all_features:
        by_sense.setdefault(gold, []).append((sent, feat))
    
    for sense, items in sorted(by_sense.items()):
        print(f"\n  义项【{sense}】({len(items)} 句):")
        for sent, feat in items:
            sememes = feat.get('sememes', {})
            context = feat.get('context_words', [])
            # 显示前5个义原
            top5 = dict(Counter(sememes).most_common(5))
            print(f"    「{sent[:30]}」")
            print(f"      上下文词: {context}")
            if top5:
                print(f"      义原: {top5}")
            else:
                print(f"      义原: (空)")
    
    # 看不同义项之间是否有义原交集
    print(f"\n  义原交集分析:")
    sense_sememes = {}
    for sense, items in by_sense.items():
        sememe_set = set()
        for _, feat in items:
            sememe_set.update(feat.get('sememes', {}).keys())
        sense_sememes[sense] = sememe_set
        print(f"    {sense}: {len(sememe_set)} 个唯一义原")
    
    if len(sense_sememes) >= 2:
        senses = list(sense_sememes.keys())
        s1, s2 = senses[0], senses[1]
        common = sense_sememes[s1] & sense_sememes[s2]
        only_s1 = sense_sememes[s1] - sense_sememes[s2]
        only_s2 = sense_sememes[s2] - sense_sememes[s1]
        print(f"    共同义原: {len(common)} 个")
        if common:
            print(f"      共同部分: {list(common)[:8]}")
        print(f"    义项1独有: {len(only_s1)} 个")
        print(f"    义项2独有: {len(only_s2)} 个")

print(f"\n{'='*80}")
print("LLM 语料抽样：每个词取第一个句子看围合效果")
print(f"{'='*80}")

with open(LLM_DATA, "r", encoding="utf-8") as f:
    llm_data = json.load(f)

for word, senses in sorted(llm_data.items()):
    print(f"\n{'─'*80}")
    print(f"【{word}】（{len(senses)} 个义项）")
    print(f"{'─'*80}")
    
    sense_sememes = {}
    for sense, sentences in senses.items():
        sememe_set = set()
        sample_count = min(3, len(sentences))
        print(f"\n  义项【{sense}】({len(sentences)} 句，抽样 {sample_count}):")
        for i in range(sample_count):
            sent = sentences[i]
            feat = extractor.extract_features(
                sentence=sent,
                target_word=word,
                occurrence=1,
            )
            sememes = feat.get('sememes', {})
            context = feat.get('context_words', [])
            top5 = dict(Counter(sememes).most_common(5))
            sememe_set.update(sememes.keys())
            print(f"    「{sent[:50]}」")
            print(f"      上下文: {context}")
            print(f"      义原: {top5 if top5 else '(空)'}")
        sense_sememes[sense] = sememe_set
        print(f"    该义项累计 {len(sememe_set)} 个唯一义原")
    
    # 义项间差异
    sense_list = list(sense_sememes.keys())
    if len(sense_list) >= 2:
        print(f"\n  义项间差异（第一 vs 第二义项）:")
        s1, s2 = sense_list[0], sense_list[1]
        common = sense_sememes[s1] & sense_sememes[s2]
        only_s1 = sense_sememes[s1] - sense_sememes[s2]
        only_s2 = sense_sememes[s2] - sense_sememes[s1]
        print(f"    共同义原: {len(common)} 个")
        print(f"    {s1}独有: {len(only_s1)} 个 — {list(only_s1)[:8]}")
        print(f"    {s2}独有: {len(only_s2)} 个 — {list(only_s2)[:8]}")
