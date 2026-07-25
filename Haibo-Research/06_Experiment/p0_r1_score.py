"""P0-R1: 评分脚本 (score 阶段)
读取预测文件 + 金标, 输出正确率
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")

LABEL_MAP_DEV = {
    "苹果科技公司": "A", "苹果水果": "B",
    "纸文书办公": "A", "纸日用包装": "B",
    "光物理光线": "A", "光修辞评价": "B",
    "花植物": "A", "花消费": "B",
    "行评价可以": "A", "行排列行列": "B", "行行业银行": "C",
    "口人体嘴巴": "A", "口出入口空间": "B",
}

LABEL_MAP_BLIND = {
    "门实体": "A", "门领域": "B",
    "纸日用包装": "B",
    "结具体": "A", "结抽象": "B",
    "心器官": "A", "心抽象": "B",
    "方形状": "A", "方方向": "B",
    "口人体": "A", "口空间": "B",
    "界边界": "A", "界领域": "B",
    "苹果科技公司": "A", "苹果水果": "B",
    "华为个人": "B", "华为科技": "A",
    "杜鹃鸟": "B", "杜鹃花": "A",
    "花消费": "B", "花植物": "A",
    "季节自然": "A", "季节行业": "B",
    "行排列": "B", "行评价": "A",
    "角几何": "A", "角动物角": "C",
    "小米谷物": "B", "小米科技": "A",
    "光修辞评价": "B", "光物理光线": "A",
    "头起始": "C", "头人体": "A", "头领导": "B",
    "长城公司": "B", "长城建筑": "A",
    "面抽象": "B",
    "点空间": "A", "点时间": "B",
}

def score(set_name):
    pred_file = os.path.join(OUTPUT, f"{set_name}_predictions.json")
    gold_file = os.path.join(DATA, f"{set_name}_gold.json")
    
    with open(pred_file, "r") as f:
        preds = json.load(f)
    with open(gold_file, "r") as f:
        golds = json.load(f)
    
    pred_map = {p["id"]: p["prediction"] for p in preds}
    gold_map = {g["id"]: (g["target_word"], g["_gold_label"], g["candidate_senses"]) for g in golds}
    
    label_map = LABEL_MAP_BLIND if set_name == "blind" else LABEL_MAP_DEV
    
    correct = 0
    total = len(preds)
    for pid, pred in pred_map.items():
        word, gold_label, senses = gold_map[pid]
        # 把 pred (义项名) 映射到 A/B/C
        key = f"{word}{pred}"
        pred_label = label_map.get(key, "A")
        if pred_label == gold_label:
            correct += 1
    
    acc = correct / total * 100
    print(f"{set_name}: {correct}/{total} ({acc:.1f}%)")
    return correct, total, acc

if __name__ == "__main__":
    print("=" * 50)
    print("P0-R1 评分")
    print("=" * 50)
    dev = score("dev")
    blind = score("blind")
    print(f"\n总计: dev={dev[2]:.1f}%, blind={blind[2]:.1f}%")
