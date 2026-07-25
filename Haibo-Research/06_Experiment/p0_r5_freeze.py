"""P0-R5 冻结 + 自包含 dry-run + 测试"""
import json, os, sys, hashlib, subprocess, shutil

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
FROZEN = os.path.join(BASE, "output", "p0_r5_frozen")
OUT_RUN = os.path.join(BASE, "output", "p0_r5_run")
MANIFEST_PATH = os.path.join(FROZEN, "P0-R5_frozen_manifest.sha256")

# 1. 复制预测器和验证器到冻结目录
for fn in ["p0_r5_predict.py", "p0_r5_validate_input.py", "p0_r5_validate_output.py", "p0_r5_score.py"]:
    shutil.copy2(os.path.join(BASE, fn), os.path.join(FROZEN, fn))

# 2. 从冻结目录运行 dry-run
# 构建 120 条合成无标签输入
import p0_r5_validate_input as vi
vi.SENSE_IDS  # ensure loaded
dry_items = []
for word, count in vi.TARGET_COUNTS.items():
    cands = vi.SENSE_IDS[word]
    for i in range(count):
        dry_items.append({"id":f"dry-{word}-{i:03d}","sentence":f"测试{word}句子{i}。","target_surface":word,
                          "target_start":2,"target_end":2+len(word),"candidate_sense_ids":cands})
dry_path = os.path.join(OUT_RUN, "dry_input.json")
with open(dry_path, "w", encoding="utf-8") as f:
    json.dump(dry_items, f, ensure_ascii=False, indent=2)

print("运行 dry-run（从冻结目录）...")
env = os.environ.copy()
env["PYTHONPATH"] = FROZEN + os.pathsep + env.get("PYTHONPATH", "")
result = subprocess.run([sys.executable, os.path.join(FROZEN, "p0_r5_predict.py"), dry_path],
                        capture_output=True, text=True, env=env, timeout=120)
print(result.stderr)
if result.returncode != 0:
    print(f"❌ dry-run 失败, 退出码 {result.returncode}")
    print(result.stdout[:2000])
    print(result.stderr[:2000])
    sys.exit(1)
print(f"✅ dry-run 通过")

# 3. 记录依赖版本
deps = {"python": sys.version, "executable": sys.executable}
try:
    import importlib.metadata as ilm
    for pkg in ["jieba","gensim","numpy","OpenHowNet"]:
        try: deps[pkg] = ilm.version(pkg)
        except: deps[pkg] = "?"
except: pass
import hashlib as hl2
for fp_desc, fp in [(f"腾讯词向量", r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"),
                     (f"HowNet 数据", os.path.join(os.path.dirname(__import__('OpenHowNet').__file__), "resources"))]:
    try:
        if os.path.isfile(fp):
            with open(fp, "rb") as f: deps[fp_desc] = hl2.sha256(f.read()).hexdigest()[:16]
        else:
            deps[fp_desc] = "目录"
    except: deps[fp_desc] = "?"
with open(os.path.join(FROZEN, "dependencies.json"), "w") as f:
    json.dump(deps, f, indent=2)

# 4. 生成 manifest
files = sorted(os.listdir(FROZEN))
manifest = {}
for fn in files:
    if fn == "P0-R5_frozen_manifest.sha256":
        continue
    fp = os.path.join(FROZEN, fn)
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            manifest[fn] = hashlib.sha256(f.read()).hexdigest()
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

# 5. 独立校验器
print(f"\nManifest: {MANIFEST_PATH}")
all_ok = True
for fn, expected in manifest.items():
    fp = os.path.join(FROZEN, fn)
    if not os.path.exists(fp):
        print(f"  ❌ {fn}: 缺失"); all_ok = False; continue
    with open(fp, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    ok = actual == expected
    if not ok: print(f"  ❌ {fn}: 哈希不匹配"); all_ok = False
    else: print(f"  ✅ {fn}: {actual[:16]}...")
print(f"\n{'✅ 全部通过' if all_ok else '❌ 有错误'}")
if not all_ok: sys.exit(1)
print(f"\nP0-R5 已冻结。等待 PI 验收。")
