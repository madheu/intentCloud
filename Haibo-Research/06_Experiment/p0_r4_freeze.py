"""P0-R4 冻结 — 生成 manifest + 运行测试"""
import json, os, sys, hashlib, subprocess

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
OUTPUT = os.path.join(BASE, "output")
FROZEN_DIR = os.path.join(BASE, "output", "p0_r4_frozen")
os.makedirs(FROZEN_DIR, exist_ok=True)
MANIFEST_PATH = os.path.join(FROZEN_DIR, "P0-R4_frozen_manifest.sha256")
TEST_LOG = os.path.join(FROZEN_DIR, "test_log.txt")

# 1. 运行测试（失败时非零退出）
print("运行测试...")
result = subprocess.run([sys.executable, os.path.join(BASE, "p0_r4_tests.py")],
                        capture_output=True, text=True, cwd=BASE)
with open(TEST_LOG, "w", encoding="utf-8") as f:
    for line in (result.stdout + "\n--- stderr ---\n" + result.stderr).split("\n"):
        if "TreatControl" not in line and "设置" not in line and "句柄" not in line:
            f.write(line + "\n")
assert result.returncode == 0, f"测试失败, 退出码 {result.returncode}"
print(f"测试通过, 日志 → {TEST_LOG}")

# 2. 复制冻结产物
import shutil
for fn in ["p0_r4_config.json", "p0_r4_predict.py", "p0_r4_validate_input.py",
           "p0_r4_validate_output.py", "p0_r4_score.py", "p0_r4_tests.py",
           "p0_r4_build.py"]:
    shutil.copy2(os.path.join(BASE, fn), os.path.join(FROZEN_DIR, fn))
for fn in ["p0_r4_sememe_map.json", "p0_r4_prototypes.npz"]:
    shutil.copy2(os.path.join(OUTPUT, fn), os.path.join(FROZEN_DIR, fn))

# 3. 记录依赖版本
deps = {}
try:
    import importlib.metadata as ilm
    for pkg in ["jieba", "gensim", "numpy", "OpenHowNet"]:
        try: deps[pkg] = ilm.version(pkg)
        except: deps[pkg] = "?"
except: pass
deps["python"] = sys.version
deps["wordvec"] = "Tencent_AILab_ChineseEmbedding light 143613x200" 
deps["hownet"] = "OpenHowNet bundled"
with open(os.path.join(FROZEN_DIR, "dependencies.json"), "w") as f:
    json.dump(deps, f, indent=2)

# 4. 生成 manifest（真实相对路径）
files = ["p0_r4_config.json", "p0_r4_predict.py", "p0_r4_validate_input.py",
         "p0_r4_validate_output.py", "p0_r4_score.py", "p0_r4_tests.py",
         "p0_r4_build.py", "p0_r4_sememe_map.json", "p0_r4_prototypes.npz",
         "test_log.txt", "dependencies.json"]
manifest = {}
for fn in files:
    fp = os.path.join(FROZEN_DIR, fn)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            manifest[fn] = hashlib.sha256(f.read()).hexdigest()
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

# 5. 独立校验器重新核验
print(f"\nManifest: {MANIFEST_PATH}")
all_ok = True
for fn, expected in manifest.items():
    fp = os.path.join(FROZEN_DIR, fn)
    if not os.path.exists(fp):
        print(f"  ❌ {fn}: 文件缺失")
        all_ok = False; continue
    with open(fp, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    ok = actual == expected
    if not ok:
        print(f"  ❌ {fn}: 哈希不匹配")
        all_ok = False
    else:
        print(f"  ✅ {fn}: {actual[:16]}...")
print(f"\n所有文件 {'✅' if all_ok else '❌'}")
print(f"\nP0-R4 已冻结。等待 PI 验收，不再读取任何盲测数据。")
