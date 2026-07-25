"""生成 revision-2 冻结 manifest"""
import json, os, sys, hashlib, numpy as np

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
MANIFEST_PATH = os.path.join(BASE, "P0-R3_frozen_manifest_rev2.sha256")
TEST_LOG = os.path.join(BASE, "output", "test_log.txt")
PROTO_PATH = os.path.join(BASE, "p0_r3_prototypes.npz")

# 保存测试日志
import subprocess
result = subprocess.run([sys.executable, os.path.join(BASE, "p0_r3_tests.py")], capture_output=True, text=True, cwd=BASE)
with open(TEST_LOG, "w") as f:
    f.write(result.stdout)
    if result.stderr:
        for line in result.stderr.split("\n"):
            if "TreatControl" not in line and "设置" not in line and "句柄" not in line:
                f.write(line + "\n")

# 保存原型
from p0_r3_predict import build_prototypes
protos = build_prototypes()
np.savez(PROTO_PATH, **{k:v for k,v in protos.items()})
print(f"原型保存: {PROTO_PATH}")

files = ["p0_r3_predict.py", "p0_r3_config.json", "p0_r3_main.py",
         "p0_r3_validate_input.py", "p0_r3_validate_output.py", "p0_r3_score.py",
         "p0_r3_tests.py"]
manifest = {}
for fn in files:
    fp = os.path.join(BASE, fn)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            manifest[fn] = hashlib.sha256(f.read()).hexdigest()
with open(PROTO_PATH, "rb") as f:
    manifest["p0_r3_prototypes.npz"] = hashlib.sha256(f.read()).hexdigest()
with open(TEST_LOG, "rb") as f:
    manifest["test_log.txt"] = hashlib.sha256(f.read()).hexdigest()

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print(f"\nRevision-2 manifest: {MANIFEST_PATH}")
for fn, h in manifest.items():
    print(f"  {fn}: {h[:16]}...")
print("\n✅ 已冻结 revision-2。等待 PI 验收，不再读取任何盲测数据。")
