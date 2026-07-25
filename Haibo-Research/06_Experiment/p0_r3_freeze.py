"""保存原型 + 生成冻结 manifest"""
import json, os, hashlib, numpy as np

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
PROTO_PATH = os.path.join(BASE, "p0_r3_prototypes.npz")
MANIFEST_PATH = os.path.join(BASE, "P0-R3_frozen_manifest.sha256")

# 重新构建原型并保存
from p0_r3_predict import build_prototypes
prototypes = build_prototypes()

# 保存为 npz
proto_data = {k: v for k, v in prototypes.items()}
np.savez(PROTO_PATH, **proto_data)
print(f"原型保存: {PROTO_PATH} ({len(prototypes)} 个)")

# 生成 SHA-256
files = [
    "p0_r3_predict.py", "p0_r3_config.json", "p0_r3_main.py",
    "P0-R3_frozen_manifest.sha256",
]
# 先不包括自身
files.remove("P0-R3_frozen_manifest.sha256")

manifest = {}
for fn in files:
    fp = os.path.join(BASE, fn)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        manifest[fn] = h

# 原型文件
with open(PROTO_PATH, "rb") as f:
    manifest["p0_r3_prototypes.npz"] = hashlib.sha256(f.read()).hexdigest()

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print(f"\n冻结 manifest: {MANIFEST_PATH}")
for fn, h in manifest.items():
    print(f"  {fn}: {h[:16]}...")
print("\n✅ 已冻结。等待 PI 提供 blind_input.json 后进入阶段 B3。")
