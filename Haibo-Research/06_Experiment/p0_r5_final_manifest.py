"""重新生成 P0-R5-final manifest（排除自引用）"""
import json, os, hashlib

FROZEN = r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen"
MP = os.path.join(FROZEN, "P0-R5_frozen_manifest.sha256")

manifest = {}
for fn in sorted(os.listdir(FROZEN)):
    if fn == "P0-R5_frozen_manifest.sha256":
        continue
    fp = os.path.join(FROZEN, fn)
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            manifest[fn] = hashlib.sha256(f.read()).hexdigest()
with open(MP, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

# 自校验
all_ok = True
for fn, expected in manifest.items():
    fp = os.path.join(FROZEN, fn)
    with open(fp, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    ok = actual == expected
    if not ok: print(f"  ❌ {fn}"); all_ok = False
    else: print(f"  ✅ {fn}: {actual[:16]}...")

print(f"\n{'✅ P0-R5-final 全部通过' if all_ok else '❌ 有错误'}")
if all_ok:
    print(f"\nManifest: {MP}")
    print(f"共 {len(manifest)} 个条目")
