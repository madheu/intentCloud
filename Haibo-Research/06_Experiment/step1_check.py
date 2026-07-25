import json, hashlib
F = r"output/p0_r5_frozen"
m = json.load(open(f"{F}/P0-R5_frozen_manifest.sha256"))
ok = True
for fn, exp in sorted(m.items()):
    act = hashlib.sha256(open(f"{F}/{fn}", "rb").read()).hexdigest()
    match = act == exp
    if not match: ok = False
    print(f"  {"OK" if match else "NO"} {fn}: {act[:16]}...")
print(f"\n{'PASS' if ok else 'FAIL'}")
