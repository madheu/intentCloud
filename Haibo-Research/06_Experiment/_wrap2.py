import OpenHowNet, sys, os
import importlib
hn = OpenHowNet.HowNetDict()
sys.path.insert(0, r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen")
os.chdir(r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen")
sys.argv = ["p0_r5_predict.py", r"E:\intentCloud\Haibo-Research\06_Experiment\data\p0_r5\blind_input.json"]
__file__ = r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen\p0_r5_predict.py"
with open(__file__, encoding="utf-8") as f:
    code = f.read()
exec(code)
