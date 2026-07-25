import OpenHowNet, sys, os
_HN_INSTANCE = OpenHowNet.HowNetDict()
from OpenHowNet import HowNetDict as _orig_HowNetDict
def _singleton_HowNetDict():
    return _HN_INSTANCE
OpenHowNet.HowNetDict = _singleton_HowNetDict

sys.path.insert(0, r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen")
os.chdir(r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen")
sys.argv = ["p0_r5_predict.py", r"E:\intentCloud\Haibo-Research\06_Experiment\data\p0_r5\blind_input.json"]
__file__ = r"E:\intentCloud\Haibo-Research\06_Experiment\output\p0_r5_frozen\p0_r5_predict.py"
with open(__file__, encoding="utf-8") as f:
    exec(f.read())
