import OpenHowNet
hn = OpenHowNet.HowNetDict()
print("HowNet 预加载完成")
import sys
sys.argv = ["p0_r5_predict.py", r"E:\intentCloud\Haibo-Research\06_Experiment\data\p0_r5\blind_input.json"]
exec(open(r"E:\intentCloud\Haibo-Research\06_Experiment\p0_r5_predict.py").read())
