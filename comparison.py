import pandas as pd
import matplotlib.pyplot as plt

df26s = pd.read_csv('runs/detect/yolo26s_rtsd_dataset/results.csv')
dfv8s = pd.read_csv('runs/detect/yolov8s_rtsd_dataset/results.csv')
df26n = pd.read_csv('runs/detect/yolo26n_rtsd_dataset/results.csv')
df26n2 = pd.read_csv('runs/detect/road_signs_yolo26l_1280_b12-28/results.csv')


print(df26s['metrics/mAP50-95(B)'].tail())
print(dfv8s['metrics/mAP50-95(B)'].tail())

ax = df26s.plot(x='epoch', y='metrics/mAP50-95(B)', label='YOLO26s')
dfv8s.plot(ax=ax, x='epoch', y='metrics/mAP50-95(B)', label='YOLOv8s')
df26n.plot(ax=ax, x='epoch', y='metrics/mAP50-95(B)', label='YOLO26n(640)')
df26n2.plot(ax=ax, x='epoch', y='metrics/mAP50-95(B)', label='YOLO26n(1024)')

ax.set_title("mAP50-95(B)")

plt.grid()
plt.savefig('plot.pdf', format='pdf')
plt.show()
