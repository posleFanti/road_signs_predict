import pandas as pd
import matplotlib.pyplot as plt

df26s = pd.read_csv('runs/detect/yolo26s_rtsd_dataset/results.csv')
dfv8s = pd.read_csv('runs/detect/yolov8s_rtsd_dataset/results.csv')

print(df26s['metrics/mAP50-95(B)'].tail())
print(dfv8s['metrics/mAP50-95(B)'].tail())

ax = df26s.plot(x='epoch', y='metrics/mAP50-95(B)')
dfv8s.plot(ax=ax, x='epoch', y='metrics/mAP50-95(B)')


plt.savefig('plot.pdf', format='pdf')
plt.show()
