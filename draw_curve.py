"""
利用不同token长度测出来的时间来绘制实测的Roofline模型曲线。
"""


import matplotlib.pyplot as plt
import numpy as np

prefill_lengths = [5,10,20,30,40,50,100,200,250,300,350,400,450,500,550,600,800,1000,1200,1400,1600,1800,2000,]
times = [12.76,12.67,13.01,13.14,13.54,13.59,16.50,24.05,24.55,30.84,31.68,40.47,42.16,43.90,50.95,53.14,70.56,85.74,99.77,109.8,129.42,149.78,160.98,]
performance = np.array(prefill_lengths) / np.array(times) * 1000 # tokens/s

plt.figure(figsize=(10,6))
plt.plot(prefill_lengths, performance, marker='o', linestyle='-')
plt.xlabel('Prefill length (tokens)')
plt.ylabel('Performance (tokens / s)')
plt.title('Tested Roofline model')
plt.grid(True)
plt.show()
plt.savefig("roofline_model.png")