import csv
from matplotlib import pyplot as plt
import numpy as np

episodes = 150
error_1 = []
#error_2 = []
#error_3 = []
# 读取CSV文件数据
filename = 'error_avg.csv'
with open(filename) as f:  # 打开这个文件，并将结果文件对象存储在f中
    reader = csv.reader(f)  # 创建一个阅读器reader
    next(reader)  # 文件中的下一行 跳过第一行
    next(reader)

    for row in reader:
        error_1.append(row[0])
        #error_2.append(row[1])
        #error_3.append(row[2])
        #print(error_avg)

error_1_float = []
#error_2_float = []
#error_3_float = []
for num in error_1:
    error_1_float.append(float(num))
# for num in error_2:
#     error_2_float.append(float(num))
# for num in error_3:
#     error_3_float.append(float(num))
#print(error_avg_float)

plt.figure(0)

plt.subplot(3,1,1)
plt.plot(error_1_float, '-b.', markersize=1)
plt.axis([0, episodes, 0, 1.1])
plt.xlabel('episodes')
plt.ylabel('error')
plt.title('C = 1, K = 3, '+r'$\rho$'+' = 0.99')
plt.yscale('symlog', nonposy='clip', linthreshy=10**-6)
plt.grid()

# plt.subplot(3,1,2)
# plt.plot(error_2_float, '-b.', markersize=1)
# plt.axis([0, episodes, 0, 1.1])
# plt.xlabel('episodes')
# plt.ylabel('error')
# plt.title('C = 2, K = 3, '+r'$\rho$'+' = 0.99')
# plt.yscale('symlog', nonposy='clip', linthreshy=10**-4)
# plt.grid()

# plt.subplot(3,1,3)
# plt.plot(error_3_float, '-b.', markersize=1)
# plt.axis([0, episodes, 0, 1.1])
# plt.xlabel('cycle')
# plt.ylabel('error')
# plt.title('C = 3, K = 3, '+r'$\rho$'+' = 0.99')
# plt.yscale('symlog', nonposy='clip', linthreshy=10**-6)
# plt.grid()

plt.tight_layout()
plt.show()
# # 根据数据绘制图形
# fig = plt.figure(dpi=128, figsize=(10, 6))
# plt.plot(dates, highs, c='red', alpha=0.5)  # 实参alpha指定颜色的透明度，0表示完全透明，1（默认值）完全不透明
# plt.plot(dates, lows, c='blue', alpha=0.5)
# plt.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)  # 给图表区域填充颜色
# plt.title('Daily high and low temperature-2004', fontsize=24)
# plt.xlabel('', fontsize=16)
# plt.ylabel('Temperature(F)', fontsize=16)
# plt.tick_params(axis='both', which='major', labelsize=16)
# fig.autofmt_xdate()  # 绘制斜的日期标签
# plt.show()







