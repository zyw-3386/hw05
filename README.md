# hw05 极简 CNN 手写数字识别与 LeNet-5 实现
1、目录结构
debug_notes.md         调试日志
report.md              实验报告
requirements.txt       依赖与环境
lenet5.py              任务二：LeNet-5实现代码
simple_cnn.py          任务一：极简CNN代码（参考公众号文章：https://mp.weixin.qq.com/s/iBNvhk-uAeAfTuanxiLs9Q）
2、安装依赖
pip install -r requirements.txt
3、运行任务一（极简CNN）
打开 task1_simple_cnn.ipynb，按顺序执行所有单元格。
4、运行任务二（LeNet-5）
打开 task2_lenet5.ipynb，按顺序执行所有单元格。
##
代码会自动从 torchvision.datasets.MNIST 下载数据集到 ./data/ 目录。如遇网络问题，可手动下载放入该目录。
自动使用GPU，无GPU则使用CPU
训练5轮，自动输出测试准确率
