<h1 align="center">第二课第一周“深度学习的实用层面”</h1>

# 文件夹结构

名称 | 解释
---- | ---
datasets | 本周编程作业数据集，由于GitHub单文件大小限制，因此未上传该文件夹啊，该文件夹有两个文件：“test_catvnoncat.h5”、“train_catvnoncat.h5”，可自行网上下载
images |  编程作业里边的一些图片源文件
md_images |  README.md内的图片源文件
answer-Initialization.ipynb |  本周第1个编程作业文件（内含答案）
answer-Regularization.ipynb |  本周第2个编程作业文件（内含答案）
answer-Gradient+Checking.ipynb |  本周第3个编程作业文件（内含答案）  
gc_utils.py |  编程作业需要使用的py文件
init_utils.py |  编程作业需要使用的py文件
reg_utils.py |  编程作业需要使用的py文件   
testCases.py |  编程作业需要使用的py文件   
1.Initialization.ipynb |  本周第1个编程作业文件（无答案）
2.Regularization.ipynb |  本周第2个编程作业文件（无答案）
3.Gradient+Checking.ipynb |  本周第3个编程作业文件（无答案）  

# 笔记

## 目录 

* [笔记](#笔记)
   * [目录](#目录)
   * [训练、验证、测试集](#训练验证测试集)
   * [偏差、方差](#偏差方差)
   * [机器学习基础](#机器学习基础)
   * [正则化](#正则化)
   * [为什么正则化有利于防止过拟合](#为什么正则化有利于防止过拟合)
   * [dropout正则化](#dropout正则化)
   * [理解dropout](#理解dropout)
   * [其他正则化方法](#其他正则化方法)
   * [标准化输入](#标准化输入)
   * [梯度消失和梯度爆炸](#梯度消失和梯度爆炸)
   * [神经网络的权重初始化](#神经网络的权重初始化)
   * [梯度的数值逼近](#梯度的数值逼近)
   * [梯度检验](#梯度检验)
   * [梯度验证应用的注意事项](#梯度验证应用的注意事项)
