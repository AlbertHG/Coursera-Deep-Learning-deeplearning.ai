<h1 align="center">第一课第一、二周“神经网络的基础 ”</h1>

# 笔记

## 目录 

* [笔记](#笔记)
   * [目录](#目录)
   * [Logistic 回归](#logistic-回归)
   * [Logistic 回归的损失函数](#logistic-回归的损失函数)
   * [梯度下降法](#梯度下降法)
   * [逻辑回归的梯度下降法](#逻辑回归的梯度下降法)
   * [m个样本的梯度下降法](#m个样本的梯度下降法)
   * [向量化](#向量化)
   * [Python 广播](#python-广播)


## Logistic 回归

Logistic 回归是一个用于二分分类的算法。

Logistic 回归中使用的参数如下：

* 输入的特征向量：$x \in R^{n_x}$，其中 ${n_x}$是特征数量；
* 用于训练的标签：$y \in 0,1$
* 权重：$w \in R^{n_x}$
* 偏置： $b \in R$
* 输出：$\hat{y} = \sigma(w^Tx+b)$
* Sigmoid 函数：$$s = \sigma(w^Tx+b) = \sigma(z) = \frac{1}{1+e^{-z}}$$

为将 $w^Tx+b$ 约束在 $[0, 1]$ 间，引入 Sigmoid 函数，Sigmoid 函数的值域为 $[0, 1]$。

Logistic 回归可以看作是一个非常小的神经网络。下图是一个典型例子：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/01-Neural%20Networks%20and%20Deep%20Learning/week2/md_images/01.png)

## Logistic 回归的损失函数

**损失函数（loss function）** 用于衡量预测结果与真实值之间的误差。

最简单的损失函数定义方式为平方差损失：$$L(\hat{y},y) = \frac{1}{2}(\hat{y}-y)^2$$

这是因为上面的平方错误损失函数一般是非凸函数（non-convex），其在使用低度下降算法的时候，容易得到局部最优解，而不是全局最优解。因此要选择凸函数。

一般使用$$L(\hat{y},y) = -(y\log\hat{y})+(1-y)\log(1-\hat{y})$$

- 当 $y=1$ 时，$ L(\hat y, y)=-\log \hat y$ 。如果 $\hat y $越接近 1， $L(\hat y, y) \approx 0$ ，表示预测效果越好；如果 $\hat y$ 越接近 0， $L(\hat y, y) \approx +\infty$ ，表示预测效果越差；
- 当 $y=0$ 时， $L(\hat y, y)=-\log (1-\hat y)$ 。如果 $\hat y$ 越接近 0， $L(\hat y, y) \approx 0$ ，表示预测效果越好；如果 $\hat y$ 越接近 1， $L(\hat y, y) \approx +\infty$ ，表示预测效果越差；
- 我们的目标是最小化样本点的损失Loss Function，损失函数是针对单个样本点的。

损失函数是在单个训练样本中定义的，它衡量了在 **单个** 训练样本上的表现。而**成本函数（cost function，或者称作成本函数）** 衡量的是在 **全体** 训练样本上的表现，即衡量参数 $w $和 $b$ 的效果。

$$J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}^{(i)},y^{(i)})$$

## 梯度下降法

函数的 **梯度（gradient）** 指出了函数的最陡增长方向。即是说，按梯度的方向走，函数增长得就越快。那么按梯度的负方向走，函数值自然就降低得最快了。

模型的训练目标即是寻找合适的 $w$与 $b$ 以最小化代价函数值。简单起见我们先假设$ w$与 $b$ 都是一维实数，那么可以得到如下的 $J$ 关于$ w $与$ b $的图：


![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/01-Neural%20Networks%20and%20Deep%20Learning/week2/md_images/02.png)

可以看到，成本函数 $J$ 是一个 **凸函数** ，与非凸函数的区别在于其不含有多个局部最低点；选择这样的代价函数就保证了无论我们初始化模型参数如何，都能够寻找到合适的最优解。

参数 $w$ 的更新公式为：

$$w := w - \alpha\frac{dJ(w, b)}{dw}$$

其中 $α$ 表示学习速率，即每次更新的 $w$ 的步伐长度。

当 $w$ 大于最优解 w′ 时，导数大于 0，那么 $w$ 就会向更小的方向更新。反之当 $w$ 小于最优解 w′ 时，导数小于 0，那么 $w$ 就会向更大的方向更新。迭代直到收敛。

在成本函数 $J(w, b)$ 中还存在参数 $b$，因此也有：

$$b := b - \alpha\frac{dJ(w, b)}{db}$$

在程序代码中，我们通常使用`dw` 来表示 $\dfrac{\partial J(w,b)}{\partial w}$ ，用`db`来表示 $\dfrac{\partial J(w,b)}{\partial b}$ 。

## 逻辑回归的梯度下降法

假设输入的特征向量维度为 2，即输入参数共有 $x_1, w_1, x_2, w_2, b$ 这五个。可以推导出如下的计算图：


![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/01-Neural%20Networks%20and%20Deep%20Learning/week2/md_images/03.png)

首先反向求出 $L$ 对于 $a$ 的导数：

$$da = \dfrac{\partial L}{\partial a}=-\dfrac{y}{a}+\dfrac{1-y}{1-a}$$

然后继续反向求出 $L$ 对于 $z$ 的导数：

$$dz = \dfrac{\partial L}{\partial z}=\dfrac{\partial L}{\partial a}\cdot\dfrac{\partial a}{\partial z}=(-\dfrac{y}{a}+\dfrac{1-y}{1-a})\cdot a(1-a)=a-y$$

再对$w_1$，$w_2$和$b$进行求导：

$$dw_{1} = \dfrac{\partial L}{\partial w_{1}}=\dfrac{\partial L}{\partial z}\cdot\dfrac{\partial z}{\partial w_{1}}=x_{1}\cdot dz=x_{1}(a-y)$$

$$db = \dfrac{\partial L}{\partial b }=\dfrac{\partial L}{\partial z}\cdot\dfrac{\partial z}{\partial b }=1\cdot dz=a-y $$

最后对参数进行更新：

$$w_{1}:=w_{1}-\alpha dw_{1}$$

$$w_{2}:=w_{2}-\alpha dw_{2} $$

$$b:=b-\alpha db $$

## m个样本的梯度下降法

接下来我们需要将对于单个用例的损失函数扩展到整个训练集的成本函数：

$$a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})=\sigma(w^Tx^{(i)}+b)$$

$$J(w,b)=\dfrac{1}{m}\sum_{i=1}^{m}L(\hat y^{(i)}, y^{(i)})=-\dfrac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\hat y^{(i)}+(1-y^{(i)})\log(1-\hat y^{(i)})\right]$$


我们可以对于某个权重参数 $w_1$，其导数计算为：

$$dw_{1} = \frac{\partial J(w,b)}{\partial{w_1}}=\frac{1}{m}\sum^m_{i=1}\frac{\partial}{\partial{w_1}L(a^{(i)},y^{(i)})}$$

$$db = \dfrac{1}{m}\sum_{i=1}^{m}(a^{(i)}-y^{(i)}) $$

## 向量化

在 Logistic 回归中，需要计算 $$z=w^Tx+b$$如果是非向量化的循环方式操作，代码可能如下：

```py
z = 0;
for i in range(n_x):
    z += w[i] * x[i]
z += b
```

而如果是向量化的操作，代码则会简洁很多，并带来近百倍的性能提升（并行指令）：

```py
z = np.dot(w, x) + b
```

不用显式 for 循环，实现 Logistic 回归的梯度下降一次迭代（这里公式和 NumPy 的代码混杂，注意分辨）：

$$Z=w^TX+b=np.dot(w.T, x) + b$$
$$A=\sigma(Z)$$
$$dZ=A-Y$$
$$dw=\frac{1}{m}XdZ^T$$
$$db=\frac{1}{m}np.sum(dZ)$$
$$w:=w-\sigma dw$$
$$b:=b-\sigma db$$

正向和反向传播尽管如此，多次迭代的梯度下降依然需要 for 循环。

## Python 广播

Numpy 的 Universal functions 中要求输入的数组 shape 是一致的。当数组的 shape 不相等的时候，则会使用广播机制，调整数组使得 shape 一样，满足规则，则可以运算，否则就出错。

四条规则：

1. 让所有输入数组都向其中 shape 最长的数组看齐，shape 中不足的部分都通过在前面加 1 补齐；
2. 输出数组的 shape 是输入数组 shape 的各个轴上的最大值；
3. 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错；
4. 当输入数组的某个轴的长度为 1 时，沿着此轴运算时都用此轴上的第一组值。

简而言之，就是python中可以对不同维度的矩阵进行四则混合运算，但至少保证有一个维度是相同的。下面给出几个广播的例子，具体细节可参阅python的相关手册。


![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/01-Neural%20Networks%20and%20Deep%20Learning/week2/md_images/04.jpg)

值得一提的是，在python程序中为了保证矩阵运算正确，可以使用reshape()函数来对矩阵设定所需的维度。这是一个很好且有用的习惯。
