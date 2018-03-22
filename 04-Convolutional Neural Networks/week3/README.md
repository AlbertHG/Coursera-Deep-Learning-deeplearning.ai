<h1 align="center">第四课第三周“目标检测”</h1>

# 笔记

## 目录 

* [笔记](#笔记)
   * [目录](#目录)
   * [目标定位(Object localization)](#目标定位object-localization)
   * [特征点检测(Landmark detection)](#特征点检测landmark-detection)
   * [目标检测(Object detection)](#目标检测object-detection)
   * [卷积的滑动窗口实现(Convolutional implementation of sliding windows)](#卷积的滑动窗口实现convolutional-implementation-of-sliding-windows)
   * [Bounding Box预测(Bounding box predictions)](#bounding-box预测bounding-box-predictions)
   * [交并比(Intersection over union)](#交并比-intersection-over-union)
   * [非极大值抑制(Non-max suppression)](#非极大值抑制non-max-suppression)
   * [Anchor Boxes](#anchor-boxes)
   * [YOLO 算法(Putting it together: YOLO algorithm)](#yolo-算法putting-it-together-yolo-algorithm)
   * [候选区域(选修)(Region proposals (Optional))](#候选区域选修region-proposals-optional)

## 目标定位(Object localization)

定位分类问题这意味着，不仅要用算法判断图片中是不是一辆汽车，还要在图片中标记出它的位置，用边框（Bounding Box）把目标物体圈起来。一般来说，定位分类问题通常只有一个较大的对象位于图片中间位置；而在对象检测问题中，图片可以含有多个对象，甚至单张图片中会有多个不同分类的对象。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/01.jpg)

- 对于目标定位和目标检测问题，其模型如上所示，它会输出一个特征向量（图中右边列向量第6、7、8行），并反馈给softmax单元来预测图片类型：

- 为了定位图片中汽车的位置，可以让神经网络多输出 4 个数字（图中右边列向量第2、3、4、5行），标记为 $b_x$、$b_y$、$b_h$、$b_w$。将图片左上角标记为 (0, 0)，右下角标记为 (1, 1)，则有：

    * 红色方框的中心点：($b_x$，$b_y$)
    * 边界框的高度：$b_h$
    * 边界框的宽度：$b_w$

- 同时还输出$P_c$（图中右边列向量第1行），表示矩形区域是目标的概率，数值在0-1之间。

综上，目标的标签$Y$如下几种形式：

$$\left[\begin{matrix}P\_c\\\ b\_x\\\ b\_y\\\ b\_h\\\ b\_w\\\ c\_1\\\ c\_2\\\ c\_3\end{matrix}\right]
,when P\_c=1:\left[\begin{matrix}1\\\ b\_x\\\ b\_y\\\ b\_h\\\ b\_w\\\ c\_1\\\ c\_2\\\ c\_3\end{matrix}\right] ,when P_c=0:\left[\begin{matrix}0\\\ ?\\\ ?\\\ ?\\\ ?\\\ ?\\\ ?\\\ ?\end{matrix}\right]
$$

若$P_c$=0，表示没有检测到目标，则输出label后面的7个参数都可以忽略(用 ? 来表示)。

损失函数可以表示为 $L(\hat y, y)$，如果使用平方误差形式，对于不同的 $P\_c$有不同的损失函数（注意下标 $i$指标签的第 $i$个值）：

1. $P\_c=1$，即$y\_1=1$：

    $L(\hat y,y)=(\hat y\_1-y\_1)^2+(\hat y\_2-y\_2)^2+\cdots+(\hat y\_8-y\_8)^2$

    损失值就是不同元素的平方和。

2. $P\_c=0$，即$y\_1=0$：

    $L(\hat y,y)=(\hat y\_1-y\_1)^2$

    *对于这种情况，不用考虑其它元素，只需要关注神经网络输出的准确度即可。*

## 特征点检测(Landmark detection)

神经网络可以像标识目标的中心点位置那样，通过输出图片上的特征点，来实现对目标特征的识别。在标签中，这些特征点以多个二维坐标的形式表示。

举个例子：假设需要定位一张人脸图像，同时检测其中的64个特征点，这些点可以帮助我们定位眼睛、嘴巴等人脸特征。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/02.jpg)

- 具体的做法是：准备一个卷积网络和一些特征集，将人脸图片输入卷积网络，输出1或0（1表示有人脸，0表示没有人脸）然后输出$(l_{1x},l_{1y})$  ……直到$(l_{64x},l_{64y})$。这里用$l$代表一个特征，这里有129个输出单元，其中1表示图片中有人脸，因为有64个特征，64×2=128，所以最终输出128+1=129个单元，由此实现对图片的人脸检测和定位。

## 目标检测(Object detection)

想要实现目标检测，可以采用 **基于滑动窗口的目标检测（Sliding Windows Detection）** 算法。

实现目标检测的要点：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/03.jpg)

1. 训练集X：将有汽车的图片进行适当的剪切，剪切成整张几乎都被汽车占据的小图或者没有汽车的小图；
2. 训练集Y：对X中的图片进行标注，有汽车的标注1，没有汽车的标注0。

使用符合上述要求的这些训练集构建CNN模型，使得模型有较高的识别率。

选择大小适宜的窗口与合适的固定步幅，对测试图片进行从左到右、从上倒下的滑动遍历。每个窗口区域使用已经训练好的 CNN 模型进行识别判断。

可以选择更大的窗口，然后重复第三步的操作。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/04.jpg)

滑动窗算法的优点是原理简单，且不需要人为选定目标区域（检测出目标的滑动窗即为目标区域）。但是其缺点也很明显，首先滑动窗的大小和步进长度都需要人为直观设定。滑动窗过小或过大，步进长度过大均会降低目标检测正确率。而且，每次滑动窗区域都要进行一次CNN网络计算，如果滑动窗和步进长度较小，整个目标检测的算法运行时间会很长。所以，滑动窗算法虽然简单，但是性能不佳，不够快，不够灵活。

## 卷积的滑动窗口实现(Convolutional implementation of sliding windows)

滑动窗算法可以使用卷积方式实现，以提高运行速度，节约重复运算成本

首先，单个滑动窗口区域进入CNN网络模型时，包含全连接层。那么滑动窗口算法卷积实现的第一步就是将全连接层转变成为卷积层，如下图所示：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/05.jpg)

全连接层转变成卷积层的操作很简单，只需要使用与上层尺寸一致的滤波算子进行卷积运算即可。最终得到的输出层维度是1 x 1 x 4，代表4类输出值。

单个窗口区域卷积网络结构建立完毕之后，对于待检测图片，即可使用该网络参数和结构进行运算。

事实上，我们不用像上一节中那样，自己去滑动窗口截取图片的一小部分然后检测，卷积这个操作就可以实现滑动窗口。

如下图中间一行卷积的初始图，我们假设输入的图像是16 x 16 x 3的，而窗口大小是14 x 14 x 3，我们要做的是把蓝色区域输入卷积网络，生成0或1分类，接着向右滑动2个元素，形成的区域输入卷积网络，生成0或1分类，然后接着滑动，重复操作。我们在16 x 16 x 3的图像上卷积了4次，输出了4个标签，我们会发现这4次卷积里很多计算是重复的。

而实际上，直接对这个16 x 16 x 3的图像进行卷积，如下图中间一行的卷积的整个过程，这个卷积就是在计算我们刚刚提到的很多重复的计算，过程中蓝色的区域就是我们初始的时候用来卷积的第一块区域，到最后它变成了2 x 2 x 4的块的左上角那一块，我们可以看到最后输出的2 x 2，刚好就是4个输出，对应我们上面说的输出4个标签。

这两个过程刚好可以对应的上。所以我们不需要把原图分成四个部分，分为用卷积去检测，而是把它们作为一张图片输入给卷积网络进行计算，其中的公有区域可以共享很多计算。

同样的，当图片大小是28 x 28 x 3的时候，CNN网络得到的输出层为8 x 8 x 4，共64个窗口结果。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/06.jpg)
*蓝色窗口表示卷积窗，黄色的表示图片*

我们不用依靠连续的卷积操作来识别图片中的汽车，我们可以对整张图片进行卷积，一次得到所有的预测值，如果足够幸运，神经网络便可以识别出汽车的位置。

## Bounding Box预测(Bounding box predictions)

卷积方式实现的滑动窗口算法，使得在预测时计算的效率大大提高。但是其存在的问题是：不能输出最精准的边界框（Bounding Box）。

假设窗口滑动到蓝色方框的地方，这不是一个能够完美匹配汽车位置的窗口，所以我们需要寻找更加精确的边界框。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/07.jpg)

YOLO（You Only Look Once）算法可以解决这类问题，生成更加准确的目标区域（如上图红色窗口）。

YOLO算法首先将原始图片分割成n x n网格，每个网格代表一块区域。为简化说明，下图中将图片分成3 x 3网格。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/08.jpg)

然后，利用上一节卷积形式实现滑动窗口算法的思想，对该原始图片构建CNN网络，得到的的输出层维度为3 x 3 x 8。其中，3 x 3对应9个网格，每个网格的输出包含8个元素：
$$y=\left[\begin{matrix}P\_c\\\ b\_x\\\ b\_y\\\ b\_h\\\ b\_w\\\ c\_1\\\ c\_2\\\ c\_3\end{matrix}\right]
$$

如果目标中心坐标 $(b_x,b_y)$ 不在当前网格内，则当前网格Pc=0；相反，则当前网格$P_c=1$（即只看中心坐标是否在当前网格内）。判断有目标的网格中， $b_x,b_y,b_h,b_w$ 限定了目标区域。

- 值得注意的是，当前网格左上角坐标设定为$(0, 0)$，右下角坐标设定为$(1, 1)$， $(b_x,b_y)$ 表示坐标值，范围限定在$[0,1]$之间，
- 但是 $b_h,b_w$ 表示比例值,可以大于 1。因为目标可能超出该网格，横跨多个区域，

如上图所示。目标占几个网格没有关系，目标中心坐标必然在一个网格之内。

划分的网格可以更密一些。网格越小，则多个目标的中心坐标被划分到一个网格内的概率就越小，这恰恰是我们希望看到的。

这是一个总结：

- 首先这和图像分类和定位算法非常像，就是它显式地输出边界框坐标，所以这能让神经网络输出边界框，可以具有任意宽高比，并且能输出更精确的坐标，不会受到滑动窗口分类器的步长大小限制。

- 其次，这是一个卷积实现，你并没有在3 × 3网格上跑9次算法，或者，如果用的是19 × 19的网格，19平方是361次，所以不需要让同一个算法跑361次。相反，这是单次卷积实现，但使用了一个卷积网络，有很多共享计算步骤，在处理这3 × 3(或者19 × 19)计算中很多计算步骤是共享的，所以这个算法效率很高。

## 交并比(Intersection over union)

IoU，即交集与并集之比，可以用来评价目标检测区域的准确性。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/09.jpg)

如上图所示，红色方框为真实目标区域，蓝色方框为检测目标区域。两块区域的交集为绿色部分，并集为紫色部分。蓝色方框与红色方框的接近程度可以用IoU比值来定义：

$IoU=\frac{I}{U}$

IoU可以表示任意两块区域的接近程度。IoU值介于0～1之间，且越接近1表示两块区域越接近。

一般在目标检测任务中，约定如果 $IoU>=0.5$ ，那么就说明检测正确。当然标准越大，则对目标检测算法越严格。得到的IoU值越大越好。

## 非极大值抑制(Non-max suppression)

对于汽车目标检测的例子中，我们将图片分成很多精细的格子。最终预测输出的结果中，可能会有相邻的多个格子里均检测出都具有同一个对象。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/10.png)

对于每个格子都运行一次，所以格子（编号1）可能会认为这辆车中点应该在格子内部，这几个格子（编号2、3）也会这么认为。对于左边的车子也一样，格子（编号4）会认为它里面有车，格子（编号5）和这个格子（编号6）也会这么认为.

那如何判断哪个网格最为准确呢？方法是使用非极大值抑制算法。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/11.jpg)

- 算法首先审视每一个概率高（$P_c>=0.6$）的格子，在上图中，分别由5个格子表示检测出车子的概率很高，分别是0.8、0.7、0.9、0.6、0.7。
- 首先看概率最大的，这里是0.9，，然后将剩下的4个格子分别和0.9的这个格子做交并比计算，发现右边为0.6和0.7的两个格子和0.9的交并比很大，则将这两个格子抑制（舍弃）。
- 接下来，再次审视剩下的格子（0.8，0.7），找出概率最大的那一个，这里是0.8，然后再次将该格子和剩下的格子（0.7）进行交并比计算，舍弃掉交并比很大的格子，这里剩下的0.7被舍弃。
- 最后剩下左边的0.8和右边的0.9两个格子。

这就是非极大值抑制，非极大值意味着你只输出概率最大的分类结果，同时抑制那些不是极大值，却比较接近极大值的边界框。

上述是单对象检测，对于多对象检测，输出标签中就会有多个分量。正确的做法是：对每个输出类别分别独立进行一次非极大值抑制。

## Anchor Boxes

到目前为止，我们介绍的都是一个网格至多只能检测一个目标。那对于多个目标重叠的情况，例如一个人站在一辆车前面，该如何使用YOLO算法进行检测呢？方法是使用不同形状的Anchor Boxes。

如下图所示，同一网格出现了两个目标：人和车。为了同时检测两个目标，我们可以设置两个Anchor Boxes，Anchor box 1检测人，Anchor box 2检测车。也就是说，每个网格多加了一层输出。原来的输出维度是 3 x 3 x 8，现在是3 x 3 x 2 x 8（也可以写成3 x 3 x 16的形式）。这里的2表示有两个Anchor Boxes，用来在一个网格中同时检测多个目标。每个Anchor box都有一个$P_c$值，若两个$P_c$值均大于某阈值，则检测到了两个目标。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/12.jpg)

对于重叠的目标，这些目标的中点有可能会落在同一个网格中，对于我们之前定义的输出： $y_{i} = \left[ \begin{array}{l} P_{c}\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ c_{1}\ c_{2}\ c_{3} \end{array} \right]^T$ ，只能得到一个目标的输出。

而Anchor box 则是预先定义多个不同形状的Anchor box，我们需要把预测目标对应地和各个Anchor box 关联起来，所以我们重新定义目标向量(如上图右边列向量所示)：

$y_{i} = \left[ P_{c}\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ c_{1}\ c_{2}\ c_{3}\ P_{c}\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ c_{1}\ c_{2}\ c_{3}\cdots\right]^T$

用这样的多目标向量分别对应不同的Anchor box，从而检测出多个重叠的目标。

如下面的图片，里面有行人和汽车，在经过了极大值抑制操作之后，最后保留了两个边界框（Bounding Box）。对于行人形状更像Anchor box 1，汽车形状更像Anchor box 2，所以我们将人和汽车分配到不同的输出位置。具体分配，对应下图颜色。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/13.jpg)

那么如何判断边界框和Anchor box匹配呢？

方法很简单：将边界框和Anchor box进行交并比计算，将交并比高的边界框和Anchor box组队。如下图中，边界框（编号1）和Anchor box(编号2)匹配。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/18.jpg)

当然，如果格子中只有汽车的时候，我们使用了两个Anchor box，那么此时我们的目标向量就成为：

$y_{i} = \left[ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\ 1\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ 0\ 1\ 0\right]^T$

其中，“？”代表的是该位置是什么样的参数我们都不关心。

难点问题：

- 如果我们使用了两个Anchor box，但是同一个格子中却有三个对象的情况，此时只能用一些额外的手段来处理；
- 同一个格子中存在两个对象，但它们的Anchor box 形状相同，此时也需要引入一些专门处理该情况的手段。

但是以上的两种问题出现的可能性不会很大，对目标检测算法不会带来很大的影响。

Anchor box 的选择：

- 一般人工指定Anchor box 的形状，选择5~10个以覆盖到多种不同的形状，可以涵盖我们想要检测的对象的形状；
- 高级方法：K-means 算法：将不同对象形状进行聚类，用聚类后的结果来选择一组最具代表性的Anchor box，以此来代表我们想要检测对象的形状。

## YOLO 算法(Putting it together: YOLO algorithm)

这节将上述关于YOLO算法组件组装在一起构成YOLO对象检测算法。

假设我们要在图片中检测三种目标：行人、汽车和摩托车，同时使用两种不同的Anchor box。

1. 构造训练集：
    - 根据工程目标，将训练集做如下规划。
    - 输入X：同样大小的完整图片；
    - 目标Y：使用 $3\times3$ 网格划分，输出大小 $3\times3\times2\times8$(其中3 × 3表示3×3个网格，2是anchor box的数量，8是向量维度) ，或者 $3\times3\times16$。
    - 对不同格子中的小图，定义目标输出向量Y，如下图示例。
        - 对于格子1的目标y就是这样的$y = \left[ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\right]^T$。
        - 而对于格子2的目标y则应该是这样：$y = \left[ 0\ ?\ ?\ ?\ ?\ ?\ ?\ ?\ 1\ b_{x}\ b_{y}\ b_{h}\ b_{w}\ 0\ 1\ 0\right]^T$。
        - 训练集中，对于车子有这样一个边界框（编号3），水平方向更长一点。所以如果这是你的anchor box，这是anchor box 1（编号4），这是anchor box 2（编号5），然后红框和anchor box 2的交并比更高，那么车子就和向量的下半部分相关。要注意，这里和anchor box 1有关的$P_c$是0，剩下这些分量都是don’t care-s，然后你的第二个 ，然后你要用这些($b_x,b_y,b_h,b_w$)来指定红边界框的位置

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/14.png)

2. 模型预测：
    - 输入与训练集中相同大小的图片，同时得到每个格子中不同的输出结果： $3\times3\times2\times8$ 。
    - 输出的预测值，以下图为例：
        - 对于左上的格子（编号1）对应输出预测y（编号3）
        - 对于中下的格子（编号2）对应输出预测y（编号4）

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/15.png)

3. 运行非最大值抑制（NMS）(为展示效果，换一张复杂的图)：

    - （编号1）假设使用了2个Anchor box，那么对于每一个网格，我们都会得到预测输出的2个bounding boxes，其中一个$P_c$比较高；
    - （编号2）抛弃概率$P_c$值低的预测bounding boxes；
    - （编号3）对每个对象（如行人、汽车、摩托车）分别使用NMS算法得到最终的预测边界框。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/16.png)

## 候选区域(选修)(Region proposals (Optional))

R-CNN（Regions with convolutional networks），会在我们的图片中选出一些目标的候选区域，从而避免了传统滑动窗口在大量无对象区域的无用运算。

所以在使用了R-CNN后，我们不会再针对每个滑动窗口运算检测算法，而是只选择一些候选区域的窗口，在少数的窗口上运行卷积网络。

具体实现：运用图像分割算法，将图片分割成许多不同颜色的色块，然后在这些色块上放置窗口，将窗口中的内容输入网络，从而减小需要处理的窗口数量。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/04-Convolutional%20Neural%20Networks/week3/md_images/17.png)

这就是R-CNN或者区域CNN的特色概念，现在R-CNN算法还是很慢的。所以有一系列的研究工作去改进这个算法，所以基本的R-CNN算法是使用某种算法求出候选区域，然后对每个候选区域运行一下分类器，每个区域会输出一个标签，并输出一个边界框，这样你就能在确实存在对象的区域得到一个精确的边界框。


参考文献：

[Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640)

[Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)

[Allan Zelener - YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)

[The official YOLO website](https://pjreddie.com/darknet/yolo/)
