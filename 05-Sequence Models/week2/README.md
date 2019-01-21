<h1 align="center">第五课第二周“自然语言处理与词嵌入”</h1>

# 笔记

## 目录 

* [笔记](#笔记)
   * [目录](#目录)
   * [词汇表征](#词汇表征)
   * [使用词嵌入](#使用词嵌入)
   * [词嵌入的特性](#词嵌入的特性)
   * [嵌入矩阵](#嵌入矩阵)
   * [学习词嵌入](#学习词嵌入)
   * [Word2Vec](#Word2Vec)
   * [负采样](#负采样)
   * [GloVe词向量](#GloVe词向量)
   * [情感分类](#情感分类)
   * [词向量除偏](#词向量除偏)

## 词汇表征

自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的学科分支，它研究实现人与计算机之间使用自然语言进行有效通信的各种理论和方法。在前面学习的内容中，我们表征词汇是直接使用英文单词来进行表征的，但是对于计算机来说，是无法直接认识单词的。为了让计算机能够能更好地理解我们的语言，建立更好的语言模型，我们需要将词汇进行表征。下面是几种不同的词汇表征方式。

在前面的一节课程中，已经使用过了 one-hot 表征的方式对模型字典中的单词进行表征，对应单词的位置用 1 表示，其余位置用 0 表示，如下图所示：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/01.png)

这种 one-hot 表征单词的方法最大的缺点就是每个单词都是独立的、正交的，无法知道不同单词之间的相似程度。例如 apple 和 orange 都是水果，词性相近，但是单从 one-hot 编码上来看，内积为零，无法知道二者的相似性。在 NLP 中，我们更希望能掌握不同单词之间的相似程度。

词嵌入（Word Embedding）是 NLP 中语言模型与表征学习技术的统称，概念上而言，它是指把一个维数为所有词的数量的高维空间（one-hot形式表示的词）“嵌入”到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/02.jpg)

如上图中，各列分别组成的向量是词嵌入后获得的第一行中几个词的词向量的一部分。这些向量中的值，可代表该词与第一列中几个词的相关程度。举个例子，对于这些词，比如我们想知道这些词与 Gender（性别）的关系。假定男性的性别为 -1，女性的性别为 +1，那么 man 的性别值可能就是-1，而 woman 就是 +1。最终根据经验 king 就是 -0.95，queen 是+0.97，apple 和 orange 没有性别可言，故值近似于 0。

特征向量的长度依情况而定，特征元素越多则对单词表征得越全面，为了说明，我们假设有300个不同的特征，这样的话就有了这一列数字（以第一列为例子），上图只写了4个，实际上是300个数字，这样就组成了一个300维的向量来表示 man 这个词。使用特征表征之后，词汇表中的每个单词都可以使用对应的 $300×1$ 的向量来表示，该向量的每个元素表示该单词对应的某个特征值。每个单词用 $e_{词汇表}$ 索引的方式标记，例如 $e_{5391},e_{9853},e_{4914},e_{7157},e_{456},e_{6257}$ 。

这种特征表征的优点是根据特征向量能清晰知道不同单词之间的相似程度，例如 apple 和 orange 之间的相似度较高，很可能属于同一类别。这种单词“类别”化的方式，大大提高了有限词汇量的泛化能力。这种特征化单词的操作被称为 Word Embeddings，即单词嵌入。

每个单词都由高维特征向量表征，为了可视化不同单词之间的相似性，可以使用降维操作，常用的一种可视化算法是 t-SNE 算法。在通过复杂而非线性的方法映射到二维空间后，每个词会根据语义和相关程度聚在一起。例如 t-SNE 算法，将 300D 降到 2D 平面上。进而对词向量进行可视化，很明显我们可以看出对于相似的词总是聚集在一块儿：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/03.png)

观察上图，你会发现 man 和 woman 这些词聚集在一块（上图编号1所示），king 和 queen 聚集在一块（上图编号2所示），这些都是人，也都聚集在一起（上图编号3所示）。动物都聚集在一起（上图编号4所示），水果也都聚集在一起（上图编号5所示），像1、2、3、4这些数字也聚集在一起（上图编号6所示）。如果把这些生物看成一个整体，他们也聚集在一起（上图编号7所示）。

## 使用词嵌入

之前我们介绍过 Named entity 识别的例子，每个单词采用的是 one-hot 编码。如下图所示，因为“orange farmer”是份职业，很明显“Sally Johnson”是一个人名。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/04.jpg)

如果我们用特征化表示方法对每个单词进行编码，再构建该RNN模型。对于一个新的句子：Robert Lin is an apple farmerRobert Lin is an apple farmer

在这两个句子中，“apple”与“orange”特征向量很接近，很容易能判断出“Robert Lin”也是一个人名。这就是特征化表示方法的优点之一。

可以看出，特征化表示方法的优点是可以减少训练样本的数目，前提是对海量单词建立词嵌入（word embedding）。这样，即使训练样本不够多，测试时遇到陌生单词，例如“durian cultivator”，根据之前海量词汇特征向量就判断出“durian”也是一种水果，与“apple”类似，而“cultivator”与“farmer”也很相似。从而得到与“durian cultivator”对应的应该也是一个人名。这种做法将单词用不同的特征来表示，即使是训练样本中没有的单词，也可以根据词嵌入的结果得到与其词性相近的单词，从而得到与该单词相近的结果，有效减少了训练样本的数量。

总而言之，使用词嵌入做迁移学习的步骤：

1. 先从大量的文本集中学习词嵌入，或者可以下载网上预训练好的词嵌入模型。
2. 然后就可以将这些词嵌入模型迁移到新的只有少量标注训练集的任务中，比如说用这个 300 维的词嵌入来表示你的单词。这样做的一个好处就是可以用更低维度的特征向量代替原来的 10000 维的 one-hot 向量。
3. 可以选择是否微调词嵌入。当标记数据集不是很大时可以省下这一步。

有趣的是，词嵌入与[第四课第四周——特殊应用：人脸识别和神经风格转换](https://alberthg.github.io/2018/04/12/deeplearning-ai-c4w4/)中介绍的人脸特征编码有很多相似性。人脸图片经过 Siamese 网络，得到其特征向量 $f(x)$ ，这点跟词嵌入是类似的。二者不同的是 Siamese 网络输入的人脸图片可以是数据库之外的；而词嵌入一般都是已建立的词汇库中的单词，非词汇库单词统一用 < UNK > 表示。

## 词嵌入的特性

词嵌入可用于类比推理，帮助我们找到不同单词之间的相似类别关系。例如，给定对应关系“男性（Man）”对“女性（Woman）”，想要类比出“国王（King）”对应的词汇。则可以有:

$$e_{man}−e_{woman} ≈ e_{king}−e_{?}$$ 

常识地，“Man”与“Woman”的关系类比于“King”与“Queen”的关系。而根据上述公式。我们将“Man”的 embedding vector 与“Woman”的 embedding vector 相减：

$$e_{?}≈ e_{king}−e_{man}+e_{woman}$$

之后的目标就是找到词向量 $w$，来找到使相似度 $sim(e_{w},e_{king}−e_{man}+e_{woman})$ 最大。在本例中，$e_{?} = e_{w} = e_{queue}$

一个最常用的相似度计算函数是余弦相似度（cosine similarity）。公式为：

$$
sim(u, v) = \frac{u^T v}{|| u ||_2 || v ||_2}
$$

当然，常见的还可以是用欧式距离（euclidian distance）来比较相似性，距离越大，相似性越小。即:

$$||u-v||^2$$

相关论文：[Mikolov et. al., 2013, Linguistic regularities in continuous space word representations](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)

## 嵌入矩阵

当应用算法来学习词嵌入时，实际上是学习一个嵌入矩阵（Embedding Matrix），假设某个词汇库包含了 10000 个单词，每个单词包含的特征维度为 300，那么表征所有单词的嵌入矩阵的维度则为 $300×10000$，我们用 $E$ 来表示。某单词位置为 $i$ 的词的 one-hot 向量表示为 $O_i$，其维度为 $10000×1$，词嵌入后生成的词嵌入向量（embedding vactors）用 $e_i$ 表示，则有：

$$E \cdot O_i = e_i$$

但在实际情况下一般不这么做。因为 one-hot 向量维度很高，且几乎所有元素都是 0，这样做的效率太低。因此，实践中直接用专门的函数查找矩阵 $E$ 的特定列。

## 学习词嵌入

神经概率语言模型（Neural Probabilistic Language Model）构建了一个能够通过上下文来预测未知词的神经网络，在利用梯度下降算法训练这个语言模型的同时学习词嵌入。

假设，输入句子：I want a glass of orange (juice)。通过这句话的前 6 个单词，预测最后的单词“juice”。构建的神经网络模型结构如下图所示：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/05.jpg)

神经网络输入层包含6个嵌入向量,每个嵌入向量的维度为 300，则输入层总共有 1800 个输入。softmax 层有 10000 个概率输出，与词汇表包含的单词数目一致。正确的输出 label 是“juice”。其中 $E,W^{[1]},b^{[1]},W^{[2]},b^{[2]}$ 为待求值。对足够的训练例句样本，运用梯度下降算法，迭代优化，最终求出嵌入矩阵 $E$。

为了让神经网络输入层数目固定，可以选择只取预测单词的前 4 个单词作为输入，例如该句中只选择“a glass of orange”四个单词作为输入。当然，这里的 4 是超参数，可调。用一个固定的历史窗口就意味着你可以处理任意长度的句子，因为输入的维度总是固定的。

一般地，我们把输入叫做 context，输出叫做 target。对应到上面这句话里：

- context: a glass of orange
- target: juice

关于 context 的选择有多种方法：

- target 前 n 个单词或后n个单词，n可调
- target 前 1 个单词
- target 附近某 1 个单词（Skip-Gram）

事实证明，不同的 context 选择方法都能计算出较准确的 $E$ 。

相关论文：[Bengio et. al., 2003, A neural probabilistic language model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## Word2Vec

Word2Vec 是一种简单高效的词嵌入学习算法，包括 2 种模型：

- Skip-Gram (SG)：根据词预测目标上下文；
- Continuous Bag of Words (CBOW)：根据上下文预测目标词；

在 Skip-Gram 模型中，我们要做的是抽取上下文（context）和目标词（target）配对，来构造一个监督学习问题。我们的做法是：随机选一个词作为上下文词，然后随机在一定词距内选另一个词，比如在上下文词前后5个词内或者前后10个词内选择目标词，这将是一个监督学习问题，训练一个如下图结构的网络。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/06.jpg)

值得一提的是，构造这个监督学习问题的目标并不是想要解决这个监督学习问题本身，而是想要使用这个学习问题来学到一个好的词嵌入模型。

以下面这句话为例：…before of life he rests in peace! if you hadn't nailed him …

首先随机选择一个单词作为上下文，例如“rests”；然后使用一个宽度为 5 或 10（自定义）的滑动窗，在上下文附近选择一个单词作为目标词，可以是“life”、“he”、“in”、“peace”等等。最终得到了多个“上下文—目标词对”作为监督式学习样本。

一个上下文词有一个 softmax 单元，输出以上下文词为条件下目标词出现的条件概率：

$$
softmax:p\left( t \middle| c \right) = \frac{e^{\theta_{t}^{T}e_{c}}}{\sum_{j = 1}^{10,000}e^{\theta_{j}^{T}e_{c}}}
$$

其中，$\theta_{t}$ 为目标词对应的参数，$e_{c}$ 为上下文词的嵌入向量，且 $e_{c} = E · O_c$，$O_c$ 为上下文词的 one-hot 向量。

损失函数仍选用交叉熵：

$$L(\hat y, y) = -\sum^{10,000}_{i=1}y_ilog\hat y_i$$

通过反向传播梯度下降的训练过程，可以得到模型的参数 $E$ 和 softmax 的参数。

然而这种算法计算量大，影响运算速度：在上面的 softmax 单元中，我们需要对所有 10000 的所有词做求和计算，计算量庞大。解决的办法之一是使用 hierarchical softmax classifier，即树形分类器。在实践中，一般采用霍夫曼树（Huffman Tree）而非平衡二叉树，常用词在顶部。

实际上有两个不同版本的 Word2Vec 模型，Skip-Gram 只是其中的一个，另一个叫做 CBOW，即连续词袋模型（Continuous Bag-Of-Words Model），它获得中间词两边的的上下文，然后用周围的词去预测中间的词，这个模型也很有效，也有一些优点和缺点。

总结下：CBOW 是从原始语句推测目标字词；而 Skip-Gram 正好相反，是从目标字词推测出原始语句。CBOW 对小型数据库比较合适，而 Skip-Gram 在大型语料中表现更好。

相关论文：[Mikolov et. al., 2013. Efficient estimation of word representations in vector space.](https://arxiv.org/pdf/1301.3781.pdf)

## 负采样

为了解决 softmax 计算较慢的问题，Word2Vec 的作者后续提出了负采样（Negative Sampling）模型。这个算法中要做的是构造一个新的监督学习问题：给定一对单词，比如 orange 和 juice，然后来预测这是否是一对上下文词-目标词（context-target）。

训练过程中，如下图所示，从语料库中选定“上下文词-目标词对”，并将标签设置为 1。另外任取 $k$ 对非“上下文词-目标词对”，作为负样本，标签设置为 0。（若只有较少的训练数据，$k$ 的值取 5-20，能达到比较好的效果；若拥有大量训练数据，$k$ 的取值取 2-5 较为合适。）

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/07.png)

原网络中的 softmax 变成多个 sigmoid 单元输出$上下文-目标词对$ $(c, t)$ 为正样本 $(y=1)$ 的概率：

$$P(y=1 | c, t) = \sigma(\theta_t^Te_c)$$

其中，$θ_t$、$e_c$ 分别代表目标词和上下文词的词向量。通过这种方法将之前的一个复杂的多分类问题变成了多个简单的二分类问题，而降低计算成本。

之前训练中每次要更新 $n$ 维的多分类 softmax 单元（$n$ 为词典中词的数量）。现在每次只需要更新 $k+1$ 维的二分类 sigmoid 单元，计算量大大降低。

在选定了上下文（Content）后，在确定正样本的情况下，我们还需要选择 $k$ 个负样本以训练每个上下文的分类器。

- 通过单词出现的频率进行采样：导致一些类似 a、the、of 等词的频率较高；
- 均匀随机地抽取负样本：没有很好的代表性；

关于计算选择某个词作为负样本的概率，作者推荐采用以下公式（而非经验频率或均匀分布）：

$$p(w_i) = \frac{f(w_i)^{\frac{3}{4}}}{\sum^m_{j=0}f(w_j)^{\frac{3}{4}}}$$

这种方法处于上面两种极端采样方法之间，即不用频率分布，也不用均匀分布，其中，$f(w_i)$ 代表语料库中单词 $w_i$ 出现的频率。上述公式更加平滑，能够增加低频词的选取可能。

相关论文：[Mikolov et. al., 2013. Distributed representation of words and phrases and their compositionality](https://arxiv.org/pdf/1310.4546.pdf)

## GloVe词向量

GloVe（global vectors for word representation）词向量模型是另外一种计算词嵌入的方法，虽然相比下没有 Skip-Gram 模型用的多，但是相比这种模型却更加简单。

Glove 模型基于语料库统计了词的共现矩阵 $X$，$X$ 中的元素 $X_{ij}$ 表示单词 $i$ 和单词 $j$ “为上下文-目标词对”的次数。（一般地，如果不限定上下文词一定在目标词的前面，则有对称关系 $X_{ij}=X_{ji}$ ；如果有限定先后，则$X_{ij}\neq  X_{ji}$ 。接下来的讨论中，我们默认存在对称关系 $X_{ij}=X_{ji}$ ）。

GloVe 模型的损失函数为：

$$J = \sum^N_{i=1}\sum^N_{j=1}(\theta^t_ie_j - log(X_{ij}))^2$$

从上式可以看出，当两个词的嵌入向量越相近，同时出现的次数越多，则对应的loss越小。

为了避免出现“两个单词不会同时出现、无相关性”的情况，即 $X_{ij}=0$ 时 $log(X_{ij})$ 为负无穷大，从而引入一个权重因子 $f(X_{ij})$：

$$J = \sum^N_{i=1}\sum^N_{j=1}f(X_{ij})(\theta^t_ie_j - log(X_{ij}))^2$$

$X_{ij}=0$ 时，$f(X_{ij}) = 0$。这种做法直接忽略了无任何相关性的上下文词和目标词，只考虑 $X_{ij}>0$ 的情况。

一般地，引入偏移量，则最终损失函数表达式为：

$$J = \sum^N_{i=1}\sum^N_{j=1}f(X_{ij})(\theta^t_ie_j + b_i + b_j - log(X_{ij}))^2$$

其中，$\theta_i$、$e_j$是单词 $i$ 和单词 $j$ 的词向量；$b_i$、$b_j$ 为偏移量。

值得注意的是，参数 $\theta_i$ 和 $e_j$ 都是需要学习的差数，在这个目标算法中二者是对称的关系，所以我们可以一致地初始化 $\theta_i$ 和 $e_j$，然后用梯度下降来最小化输出，在处理完所有词后，直接取二者的平均值作为词嵌入向量，最终的 $e_w$ 可表示为：

$$e_{w}^{(final)}= \frac{e_{w} +\theta_{w}}{2}$$

从上面的目标中，可以看出我们想要学习一些向量，他们的输出能够对上下文和目标两个词同时出现的频率进行很好的预测，从而得到我们想要的词嵌入向量。

相关论文：[Pennington st. al., 2014. Glove: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

## 情感分类

情感分类是指分析一段文本对某个对象的情感是正面的还是负面的，实际应用包括舆情分析、民意调查、产品意见调查等等。训练情感分类模型时，面临的挑战之一可能是标记好的训练数据不够多。然而有了词嵌入得到的词向量，只需要中等数量的标记好的训练数据，就能构建出一个表现出色的情感分类器。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/08.png)

这是一个情感分类问题的一个例子，输入 $x$ 是一段文本，而输出 $y$ 是你要预测的相应情感。我们要做的就是：训练一个将左边的餐厅评价转换为右边评价所属星级的情感分类器，也就是实现 $x$ 到 $y$ 的映射。有了用词嵌入方法获得的嵌入矩阵 $E$，一种简单的实现方法如下：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/09.png)

如上图所示，用词嵌入方法获得嵌入矩阵 $E$ 后，计算出句中每个单词的词向量并取平均值，输入给一个 softmax 单元，输出预测结果 $\hat y$。

这种方法的优点是适用于任何长度的文本；缺点是没有考虑词的顺序，对于包含了多个正面评价词的负面评价，很容易预测到错误结果。比如句子："Completely lacking in good taste, good service, and good ambiance."，虽然 good这个词出现了很多次，有 3 个 good，如果如上图方法一般忽略词序，仅仅把所有单词的词嵌入加起来或者平均下来，你最后的特征向量会有很多 good 的表示，你的分类器很可能认为这是一个好的评论，然而事实上这是一个差评，只有一星的评价。

为了解决这一问题，情感分类的另一种模型是 RNN。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/10.png)

如上图所示，用词嵌入方法获得嵌入矩阵 $E$ 后，然后输入到 many-to-one 的 RNN 模型中，最后通过最后的 softmax 分类器中，输出预测结果 $\hat y$。由于词向量是从一个大型的语料库中获得的，这种方法将保证了词的顺序的同时能够对一些词作出泛化。

## 词向量除偏

现在机器学习和人工智能算法正渐渐地被用以辅助或是制定极其重要的决策，因此我们想尽可能地确保它们不受非预期形式偏见影响，比如说性别歧视、种族歧视等等。

例如，使用未除偏的词嵌入结果进行类比推理时，"Man" 对 "Computer Programmer" 可能得到 "Woman" 对 "Housemaker" 等带有性别偏见的结果。词嵌入除偏的方法有以下几种。

以性别偏见为例，我们来探讨下如何消除词嵌入中的偏见。

**中和本身与性别无关词汇** 

对于“医生（doctor）”、“老师（teacher）”、“接待员（receptionist）”等本身与性别无关词汇，可以 **中和（Neutralize）** 其中的偏见。首先用“女性（woman）”的词向量减去“男性（man）”的词向量，得到的向量 $g=e_{woman}−e_{man}$ 就代表了“性别（gender）”。假设现有的词向量维数为 50，那么对某个词向量，将 50 维空间分成两个部分：与性别相关的方向 $g$ 和与 $g$ **正交** 的其他 49 个维度 $g_{\perp}$。如下左图：

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/11.jpg)

而除偏的步骤，是将要除偏的词向量（左图中的 $e_{receptionist}$）在向量 $g$ 方向上的值置为 0，变成右图所示的 $e_{receptionist}^{debiased}$。

公式如下：

$$e_{component}^{bias} = \frac{e · g}{||g||^2_2} × g$$

$$e_{receptionist}^{debiased} = e - e_{component}^{bias}$$

**均衡本身与性别有关词汇** 

对于“男演员（actor）”、“女演员（actress）”、“爷爷（grandfather）”等本身与性别有关词汇，中和“婴儿看护人（babysit）”中存在的性别偏见后，还是无法保证它到“女演员（actress）”与到“男演员（actor）”的距离相等。对这样一对性别有关的词，除偏的过程是 **均衡（Equalization）** 它们的性别属性。其核心思想是确保一对词（actor 和 actress）到 $g_{\perp}$ 的距离相等。

![](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week2/md_images/12.jpg)

对需要除偏的一对词 $w_1$、$w_2$，选定与它们相关的某个未中和偏见的单词 $B$ 之后，均衡偏见的过程如下公式：

$$\mu = \frac{e_{w1} + e_{w2}}{2}$$

$$\mu_{B} = \frac {\mu · bias\\\_axis}{||bias\\\_axis||_2} + ||bias\\\_axis||_2 · bias\\\_axis$$

$$\mu_{\perp} = \mu - \mu_{B}$$

$$e_{w1B} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{(e_{\text{w1}} - \mu_{\perp}) - \mu_B} {|(e_{w1} - \mu_{\perp}) - \mu_B)|}$$

$$e_{w2B} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{(e_{\text{w2}} - \mu_{\perp}) - \mu_B} {|(e_{w2} - \mu_{\perp}) - \mu_B)|}$$

$$e_1 = e_{w1B} + \mu_{\perp}$$

$$e_2 = e_{w2B} + \mu_{\perp}$$


相关论文：[Bolukbasi et. al., 2016. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings](https://arxiv.org/pdf/1607.06520.pdf)