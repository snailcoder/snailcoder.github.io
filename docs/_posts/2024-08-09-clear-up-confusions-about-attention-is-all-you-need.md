---
layout: post
title:  Attention Is All You Need答疑解惑
date:   2024-08-09 22:03:00 +0800
tags:   transformers
---

最近一直和同事骑车下班，路上他问了我不少关于transformer的问题。正好也看到知乎上有人在问Attention is all you need论文的一些细节，不禁回想起自己第一次读到此文的感受：糊里糊涂。不能怪自己的理解力差，论文写作真的不咋地。你看，七年过去了，网上介绍transformers的文章早已汗牛充栋，到如今也有不少人看不懂论文细节。

多年以前曾在自己团队内部讲过这篇论文，这里旧案重翻，把一些令人困惑的细节整理出来，希望能帮更多人答疑解惑。

1. 注意力公式$$ Attention(Q,K,V)=softmax(\frac{QK^{\mathsf{T}}}{\sqrt{d_{k}}}) $$中，为什么要除以$$\sqrt{d_{k}}$$？

    原文是这么说的：
    > We suspect that for large values of $$d_{k}$$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $$ \frac{1}{\sqrt{d_{k}}} $$

    其中，$$d_{k}$$是Q和K向量的维度。论文作者认为，$$d_{k}$$越大，Q向量和K向量的点积越大，它们主要分布在softmax函数的***饱和区域***，softmax在这些区域的梯度趋于0，会导致***梯度消失***。

    那么，为什么$$d_{k}$$越大，点积越大呢？

    设向量$$\mathbf{q}$$和$$\mathbf{k}$$的各分量都是独立同分布的随机变量，均值为0，方差为1。$$\mathbf{q}$$和$$\mathbf{k}$$点积的期望为：

    $$E(\mathbf{q}\cdot\mathbf{k})=E(q_{1}\cdot k_{1}+\cdots +q_{k}\cdot k_{d_{k}})$$

    根据数学期望的线性性质（即和的期望等于期望的和），有：

    $$E(q_{1}\cdot k_{1}+\cdots +q_{k}\cdot k_{d_{k}})=E(q_{1}\cdot k_{1})+\cdots+E(q_{k}\cdot k_{d_{k}})=d_{k}\cdot 0=0$$

    $$\mathbf{q}$$和$$\mathbf{k}$$点积的方差为：

    $$var(\mathbf{q}\cdot\mathbf{k})=var(q_{1}\cdot k_{1}+\cdots +q_{k}\cdot k_{d_{k}})$$

    当两个变量相互独立时，和的方差等于各自方差的和，因此：
    
    $$var(q_{1}\cdot k_{1}+\cdots +q_{k}\cdot k_{d_{k}})=var(q_{1}\cdot k_{1})+\cdots+var(q_{k}\cdot k_{d_{k}})=d_{k}\cdot 1=d_{k}$$

    可见，$$\mathbf{q}\cdot\mathbf{k}$$的期望为0，方差为$$d_{k}$$。所以，$$d_{k}$$越大，方差越大，$$\mathbf{q}\cdot\mathbf{k}$$的波动越大，出现较大值的概率越大。

    按照论文的方法，给$$\mathbf{q}\cdot\mathbf{k}$$乘以系数$$\frac{1}{\sqrt{d_{k}}}$$，重新计算方差：

    $$var(\frac{q_{1}\cdot k_{1}}{\sqrt{d_{k}}})+\cdots+var(\frac{q_{k}\cdot k_{d_{k}}}{\sqrt{d_{k}}})=\frac{1}{d_{k}}(var(q_{1}\cdot k_{1})+\cdots+var(q_{k}\cdot k_{d_{k}}))=1$$

    这样，$$\mathbf{q}\cdot\mathbf{k}$$的方差不再受$$d_{k}$$影响，梯度消失问题得以避免。

2. 为什么使用三角函数位置编码？

    论文提供的位置编码公式如下：

    $$PE_{(pos,2i)}=\sin(\frac{pos}{10000^{2i/d_{model}}}), PE_{(pos, 2i+1)}=\cos(\frac{pos}{10000^{2i/d_{model}}})$$

    其中，pos是token在序列中的位置，2i和2i+1分别表示输入维度的偶数位置和奇数位置。原文解释了使用这种位置编码的原因：

    > We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, $$PE_{pos+k}$$ can be represented as a linear function of $$PE_{pos}$$.

    也就是说，$$PE_{pos+k}$$可以用$$PE_{pos}$$线性表示。利用三角函数的和差角公式，有：

    $$\begin{align}
      PE_{(pos+k,2i)}&=\sin(\frac{pos+k}{10000^{2i/d_{model}}}) \\
      &=\sin(\frac{pos}{10000^{2i/d_{model}}})\cos(\frac{k}{10000^{2i/d_{model}}})+\cos(\frac{pos}{10000^{2i/d_{model}}})\sin(\frac{k}{10000^{2i/d_{model}}}) \\
      &=PE_{(pos,2i)}\cos(\frac{k}{10000^{2i/d_{model}}})+PE_{(pos,2i+1)}\sin(\frac{k}{10000^{2i/d_{model}}})
      \end{align}$$

    $$\begin{align}
      PE_{(pos+k,2i+1)}&=\cos(\frac{pos+k}{10000^{2i/d_{model}}}) \\
      &=\cos(\frac{pos}{10000^{2i/d_{model}}})\cos(\frac{k}{10000^{2i/d_{model}}})-\sin(\frac{pos}{10000^{2i/d_{model}}})\sin(\frac{k}{10000^{2i/d_{model}}}) \\
      &=PE_{(pos,2i+1)}\cos(\frac{k}{10000^{2i/d_{model}}})-PE_{(pos,2i)}\sin(\frac{k}{10000^{2i/d_{model}}})
      \end{align}$$

    论文希望这种位置编码能够帮助模型理解两个token之间的相对位置关系，使它们之间的注意力更加合理。想法挺好的，但效果未必好，不然BERT、GPT等模型也不会抛弃三角函数位置编码。

    还有一个让人费劲的地方，为什么三角函数内部要用$$\frac{pos}{10000^{2i/d_{model}}}$$这么一个奇怪的形式呢？

    假设我们把位置编码写为$$PE_{(pos,2i)}=\sin(pos)$$。由于三角函数具有周期性，导致$$PE_{(pos,2i)}=PE_{(pos+2\pi,2i)}\approx PE_{(pos+6,2i)}$$，即每隔5个token，就会出现一个相同的位置编码。显然，这并不合理。如果我们把位置编码写为$$PE_{(pos,2i)}=\sin(\frac{pos}{10000})$$，就不会出现上面的情况了，因为下一个相同的位置编码为$$PE_{(pos,2i)}=\sin(\frac{pos}{10000}+2\pi)=\sin(\frac{pos+2\pi\cdot 10000}{10000})$$，出现位置是$$pos+2\pi\cdot 10000$$，即该正弦函数的**波长**为$$2\pi\cdot 10000$$，远大于token序列长度。

    然而，$$PE_{(pos,2i)}=\sin(\frac{pos}{10000})$$忽略了维度信息，如果把它当作位置编码，得到的位置向量的偶数维全部相等，奇数维也全部相等。为了避免这种“奇怪”的向量，在公式中引入向量的维度信息，得到$$PE_{(pos,2i)}=\sin(\frac{pos}{10000^{2i/d_{model}}})$$。由于$$2i\in [0,d_{model}]$$，考虑两个极端情况：

    - 当$$2i=0$$时，$$PE_{(pos,2i)}=\sin(pos)$$，正弦函数的波长为$$2\pi$$；
    - 当$$2i=d_{model}$$时，$$PE_{(pos,2i)}=\sin(\frac{pos}{10000})$$，正弦函数的波长为$$2\pi\cdot 10000$$。

    可以看出，位置编码向量在不同位置的低维度上容易出现相同数值，在高维度上几乎不会出现相同数值。这一现象是好是坏，这里不做探究。

3. 为什么使用多头注意力？

    原文是这样说的：

    > Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.

    可类比于CNN有多个卷积核，不同卷积核负责捕获不同特征，例如一个核捕获水平边缘检测特征，一个核捕获垂直边缘检测特征等。Transformer的每个头都有自己的Q、K、V矩阵，一个头负责建立从动词到宾语的注意力，一个头负责建立从当前单词到下一个单词的注意力等，可以参考[这篇论文](https://aclanthology.org/W19-4828.pdf)。

也许还有其他关于这篇论文的奇奇怪怪的问题，有时间再总结吧。

心情已经连续糟糕了半年，不知道将来会不会好。前两天立秋了。去年立秋心情还不错，甚至下午开车到独墅湖然后绕湖骑行一圈。如今终于读懂一句诗：欲买桂花同载酒，终不似、少年游。

