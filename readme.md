# 小作业
## BPTT算法实现
    - 代码：BPTT-Test.ipynb
    - 根据老师给出的公式计算即可
    - 疑问：作业给出了W，U，V的结果，但是bo,bs的结果没有给出，不知道写的是否正确
## RNN中num_steps和state_size对效果的影响
    - num_steps 表示网络记忆的长度，值越大记忆长度越大，学习到的上下文联系更多，值太小会学不到数据前后的联系
    - state_size 表示特征向量的长度，特征向量越长，对文字的表述更清晰，太小无法学习到文字的全部信息
# Embedding作业
    - 目的：将自然语言间的联系用数学向量在表示，方便计算机操作
    - 作法：普通的自然语言处理是one hot数据，维度太高，计算量太大，Embedding是将维度降低，将自然语言数据特征映射到低维空间。word2vec是一种Embedding算法，使用稠密的表达方式，对one hot降维，得到低维实数向量，将特征向量映射到多维空间，不仅仅是坐标轴上，避免产生维度爆炸。
    - 原理：完全通过给的文本来计算语言文字间的联系，并生成一个低维向量，不需要对各种语言进行单独设计模型。只要给的数据量足够大就可以得到足够的语义联系向量
    - 代码：word2vec.py
    - 结果：tsne_w11.png 语义相似度，由其对应向量的余弦相似度求得，所以相似的词汇其向量将聚集为一处
# RNN训练
    - RNN--循环神经网络，与全连接及卷积神经网络的区别是增加了时间维度的考虑。取一段时间的数据做为一个数据集，在这一数据集里每一步的权重计算时都要将之前一步的权重参与计算。在反向计算中，也同样将后一个权重的因子参与到前一个权重的计算中来更新前一个权重，将所有权重串起来，从而达到权重W在时间维度上的体现。
    - RNN的算法比之前学习的卷积网络要复杂一些，不像卷积网络那个结构清晰。有一些计算，如outputs，outputs_state的含义,以及concat，reshape等操作的意义吃的不是很透，总感觉有些似是而非
    - 本次作业没有实现较好的写诗机器人，个人认为主要因素有以下几个：
        1.数据未进行处理，有很多乱码、文字缺失，影响结果。
        2.标点，空格，换行等字符未处理
        3.所给的数据量较小
        4.label设定的方式过于简单
        5.由于tenserflow版本问题模型一直运行不起来，浪费太多时间，没有进行超参数调优
    - 本次作业最大的疑问就是为何代码在tenserflow 1.5版本上无法运行，此问题一直没搞定。希望老师给1.5版本上的参考答案
    
        
