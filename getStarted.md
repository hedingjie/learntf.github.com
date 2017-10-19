##TensorFlow入门

这份指导教程将开始你的TensorFlow编程生涯。在开始阅读本教程之前，请先确保您[安装了TensorFlow](https://www.tensorflow.org/install/index)。要弄明白这个教程，你需要具有以下知识储备：
* Python编程基础
* 了解数组的一些知识
* 最好能了解一些关于机器学习的东西。然而，如果你没有一点基础的话，这仍然是一本你入门机器学习的好教材

TensorFlow提供了多种API。最底层的API——TensorFlow核心（TensorFlow Core）——为你提供了完整的程序控制功能。我们向研究机器学习的人员或者是有志于提升自己对于自己模型控制水平的人员强烈推荐TensorFlow核心。更高层次的API是建立在TensorFlow核心之上，但它们学起来和用起来通常都比TensorFlow核心简单。补充一点，这些相对顶层的API让不同用户之间重复的任务变得简单而稳定。像tf.estimator这样的API可以帮助你管理数据集（data sets），评估器（estimators），训练过程（training）以及接口(inference)。这个指导将首先从TensorFlow核心的教程开始。之后，我们将演示如何实现tf.estimator中的相同的模型。了解TensorFlow核心的原则有助于你在调用更加复杂的高层次API的时候，能对其内部工作过程有一个很好的理解。
###张量（Tensors）
TensorFlow中主要的数据单元是张量（tensors）。一个张量包含了一个原始值集合，但是它被转化成了任意维度的数组。一个张量的等级（rank）是它的维度数。这有一些张量的例子：
```python
3 # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```
##TensorFlow核心指导
###导入TensorFlow
下面是导入TensorFlow的标准的做法：
```python
import tensorflow as tf
```
这使得Python能够访问TensorFlow的所有类（classes），方法（methods）以及变量（Symbols）。如果你已经完成了上述步骤，请继续阅读。
###计算图（The Computational Graph）
你可以认为TensorFlow核心程序包含以下两个独立的部分：
1. 构建计算图
2. 运行计算图

计算图是由一系列的TensorFlow操作作为节点，所组成的一个图（graph）。接下来让我们构建一个简单的计算图。它的每一个节点都都由0个以上的张量作为输入，并且产生一个张量作为输出。节点中有一种常量（constant）类型节点。正如所有TensorFlow中的常量一样，它不接受任何输入，而是将它内部存储的值作为输出。我们可以创建两个浮点类型张量node1和node2，示例如下：
```python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```
最后那条print输出语句输出的结果如下:
```python
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```
注意，输出的结果并不是你所想象的是3.0和4.0这样的数值。