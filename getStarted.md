## TensorFlow入门

这份指导教程将开始你的TensorFlow编程生涯。在开始阅读本教程之前，请先确保您[安装了TensorFlow](https://www.tensorflow.org/install/index)。要弄明白这个教程，你需要具有以下知识储备：
* Python编程基础
* 了解数组的一些知识
* 最好能了解一些关于机器学习的东西。然而，如果你没有一点基础的话，这仍然是一本你入门机器学习的好教材

TensorFlow提供了多种API。最底层的API——TensorFlow核心（TensorFlow Core）——为你提供了完整的程序控制功能。我们向研究机器学习的人员或者是有志于提升自己对于自己模型控制水平的人员强烈推荐TensorFlow核心。更高层次的API是建立在TensorFlow核心之上，但它们学起来和用起来通常都比TensorFlow核心简单。补充一点，这些相对顶层的API让不同用户之间重复的任务变得简单而稳定。像tf.estimator这样的API可以帮助你管理数据集（data sets），评估器（estimators），训练过程（training）以及接口(inference)。这个指导将首先从TensorFlow核心的教程开始。之后，我们将演示如何实现tf.estimator中的相同的模型。了解TensorFlow核心的原则有助于你在调用更加复杂的高层次API的时候，能对其内部工作过程有一个很好的理解。
### 张量（Tensors）
TensorFlow中主要的数据单元是张量（tensors）。一个张量包含了一个原始值集合，但是它被转化成了任意维度的数组。一个张量的等级（rank）是它的维度数。这有一些张量的例子：

```python
3 # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

## TensorFlow核心指导
### 导入TensorFlow
下面是导入TensorFlow的标准的做法：

```python
import tensorflow as tf
```

这使得Python能够访问TensorFlow的所有类（classes），方法（methods）以及变量（Symbols）。如果你已经完成了上述步骤，请继续阅读。
### 计算图（The Computational Graph）
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
Tensor("Const:0", shape=(), dtype=float32) 
Tensor("Const_1:0", shape=(), dtype=float32)
```

注意，输出的结果并不是你所想象的是3.0和4.0这样的数值。相反，当我们仔细观察时，发现其实输出的是产生分别3.0和4.0这样的数值的节点。为了直观地观察节点，我们必须在session中运行计算图。session中封装了TensorFlow运行环境的操作和状态。

下面的代码创建了一个Session对象并且调用了它的run方法，运行整张计算图来评估node1和node2。运行整张图的代码如下：

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

这时我们会得到所期望的结果

```python
[3.0, 4.0]
```

我们可以通过组合张量节点和操作（operations）（操作也是节点）构建更为复杂的计算图。例如我们可以将两个常量节点相加以产生一个新的图，如下所示：

```python
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
```

TensorFlow提供了一个称之为TensorBoard的使用工具，它可以将计算图以可视化的方式呈现出来，如下所示：

![TensorBoard](https://www.tensorflow.org/images/getting_started_add.png)

这个图其实并没有多大意义，因为它总是产生一个常量结果。而一个能接受外界输入参数的节点称为预留节点（placeholder）。预留节点是的我们能在之后为其指派值而不用预先给定。

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

以上三行有点像一个我们定义了一个函数或者lambda表达式，它接收两个参数（a和b），然后进行运算。在计算这个图的时候，我们可以在run函数中通过feed_dict参数来为预留节点赋值：

```python
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
```

输出结果：

```python
7.5
[ 3.  7.]
```

我们可以借助TensorBoard看到其计算图如下：

![TensorBoard中的计算图](https://www.tensorflow.org/images/getting_started_adder.png)

我们可以通过添加其他操作使得原有计算图更为复杂，如下所示：

```python
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
```

输出结果如下：

```python
22.5
```

而之前的计算图将变成下面的样子：

![图3](https://www.tensorflow.org/images/getting_started_triple.png)

在机器学习中，我们通常希望模型能够像上面的示例接受任意输入。为了使得模型便于训练，在相同输入的情况下，我们需要通过控制图来获得不同的输出。变量（Variables）允许我们向计算图中添加可训练的参数。而在初始化变量的时候，我们通常为变量设定一个类型并赋一个初值：

```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

常量在你调用tf.constant时便被初始化了，之后它们的值将不再发生改变。而变量则形成了鲜明的对比，在你调用tf.Variable时，变量并没有被初始化。为了初始化TensorFlow程序中的所有值，你必须明确地调用如下操作：

```python
init = tf.global_variables_initializer()
sess.run(init)
```

很重要的一点是你要意识到，```init```是TensorFlow子图中初始化所有全局变量的句柄。直到我们调用```sess.run```之前，变量是不会被初始化的。

因为```x```是一个预留节点，所以我们针对```x```的不同的值计算```linear_model```的值，如下所示：

```python
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```

输出如下：

```python
[ 0.          0.30000001  0.60000002  0.90000004]
```

我们已经创建了一个模型，但是到目前为止我们并不知道它是否足够好。为了评估这个模型，我们需要一个预留变量```y```表示所期望的结果，同时，我们还需要编写一个损失函数（loss function）。

损失函数是用来衡量当前模型与所提供的数据的差异大小。我们将使用标准的损失模型作为线性回归的模型，它是当前模型与所提供的数据的差的平方求和得到。```linear_model - y```得到了一个向量，它的每一个元素都对应着例子中的偏差。我们调用```tf.square```来平方话这些偏差。之后我们通过```tf.reduce_sum```将这些偏差的平方求和构成一个标量，它抽象地反映了所有样本的误差：

```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```

输出损失值如下：

```python
23.66
```

我们可以通过手动为```W```和```b```赋```-1```和```1```或者改进这个值。变量可以通过```tf.initialized```赋值，但是可以通过```tf.assign```这样的操作进行更改。例如```W=-1```和```b=1```使我们这个模型最佳的参数。我们可以按照```W```和```b```进行相应的更改：

```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```

更改之后输出的损失值为0。

```python
0.0
```

```W```和```b```的“最佳”值是我们猜测得到的，但是机器学习的关键点是自动地找到正确的参数。我们将在下面的部分展示如何实现这个的。

## tf.train API

对于机器学习的详尽的讨论已经超出了本教程的范围。然而，TensorFlow提供了优化器，它可以通过逐渐改变每一个变量来使得损失函数最小化。最简单的损失函数是梯度下降函数。它根据相应变量损失的微分梯度来修改每一个变量的值。通常，手工计算导数繁琐且容易出错。所以，TensorFlow通过```tf.gradients```函数根据模型的描述自动进行求导。为了简单起见，优化器通常自动为你做了这些。示例如下：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

```python
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
```

这将使得模型的最终参数如下：

```python
[array([-0.9999969], dtype=float32), array([ 0.99999082],
 dtype=float32)]
```

现在，我们已经完成了一个真正的机器学习过程了！在这个简单的线性回归模型中，我们并没有太多地用到TensorFlow的核心代码，但是如果你需要向你的更为复杂的模型和方法中反馈数据就需要更多的代码。因此，TensorFlow针对一些常用的模式，结构以及功能提供了更高层次的抽象（封装）。我们接下来将要学习如何如何使用其中的一些抽象方法。

### 完整的程序

完整的线性回归训练模型如下所示：

```python
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

运行后，它将产生如下输出：

```python
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

注意这里面的损失值已经很小了（接近于0）。如果你亲自运行这个程序，你的损失值可能会和例子的结果不一样，因为模型是用伪随机数进行初始化的。

这个更加复杂的程序依然可以在TensorBoard上展示出来。

![图4](https://www.tensorflow.org/images/getting_started_final.png)

### tf.estimator

tf.estimator是高层次的TensorFlow库，它从以下方面简化机器学习的机制：
* 训练循环过程
* 求值循环过程
* 管理数据集
tf.estimator还提供了常用的模型。

### 基本用法
注意观察使用```tf.estimator```是如何使得线性回归程序变得更简单的。

```python
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

```
运行后，将产生如下结果：

```python
train metrics: {'loss': 1.2712867e-09, 'global_step': 1000}
eval metrics: {'loss': 0.0025279333, 'global_step': 1000}
```

注意我们的eval数据有一个更高的损失，但是它仍然很靠近于0.那意味着我们的学习过程是正确的。

### 定制一个模型

```tf.estimator```并不限制你必须使用它预制的模型。假定我们想要构建一个不同于TensorFlow内置内置模型的定制模型。我们依然需要维持```tf.estimator```数据集，反馈以及训练等的高层次的抽象。为了阐明这个情况，我们将自己利用已有的低层次TensorFlow API知识展示如何实现与之前所用到的线性回归的模型相等价的模型。

为了使用```tf.estimator```定义一个定制的模型，我们需要使用```tf.estimator.Estimator```。```tf.estimator.LinearRegressor```实际上是```tf.estimator.Estimator```的子类。我们简单地提供一个函数```model_fn```给```tf.estimator```来指明如何进行计算预测，训练以及损失而不是使用子类估计器。代码如下：

```python
import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```

运行后，产生如下结果：

```python
train metrics: {'loss': 1.227995e-11, 'global_step': 1000}
eval metrics: {'loss': 0.01010036, 'global_step': 1000}
```

### 下一步
现在，你已经对TensorFlow的基本知识有了一个大致的了解。我们也有多个教程让你来学习更多。如果你是机器学习的小白，请阅读MNIST新手入门，否则请看MNIST专家教程。


