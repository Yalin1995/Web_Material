# coding:utf-8
# Author：Yalin Yang
# __file__ : vistualize.py
# __time__ : 2019/1/6 19:25

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义一个添加神经层的函数， 参数有输入，输入大小，输出大小 (指矩阵的维度) 和 激励函数（默认为None）


def add_layer(inputs,in_size,out_size,
              activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    # 计算预测值
    Wx_plus_b = tf.matmul(inputs,Weights) +biases

    # 若是非线性关系，必须加上激励函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 定义一个 -1 ~ 1 ， 300 行的 x 值

x_data = np.linspace(-1,1,300)[:,np.newaxis]

# 增加一点 noise ，使得 y 不完全遵循 二次函数
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# # 建立输入层神经元为 1 （根据输入对象的属性而定，不是个数）， 隐藏层神经元为 10 ， 输入层神经元为 1 （根据输出对象的属性而定）的神经网络。
# 定义隐藏层
# 定义占位符用于隐藏层的参数传递
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# input_size 衍于输入神经元个数， output_size衍于该层本身神经元个数
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function= None)

# 计算真实值和预测值之间的差异，对每个例子进行求和后求平均
loss= tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction)
                    ,reduction_indices=[1]))

# 建立优化器，第一个参数为学习效率，用它来减小误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化所有的变量
init = tf.global_variables_initializer()

# 建立对话，激活神经网络
sess = tf.Session()
sess.run(init)   # Very important

# 创建一个图片框
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# 以点的形式 plot 真实数据
ax.scatter(x_data,y_data)
# 将图像显示为动态
plt.ion()
plt.show()

for step in range(5000):
    sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
    if step % 100 == 0:
        # print (step,sess.run(loss,feed_dict = {xs:x_data,ys:y_data}))
        try:
            # 每次显示后均去除预测线段
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict = {xs:x_data})
        # 以曲线的形式 plot 出 prediction 的值, color 为 red （"r-"）, 线宽为5
        lines = ax.plot(x_data,prediction_value,"r-",lw=5)
        plt.pause(0.5)
