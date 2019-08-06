import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成带随机噪声的一次函数
x_data = np.linspace(-10, 10, 100)[:, np.newaxis]
noise = np.random.normal(0., 1., x_data.shape)
y_data = 5*x_data + noise + 7

# # 原始数据可视化
# fig = plt.figure()
# plt.scatter(x_data, y_data)
# plt.show()

# 数据处理
x_data = tf.convert_to_tensor(tf.cast(x_data, tf.float32))
y_data = tf.convert_to_tensor(tf.cast(y_data, tf.float32))
db_train = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(5)

# 构建一个单层网络
w = tf.Variable(tf.zeros([1, 1]), tf.float32)
b = tf.Variable(tf.zeros([1]), tf.float32)

epochs = 50
lr = 0.01

cost = tf.keras.metrics.Mean()

for epoch in range(1, epochs +1):
    for step, (x, y) in enumerate(db_train):
        x = np.matrix(x)
        # 计算数据传输
        with tf.GradientTape() as tape:
            y_hat = x@w + b
            loss = tf.reduce_sum(tf.square(y_hat - y)) / x.shape[0]

        # 计算梯度
        grads = tape.gradient(loss, [w, b])
        # 更新梯度
        w.assign_sub(lr*grads[0])
        b.assign_sub(lr * grads[1])

        cost.update_state(loss)

        # 输出损失函数值
        if step % 5 == 0:
            print('step: ', step, 'loss: ', cost.result().numpy())
            cost.reset_states()

# 绘制回归结果曲线
y_pred = x_data@w + b

fig =plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, c='red')
plt.show()