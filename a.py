
import numpy as np
# 预测模型：X(K) = X(K - 1) + X'(K-1)*dt + X"(K-1)*dt^2 * (1/2!) + Q2 # 匀加速直线运动
# 观测模型：Y(K) = X(K) + R  R~N(0,1)
#      此时状态变量 X = [X(K)  X'(K)  X"(K)].T 列向量 X = [位移 速度 加速度]
# Y(K) = H * X + R             
#                  H = [1 0 0] 行向量
# refer to state_transition_matrix.jpg
dt = t[1] - t[0]
F4 = np.array([[1, dt, 0.5*dt**2],
               [0, 1,         dt],
               [0, 0,         1]]) # 注意dt是否小于计算机所能表示的精度

H4 = np.array([[1, 0, 0]
               [1, 0, 0]])
               
Q4 = np.array([[1, 0,    0],
               [0, 0.01, 0],
               [0, 0,    0.0001]])

R4 = np.array([[3, 0],
               [0, 3]]) # 有两个传感器，这两个传感器的误差都是假设服从高斯分布的，不相关，且独立 

# 设置初值
# 期望
M_plus_4 = np.zeros((3, L))
M_plus_4[0,0] = 0.1 ** 2 # 初始位移
M_plus_4[1,0] = 0        # 初始速度
M_plus_4[2,0] = 0        # 初始加速度

# 方差
S_plus_4 = np.array([[0.01, 0,     0],
                     [0,    0.01,  0],
                     [0,    0,     0.0001]])


for i in range(1, L): # i 就是当前的时刻（即jpg中的k）
    # 预测步
    M_minus_4      = F4 @ M_plus_4[:, i - 1]
    S_minus_4      = F4 @ S_plus_4 @ F4.T + Q4
    K4             = S_minus_4 @ H4.T @ inv(H4 @ S_minus_4 @ H4.T + R4)
    # 更新步
    Y = np.zeros((2,1))
    Y[0,0] = y[i]
    Y[1,0] = y2[i]
    M_plus_4[:,i]  = M_minus_4 + K4@(Y - H4 @ M_minus_4)
    S_plus_4       = (np.identity(3) - K4@H4)@S_minus_4


plt.plot(t, x, label = "gt")
plt.plot(t, y, label = "observed")
plt.plot(t, M_plus_2, label = "m1")
plt.plot(t, M_plus_4[0,:], label = "two sensors fusion"), 

plt.xlabel("t")
plt.ylabel("signal")
plt.title("gt and sensor data")
plt.legend()
plt.show()