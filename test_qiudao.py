# 示例 3：用于强化学习中的目标计算
# 具体可以参考笔者的另一篇博客：PyTorch 中detach的使用：以强化学习中Q-Learning的目标值计算为例
# 强化学习中通常需要用 detach 分离目标值的计算，例如 Q-learning：
#  在强化学习中经常需要用这个detach操作,来做 pi_old的模拟.

import torch
# 假设 q_values 是当前 Q 网络的输出
q_values = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
next_q_values = torch.tensor([15.0, 25.0, 35.0], requires_grad=True)

# 使用 detach 防止目标值的梯度传播
target_q_values = (next_q_values.detach() * 0.9) + 1
print(target_q_values)
# 损失计算
loss = ((q_values - target_q_values) ** 2).mean() # 导数等于 2/3 *(q-tar_q)
print("q_values 的梯度：", q_values.grad)  
loss.backward()

print("q_values 的梯度：", q_values.grad)  # q_values 会有梯度

#==测试一下grpo_ref_split.py : 180行



per_token_logps= torch.tensor([10.0, 20.0, 30.0], requires_grad=True)

# advantages= torch.tensor([1, -1, 1])
loss = (torch.exp(per_token_logps - per_token_logps.detach()) ).mean()
loss.backward()
print("per_token_logps 的梯度：", per_token_logps.grad)  # q_values 会有梯度

print('能看出来即使一样的减完,梯度也不是0,也可以逆向传导')


