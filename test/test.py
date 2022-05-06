# _*_ coding : utf-8 _*_
# @Time : 2022/5/4 0:13
# @Author : Cosmica
# @File : test
# @Project : PytorchMario
import math
import torch


# 其中cfg表示训练过程中的参数，env表示训练的交互环境，agent表示一个DQN的类对象
def train(cfg, env, agent):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward,
                              next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    env.close()
    return rewards, ma_rewards


if __name__ == '__main__':
    print(torch.cuda.is_available())















