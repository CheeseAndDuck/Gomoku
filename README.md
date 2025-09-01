# Gomoku
Gomoku AI game models

一共包含三个版本
original 和 originalV2是参考了AlphaGo，其中所采用的MCTS是“选择→扩展→模拟→回溯”
newV5是参考了AlphaGo Zero，其中所采用的MCTS是“选择→扩展→评估→回溯”

originalV2和newV5都是结合价值评估网络和累积折扣奖惩
