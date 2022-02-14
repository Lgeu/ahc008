# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Tic-Tac-Toe

import copy
import random
from subprocess import Popen, PIPE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import BaseEnvironment


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {'policy': p, 'value': v}


class Environment(BaseEnvironment):
    X, Y = 'ABC',  '123'
    BLACK, WHITE = 1, -1
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self, args=None):
        super().__init__()
        self.reset()

    def reset(self, args=None):
        self.p_game = Popen("")  # TODO



        # self.board = np.zeros((3, 3))  # (x, y)
        # self.color = self.BLACK
        # self.win_color = 0
        # self.record = []

    # def action2str(self, a, _=None):
    #     return self.X[a // 3] + self.Y[a % 3]

    # def str2action(self, s, _=None):
    #     return self.X.find(s[0]) * 3 + self.Y.find(s[1])

    # def record_string(self):
    #     return ' '.join([self.action2str(a) for a in self.record])

    # def __str__(self):
    #     s = '  ' + ' '.join(self.Y) + '\n'
    #     for i in range(3):
    #         s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
    #     s += 'record = ' + self.record_string()
    #     return s

    def play(self, action, _=None):
        # action: "udlrUDLR." 通行不能にする、移動する、何もしない
        self.p_game.write()


        # state transition function
        # action is integer (0 ~ 8)
        x, y = action // 3, action % 3
        self.board[x, y] = self.color

        # check winning condition
        win = self.board[x, :].sum() == 3 * self.color \
            or self.board[:, y].sum() == 3 * self.color \
            or (x == y and np.diag(self.board, k=0).sum() == 3 * self.color) \
            or (x == 2 - y and np.diag(self.board[::-1, :], k=0).sum() == 3 * self.color)

        if win:
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)

    def diff_info(self, _):
        if len(self.record) == 0:
            return ""
        return self.action2str(self.record[-1])

    def update(self, info, reset):
        if reset:
            self.reset()
        else:
            action = self.str2action(info)
            self.play(action)

    def turn(self):
        return self.players()[len(self.record) % 2]

    def terminal(self):
        # check whether the state is terminal
        return self.win_color != 0 or len(self.record) == 3 * 3

    def outcome(self):
        # terminal outcome
        outcomes = [0, 0]
        if self.win_color > 0:
            outcomes = [1, -1]
        if self.win_color < 0:
            outcomes = [-1, 1]
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}

    def legal_actions(self, _=None):
        # legal action list
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def players(self):
        return [0, 1]

    def net(self):
        return Model()

    def observation(self, player=None):
        # input feature for neural nets
        turn_view = player is None or player == self.turn()
        color = self.color if turn_view else -self.color
        a = np.stack([
            np.ones_like(self.board) if turn_view else np.zeros_like(self.board),
            self.board == color,
            self.board == -color
        ]).astype(np.float32)
        return a


if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = e.legal_actions()
            print([e.action2str(a) for a in actions])
            e.play(random.choice(actions))
        print(e)
        print(e.outcome())
