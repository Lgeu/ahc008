import math
import random
import socket
from subprocess import Popen, PIPE, STDOUT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import BaseEnvironment


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MBConvBlock(nn.Module):
    def __init__(self, input_filters, output_filters, expand_ratio, kernel_size, stride, image_size):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.stride = stride

        # Expansion phase (Inverted Bottleneck)
        inp = input_filters  # number of input channels
        oup = input_filters * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False, padding="same")
            self._bn0 = nn.BatchNorm2d(num_features=oup)
            
        # Depthwise convolution phase        
        self._depthwise_conv = Conv2dStaticSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=kernel_size, stride=stride, bias=False, image_size=image_size)
        self._bn1 = nn.BatchNorm2d(num_features=oup)
        image_size //= stride

        # Squeeze and Excitation layer, if desired
        num_squeezed_channels = max(1, input_filters // 4)
        self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1, padding="same")
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1, padding="same")

        # Pointwise convolution phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=output_filters, kernel_size=1, bias=False, padding="same")
        self._bn2 = nn.BatchNorm2d(num_features=output_filters)
        self._swish = MemoryEfficientSwish()


    def forward(self, inputs):
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)  # global average pooling
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection
        input_filters, output_filters = self.input_filters, self.output_filters
        if self.stride == 1 and input_filters == output_filters:
            x = x + inputs  # skip connection
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        image_size = 32
        global_feature_dims = 25
        local_feature_dims = 30
        out_dims = 9 + 1 + 1 + 1  # 方策 (人)、reward (ペット)、面積 (人)、 捕まる確率 (ペット)

        dim1 = 32

        self.global_feature_linear_1 = nn.Linear(global_feature_dims, dim1)
        self.global_feature_bn_1 = nn.BatchNorm1d(dim1)
        self.global_feature_linear_2 = nn.Linear(dim1, dim1 * 10)
        self.global_feature_bn_2 = nn.BatchNorm1d(dim1 * 10)
        self.global_feature_mapping = torch.tensor([
            [0, 1, 2, 3, 3, 2, 1, 0],
            [1, 4, 5, 6, 6, 5, 4, 1],
            [2, 5, 7, 8, 8, 7, 5, 2],
            [3, 6, 8, 9, 9, 8, 6, 3],
            [3, 6, 8, 9, 9, 8, 6, 3],
            [2, 5, 7, 8, 8, 7, 5, 2],
            [1, 4, 5, 6, 6, 5, 4, 1],
            [0, 1, 2, 3, 3, 2, 1, 0],
        ])

        self.pre_conv = nn.Conv2d(local_feature_dims, dim1, kernel_size=3, stride=1, padding="same", bias=False)
        self.pre_bn = nn.BatchNorm2d(dim1)
        self.swish = MemoryEfficientSwish()

        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))
        
        self.encoder_blocks.append(MBConvBlock(kernel_size=3, stride=2, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))
        image_size //= 2  # 16
        self.encoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))
        
        self.encoder_blocks.append(MBConvBlock(kernel_size=3, stride=2, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))
        image_size //= 2  # 8
        
        self.decoder_blocks = nn.ModuleList()
        self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1 * 2, output_filters=dim1, image_size=image_size))
        #self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))

        image_size *= 2  # 16
        self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1 * 2, output_filters=dim1, image_size=image_size))
        self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))
        image_size *= 2  # 32
        self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1 * 2, output_filters=dim1, image_size=image_size))
        #self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1, output_filters=dim1, image_size=image_size))
        self.decoder_blocks.append(MBConvBlock(kernel_size=3, stride=1, expand_ratio=5, input_filters=dim1, output_filters=out_dims, image_size=image_size))

    def forward(self, local_features, global_features):
        # TODO: 入出力
        print("in")
        skips = []

        device = next(self.parameters()).device

        self.global_feature_mapping.to(device)

        batch_size = global_features.size(0)

        # [batch_size, global_dims] -> [batch_size, dim1]
        g = self.swish(self.global_feature_bn_1(self.global_feature_linear_1(global_features)))
        # [batch_size, dim1] -> [batch_size, dim1, 10]
        g = self.swish(self.global_feature_bn_2(self.global_feature_linear_2(g))).view(batch_size, -1, 10)
        # [batch_size, dim1, 10] -> [batch_size, dim1, 8, 8]
        g = g[:, :, self.global_feature_mapping]
        print(g.shape)

        # [batch_size, in_dims, 32, 32] -> [batch_size, dim1, 32, 32]
        x = self.swish(self.pre_bn(self.pre_conv(local_features)))
        # [batch_size, dim1, 32, 32] -> [batch_size, dim1, 8, 8]
        for block in self.encoder_blocks:
            print(x.shape)
            if block.stride == 2:
                skips.append(x)
            x = block(x)
        # [batch_size, dim1, 8, 8] -> [batch_size, out_dims, 32, 32]
        print()
        x = torch.cat([x, g], dim=1)
        for block in self.decoder_blocks:
            print(x.shape)
            if x.size(1) != block.input_filters:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        assert len(skips) == 0
        return x


class Environment(BaseEnvironment):
    N_GLOBAL_FEATURES = 25
    N_LOCAL_FEATURES = 30

    def __init__(self, args=None):
        super().__init__()
        self.port = random.randint(10000, 60000)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("127.0.0.1", self.port))
        self.server_socket.listen(1)  # 接続待ち
        self.p_game = None
        self.reset()

    def _send(self, data):
        sum_sent = 0
        while sum_sent < len(data):
            sent = self.sock.send(data[sum_sent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            sum_sent = sum_sent + sent

    def _recv(self, length):
        chunks = []
        bytes_recd = 0
        while bytes_recd < length:
            chunk = self.sock.recv(min(length - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)

    def reset(self, args=None):
        if self.p_game is not None:
            self.p_game.kill()
        # TODO: ランダムシード
        self.p_game = Popen(f"../../../tools/target/release/tester ../../../env.out {self.port} < ../../../tools/in/0000.txt", stdin=PIPE, stdout=PIPE, stderr=STDOUT)  # TODO
        self.sock, address = self.server_socket.accept()
        self.n_people = int(np.frombuffer(self._recv(1), dtype=np.int8))
        self.current_turn = 0
        self.obs = np.frombuffer(self._recv((self.N_GLOBAL_FEATURES + self.N_LOCAL_FEATURES * 32 * 32) * 4), dtype=np.float32)

    def step(self, actions):
        arr = []
        for player_id, action in sorted(actions.items()):  # type: (int, int)
            # player_id: int
            arr.append(action)
        arr = np.array(arr, dtype=np.int8)
        self._send(arr.tobytes())
        self.current_turn += 1
        # 注: C++ 側で、最終ターンではすぐに終わらないようにする
        if self.terminal():
            self.reward_arr = np.zeros(self.n_people, dtype=np.float32)
            self.outcome_arr = np.frombuffer(self._recv(self.n_people * 4), dtype=np.float32)
        else:
            self.reward_arr = np.frombuffer(self._recv(self.n_people * 4), dtype=np.float32)
            self.outcome_arr = np.zeros(self.n_people, dtype=np.float32)
            self.obs_arr = np.frombuffer(self._recv((self.N_GLOBAL_FEATURES + self.N_LOCAL_FEATURES * 32 * 32) * 4), dtype=np.float32)
            self.legal_actions_arr = np.frombuffer(self._recv(self.n_people * 2), dtype=np.int16)

    def terminal(self):
        return self.current_turn == 300

    def players(self):
        return list(range(10))
    
    def turns(self):
        return list(range(len(self.n_people)))

    def reward(self):
        return {p: float(self.reward_arr[p]) if p < self.n_people else 0.0 for p in self.players()}

    def outcome(self):
        return {p: float(self.outcome_arr[p]) if p < self.n_people else 0.0 for p in self.players()}

    def legal_actions(self, player):
        if player < self.n_people:
            return [a for a in range(9) if self.legal_actions_arr[player] >> a & 1]
        else:
            return []
    
    def observation(self, player=None):
        return self.obs_arr
    
    def net(self):
        return Model()

    def action2str(self, a, player=None):
        return "udlrUDLR."[a]

    def str2action(self, s, player=None):
        return "udlrUDLR.".index(s)


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
