# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# training

import os
import time
import copy
import threading
import random
import bz2
import pickle
import warnings
import itertools
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import psutil

from .environment import prepare_env, make_env
from .util import map_r, bimap_r, trimap_r, rotate
from .model import to_torch, to_gpu, ModelWrapper
from .losses import compute_target
from .connection import MultiProcessJobExecutor
from .worker import WorkerCluster, WorkerServer


def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes / Batcher.select_episode で作成したもので、中身の moment は Generator が作成したもの
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (B, T, P, ...) .
        (B is batch size, T is time length, P is player count)
    """

    obss, datum = [], []

    def replace_none(a, b):
        return a if a is not None else b

    for ep in episodes:
        # ep["moment"] は、step で分割してちょっとずつ compress したものになってる (なんで？) ので、戻す
        #moments_ = sum([pickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])  # O(N^2)
        moments_ = list(itertools.chain.from_iterable([pickle.loads(bz2.decompress(ms)) for ms in ep['moment']]))
        moments = moments_[ep['start'] - ep['base']:ep['end'] - ep['base']]
        players = list(moments[0]['observation'].keys())
        if not args['turn_based_training']:  # solo training
            players = [random.choice(players)]
        
        obs_zeros = map_r(moments[0]['observation'][moments[0]['turn'][0]], lambda o: np.zeros_like(o))  # template for padding
        amask_zeros = np.zeros_like(moments[0]['action_mask'][moments[0]['turn'][0]])  # template for padding

        # moments は moments[ステップ][属性][プレイヤー ID] みたいになっていて、ターン制の場合はプレイヤー ID の部分が冗長なので、外す
        # 各変数は hoge[ステップ][プレイヤー][行動] の形式になる
        if args['turn_based_training'] and not args['observation']:
            obs = [[m['observation'][m['turn'][0]]] for m in moments]  # m: moments の 1 ステップ
            prob = np.array([[[m['selected_prob'][m['turn'][0]]]] for m in moments])
            act = np.array([[m['action'][m['turn'][0]]] for m in moments], dtype=np.int64)[..., np.newaxis]  # [T, P, 1]  # newaxis を入れるのは tmask に合わせるため？
            amask = np.array([[m['action_mask'][m['turn'][0]]] for m in moments])
        else:
            obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
            prob = np.array([[[replace_none(m['selected_prob'][player], 1.0)] for player in players] for m in moments])
            act = np.array([[replace_none(m['action'][player], 0) for player in players] for m in moments], dtype=np.int64)[..., np.newaxis]
            amask = np.array([[replace_none(m['action_mask'][player], amask_zeros + 1e32) for player in players] for m in moments])


        # reshape observation
        # rotate が回転させるものはあくまで tuple, list, dict で、np.ndarray は回転させないことに注意
        obs = rotate(rotate(obs))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))

        # datum that is not changed by training configuration
        v = np.array([[replace_none(m['value'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        rew = np.array([[replace_none(m['reward'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        ret = np.array([[replace_none(m['return'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        oc = np.array([ep['outcome'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)

        emask = np.ones((len(moments), 1, 1), dtype=np.float32)  # episode mask
        tmask = np.array([[[m['selected_prob'][player] is not None] for player in players] for m in moments], dtype=np.float32)  # [T, P, 1]
        omask = np.array([[[m['observation'][player] is not None] for player in players] for m in moments], dtype=np.float32)

        progress = np.arange(ep['start'], ep['end'], dtype=np.float32)[..., np.newaxis] / ep['total']

        # pad each array if step length is short
        batch_steps = args['forward_steps']
        if len(tmask) < batch_steps:
            pad_len_b = - (ep['train_start'] - ep['start'])
            pad_len_a = batch_steps - len(tmask) - pad_len_b
            # np.pad はやたら重い
            def pad(array, value):
                first_dim = array.shape[0]
                result_array = np.full((pad_len_b + first_dim + pad_len_a,) + array.shape[1:], value, dtype=array.dtype)
                result_array[pad_len_b:pad_len_b + first_dim] = array
                return result_array
            obs = map_r(obs, lambda o: pad(o, 0))
            prob = pad(prob, 1)
            v = np.concatenate([np.pad(v, [(pad_len_b, 0), (0, 0), (0, 0)], 'constant', constant_values=0), np.tile(oc, [pad_len_a, 1, 1])])
            act = pad(act, 0)
            rew = pad(rew, 0)
            ret = pad(ret, 0)
            emask = pad(emask, 0)
            tmask = pad(tmask, 0)
            omask = pad(omask, 0)
            amask = pad(amask, 1e32)
            progress = pad(progress, 1)

        obss.append(obs)
        datum.append((prob, v, act, oc, rew, ret, emask, tmask, omask, amask, progress))

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    prob, v, act, oc, rew, ret, emask, tmask, omask, amask, progress = [to_torch(np.array(val)) for val in zip(*datum)]
    #print("awsedrftgyhujikolp;", v.shape, omask.shape)

    return {
        'observation': obs,
        'selected_prob': prob, 'value': v,
        'action': act, 'outcome': oc,
        'reward': rew, 'return': ret,
        'episode_mask': emask,
        'turn_mask': tmask, 'observation_mask': omask,
        'action_mask': amask,
        'progress': progress,
    }


def forward_prediction(model, hidden, batch, args):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: batch outputs of neural network
    """

    observations = batch['observation']  # (..., B, T, P or 1, ...)
    batch_shape = batch['action'].size()[:3]  # (B, T, P or 1)

    # feed-forward neural network
    obs = map_r(observations, lambda o: o.flatten(0, 2))  # (..., B * T * P or 1, ...)
    outputs = model(obs, None)  # type:dict
    outputs = map_r(outputs, lambda o: o.unflatten(0, batch_shape))  # (..., B, T, P or 1, ...)

    for k, o in outputs.items():
        if k == 'policy':
            o = o.mul(batch['turn_mask'])  # o: [B, T, P or 1, n_actions], turn_mask: [B, T, P or 1, 1]
            if o.size(2) > 1 and batch_shape[2] == 1:  # turn-alternating batch  # そうなるか？？？
                o = o.sum(2, keepdim=True)  # gather turn player's policies
            outputs[k] = o - batch['action_mask']
        else:
            # mask valid target values and cumulative rewards
            #print(k, outputs[k].shape, o.shape, batch['observation_mask'].shape)
            outputs[k] = o.mul(batch['observation_mask'])

    return outputs


def compose_losses(outputs, log_selected_policies, total_advantages, targets, batch, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    tmasks = batch['turn_mask']
    omasks = batch['observation_mask']

    losses = {}
    dcnt = tmasks.sum().item()
    turn_advantages = total_advantages.mul(tmasks).sum(2, keepdim=True)

    losses['p'] = (-log_selected_policies * turn_advantages).sum()
    # value で rewards は考慮されない
    if 'value' in outputs:
        losses['v'] = ((outputs['value'] - targets['value']) ** 2).mul(omasks).sum() / 2
    # return ... 将来の reward の合計みたいなやつ、outcome は考慮されない
    if 'return' in outputs:
        losses['r'] = F.smooth_l1_loss(outputs['return'], targets['return'], reduction='none').mul(omasks).sum()

    entropy = dist.Categorical(logits=outputs['policy']).entropy().mul(tmasks.sum(-1))
    losses['ent'] = entropy.sum()

    base_loss = losses['p'] + losses.get('v', 0) + losses.get('r', 0)
    entropy_loss = entropy.mul(1 - batch['progress'] * (1 - args['entropy_regularization_decay'])).sum() * -args['entropy_regularization']
    losses['total'] = base_loss + entropy_loss

    return losses, dcnt


def compute_loss(batch, model, hidden, args):
    outputs = forward_prediction(model, hidden, batch, args)
    actions = batch['action']
    emasks = batch['episode_mask']
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    log_selected_b_policies = torch.log(torch.clamp(batch['selected_prob'], 1e-16, 1)) * emasks

    log_selected_t_policies = F.log_softmax(outputs['policy'], dim=-1).gather(-1, actions) * emasks

    # 重要度サンプリングの閾値 / thresholds of importance sampling
    log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
    cs = torch.clamp(rhos, 0, clip_c_threshold)
    outputs_nograd = {k: o.detach() for k, o in outputs.items()}

    if 'value' in outputs_nograd:
        values_nograd = outputs_nograd['value']
        if args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
        outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

    # targets と advantages を計算 / compute targets and advantage
    # compute_target ... 将来に渡る values と rewards を gamma と lambda で減衰させながら計算 (最後のターンは returns)
    targets = {}
    advantages = {}
    # これは最後あたりは outcome の予測、途中までは将来の自分の予測 value の予測になる
    # outcome が 0 で value の予測値が無ければ policy の計算に影響しない？
    targets['value'], advantages['value'] = compute_target(
        args['value_target'],         # algorithm
        outputs_nograd.get('value'),  # values
        batch['outcome'],             # returns
        None,                         # rewards
        args['lambda'],               # lmb
        1,                            # gamma
        clipped_rhos,                 # rhos
        cs,                           # cs
    )
    # これは最後あたりは batch["return"] の予測、途中までは rewards と将来の自分の予測 return の予測になる
    # batch["return"] は gamma で減衰する将来の rewards、outputs["return"] はそのターンの自分の出力なので注意
    # env に reward が無くてモデルの出力にも return が無ければ 0 になるはずで、policy の計算に影響しなくなる
    targets['return'], advantages['return'] = compute_target(
        args['value_target'],          # algorithm
        outputs_nograd.get('return'),  # values   # ここ違う
        batch['return'],               # returns  # ここ違う
        batch['reward'],               # rewards  # ここ違う
        args['lambda'],                # lmb
        args['gamma'],                 # gamma    # ここ違う
        clipped_rhos,                  # rhos
        cs,                            # cs
    )

    # policy advantage を計算 / compute policy advantage
    # advantages['value'] + advantages['return']
    total_advantages = clipped_rhos * sum(advantages.values())

    return compose_losses(outputs, log_selected_t_policies, total_advantages, targets, batch, args)


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.shutdown_flag = False

        self.executor = MultiProcessJobExecutor(self._worker, self._selector(), self.args['num_batchers'], num_receivers=2)

    def _selector(self):
        while True:
            yield [self.select_episode() for _ in range(self.args['batch_size'])]

    def _worker(self, conn, bid):
        print('started batcher %d' % bid)
        while not self.shutdown_flag:
            episodes = conn.recv()
            batch = make_batch(episodes, self.args)
            conn.send(batch)
        print('finished batcher %d' % bid)

    def run(self):
        self.executor.start()

    def select_episode(self):
        while True:
            ep_idx = random.randrange(min(len(self.episodes), self.args['maximum_episodes']))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                break
        ep = self.episodes[ep_idx]
        turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        train_st = random.randrange(turn_candidates)
        st = train_st
        ed = min(train_st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        ep_minimum = {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'start': st, 'end': ed, 'train_start': train_st, 'total': ep['steps'],
        }
        return ep_minimum

    def batch(self):
        return self.executor.recv()

    def shutdown(self):
        self.shutdown_flag = True
        self.executor.shutdown()


class Trainer:
    def __init__(self, args, model):
        self.episodes = deque()
        self.args = args
        self.gpu = torch.cuda.device_count()
        self.model = model
        self.default_lr = 3e-8
        self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
        self.params = list(self.model.parameters())
        lr = self.default_lr * self.data_cnt_ema
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        self.lock = threading.Lock()
        self.batcher = Batcher(self.args, self.episodes)
        self.updated_model = None, 0
        self.update_flag = False
        self.shutdown_flag = False

        self.wrapped_model = ModelWrapper(self.model)
        self.trained_model = self.wrapped_model
        if self.gpu > 1:
            self.trained_model = nn.DataParallel(self.wrapped_model)

    def update(self):
        if len(self.episodes) < self.args['minimum_episodes']:
            return None, 0  # return None before training
        self.update_flag = True
        while True:
            time.sleep(0.1)
            model, steps = self.recheck_update()
            if model is not None:
                break
        return model, steps

    def report_update(self, model, steps):
        self.lock.acquire()
        self.update_flag = False
        self.updated_model = model, steps
        self.lock.release()

    def recheck_update(self):
        self.lock.acquire()
        flag = self.update_flag
        self.lock.release()
        return (None, -1) if flag else self.updated_model

    def shutdown(self):
        self.shutdown_flag = True
        self.batcher.shutdown()

    def train(self):
        if self.optimizer is None:  # non-parametric model
            time.sleep(0.1)
            return self.model

        batch_cnt, data_cnt, loss_sum = 0, 0, {}
        if self.gpu > 0:
            self.trained_model.cuda()
        self.trained_model.train()

        while data_cnt == 0 or not (self.update_flag or self.shutdown_flag):
            batch = self.batcher.batch()
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = self.wrapped_model.init_hidden([batch_size, player_count])
            if self.gpu > 0:
                batch = to_gpu(batch)
                hidden = to_gpu(hidden)

            losses, dcnt = compute_loss(batch, self.trained_model, hidden, self.args)

            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.step()

            batch_cnt += 1
            data_cnt += dcnt
            for k, l in losses.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + l.item()

            self.steps += 1

        print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))

        self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.default_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        self.model.cpu()
        self.model.eval()
        return copy.deepcopy(self.model)

    def run(self):
        print('waiting training')
        while not self.shutdown_flag:
            if len(self.episodes) < self.args['minimum_episodes']:
                time.sleep(1)
                continue
            if self.steps == 0 and self.optimizer is not None:
                self.batcher.run()
                print('started training')
            model = self.train()
            self.report_update(model, self.steps)
        print('finished training')


class Learner:
    def __init__(self, args, net=None, remote=False):
        train_args = args['train_args']
        env_args = args['env_args']
        train_args['env'] = env_args
        args = train_args

        self.args = args
        random.seed(args['seed'])

        self.env = make_env(env_args)
        eval_modify_rate = (args['update_episodes'] ** 0.85) / args['update_episodes']
        self.eval_rate = max(args['eval_rate'], eval_modify_rate)
        self.shutdown_flag = False
        self.flags = set()

        # trained datum
        self.model_epoch = self.args['restart_epoch']
        self.model = net if net is not None else self.env.net()
        if self.model_epoch > 0:
            self.model.load_state_dict(torch.load(self.model_path(self.model_epoch)), strict=False)

        # generated datum
        self.generation_results = {}
        self.num_episodes = 0

        # evaluated datum
        self.results = {}
        self.results_per_opponent = {}
        self.num_results = 0

        # multiprocess or remote connection
        self.worker = WorkerServer(args) if remote else WorkerCluster(args)

        # thread connection
        self.trainer = Trainer(args, self.model)

    def shutdown(self):
        self.shutdown_flag = True
        self.trainer.shutdown()
        self.worker.shutdown()
        self.thread.join()

    def model_path(self, model_id):
        return os.path.join('models', str(model_id) + '.pth')

    def latest_model_path(self):
        return os.path.join('models', 'latest.pth')

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_epoch += 1
        self.model = model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), self.model_path(self.model_epoch))
        torch.save(model.state_dict(), self.latest_model_path())

    def feed_episodes(self, episodes):
        # analyze generated episodes
        for episode in episodes:
            if episode is None:
                continue
            for p in episode['args']['player']:
                model_id = episode['args']['model_id'][p]
                outcome = episode['outcome'][p]
                n, r, r2 = self.generation_results.get(model_id, (0, 0, 0))
                self.generation_results[model_id] = n + 1, r + outcome, r2 + outcome ** 2

        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])

        mem_percent = psutil.virtual_memory().percent
        mem_ok = mem_percent <= 95
        maximum_episodes = self.args['maximum_episodes'] if mem_ok else int(len(self.trainer.episodes) * 95 / mem_percent)

        if not mem_ok and 'memory_over' not in self.flags:
            warnings.warn("memory usage %.1f%% with buffer size %d" % (mem_percent, len(self.trainer.episodes)))
            self.flags.add('memory_over')

        while len(self.trainer.episodes) > maximum_episodes:
            self.trainer.episodes.popleft()

    def feed_results(self, results):
        # store evaluation results
        for result in results:
            if result is None:
                continue
            for p in result['args']['player']:
                model_id = result['args']['model_id'][p]
                res = result['result'][p]
                n, r, r2 = self.results.get(model_id, (0, 0, 0))
                self.results[model_id] = n + 1, r + res, r2 + res ** 2

                if model_id not in self.results_per_opponent:
                    self.results_per_opponent[model_id] = {}
                opponent = result['opponent']
                n, r, r2 = self.results_per_opponent[model_id].get(opponent, (0, 0, 0))
                self.results_per_opponent[model_id][opponent] = n + 1, r + res, r2 + res ** 2

    def update(self):
        # call update to every component
        print()
        print('epoch %d' % self.model_epoch)

        if self.model_epoch not in self.results:
            print('win rate = Nan (0)')
        else:
            def output_wp(name, results):
                n, r, r2 = results
                mean = r / (n + 1e-6)
                name_tag = ' (%s)' % name if name != '' else ''
                print('win rate%s = %.3f (%.1f / %d)' % (name_tag, (mean + 1) / 2, (r + n) / 2, n))

            if len(self.args.get('eval', {}).get('opponent', [])) <= 1:
                output_wp('', self.results[self.model_epoch])
            else:
                output_wp('total', self.results[self.model_epoch])
                for key in sorted(list(self.results_per_opponent[self.model_epoch])):
                    output_wp(key, self.results_per_opponent[self.model_epoch][key])

        if self.model_epoch not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_epoch]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))

        model, steps = self.trainer.update()
        if model is None:
            model = self.model
        self.update_model(model, steps)

        # clear flags
        self.flags = set()

    def server(self):
        # central conductor server
        # returns as list if getting multiple requests as list
        print('started server')
        prev_update_episodes = self.args['minimum_episodes']
        while self.model_epoch < self.args['epochs'] or self.args['epochs'] < 0:
            # no update call before storing minimum number of episodes + 1 age
            next_update_episodes = prev_update_episodes + self.args['update_episodes']
            while not self.shutdown_flag and self.num_episodes < next_update_episodes:
                conn, (req, data) = self.worker.recv()
                multi_req = isinstance(data, list)
                if not multi_req:
                    data = [data]
                send_data = []

                if req == 'args':
                    for _ in data:
                        args = {'model_id': {}}

                        # decide role
                        if self.num_results < self.eval_rate * self.num_episodes:
                            args['role'] = 'e'
                        else:
                            args['role'] = 'g'

                        if args['role'] == 'g':
                            # genatation configuration
                            args['player'] = self.env.players()
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_epoch
                                else:
                                    args['model_id'][p] = -1
                            self.num_episodes += 1
                            if self.num_episodes % 100 == 0:
                                print(self.num_episodes, end=' ', flush=True)

                        elif args['role'] == 'e':
                            # evaluation configuration
                            args['player'] = [self.env.players()[self.num_results % len(self.env.players())]]
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_epoch
                                else:
                                    args['model_id'][p] = -1
                            self.num_results += 1

                        send_data.append(args)

                elif req == 'episode':
                    # report generated episodes
                    self.feed_episodes(data)
                    send_data = [None] * len(data)

                elif req == 'result':
                    # report evaluation results
                    self.feed_results(data)
                    send_data = [None] * len(data)

                elif req == 'model':
                    for model_id in data:
                        model = self.model
                        if model_id != self.model_epoch and model_id > 0:
                            try:
                                model = copy.deepcopy(self.model)
                                model.load_state_dict(torch.load(self.model_path(model_id)), strict=False)
                            except:
                                # return latest model if failed to load specified model
                                pass
                        send_data.append(pickle.dumps(model))

                if not multi_req and len(send_data) == 1:
                    send_data = send_data[0]
                self.worker.send(conn, send_data)
            prev_update_episodes = next_update_episodes
            self.update()
        print('finished server')

    def run(self):
        try:
            # open training thread
            self.thread = threading.Thread(target=self.trainer.run)
            self.thread.start()
            # open generator, evaluator
            self.worker.run()
            self.server()

        finally:
            self.shutdown()


def train_main(args):
    prepare_env(args['env_args'])  # preparing environment is needed in stand-alone mode
    learner = Learner(args=args)
    learner.run()


def train_server_main(args):
    learner = Learner(args=args, remote=True)
    learner.run()
