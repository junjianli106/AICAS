import os.path
import time
import numpy as np
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch import einsum

import itertools

def print_time_delta(desc, start_ts):
    if start_ts is not None:
        now = time.time()
        delta = now - start_ts[0]
        print(desc, '%.3f'%delta)
        
def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data

class LabelEncoder():
    def __init__(self):
        pass

    def fit(self, vals, inc=False):
        if inc:
            old_sz = len(self.map)
            new_map     = dict((i, v) for i, v in enumerate(sorted(set([_ for _ in vals if _ not in self.map]))))
            new_inv_map = dict((v, i) for i, v in enumerate(sorted(set([_ for _ in vals if _ not in self.map]))))
            self.map.update(new_map)
            self.inv_map.update(new_inv_map)
            print(f'old_sz shape:{old_sz}, new_map shape:{len(new_map)}')
        else:
            self.map     = dict((i, v) for i, v in enumerate(sorted(set([_ for _ in vals]))))
            self.inv_map = dict((v, i) for i, v in enumerate(sorted(set([_ for _ in vals]))))

        print('LabelEncoder fit done')

    def transform(self, vals):
        return [self.map[val] for val in vals]

    def inv_transform(self, keys):
        return [self.inv_map[key] for key in keys]

class FeatureEncoder():
    def __init__(self, use_cols):
        self.UsedEncoder = {}
        self.use_cols = use_cols

    def fit(self, df):
        for col in self.use_cols:
            print(f'Label Encoder for feature:{col}')
            self.UsedEncoder[col] = LabelEncoder().fit(df[col].values)
        result_path = './data/result/'
        if not os.path.isdir(result_path):
            os.makedirs(result_path, exist_ok=True)
        save_pickle('./data/result/UsedEncoder.pkl')

        return self.UsedEncoder

    def load(self, UsedEncoder_path):
        if os.path.isfile(UsedEncoder_path):
            self.UsedEncoder = load_pickle(UsedEncoder_path)
        return self.UsedEncoder

class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]

    Output shape:
        - Same shape as input.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer



class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=False, layer_norm=True, use_activation=True,
                 dropout_rate=0.3, activation='relu', sigmoid=False):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_size))
        if use_activation:
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            elif activation.lower() == 'none':
                pass
            else:
                raise NotImplementedError
            layers.append(nn.Dropout(p=dropout_rate))
        if layer_norm:
            layers.append(nn.LayerNorm(input_size))

        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            elif activation.lower() == 'none':
                pass
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))

            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)

        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0001)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class SENETLayer(nn.Module):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, reduction_ratio=3):
        super(SENETLayer, self).__init__()
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))

        return V


class BilinearInteraction(nn.Module):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2, embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **embedding_size** : Positive integer, embedding size of sparse features.
        - **bilinear_type** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, embedding_size, bilinear_type="interaction"):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == "each":
            for _ in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == "interaction":
            for _, _ in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)


class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function name used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = nn.ReLU(inplace=True)

        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)

        return result


class CrossNetMix(nn.Module):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **device**: str, e.g. ``"cpu"`` or ``"cuda:0"``
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, low_rank=64, num_experts=8, layer_num=2):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # V: (in_features, low_rank)
        self.V_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # C: (low_rank, low_rank)
        self.C_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, low_rank, low_rank))
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        init_para_list = [self.U_list, self.V_list, self.C_list]
        for para in init_para_list:
            for i in range(self.layer_num):
                nn.init.xavier_normal_(para[i])

        for i in range(len(self.bias)):
            nn.init.zeros_(self.bias[i])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term