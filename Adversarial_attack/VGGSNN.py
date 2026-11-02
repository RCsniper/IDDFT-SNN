import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import torchvision
from biotorch.layers.usf import Conv2d as BioConv2d, Linear as BioLinear

from snntorch import surrogate

import torch
import torch.nn as nn
from warnings import warn

class Augment_spike_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem_thr, v, bfunc):
        spk = (mem_thr > 0).to(mem_thr.dtype)
        ctx.bfunc = bfunc
        ctx.save_for_backward(v)
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        g = ctx.bfunc(v)
        grad_v = grad_output * g
        return None, grad_v, None

# class Augment_spike_function(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mem_thr, v, bfunc):
#         spk = (mem_thr > 0).to(mem_thr.dtype)
#         ctx.save_for_backward(mem_thr)
#         return spk
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         (mem,) = ctx.saved_tensors
#         g = (
#             2
#             / 2
#             / (1 + (torch.pi / 2 * 2 * mem).pow_(2))
#
#         )
#         grad = grad_output * g
#         return grad, None, None


class Leaky_augment(nn.Module):

    reset_dict = {"subtract": 0, "zero": 1, "none": 2}

    def __init__(
        self,
        beta,
        bfunc,
        threshold=1.0,
        reset_mechanism="subtract",
        reset_delay=True,
        graded_spikes_factor=1.0,
        inhibition=False,
        output=False,
        init_hidden=False,
    ):
        super().__init__()
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.register_buffer("graded_spikes_factor", torch.as_tensor(graded_spikes_factor))

        self.bfunc = bfunc
        self.inhibition = inhibition
        self.output = output
        self.init_hidden = init_hidden
        self.reset_delay = reset_delay

        if reset_mechanism not in self.reset_dict:
            raise ValueError("reset_mechanism must be 'subtract', 'zero', or 'none'")
        self.register_buffer("reset_mechanism_val", torch.as_tensor(self.reset_dict[reset_mechanism]))
        self._reset_mechanism = reset_mechanism


        self.register_buffer("mem", torch.zeros(0), persistent=False)
        if self.reset_mechanism_val == 0:
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:
            self.state_function = self._base_zero
        else:
            self.state_function = self._base_int

        if inhibition:
            warn(
                "Inhibition unstable",
                UserWarning,
            )

    @property
    def reset_mechanism(self):
        return self._reset_mechanism

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def init_leaky(self):
        return self.reset_mem()

    def mem_reset(self, mem):
        mem_shift = mem - self.threshold
        return (mem_shift > 0).to(mem.dtype).detach()

    def _base_state_function(self, input_):
        return self.beta.clamp(0, 1) * self.mem + input_

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def forward(self, input_, mem=None):

        if mem is not None:
            self.mem = mem


        if self.init_hidden and mem is not None:
            raise TypeError("`mem` should not be passed while `init_hidden=True`")


        if self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)


        self.reset = self.mem_reset(self.mem)


        self.mem = self.state_function(input_)


        mem_shift = self.mem - self.threshold
        spk = Augment_spike_function.apply(mem_shift, input_, self.bfunc)
        spk = spk * self.graded_spikes_factor


        if not self.reset_delay:
            do_reset = (spk / self.graded_spikes_factor - self.reset)
            if self.reset_mechanism_val == 0:      # subtract
                self.mem = self.mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:    # zero
                self.mem = self.mem - do_reset * self.mem

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem


class BN1d_SNN(nn.Module):
    def __init__(self, num_features, alpha=1.0, Vth=1.0):
        super().__init__()
        self.alpha, self.Vth = alpha, Vth
        self.tdBN = nn.BatchNorm1d(num_features)
        torch.nn.init.constant_(self.tdBN.weight, alpha * Vth)

    def forward(self, x):
        # x: [B,T,C] -> [B,C,T] -> BN1d -> [B,T,C]
        y = x.transpose(1, 2).contiguous()
        y = self.tdBN(y)
        y = y.transpose(1, 2).contiguous()
        return y


class BN2d_SNN(nn.Module):
    def __init__(self, num_features, alpha=1.0, Vth=1.0):
        super().__init__()
        self.alpha, self.Vth = alpha, Vth
        self.tdBN = nn.BatchNorm3d(num_features)
        torch.nn.init.constant_(self.tdBN.weight, alpha * Vth)

    def forward(self, x):
        # x: [B,T,C,H,W] -> [B,C,T,H,W] -> BN3d -> [B,T,C,H,W]
        y = x.transpose(1, 2).contiguous()
        y = self.tdBN(y)
        y = y.transpose(1, 2).contiguous()
        return y


class tdLayer(nn.Module):

    def __init__(self, layer, bn=None):
        super().__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x: torch.Tensor):
        # x: [B,T,...] -> [B*T,...]
        bt = x.flatten(0, 1).contiguous()
        y = self.layer(bt)
        # y: [B*T,...] -> [B,T,...]
        y = y.view(x.shape[0], x.shape[1], *y.shape[1:])
        if self.bn is not None:
            y = self.bn(y)
        return y

class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x, **kwargs):
        return x * self.scale

class SpikingNeuronWrapper(nn.Module):

    def __init__(self, lif_factory):
        super().__init__()
        self.neuron = lif_factory()

    def _reset(self):

        if hasattr(self.neuron, "reset_mem"):
            self.neuron.reset_mem()
        elif hasattr(self.neuron, "init_leaky"):
            self.neuron.init_leaky()

    def forward(self, x: torch.Tensor):
        self._reset()
        T = x.size(1)
        spikes = []
        mem = None
        for t in range(T):
            xt = x[:, t, ...]
            out = self.neuron(xt, mem) if mem is not None else self.neuron(xt)
            if isinstance(out, tuple):
                spk, mem = out
            else:
                spk, mem = out, None
            spikes.append(spk)
        return torch.stack(spikes, dim=1)

class VGGSNN(nn.Module):
    """
    VGG-SNN structure:
      64C3-LIF
      128C3-LIF-2AP
      256C3-LIF-256C3-LIF-2AP
      512C3-LIF-512C3-LIF-2AP
      512C3-LIF-512C3-LIF-2AP
      zFC

    """

    def __init__(
        self,
        T,
        num_classes=10,
        training_strategy="BP",
        beta=0.8,
        threshold=1.0,
        bfunc=None,
        spike_grad=None,
        in_ch=2,
        input_size=48,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.input_size = input_size

        # ---- conv / linear factory ----
        if training_strategy == "BP":
            conv_factory = nn.Conv2d
            linear_factory = nn.Linear
        else:
            conv_factory = BioConv2d
            linear_factory = BioLinear

        # ---- neuron factory ----
        if training_strategy == "auSF" and bfunc is not None:
            def lif_factory():
                return Leaky_augment(
                    beta=beta,
                    threshold=threshold,
                    bfunc=bfunc
                )
        else:
            def lif_factory():
                return snn.Leaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad
                )

        self._conv_factory = conv_factory
        self._linear_factory = linear_factory
        self._lif_factory = lif_factory

        # -------------------------
        # Stage 1: 64C3-LIF
        # -------------------------
        self.stage1 = nn.Sequential(
            tdLayer(
                conv_factory(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(64)
            ),
            SpikingNeuronWrapper(lif_factory),
        )

        # -------------------------
        # Stage 2: 128C3-LIF-2AP
        # -------------------------
        self.stage2 = nn.Sequential(
            tdLayer(
                conv_factory(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(128)
            ),
            SpikingNeuronWrapper(lif_factory),
            tdLayer(nn.AvgPool2d(kernel_size=2, stride=2)),
        )

        # -------------------------
        # Stage 3: 256C3-LIF-256C3-LIF-2AP
        # -------------------------
        self.stage3 = nn.Sequential(
            tdLayer(
                conv_factory(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(256)
            ),
            SpikingNeuronWrapper(lif_factory),

            tdLayer(
                conv_factory(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(256)
            ),
            SpikingNeuronWrapper(lif_factory),

            tdLayer(nn.AvgPool2d(kernel_size=2, stride=2)),
        )

        # -------------------------
        # Stage 4: 512C3-LIF-512C3-LIF-2AP
        # -------------------------
        self.stage4 = nn.Sequential(
            tdLayer(
                conv_factory(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(512)
            ),
            SpikingNeuronWrapper(lif_factory),

            tdLayer(
                conv_factory(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(512)
            ),
            SpikingNeuronWrapper(lif_factory),

            tdLayer(nn.AvgPool2d(kernel_size=2, stride=2)),
        )

        # -------------------------
        # Stage 5: 512C3-LIF-512C3-LIF-2AP
        # -------------------------
        self.stage5 = nn.Sequential(
            tdLayer(
                conv_factory(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(512)
            ),
            SpikingNeuronWrapper(lif_factory),

            tdLayer(
                conv_factory(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                BN2d_SNN(512)
            ),
            SpikingNeuronWrapper(lif_factory),

            #tdLayer(nn.AvgPool2d(kernel_size=2, stride=2)),
        )

        # 48 -> 24 -> 12 -> 6 -> 3
        feat_size = input_size // 16
        if feat_size < 1:
            raise ValueError(f"input_size={input_size} is too small for 4 pooling stages")

        self.feat_size = feat_size

        # -------------------------
        # zFC
        # -------------------------
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((3, 3)))
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.25),
            tdLayer(linear_factory(512 * 3 * 3, 1024, bias=True)),
            SpikingNeuronWrapper(lif_factory),
            nn.Dropout(p=0.25),

            # nn.Dropout(p=0.25),
            tdLayer(nn.Linear(1024, 1024, bias=True)),
            SpikingNeuronWrapper(lif_factory),
            nn.Dropout(p=0.25),
            tdLayer(linear_factory(1024, self.num_classes, bias=True)),
        )

        # self.fc = nn.Sequential(
        #     SpikingNeuronWrapper(lif_factory),
        #     tdLayer(
        #     linear_factory(512 * self.feat_size * self.feat_size, num_classes, bias=True)
        # )
        # )

        self._initialize_weights()

    def forward(self, inputs):

        if inputs.dim() == 4:
            x = inputs.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        elif inputs.dim() == 5:
            x = inputs
        else:
            raise ValueError(f"Expected input dim 4 or 5, but got shape {inputs.shape}")

        x = x.float()

        x = self.stage1(x)   # [B,T,64,H,W]
        x = self.stage2(x)   # [B,T,128,H/2,W/2]
        x = self.stage3(x)   # [B,T,256,H/4,W/4]
        x = self.stage4(x)   # [B,T,512,H/8,W/8]
        x = self.stage5(x)   # [B,T,512,H/16,W/16]

        # x = x.view(x.shape[0], x.shape[1], -1)   # [B,T,512*feat*feat]
        # x = self.fc(x)                           # [B,T,num_classes]

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc(x)

        return x

    def forward_logits(self, inputs, reduce="sum"):
        logits_T = self.forward(inputs)  # [B,T,C]

        if reduce is None:
            return logits_T
        if reduce == "sum":
            return logits_T.sum(dim=1)
        if reduce == "mean":
            return logits_T.mean(dim=1)
        if reduce == "last":
            return logits_T[:, -1, :]

        raise ValueError(f"Unknown reduce={reduce}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)