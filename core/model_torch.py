# Copyright (C) 2023  willlllllio
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License version 3
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch.nn import Conv2d, ModuleList, LeakyReLU


class DeviceMixin():
	@property
	def device(self: nn.Module) -> device:
		return next(self.parameters()).device


class ConvLin(nn.Module):
	def __init__(self,
				 xin = 1024,
				 xout = 1024,
				 src_in = 512,
				 act: bool = False,
				 ):
		super().__init__()

		# assert src_out // 2 == xout
		self.conv = nn.Conv2d(xin, xout, kernel_size = (3, 3), stride = (1, 1))
		self.lin = nn.Linear(in_features = src_in, out_features = (xout * 2), bias = True)
		self.act = act
		self.xout = xout

	def forward(self, x, source):
		x = F.pad(x, (1, 1, 1, 1), "reflect", 0)
		x = self.conv(x)

		meaned = x - x.mean((2, 3), keepdim = True)
		x = meaned.pow(2)

		x = x.mean((2, 3), keepdim = True)
		x = 1.0 / (x + 1.0000e-08).sqrt()
		x = meaned * x

		source = self.lin(source)
		source = source.unsqueeze(dim = 2).unsqueeze(dim = 2)
		src1 = source[:, :self.xout, :, :]
		src2 = source[:, self.xout:, :, :]

		x = src1 * x + src2
		if self.act:
			x = F.relu(x)

		return (x)


class ConvLinBlock(torch.nn.Module):
	def __init__(self,
				 layers = 2,
				 xin = 1024,
				 xout = 1024,
				 src_in = 512,
				 ):
		super().__init__()

		self.layers = nn.ModuleList(
			[ConvLin(xin, xout, src_in, i + 1 != layers) for i in range(layers)]
		)

	def forward(self, target, source):
		x = target
		for layer in self.layers:
			x = layer(x, source)
		return target + x


class FaceSwapper(DeviceMixin, torch.nn.Module):
	def __init__(
			self,
			src_channels = 3,
			src_size = 512,

			down_convs = (
					(128, 7, 1, 0),  # out_chan, kernel_size, stride, pad
					(256, 3, 1, 1),
					(512, 3, 2, 1),
					(1024, 3, 2, 1),
			),

			mid_blocks = 6,
			mid_layers = 2,
			mid_in = 1024,
			mid_out = 1024,

			up_convs = (
					(2.0, 512, 3, 1, 1),  # scale, out_chan, kernel_size, stride, pad
					(2.0, 256, 3, 1, 1),
					(None, 128, 3, 1, 1),
			),
			out_conv = (3, 7, 1, 3),  # out_channels, kernel, stride, pad

			up_act = lambda: LeakyReLU(negative_slope = 0.2),
			down_act = lambda: LeakyReLU(negative_slope = 0.2),

	):
		super().__init__()

		down_in = [src_channels] + [i[0] for i in down_convs[:-1]]
		self.down_convs = ModuleList([
			Conv2d(down_in[i], out, kernel_size = (k, k), stride = (s, s), padding = ((p, p) if p else 0))
			for i, (out, k, s, p) in enumerate(down_convs)
		])
		self.down_act = down_act()

		self.blocks = ModuleList(
			[ConvLinBlock(mid_layers, mid_in, mid_out, src_size) for _ in range(mid_blocks)]
		)

		up_in = [mid_out] + [i[1] for i in up_convs[:-1]]
		self.up_scales = [i[0] for i in up_convs]
		self.up_convs = ModuleList([
			Conv2d(up_in[i], out, kernel_size = (k, k), stride = (s, s), padding = ((p, p) if p else 0))
			for i, (_scale, out, k, s, p) in enumerate(up_convs)
		])
		self.up_act = up_act()

		c, k, s, p = out_conv
		self.out_pad = p
		self.out_conv = Conv2d(up_convs[-1][1], out_conv[0], kernel_size = (k, k), stride = (s, s))

		self.register_buffer('l_init', torch.zeros((src_size, src_size)))

	def forward(self, target, source, return_hidden = False):
		target = F.pad(target, (3, 3, 3, 3), "reflect", 0)

		for conv in self.down_convs:
			target = conv(target)
			target = self.up_act(target)

		for block in self.blocks:
			target = block(target, source)

		if return_hidden:
			return target

		for conv, scale in zip(self.up_convs, self.up_scales):
			if scale is not None:
				target = F.interpolate(target, scale_factor = (scale, scale), mode = "bilinear", align_corners = False)
			target = conv(target)
			target = self.down_act(target)

		if self.out_pad is not None:
			target = F.pad(target, (self.out_pad, self.out_pad, self.out_pad, self.out_pad), "reflect", 0)
		target = self.out_conv(target)
		target = F.tanh(target)

		return (target + 1) / 2


def state_dict_onnx_to_torch(sd, l_init = None):
	# convert default onnx 128 model keys to torch model
	sd = { k: v for k, v in sd.items() if not k.startswith("initializers.") }

	for base, nums in (
			("down_convs.", (40, 42, 44, 46)),
			("up_convs.", (590, 594, 596,)),
	):
		for pos, i in enumerate(nums):
			for name in (".weight", ".bias"):
				sd[f"{base}{(pos if 'out' in base else pos)}{name}"] = sd.pop(f"Conv_{i}{name}")

	for name in (".weight", ".bias"):
		sd[f"out_conv{name}"] = sd.pop(f"Conv_612{name}")

	for bpos, block in enumerate([
		(("Conv_62", "Gemm_73",), ("Conv_107", "Gemm_118")),
		(("Conv_152", "Gemm_163",), ("Conv_197", "Gemm_208")),
		(("Conv_242", "Gemm_253",), ("Conv_287", "Gemm_298")),
		(("Conv_332", "Gemm_343",), ("Conv_377", "Gemm_388")),
		(("Conv_422", "Gemm_433",), ("Conv_467", "Gemm_478")),
		(("Conv_512", "Gemm_523",), ("Conv_557", "Gemm_568")),
	]):
		for layerpos, l in enumerate(block):
			for name, newname in zip(l, ("conv", "lin")):
				for fix in (".weight", ".bias"):
					sd[f"blocks.{bpos}.layers.{layerpos}.{newname}{fix}"] = sd.pop(f"{name}{fix}")

	if l_init is not None:
		sd["l_init"] = l_init

	return sd
