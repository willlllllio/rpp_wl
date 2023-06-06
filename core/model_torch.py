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
		src_out = xout * 2
		self.conv = nn.Conv2d(xin, xout, kernel_size = (3, 3), stride = (1, 1))
		self.lin = nn.Linear(in_features = src_in, out_features = src_out, bias = True)
		self.act = act

	def forward(self, x, source):
		x = F.pad(x, (1, 1, 1, 1), "reflect", 0)
		x = self.conv(x)

		x_mean = x.mean((2, 3), keepdim = True)
		meaned = x - x_mean
		x = meaned.pow(2)

		x = x.mean((2, 3), keepdim = True)
		x = 1.0 / (x + 1.0000e-08).sqrt()
		x = meaned * x

		source = self.lin(source)
		source = source.unsqueeze(dim = 2).unsqueeze(dim = 2)
		src1 = source[:, :1024, :, :]
		src2 = source[:, 1024:, :, :]

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
	def __init__(self, blocks = 6):
		super().__init__()
		self.in_act = LeakyReLU(negative_slope = 0.2)
		self.out_act = LeakyReLU(negative_slope = 0.2)

		self.conv_in_1 = Conv2d(3, 128, kernel_size = (7, 7), stride = (1, 1))
		self.conv_in_2 = Conv2d(128, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
		self.conv_in_3 = Conv2d(256, 512, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))
		self.conv_in_4 = Conv2d(512, 1024, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))

		self.blocks = ModuleList(
			[ConvLinBlock() for _ in range(blocks)]
		)

		self.conv_out_1 = Conv2d(1024, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
		self.conv_out_2 = Conv2d(512, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
		self.conv_out_3 = Conv2d(256, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
		self.conv_out_4 = Conv2d(128, 3, kernel_size = (7, 7), stride = (1, 1))

	def forward(self, target, source):
		target = F.pad(target, (3, 3, 3, 3), "reflect", 0)
		target = self.conv_in_1(target)
		target = self.in_act(target)
		target = self.conv_in_2(target)
		target = self.in_act(target)
		target = self.conv_in_3(target)
		target = self.in_act(target)
		target = self.conv_in_4(target)
		target = self.in_act(target)

		for block in self.blocks:
			target = block(target, source)

		target = F.interpolate(target, scale_factor = (2.0, 2.0), mode = "bilinear", align_corners = False)
		target = self.conv_out_1(target)
		target = self.out_act(target)

		target = F.interpolate(target, scale_factor = (2.0, 2.0), mode = "bilinear", align_corners = False)
		target = self.conv_out_2(target)
		target = self.out_act(target)

		target = self.conv_out_3(target)
		target = self.out_act(target)
		target = F.pad(target, (3, 3, 3, 3), "reflect", 0)

		target = self.conv_out_4(target)
		target = F.tanh(target)

		return (target + 1) / 2


def state_dict_onnx_to_torch(sd):
	sd = { k: v for k, v in sd.items() if not k.startswith("initializers.") }

	for base, nums in (
			("conv_in_", (40, 42, 44, 46)),
			("conv_out_", (590, 594, 596, 612)),
	):
		for pos, i in enumerate(nums):
			for name in (".weight", ".bias"):
				sd[f"{base}{pos + 1}{name}"] = sd.pop(f"Conv_{i}{name}")

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

	return sd
