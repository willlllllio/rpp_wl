from __future__ import annotations

import os.path

import torch

from core.model_torch import state_dict_onnx_to_torch, FaceSwapper
from core.utils import Timer

from onnx2torch import convert
from onnx import numpy_helper
import onnx


def convert_swapper(onnx_model_path: str, ckpt_path: str | None, safetensors_path: str | None):
	onnx_model_path = str(onnx_model_path)
	with Timer("converting took {} secs"):
		torch_model_1 = convert(onnx_model_path)

	onnx_model = onnx.load(onnx_model_path)

	emap = torch.from_numpy(numpy_helper.to_array((onnx_model.graph.initializer[-1])))
	sd = state_dict_onnx_to_torch(torch_model_1.state_dict(), l_init = emap)

	fs = FaceSwapper()
	fs.load_state_dict(sd)

	fsd = fs.state_dict()
	if (
			False
			or True
	):

		if ckpt_path and not os.path.exists(ckpt_path):
			torch.save(sd, ckpt_path)

		if safetensors_path and not os.path.exists(safetensors_path):
			import safetensors.torch
			safetensors.torch.save_file(sd, safetensors_path)
	print("conv done")


if __name__ == '__main__':
	convert_swapper(
		"inswapper_128.onnx",
		ckpt_path = "inswapper_128.torch.ckpt",
		safetensors_path = "inswapper_128.torch.safetensors",
	)
