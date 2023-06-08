# from https://github.com/deepinsight/insightface/blob/bc19ea168beb303c757fb3fd797582ee07708e2f/python-package/insightface/model_zoo/inswapper.py#L12
#
# MIT License
#
# Copyright (c) 2022 Jiankang Deng and Jia Guo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

import torch
import numpy as np
import cv2
from insightface.utils import face_align


class TorchINSwapper():
	def __init__(self, model):
		print("TorchINSwapper", TorchINSwapper)
		self.model = model
		self.model.eval()

		emap = os.path.join(os.path.abspath(os.path.dirname(__file__)), "emap.npy")
		self.emap = np.load(emap, allow_pickle = False)

		self.input_mean = 0.0
		self.input_std = 255.0
		self.input_size = (128, 128)

	def get(self, img, target, source_face, paste_back = True):
		# print("patched get!!!~!")
		aimg, M = face_align.norm_crop2(img, target.kps, self.input_size[0])
		blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
									 (self.input_mean, self.input_mean, self.input_mean), swapRB = True)

		latent = source_face.normed_embedding.reshape((1, -1))
		latent = np.dot(latent, self.emap)
		latent /= np.linalg.norm(latent)

		import contextlib
		Timer = contextlib.nullcontext
		with Timer("torch full took {}"):
			with torch.no_grad():
				blob = torch.from_numpy(blob).to(self.model.device)
				latent = torch.from_numpy(latent).to(self.model.device)
				pred = self.model(blob, latent).cpu().numpy()

		# TODO: do this on GPU as well?

		img_fake = pred.transpose((0, 2, 3, 1))[0]
		bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
		if not paste_back:
			return bgr_fake, M
		else:
			target_img = img
			fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
			fake_diff = np.abs(fake_diff).mean(axis = 2)
			fake_diff[:2, :] = 0
			fake_diff[-2:, :] = 0
			fake_diff[:, :2] = 0
			fake_diff[:, -2:] = 0
			IM = cv2.invertAffineTransform(M)
			img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype = np.float32)
			bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue = 0.0)
			img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue = 0.0)
			fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue = 0.0)
			img_white[img_white > 20] = 255
			fthresh = 10
			fake_diff[fake_diff < fthresh] = 0
			fake_diff[fake_diff >= fthresh] = 255
			img_mask = img_white
			mask_h_inds, mask_w_inds = np.where(img_mask == 255)
			mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
			mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
			mask_size = int(np.sqrt(mask_h * mask_w))
			k = max(mask_size // 10, 10)
			# k = max(mask_size//20, 6)
			# k = 6
			kernel = np.ones((k, k), np.uint8)
			img_mask = cv2.erode(img_mask, kernel, iterations = 1)
			kernel = np.ones((2, 2), np.uint8)
			fake_diff = cv2.dilate(fake_diff, kernel, iterations = 1)
			k = max(mask_size // 20, 5)
			# k = 3
			# k = 3
			kernel_size = (k, k)
			blur_size = tuple(2 * i + 1 for i in kernel_size)
			img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
			k = 5
			kernel_size = (k, k)
			blur_size = tuple(2 * i + 1 for i in kernel_size)
			fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
			img_mask /= 255
			fake_diff /= 255
			# img_mask = fake_diff
			img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
			fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
			fake_merged = fake_merged.astype(np.uint8)
			return fake_merged
