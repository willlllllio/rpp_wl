from __future__ import annotations

import builtins
import dataclasses
import os
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import insightface
from core.utils import write_atomic, ensure
import core.globals

FACE_SWAPPER = None
FACE_ANALYSER = None

import logging as _logging

logger = _logging.getLogger(__name__)


def get_face_analyser():
	global FACE_ANALYSER
	if FACE_ANALYSER is None:
		FACE_ANALYSER = insightface.app.FaceAnalysis(name = 'buffalo_l', providers = core.globals.providers)
		FACE_ANALYSER.prepare(ctx_id = 0, det_size = (640, 640))
	return FACE_ANALYSER


def get_face(img_data):
	face = get_face_analyser().get(img_data)
	try:
		return sorted(face, key = lambda x: x.bbox[0])[0]
	except IndexError:
		return None


def load_face_swapper():
	model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
	return insightface.model_zoo.get_model(model_path, providers = core.globals.providers)


def get_face_swapper():
	global FACE_SWAPPER
	if FACE_SWAPPER is None:
		FACE_SWAPPER = load_face_swapper()
	return FACE_SWAPPER


class ProcErrorHandling(Enum):
	Log = 1
	Ignore = 2
	Symlink = 3
	Copy = 4
	Raise = 5


@dataclasses.dataclass
class ProcessSettings():
	load_own_model: bool
	skip_existing: bool
	error_handling: ProcErrorHandling = ProcErrorHandling.Log
	progprint: Any = builtins.print

	def progress(self, status):
		self.progprint(status, end = "", flush = True)


def process_video(source_img: Path, frame_paths: list[tuple[Path, Path]], settings: ProcessSettings):
	source_face = get_face(cv2.imread(str(source_img)))
	if not frame_paths:
		return

	if settings.load_own_model:
		# needed to run multiple at the same time on the GPU, seems to give better utilization on some
		swapper = load_face_swapper()
	else:
		swapper = get_face_swapper()

	for (src_frame_path, target_frame_path) in frame_paths:
		if settings.skip_existing and target_frame_path.exists():
			res = "R"
		else:
			src_frame = cv2.imread(str(src_frame_path))
			res = process_frame(swapper, source_face, src_frame)

		if not isinstance(res, str):
			is_ok, buffer = cv2.imencode(".png", res)
			if not is_ok:
				logger.error("failed encoding image, %r", type(res))
				settings.progress("P")
			else:
				settings.progress(".")
				write_atomic(buffer, target_frame_path, may_exist = False)
			del buffer, res
		else:
			settings.progress(res)
			error_handling = settings.error_handling
			if error_handling is ProcErrorHandling.Ignore:
				continue

			logger.info("processing error %r for file %s, %s", res, src_frame_path, error_handling.name)
			if error_handling is ProcErrorHandling.Symlink:
				ensure(not target_frame_path.exists(), c = target_frame_path)
				os.symlink(src_frame_path.absolute(), target_frame_path.absolute())
			else:
				raise NotImplementedError(error_handling, src_frame_path, target_frame_path)  # TODO


def process_gen(source_img: Path, frame_gen, settings: ProcessSettings):
	source_face = get_face(cv2.imread(str(source_img)))

	if settings.load_own_model:
		# needed to run multiple at the same time on the GPU, seems to give better utilization on some
		swapper = load_face_swapper()
	else:
		swapper = get_face_swapper()

	for pos, src_frame in enumerate(frame_gen):
		res = process_frame(swapper, source_face, src_frame)
		if not isinstance(res, str):
			settings.progress(".")
			yield res
		else:
			settings.progress(res)
			error_handling = settings.error_handling
			if error_handling is ProcErrorHandling.Ignore:
				continue

			logger.info("processing error %r for frame %s, %s", res, pos, error_handling.name)
			if error_handling is ProcErrorHandling.Copy:
				yield src_frame
			else:
				raise NotImplementedError(res, error_handling, pos)  # TODO


def process_frame(swapper, source_face, frame):

	try:
		face = get_face(frame)
	except Exception as ex:
		return "F"

	if not face:
		return "S"

	try:
		return swapper.get(frame, face, source_face, paste_back = True)
	except Exception as ex:
		traceback.print_exc()
		return "E"


def process_img(source_img: Path, frame_path: Path, output_file: Path):
	source_face = get_face(cv2.imread(str(source_img)))
	swapper = load_face_swapper()

	frame = cv2.imread(str(frame_path))
	face = get_face(frame)
	result = swapper.get(frame, face, source_face, paste_back = True)

	is_ok, buffer = cv2.imencode(".png", result)
	if not is_ok:
		raise ValueError("failed encoding image??")
	write_atomic(buffer, output_file, may_exist = False)
