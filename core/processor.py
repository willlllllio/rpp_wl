from __future__ import annotations

import builtins
import dataclasses
import functools
import os
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import cv2
import insightface
from core.utils import write_atomic, ensure, noop
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


def get_face(face_analyser, img_data):
	face = face_analyser.get(img_data)
	try:
		return sorted(face, key = lambda x: x.bbox[0])[0]
	except IndexError:
		return None


def get_faces(face_analyser, img_data):
	return face_analyser.get(img_data)


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


@dataclasses.dataclass
class SwState():
	settings: ProcessSettings
	swap_settings: SwapSettings
	face: Any
	swapper: Any
	face_analyser: Any


@dataclasses.dataclass
class SwapSettings():
	face_path: Path
	multi_face: bool


def _setup(settings: ProcessSettings, swap_settings: SwapSettings):
	face_analyser = get_face_analyser()
	face = get_face(face_analyser, cv2.imread(str(swap_settings.face_path)))

	if settings.load_own_model:
		# needed to run multiple at the same time on the GPU, seems to give better utilization on some
		swapper = load_face_swapper()
	else:
		swapper = get_face_swapper()

	return SwState(settings, swap_settings, face, swapper, face_analyser)


def process_gen_frame_disk(state: SwState, src_tup):
	settings = state.settings
	src_ctx, (src_frame_path, target_frame_path) = src_tup
	if settings.skip_existing and target_frame_path.exists():
		return

	src_frame = cv2.imread(str(src_frame_path))
	try:
		frame, faces_cnt = process_frame(state.swapper, state.face_analyser, state.face, src_frame, state.swap_settings.multi_face)
		is_ok, buffer = cv2.imencode(".png", frame)
		if not is_ok:
			logger.error("failed encoding image, %r, %r", src_ctx, type(frame))
		else:
			write_atomic(buffer, target_frame_path, may_exist = False)
	except ProcessingError as ex:
		error_handling = settings.error_handling
		if error_handling is ProcErrorHandling.Ignore:
			return

		logger.info("processing error %r, ctx %r, files %s, error_handling %s",
					ex, src_ctx, src_frame_path, error_handling.name)
		if error_handling is ProcErrorHandling.Symlink:
			ensure(not target_frame_path.exists(), c = target_frame_path)
			os.symlink(src_frame_path.absolute(), target_frame_path.absolute())
		else:
			raise NotImplementedError(error_handling, src_frame_path, target_frame_path)  # TODO


def process_gen_frame(state: SwState, src_tup):
	settings = state.settings
	src_ctx, src_frame = src_tup
	try:
		res = process_frame(state.swapper, state.face_analyser, state.face, src_frame, state.swap_settings.multi_face)
		return src_ctx, res
	except ProcessingError as ex:
		error_handling = settings.error_handling
		if error_handling is ProcErrorHandling.Ignore:
			return src_ctx, None

		logger.info("processing error %r, ctx %r, error_handling %s",
					ex, src_ctx, error_handling.name)
		if error_handling is ProcErrorHandling.Copy:
			return src_ctx, (src_frame, 0)
		else:
			raise NotImplementedError(ex, error_handling)  # TODO


class ProcessingError(Exception):
	pass


class NoFaceError(ProcessingError):
	pass


class FaceAnalyzerError(ProcessingError):
	pass


class SwapError(ProcessingError):
	pass


def process_frame(swapper, face_analyser, source_face, frame, multi_face):
	try:
		if multi_face:
			faces = get_faces(face_analyser, frame)
		else:
			face = get_face(face_analyser, frame)
			faces = [face] if face else []
	except Exception as ex:
		raise FaceAnalyzerError(ex)

	if not faces:
		raise NoFaceError()

	for face in faces:
		try:
			frame = swapper.get(frame, face, source_face, paste_back = True)
		except Exception as ex:
			raise SwapError(ex)

	return frame, len(faces)


_gen_state = None


def _init_gen_state_global(settings: ProcessSettings, swap_settings: SwapSettings):
	# print("_init_gen_state_global", os.getpid(), os.getppid())
	global _gen_state
	ensure(_gen_state is None)
	_gen_state = _setup(settings, swap_settings)
	return _gen_state


def process_gen_frame_global(process_fn, tup):
	global _gen_state

	init_args, src_frame = tup
	if _gen_state is None:
		_init_gen_state_global(*init_args)

	return process_fn(_gen_state, src_frame)


def process_gen(init_args, frame_gen, process_fn):
	state = _setup(*init_args)
	for src_tup in frame_gen:
		yield process_fn(state, src_tup)


def parallel_process_gen(
		use_gpu: bool, procs_cpu: int, procs_gpu: int, swap_settings: SwapSettings, frame_gen,
		process_disk = False,
):
	print(f"{procs_cpu=} {procs_gpu=} {use_gpu=}")
	settings = ProcessSettings(True, False, ProcErrorHandling.Copy, noop)
	init_args = (settings, swap_settings)
	procs = procs_gpu if use_gpu else procs_cpu

	process_fn = process_gen_frame_disk if process_disk else process_gen_frame
	if procs > 1:
		if not use_gpu:
			import multiprocessing as mp
		else:
			import multiprocessing.dummy as mp
			# Multiple parallel runs on GPU, can do just with threads.
			# init here since the init later isn't threadsafe and might be done multiple times unnecessarily,
			# not an issue with actual multiprocess.
			_init_gen_state_global(*init_args)

		pool = mp.Pool(procs)

		def _settings_gen(gen):
			for i in gen:
				yield init_args, i

		gen = _settings_gen(frame_gen)
		fn = functools.partial(process_gen_frame_global, process_fn)
		gen = pool.imap(fn, gen)
	else:
		gen = process_gen(init_args, frame_gen, process_fn)

	return gen


def process_img(
		face_img: Path, frame_path: Path, output_file: Path, multi_face: bool,
		may_exist: bool = False,
):
	frame = cv2.imread(str(frame_path))
	face_analyser = get_face_analyser()
	face = get_face(face_analyser, cv2.imread(str(face_img)))
	swapper = load_face_swapper()

	frame, faces_count = process_frame(swapper, face_analyser, face, frame, multi_face)
	is_ok, buffer = cv2.imencode(".png", frame)
	if not is_ok:
		raise ValueError("failed encoding image??")
	write_atomic(buffer, output_file, may_exist = may_exist)
