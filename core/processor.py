from __future__ import annotations

import os
import builtins
import dataclasses
import functools
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Callable

from core.mp_utils import imap_backpressure
from core.utils import write_atomic, ensure, noop

import cv2
import onnxruntime
import insightface

import logging as _logging

logger = _logging.getLogger(__name__)

FACE_SWAPPER = None
FACE_ANALYSER = None


def get_default_providers():
	return onnxruntime.get_available_providers()


def get_cpu_providers():
	return ['CPUExecutionProvider']


def get_model(model_file, local: bool, **kwargs):
	from insightface.model_zoo.model_zoo import PickableInferenceSession

	providers = kwargs.get('providers', get_default_providers())
	provider_options = kwargs.get('provider_options', None)
	session = PickableInferenceSession(model_file, providers = providers, provider_options = provider_options)

	if local:
		# print("loading local_model")
		from inswapper_local import INSwapper as model
	else:
		# print("loading modelzoo")
		from insightface.model_zoo.inswapper import INSwapper as model

	return model(model_file = model_file, session = session)


def get_face_analyser(settings: ProcessSettings):
	global FACE_ANALYSER
	if FACE_ANALYSER is None:
		providers = get_default_providers() if settings.swap_settings.use_gpu else get_cpu_providers()
		FACE_ANALYSER = insightface.app.FaceAnalysis(name = 'buffalo_l', providers = providers)
		FACE_ANALYSER.prepare(ctx_id = 0, det_size = (640, 640))
	return FACE_ANALYSER


def load_face_swapper(settings: ProcessSettings):
	model_path = '../inswapper_128.onnx'
	model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
	providers = get_default_providers() if settings.swap_settings.use_gpu else get_cpu_providers()
	return get_model(model_path, settings.swap_settings.local_model, providers = providers)


def get_face_swapper(settings: ProcessSettings):
	global FACE_SWAPPER
	if FACE_SWAPPER is None:
		FACE_SWAPPER = load_face_swapper(settings)
	return FACE_SWAPPER


def get_face(face_analyser, img_data):
	face = face_analyser.get(img_data)
	try:
		return sorted(face, key = lambda x: x.bbox[0])[0]
	except IndexError:
		return None


def get_faces(face_analyser, img_data):
	return face_analyser.get(img_data)


class ProcErrorHandling(Enum):
	Log = 1
	Ignore = 2
	Symlink = 3
	Copy = 4
	Raise = 5


@dataclasses.dataclass
class ProcessSettings():
	load_own_model: bool
	swap_settings: SwapSettings
	skip_existing: bool = False
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
	local_model: bool
	use_gpu: bool
	procs_cpu: int
	procs_gpu: int


def _setup(settings: ProcessSettings, swap_settings: SwapSettings):
	face_analyser = get_face_analyser(settings)
	face = get_face(face_analyser, cv2.imread(str(swap_settings.face_path)))

	if settings.load_own_model:
		swapper = load_face_swapper(settings)
	else:
		swapper = get_face_swapper(settings)

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

		log = logger.debug if isinstance(ex, NoFaceError) else logger.info
		log("processing error %r, ctx %r, files %s, error_handling %s",
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

		log = logger.debug if isinstance(ex, NoFaceError) else logger.info
		log("processing error %r, ctx %r, error_handling %s",
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
	print("_init_gen_state_global", os.getpid(), os.getppid())
	global _gen_state
	ensure(_gen_state is None)
	_gen_state = _setup(settings, swap_settings)
	return _gen_state


def process_gen_frame_global(process_fn, tup):
	global _gen_state

	src_frame = tup
	if _gen_state is None:
		raise ValueError("no gen state")

	return process_fn(_gen_state, src_frame)


def process_gen(init_args, frame_gen, process_fn):
	state = _setup(*init_args)
	for src_tup in frame_gen:
		yield process_fn(state, src_tup)


def _passthrough(i):
	return i[1][0], (i[1][1], 0)


def parallel_process_gen(swap_settings: SwapSettings, frame_gen, process_disk = False):
	print(f"main proc {os.getpid()}")
	print(f"procs_cpu={swap_settings.procs_cpu} use_gpu={swap_settings.use_gpu} procs_gpu={swap_settings.procs_gpu} ")
	settings = ProcessSettings(True, swap_settings, False, ProcErrorHandling.Copy, noop)
	init_args = (settings, swap_settings)
	procs = swap_settings.procs_gpu if swap_settings.use_gpu else swap_settings.procs_cpu

	process_fn = process_gen_frame_disk if process_disk else process_gen_frame
	if procs > 1:
		gen = frame_gen
		fn = functools.partial(process_gen_frame_global, process_fn)

		if not swap_settings.use_gpu:
			import multiprocessing as mp
			pool = mp.Pool(procs, initializer = _init_gen_state_global, initargs = init_args)
			gen = pool.imap(fn, gen)
		else:
			import multiprocessing.dummy as mp
			# Multiple parallel runs on GPU, can do just with threads.
			# init here since the init later isn't threadsafe and might be done multiple times unnecessarily,
			# not an issue with actual multiprocess.
			_init_gen_state_global(*init_args)
			pool = mp.Pool(procs)
			gen = imap_backpressure(pool.imap, fn, gen, procs + 30, 0.01, procs * 2 + 10, 0.01)
	else:
		gen = process_gen(init_args, frame_gen, process_fn)

	return gen


def process_img(
		face_img: Path, frame_path: Path, output_file: Path,
		gpu: bool, multi_face: bool, local_model: bool,
		overwrite: bool = False,
):
	swap_settings = SwapSettings(None, multi_face, local_model, gpu, 1, 1)
	settings = ProcessSettings(False, swap_settings, False)
	frame = cv2.imread(str(frame_path))

	face_analyser = get_face_analyser(settings)
	face = get_face(face_analyser, cv2.imread(str(face_img)))
	swapper = load_face_swapper(settings)

	frame, faces_count = process_frame(swapper, face_analyser, face, frame, multi_face)
	is_ok, buffer = cv2.imencode(".png", frame)
	if not is_ok:
		raise ValueError("failed encoding image??")
	write_atomic(buffer, output_file, may_exist = overwrite)
