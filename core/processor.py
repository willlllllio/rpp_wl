from __future__ import annotations

import os
import builtins
import dataclasses
import functools
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Callable

import torch
from insightface.app import FaceAnalysis

from core.mp_utils import imap_backpressure
from core.utils import write_atomic, ensure, noop, Timer

import cv2
import onnxruntime
import insightface

import logging as _logging

logger = _logging.getLogger(__name__)

FACE_SWAPPER = None
FACE_ANALYSER = None


def model_device(model: torch.nn.Module):
	# TODO: check _buffers if no params?
	return next(model.parameters()).device


def get_default_providers():
	return onnxruntime.get_available_providers()


def get_cpu_providers():
	return ['CPUExecutionProvider']


def load_face_analyser(settings: ProcessSettings):
	providers = get_default_providers() if settings.swap_settings.use_gpu else get_cpu_providers()
	fa = insightface.app.FaceAnalysis(name = 'buffalo_l', providers = providers)
	fa.prepare(ctx_id = 0, det_size = (640, 640))
	fa.models.pop("landmark_3d_68")
	fa.models.pop("landmark_2d_106")
	fa.models.pop("genderage")

	if settings.swap_settings.model_type == "torch":
		try:
			model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../buffalo_l_detect.ckpt")
			detect = torch.load(model_path)
			print("loaded from file")
		except:
			from onnx2torch import convert
			onnx_model_path = os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx")
			# You can pass the path to the onnx model to convert it or...
			with Timer("converting detectmodel took {} secs"):
				detect = convert(onnx_model_path)

		def my_run(*args, **kwargs):
			with torch.no_grad():
				# print('torch detect!')
				device = settings.torch_device or model_device(detect)
				blob = args[1]["input.1"]
				blob = torch.from_numpy(blob).to(device)
				res = detect(blob)
				return [i.detach().cpu().numpy() for i in res]

		if settings.torch_device is not None:
			detect.to(settings.torch_device)

		org_model = fa.models["detection"]
		org_model.session.run = my_run
		print("using torch RetinaFace")

	return fa


def get_face_analyser(settings: ProcessSettings):
	global FACE_ANALYSER
	if FACE_ANALYSER is None:
		FACE_ANALYSER = load_face_analyser(settings)
	return FACE_ANALYSER


def get_model(model_file, local: bool, **kwargs):
	from insightface.model_zoo.model_zoo import PickableInferenceSession

	providers = kwargs.get('providers', get_default_providers())
	provider_options = kwargs.get('provider_options', None)
	session = PickableInferenceSession(model_file, providers = providers, provider_options = provider_options)

	if local:
		# print("loading local_model")
		from core.inswapper_local import INSwapper as model
	else:
		# print("loading modelzoo model")
		from insightface.model_zoo.inswapper import INSwapper as model

	return model(model_file = model_file, session = session)


def load_face_swapper_torch(settings: ProcessSettings):
	from core.model_torch import FaceSwapper
	from core.model_torch_swapper import TorchINSwapper
	import torch

	if settings.swap_settings.model_path:
		model_path = settings.swap_settings.model_path
	else:
		model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../inswapper_128.torch.safetensors")
		if not os.path.isfile(model_path):
			model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../inswapper_128.torch.ckpt")

	model_path = str(model_path)
	if model_path.endswith(".safetensors"):
		import safetensors.torch
		sd = safetensors.torch.load_file(model_path)
	else:
		sd = torch.load(model_path)

	model = FaceSwapper()
	model.load_state_dict(sd)
	if settings.torch_device is not None:
		model.to(settings.torch_device)

	return TorchINSwapper(model)


def load_face_swapper(settings: ProcessSettings):
	if settings.swap_settings.model_type == "torch":
		return load_face_swapper_torch(settings)

	if settings.swap_settings.model_path:
		model_path = settings.swap_settings.model_path
	else:
		model_path = '../inswapper_128.onnx'
		model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)

	providers = get_default_providers() if settings.swap_settings.use_gpu else get_cpu_providers()
	return get_model(str(model_path), settings.swap_settings.model_type == "onnx_local", providers = providers)


def get_face_swapper(settings: ProcessSettings):
	global FACE_SWAPPER
	if FACE_SWAPPER is None:
		FACE_SWAPPER = load_face_swapper(settings)
	return FACE_SWAPPER


def get_face(face_analyser, img_data):
	return (get_faces(face_analyser, img_data) or [None])[0]


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
	torch_device: torch.device | None = None

	def progress(self, status):
		self.progprint(status, end = "", flush = True)


@dataclasses.dataclass
class SwState():
	settings: ProcessSettings
	swap_settings: SwapSettings
	face: Any
	swapper: Any
	face_analyser: Any
	face_analyser_mini: FaceAnalysisWrapper


@dataclasses.dataclass
class SwapSettings():
	face_path: Path
	multi_face: bool
	model_type: str
	model_path: Path | str | None
	use_gpu: bool
	procs_cpu: int
	procs_gpu: int
	torch_device: str | bool = False
	load_own_model: bool = True
	drop_recognition_model: bool = False


def drop_recognition_model(face_analyser):
	if "recognition" in face_analyser.models:
		face_analyser.models.pop("recognition")
		return True
	return False


class FaceAnalysisWrapper(FaceAnalysis):
	# subclass of FaceAnalysis without landmark/gender and
	#
	def __init__(self, face_analysis: FaceAnalysis, recognition: bool = False):
		models = dict(face_analysis.models)
		keep = ("detection", "recognition") if recognition else ("detection",)
		for name in list(models.keys()):
			if name not in keep:
				models.pop(name)

		self.__face_analysis = face_analysis
		self.models = models

	def __getattribute__(self, item):
		try:
			return object.__getattribute__(self, item)
		except AttributeError:
			pass

		return self.__face_analysis.__getattribute__(item)


def _setup(settings: ProcessSettings):
	swap_settings = settings.swap_settings

	if settings.load_own_model:
		face_analyser = load_face_analyser(settings)
	else:
		face_analyser = get_face_analyser(settings)

	face = get_face(face_analyser, cv2.imread(str(swap_settings.face_path)))
	if swap_settings.drop_recognition_model:
		drop_recognition_model(face_analyser)

	if settings.load_own_model:
		swapper = load_face_swapper(settings)
	else:
		swapper = get_face_swapper(settings)

	face_analyser_mini = FaceAnalysisWrapper(face_analyser, False)
	return SwState(settings, swap_settings, face, swapper, face_analyser, face_analyser_mini)


def process_gen_frame_disk(state: SwState, src_tup):
	settings = state.settings
	src_ctx, (src_frame_path, target_frame_path) = src_tup
	if settings.skip_existing and target_frame_path.exists():
		return

	src_frame = cv2.imread(str(src_frame_path))
	try:
		frame, faces_cnt = process_frame(state.swapper, state.face_analyser_mini, state.face, src_frame, state.swap_settings.multi_face)
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
			pass


def process_gen_frame(state: SwState, src_tup):
	settings = state.settings
	src_ctx, src_frame = src_tup
	try:
		res = process_frame(state.swapper, state.face_analyser_mini, state.face, src_frame, state.swap_settings.multi_face)
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


def _init_gen_state_global(settings: ProcessSettings):
	print("_init_gen_state_global", os.getpid(), os.getppid())
	global _gen_state
	ensure(_gen_state is None)
	_gen_state = _setup(settings)
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


def _torch_device(swap_settings: SwapSettings):
	if swap_settings.model_type == "torch" and swap_settings.use_gpu:
		tdev = swap_settings.torch_device
		devname = None
		if isinstance(tdev, str):
			devname = tdev
		elif tdev is not False:
			if torch.has_cuda and torch.cuda.is_available():
				devname = "cuda"
			elif torch.has_mps and torch.backends.mps.is_available():
				devname = "mps"

		if devname:
			return torch.device(devname)

	return None


def parallel_process_gen(swap_settings: SwapSettings, frame_gen, process_disk = False):
	print(f"main proc {os.getpid()}")
	print(f"procs_cpu={swap_settings.procs_cpu} use_gpu={swap_settings.use_gpu} procs_gpu={swap_settings.procs_gpu} ")

	error_handling = ProcErrorHandling.Log if process_disk else ProcErrorHandling.Copy

	dev = _torch_device(swap_settings)
	print("using torch device: ", repr(dev))
	settings = ProcessSettings(swap_settings.load_own_model, swap_settings, False, error_handling, noop, dev)
	init_args = (settings,)
	procs = swap_settings.procs_gpu if swap_settings.use_gpu else swap_settings.procs_cpu

	# TODO: add optional total_frames arg and: procs = min(procs, total_frames)
	# to not init tons of unneeded model

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
		swap_settings: SwapSettings, frame_path: Path, output_file: Path,
		overwrite: bool = False,
):
	settings = ProcessSettings(False, swap_settings, False, torch_device = _torch_device(swap_settings))
	state = _setup(settings)

	src_frame = cv2.imread(str(frame_path))
	frame, faces_count = process_frame(state.swapper, state.face_analyser_mini, state.face, src_frame, state.swap_settings.multi_face)
	is_ok, buffer = cv2.imencode(".png", frame)
	if not is_ok:
		raise ValueError("failed encoding image??")
	write_atomic(buffer, output_file, may_exist = overwrite)
