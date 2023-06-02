#!/usr/bin/env python3

from __future__ import annotations

import functools
import os
import shlex

if not os.environ.get("SKIP_EARLY_TORCH") == "1":
	import torch  # needs to be imported before onnx for GPU support to work easily apparently

import platform
import re
import subprocess
import sys
import traceback

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from core.processor import process_video, process_img, get_face, ProcessSettings, ProcErrorHandling, process_gen, _init_gen_state_global, process_gen_frame_global
from core.utils import is_img, create_video, add_audio, extract_frames, ensure, Timer, create_video_with_audio, ensure_equal, create_video_from_frame_gen, \
	tmp_path_move_ctx, noop, str_to_num, get_video_info, VidInfo
import psutil

# DEFAULT_FRAME_SUFFIX_ORG = "_org.png"
# DEFAULT_FRAME_SUFFIX_SWAPPED = "_swapped.png"

DEFAULT_FRAME_SUFFIX_ORG = ".png"
DEFAULT_FRAME_SUFFIX_SWAPPED = ".png"


def name_pattern(name: str, length: int = 5):
	return f"%0{length}d{name}"


def limit_resources(args):
	if args['max_memory'] >= 1:
		memory = args['max_memory'] * 1024 * 1024 * 1024
		if str(platform.system()).lower() == 'windows':
			import ctypes
			kernel32 = ctypes.windll.kernel32
			kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
		else:
			import resource
			try:
				resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
			except:
				traceback.print_exc()


def pre_check():
	import shutil
	import core.globals
	import torch
	if sys.version_info < (3, 8):
		quit(f'Python version is not supported - please upgrade to 3.8 or higher')
	if not shutil.which('ffmpeg'):
		quit('ffmpeg is not installed!')
	model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
	if not os.path.isfile(model_path):
		quit('File "inswapper_128.onnx" does not exist!')
	if '--gpu' in sys.argv:
		CUDA_VERSION = torch.version.cuda
		CUDNN_VERSION = torch.backends.cudnn.version()

		if 'ROCMExecutionProvider' not in core.globals.providers:
			if not torch.cuda.is_available() or not CUDA_VERSION:
				quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")
			if CUDA_VERSION > '11.8':
				quit(f"CUDA version {CUDA_VERSION} is not supported - please downgrade to 11.8.")
			if CUDA_VERSION < '11.4':
				quit(f"CUDA version {CUDA_VERSION} is not supported - please upgrade to 11.8")
			if CUDNN_VERSION < 8220:
				quit(f"CUDNN version {CUDNN_VERSION} is not supported - please upgrade to 8.9.1")
			if CUDNN_VERSION > 8910:
				quit(f"CUDNN version {CUDNN_VERSION} is not supported - please downgrade to 8.9.1")
	else:
		core.globals.providers = ['CPUExecutionProvider']


def _frames(frame_paths: list[Path], output_dir: Path, org_suffix: str, swapped_suffix: str) \
		-> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
	frames = []
	for i in frame_paths:
		outname = i.name[:-len(org_suffix)] + swapped_suffix
		outpath = output_dir / outname
		frames.append((i, outpath))

	todo = []
	done = []

	for i in frames:
		(done if i[1].exists() else todo).append(i)

	return frames, todo, done


def start_processing_cpu(face_img: Path, frame_paths: list[tuple[Path, Path]], settings: ProcessSettings, pool, procnum: int):
	n = max(len(frame_paths) // procnum, 1)
	processes = []
	for i in range(0, len(frame_paths), n):
		p = pool.apply_async(process_video, args = (settings, face_img, frame_paths[i:i + n]))
		processes.append(p)

	for p in processes:
		p.get()


def start_processing_gpu_multi(face_img: Path, frame_paths: list[tuple[Path, Path]], settings: ProcessSettings, pool, procnum: int):
	return start_processing_cpu(face_img, frame_paths, settings, pool, procnum)


def start_processing_gpu_single(face_img: Path, frame_paths: list[tuple[Path, Path]], settings: ProcessSettings):
	process_video(settings, face_img, frame_paths)


def status(string):
	print("Status: " + string)


_leading_num_reg = re.compile("^(\d+)(?:[^\d]|$)")


def get_framepaths(frames_dir: Path, filename_suffix: str, ensure_continuous: bool = True) -> list[Path]:
	with os.scandir(frames_dir) as it:
		files = [i for i in it if i.is_file() and i.name.endswith(filename_suffix)]

	with_num = [(int(_leading_num_reg.search(file.name).group(1)), file) for file in files]
	with_num = sorted(with_num)
	if ensure_continuous:
		nums = [i[0] for i in with_num]
		ensure(nums == sorted(range(1, len(files) + 1)), c = ("expected continuous frames", nums))

	return [Path(i.path) for _, i in with_num]


def makedir(path: str | Path, exist_ok = False, parents: bool | int = False):
	# like Path.mkdir but if parents is int create at most [parents] folders
	# instead of possibly all back to /

	path = Path(path)
	if isinstance(parents, bool):
		path.mkdir(exist_ok = exist_ok, parents = parents)
	else:
		ensure(parents > 0, c = parents)
		_must_exist, _parents = path, parents
		while _parents:
			_must_exist = _must_exist.parent
			_parents -= 1

		if not _must_exist.exists():
			raise FileNotFoundError("parent dir not found", _must_exist, parents, path)

		path.mkdir(exist_ok = exist_ok, parents = True)


def output_args_replace(format_str: str, face_path: Path, source_path: Path, args: dict):
	def rep(match: re.Match):
		name = match.group(1)
		if name == "src_bn":
			return source_path.name
		if name == "face_bn":
			return face_path.name
		if name == "src_bnc":
			return source_path.with_suffix("").name
		if name == "face_bnc":
			return face_path.with_suffix("").name
		if name == "format":
			return args["format"]
		if name == "plain_format":
			return args["plain_format"] or args["format"]

		raise ValueError(f"unsupported format name: {name!r}")

	return re.sub(r"{(\w+)}", rep, format_str)


def start(args):
	face_path = Path(args["face"])
	source_path = Path(args["source_vid"])
	if not face_path:
		return print("\n[WARNING] Please select an image containing a face.")

	if not face_path.is_file():
		return print("\n[WARNING] face_path not found", face_path)

	if not source_path:
		return print("\n[WARNING] Please select a video/image to swap face in.")

	if not source_path.exists():
		return print("\n[WARNING] source_path not found", source_path)

	ensure(not (args["output_vid_formatted"] and args["output_vid"]), c = "got both output_vid_formatted and output_vid")
	if args["output_vid_formatted"]:
		args["output_vid"] = _out = output_args_replace(args["output_vid_formatted"], face_path, source_path, args)
		print(f"using formatted output path: {str(_out)!r}")

	output_path = args["output_vid"]
	if output_path:
		output_path = Path(output_path)
		if output_path.is_dir():
			output_path = output_path / source_path.with_suffix(f".swapped.{args['format']}").name
			print(f"output_path is directory, saving to {str(output_path)!r}")
	else:
		output_path = source_path.with_suffix(f".swapped.{args['format']}")

	if not args["overwrite"]:
		ensure(not output_path.exists(), c = ("output_path exists", output_path))

	with Timer("setgrad took {:.2f} secs"):
		import torch
		torch.set_grad_enabled(False)

	if source_path.is_file():
		if is_img(source_path):
			process_img(face_path, source_path, output_path)
			status("swap successful!")
			return

		vid_info = get_video_info(source_path, ffprobe = args["ffprobe"])
	else:
		vid_info = VidInfo(0, 0, args["fps_source"], False)
		ensure(vid_info.fps, c = ("source_path is png sequence, manually passing --fps_source framerate argument required"))

	fps_target: int | None = args["fps_target"]
	if fps_target and vid_info.fps > fps_target:
		fps_use = fps_target
		fps_swapped = fps_target
		print("limiting fps to", fps_use)
	else:
		fps_use = None
		fps_swapped = vid_info.fps

	if args["work_dir"]:
		workdir = Path(args["work_dir"])
	elif args["work_dir_root"]:
		workdir = Path(args["work_dir_root"])
		workdir = workdir / f"{output_path.name}.tmp"
	else:
		workdir = output_path.with_name(output_path.name + ".tmp")

	stream = args["stream"]
	with Timer("full processing took {:.4f} secs"):
		if stream:
			ensure(not source_path.is_dir(), c = "directory sources not implemented for stream")
			process_streamed(args, source_path, face_path, workdir, output_path, vid_info, fps_use, fps_swapped)
		else:
			process_using_frames(args, source_path, face_path, workdir, output_path, fps_use, fps_swapped)


def _split_shell_args(args):
	if not args:
		return []
	res = []
	for i in args:
		res.extend(shlex.split(i))
	return res


def process_streamed(
		args: dict, source_path: Path, face_path: Path, workdir: Path,
		output_path: Path, vid_info: VidInfo,
		src_fps_output_max: int | float | None,  # fps to read from source, so drop frames if src fps is higher, None = use all frames.
		fps_swapped: int | float,  # fps that will be output, always same as fps_use if that isn't None
):
	vid_output_audio: bool = args["vid_output_audio"]
	overwrite = args["overwrite"]

	# Note: cv2 VideoCapture is much faster (almost 2x) but has no way to easily skip frames,
	# would have to get frame timestamps and implement skipping by hand when using fps_use to always use, or assume constant fps.

	if args["cv2_reader"] or (args["ffmpeg_reader"] is False and src_fps_output_max is None):
		gen = _frame_gen_cv2(source_path)
	else:
		gen = _frame_gen_ffmpeg(args, source_path, vid_info.width, vid_info.height, src_fps_output_max)

	settings = ProcessSettings(True, False, ProcErrorHandling.Copy, noop)

	procs_cpu = args["parallel_cpu"]
	procs_gpu = args["parallel_gpu"]
	use_gpu = args["gpu"]
	procs = procs_gpu if use_gpu else procs_cpu
	print(f"{procs_cpu=} {procs_gpu=} {use_gpu=}")
	if procs > 1:
		if not use_gpu:
			import multiprocessing as mp
		else:
			import multiprocessing.dummy as mp
			# Multiple parallel runs on GPU, can do just with threads.
			# init here since the init later isn't threadsafe and might be done multiple times unnecessarily,
			# not an issue with actual multiprocess.
			_init_gen_state_global(settings, face_path)

		pool = mp.Pool(procs)

		def _settings_gen(gen):
			for i in gen:
				yield settings, face_path, i

		gen = _settings_gen(gen)
		gen = pool.imap(process_gen_frame_global, gen)
	else:
		gen = process_gen(settings, face_path, gen)

	_tmp_file_ctx = functools.partial(tmp_path_move_ctx, trail_org_ext = True, overwrite = overwrite, overwrite_delete = overwrite)

	audio_source_path = None
	if args["direct_audio"] and vid_info.has_audio:
		output_path_gen = output_path
		audio_2stage = False
		audio_source_path = source_path
	else:
		audio_2stage = vid_output_audio and vid_info.has_audio
		if audio_2stage:
			# want audio added, create audioless .plain then combine at final output_path
			output_path_gen = output_path.with_suffix(".plain." + (args["plain_format"] or args["format"]))
			if not overwrite:
				ensure(not output_path_gen.exists(), c = ("output_path_gen exists", output_path_gen))
		else:
			# no audio, can write to final filepath right away
			output_path_gen = output_path

	with _tmp_file_ctx(output_path_gen) as tmp_path:
		ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info", *_split_shell_args(args["out_ff_args"])])
		create_video_from_frame_gen(
			gen, vid_info.width, vid_info.height, fps_swapped, tmp_path,
			preset = args["preset"], crf = args["crf"], audio_source_path = audio_source_path, audio_shortest = args["audio_shortest"],
			**ffmpeg,
		)

	if audio_2stage:
		ensure(output_path != output_path_gen)
		with _tmp_file_ctx(output_path) as tmp_path:
			ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info", *_split_shell_args(args["audio_ff_args"])])
			add_audio(output_path_gen, source_path, tmp_path, shortest = args["audio_shortest"], **ffmpeg)


def _frame_gen_ffmpeg(args, source_path: Path, width, height, fps_to_output: int | float | None, ):
	ensure(width and height, c = (width, height))

	img_size = width * height * 3
	ffmpeg = [args["ffmpeg"], "-hide_banner", "-loglevel", "info"]
	fps = ["-filter:v", f"fps=fps={fps_to_output}"] if fps_to_output else []

	com = [
		*ffmpeg,
		*_split_shell_args(args["ffmpeg_reader_args_0"]),
		"-i", str(source_path),
		*_split_shell_args(args["ffmpeg_reader_args_1"]),
		*fps,
		"-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:",
		*_split_shell_args(args["ffmpeg_reader_args_2"]),
	]
	print("com", repr(com))
	proc = subprocess.Popen(com, stdout = subprocess.PIPE, bufsize = 128 * 1024 ** 2)
	while True:
		buffer = proc.stdout.read(img_size)
		if not buffer:
			break
		ensure_equal(len(buffer), img_size)
		frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
		yield frame


def _frame_gen_cv2(source_path: Path):
	vidcap = cv2.VideoCapture(str(source_path))
	count = 0
	while True:
		ok, frame = vidcap.read()
		if not ok:
			break
		count += 1
		yield frame


def process_using_frames(
		args: dict, source_path: Path, face_path: Path, workdir: Path,
		output_path: Path,
		# fps for frames to take from source, None if using all,
		fps_use: int | float | None,
		fps_swapped: int | float,  # fps that will be output, always same as fps_use if that is given
):
	vid_output_audio: bool = args["vid_output_audio"]
	name_suffix_org = args["name_suffix_org"]
	name_suffix_swapped = args["name_suffix_swapped"]
	ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info"])

	output_path_plain = None
	if args["vid_output_plain"]:
		output_path_plain = output_path.with_suffix(".plain." + (args["plain_format"] or args["format"]))
		ensure(not output_path_plain.exists(), c = ("output_path_plain exists", output_path_plain))

	if source_path.is_file():
		if args["frames_dir"]:
			in_frames_dir = Path(args["frames_dir"])
		else:
			_root = args["frames_dir_root"] or workdir
			in_frames_dir = _root / f"f_in__{source_path.name}__F{fps_use or 'srcfps'}"
			if in_frames_dir.exists():
				status(f"frames dir exists, not extracting again, assuming okay: {str(in_frames_dir)!r}")
			else:
				makedir(in_frames_dir, exist_ok = True, parents = 2)
				status(f"extracting frames to {str(in_frames_dir)!r}")
				extract_frames(source_path, in_frames_dir, fps_use, filename_pattern = name_pattern(name_suffix_org), **ffmpeg)
	else:
		print("using png sequence as source")
		ensure(source_path.is_dir())
		in_frames_dir = source_path

	in_frame_paths = get_framepaths(in_frames_dir, name_suffix_org)
	status(f"got {len(in_frame_paths)} frames total.")

	with Timer("swap took {:.2f} secs"):
		if args["swapped_dir"]:
			swapped_frames_dir = Path(args["swapped_dir"])
		else:
			_root = args["swapped_dir_root"] or workdir
			swapped_frames_dir = _root / f"f_swapped__{source_path.name}__F{fps_use or 'srcfps'}"
			makedir(swapped_frames_dir, exist_ok = True, parents = 2)

		fp_all, fp_todo, fp_done = _frames(in_frame_paths, swapped_frames_dir, name_suffix_org, name_suffix_swapped)
		ensure(fp_all, c = ("didn't find any frames", in_frame_paths))

		del_done = False
		if args["redo_swapped"] and len(fp_todo) != len(fp_all):
			del_done = True
			print(f"redoing {len(fp_done)} already completed of {len(fp_all)}")

		if args["redo_completed_swap"] and not fp_todo:
			print(f"all {len(fp_all)} completed, redoing all")
			del_done = True

		if del_done:
			for _, dst in fp_done:
				dst.unlink(missing_ok = False)

			fp_todo = fp_all
			fp_done = []

		if fp_todo:
			status(f"swapping {len(fp_todo)} frames of {len(fp_all)} total, {len(fp_done)} finished.")
			procs_cpu = args["parallel_cpu"]
			procs_gpu = args["parallel_gpu"]
			use_gpu = args["gpu"]
			print(f"{procs_cpu=} {procs_gpu=} {use_gpu=}")

			settings = ProcessSettings(False, False, ProcErrorHandling.Log)

			pool = None
			if use_gpu:
				print("running on GPU")
				if procs_gpu > 1:
					import multiprocessing.dummy as mp
					pool = mp.Pool(procs_gpu)
					settings.load_own_model = True
					start_processing_gpu_multi(face_path, fp_todo, settings, pool, procs_gpu)
				else:
					try:
						import tqdm
						fp_todo_use = tqdm.tqdm(fp_todo)
					except ImportError:
						fp_todo_use = fp_todo
					start_processing_gpu_single(face_path, fp_todo_use, settings)
			else:
				import multiprocessing as mp
				pool = mp.Pool(procs_cpu)
				start_processing_cpu(face_path, fp_todo, settings, pool, procs_cpu)

			if pool is not None:
				pool.close()
				pool.join()
		else:
			status("skipping swapping, all finished already")

	swapped_pat = name_pattern(name_suffix_swapped)
	ffargs = dict(filename_pattern = swapped_pat, **ffmpeg)
	if vid_output_audio:
		if output_path_plain is not None:
			status(f"creating plain video with fps {fps_swapped} from {len(fp_all)} frames at {str(output_path_plain)!r} without any audio.")
			create_video(swapped_frames_dir, fps_swapped, output_path_plain, **ffargs)

		status(f"creating video with fps {fps_swapped} from {len(fp_all)} frames at {str(output_path)!r} with audio from {str(source_path)!r}")
		create_video_with_audio(swapped_frames_dir, fps_swapped, source_path, output_path, **ffargs)
	else:
		status(f"creating video with fps {fps_swapped} from {len(fp_all)} frames at {str(output_path)!r} without any audio.")
		create_video(swapped_frames_dir, fps_swapped, output_path, **ffargs)

	status("swap successful!")
	return


def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--face", help = "use this face")
	parser.add_argument("-s", "--source_vid", help = "replace this face")
	parser.add_argument("-o", "--output_vid", help = "save output to this file")
	parser.add_argument("-O", "--output_vid_formatted", help = "save output to this file with {} formatting")

	parser.add_argument("-y", "--overwrite", action = "store_true", help = "save output to this file with {} formatting")
	parser.add_argument("--gpu", action = "store_true",
						help = "use gpu")
	parser.add_argument("--stream", action = "store_true",
						help = "no frame files just do everything in mem and write directly to plain output file")
	parser.add_argument("--keep_frames", action = "store_true",
						help = "keep frames directory")
	parser.add_argument("-r", "--fps_target", type = str_to_num,
						help = "maximum source fps wanted, will drop frames if source is higher fps, does nothing if source fps is lower")
	parser.add_argument("--fps_source", type = str_to_num,
						help = "source video fps, only needed for png sequence folder sources or maybe weird file formats")

	parser.add_argument("--crf", type = int, default = 15,
						help = "output crf")
	parser.add_argument("--preset", default = "superfast",
						help = "output preset")
	parser.add_argument("--out_ff_args", action = "append",
						help = "extra output ffmpeg args")
	parser.add_argument("--audio_ff_args", action = "append",
						help = "extra audio merging ffmpeg args")

	parser.add_argument("--audio_shortest", action = "store_true",
						help = "shorten audio file to vid length, should only be needed if seeking with -XYZ")

	parser.add_argument("-R", "--ffmpeg_reader", action = "store_true",
						help = "always use ffmpeg source reader")
	parser.add_argument("--cv2_reader", action = "store_true",
						help = "always use opencv source reader")

	parser.add_argument("-X", "--ffmpeg_reader_args_0", action = "append",
						help = "arguments passed to ffmpeg reader in front of args. Note: add space in front if it starts with -")
	parser.add_argument("-Y", "--ffmpeg_reader_args_1", action = "append",
						help = "arguments passed to ffmpeg reader after input. Note: add space in front if it starts with -")
	parser.add_argument("-Z", "--ffmpeg_reader_args_2", action = "append",
						help = "arguments passed to ffmpeg reader after output. Note: add space in front if it starts with -")

	vidcontainer = lambda inp: inp.lstrip(".").strip()
	parser.add_argument("-F", "--format", default = "mp4", type = vidcontainer, help = "video container to use, default mp4")
	parser.add_argument("--plain_format", type = vidcontainer, help = "video container to use for plain files")

	parser.add_argument("--name_suffix_org", default = DEFAULT_FRAME_SUFFIX_ORG,
						help = "suffix (including extension) of original frame names")
	parser.add_argument("--name_suffix_swapped", default = DEFAULT_FRAME_SUFFIX_SWAPPED,
						help = "suffix (including extension) of original frame names")

	def existing_path(p: str):
		path = Path(p)
		if not path.exists():
			raise argparse.ArgumentTypeError(f"path not found: {p!r}")
		return path

	parser.add_argument("--frames_dir", type = existing_path, help = "source frames tmp dir")
	parser.add_argument("--frames_dir_root", type = existing_path, help = "source frames tmp root dir")

	parser.add_argument("--swapped_dir", type = existing_path, help = "swapped tmp dir")
	parser.add_argument("--swapped_dir_root", type = existing_path, help = "swapped tmp root dir")

	parser.add_argument("--work_dir", type = existing_path, help = "work tmp dir")
	parser.add_argument("--work_dir_root", type = existing_path, help = "work tmp root dir")

	parser.add_argument("-A", "--no_audio", dest = "vid_output_audio", action = "store_false",
						help = "dont try to copy audio from source")
	parser.add_argument("-P", "--no_plain", dest = "vid_output_plain", action = "store_false",
						help = "dont create plaint output video (.plain.mp4 file without original audio)")
	parser.add_argument("-d", "--direct_audio", action = "store_true",
						help = "add audio directly to output file in one go (instead of plain file and then 2nd separate file with audio merged in)")

	parser.add_argument("-S", "--redo_swapped", action = "store_true",
						help = "always redo any already swapped images")

	parser.add_argument("-C", "--redo_completed_swap", action = "store_true",
						help = "redo swapping if it has been fully completed")

	parser.add_argument("--max_memory", default = 16, type = int, help = "set max memory")
	parser.add_argument("--parallel_cpu", type = int, default = max(psutil.cpu_count(logical = True), 1),
						help = "number of cores to use")
	parser.add_argument("--parallel_gpu", type = int, default = 1,
						help = "number of instance to run in parallel on the GPU, seems faster on some GPU if enough mem")

	parser.add_argument("--ffprobe", default = "ffprobe", help = "ffprobe command/path")
	parser.add_argument("--ffmpeg", default = "ffmpeg", help = "ffprobe command/path")
	return parser


if __name__ == "__main__":
	parser = make_parser()
	args = { }
	for name, value in vars(parser.parse_args()).items():
		args[name] = value

	pre_check()
	limit_resources(args)
	start(args)
