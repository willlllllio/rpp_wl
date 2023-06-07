#!/usr/bin/env python3

from __future__ import annotations

import logging
import os

if not os.environ.get("SKIP_EARLY_TORCH") == "1":
	import torch  # needs to be imported before onnx for GPU support to work easily apparently

import collections
import functools

from typing import Iterable, Any
import shlex

import re
import subprocess

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from core.processor import process_img, parallel_process_gen, SwapSettings
from core.utils import is_img, add_audio, extract_frames, ensure, Timer, create_video_with_audio, ensure_equal, create_video_from_frame_gen, \
	tmp_path_move_ctx, str_to_num, get_video_info, VidInfo
import psutil

DEFAULT_FRAME_SUFFIX_ORG = ".png"
DEFAULT_FRAME_SUFFIX_SWAPPED = ".png"


def no_tmp_ctx(arg):
	return arg


def name_pattern(name: str, length: int = 5):
	return f"%0{length}d{name}"


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


def status(string):
	print("Status: " + string)


_leading_num_reg = re.compile("^(\d+)(?:[^\d]|$)")


def get_framepaths(frames_dir: Path, filename_suffix: str | Iterable[str], ensure_continuous: bool = True) -> list[Path]:
	with os.scandir(frames_dir) as it:
		files = [i for i in it if i.is_file() and i.name.endswith(filename_suffix)]

	with_num = [(int(_leading_num_reg.search(file.name).group(1)), file) for file in files]
	with_num = sorted(with_num)
	if ensure_continuous:
		nums = [i[0] for i in with_num]
		ensure(nums == sorted(range(1, len(files) + 1)), c = ("expected continuous frames", nums))
	return [Path(i.path) for _, i in with_num]


def get_imagepaths(frames_dir: Path, filename_suffix: str | Iterable[str]) -> list[Path]:
	with os.scandir(frames_dir) as it:
		files = [i for i in it if i.is_file() and i.name.endswith(filename_suffix)]

	return [Path(i.path) for i in files if is_img(i.path)]


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
		if name in ("format", "F", "ext"):
			return args["format"]
		if name == "plain_format":
			return args["plain_format"] or args["format"]

		raise ValueError(f"unsupported format name: {name!r}")

	return re.sub(r"{(\w+)}", rep, format_str)


def start(args):
	face_path = Path(args["face"])
	source_path = Path(args["source_vid"])
	image_mode: bool = args["image_mode"]

	if not face_path.is_file():
		return print("\n[WARNING] face_path not found", face_path)

	if not source_path.exists():
		return print("\n[WARNING] source_path not found", source_path)

	ensure(not (args["output_vid_formatted"] and args["output_vid"]), c = "got both output_vid_formatted and output_vid")
	if args["output_vid_formatted"]:
		args["output_vid"] = _out = Path(output_args_replace(args["output_vid_formatted"], face_path, source_path, args))
		print(f"using formatted output path: {str(_out)!r}")

	source_is_image = source_path.is_file() and is_img(source_path)

	output_path = args["output_vid"]
	if not image_mode or source_is_image:
		_default_format = args["img_format"] if source_is_image else args["format"]
		if output_path:
			if output_path.is_dir():
				output_path = output_path / source_path.with_suffix(f".swapped.{_default_format}").name
				print(f"output_path is directory, saving to {str(output_path)!r}")
		else:
			output_path = source_path.with_suffix(f".swapped.{_default_format}")
		if not args["overwrite"]:
			ensure(not output_path.exists(), c = ("output_path exists", output_path))

	if output_path:
		print(f"saving to {str(output_path)!r}")
	else:
		print(f"no output path")

	if source_path.is_file():
		if source_is_image:
			process_img(
				face_path, source_path, output_path, args["gpu"], args["multi_face"],
				args["model_type"], args["model"], overwrite = args["overwrite"],
			)
			status("swap successful!")
			return
		vid_info = get_video_info(source_path, ffprobe = args["ffprobe"])
	else:
		vid_info = VidInfo(0, 0, args["fps_source"], False)
		if not (image_mode and (not output_path or args["no_output"])):
			ensure(vid_info.fps, c = ("source_path is png sequence, manually passing --fps_source framerate argument required"))

	with Timer("setgrad took {:.2f} secs"):
		import torch
		torch.set_grad_enabled(False)

	fps_target: int | None = args["fps_target"]
	if fps_target and vid_info.fps > fps_target:
		fps_use = fps_target
		fps_swapped = fps_target
		print("limiting fps to", fps_use)
	else:
		fps_use = None
		fps_swapped = vid_info.fps

	with Timer("full processing took {:.4f} secs"):
		if image_mode:
			process_image_mode(args, face_path, source_path, output_path, vid_info, fps_use, fps_swapped)
		else:
			ensure(not source_path.is_dir(), c = "directory sources not implemented for stream")
			process_streamed(args, face_path, source_path, output_path, vid_info, fps_use, fps_swapped)


def _split_shell_args(args):
	if not args:
		return []
	res = []
	for i in args:
		res.extend(shlex.split(i))
	return res


def _get_face(gen):
	for ctx, (frame, faces_cnt) in gen:
		yield frame


def _swap_settings(face_path: str | Path, args: dict):
	return SwapSettings(
		Path(face_path), args["multi_face"], args["model_type"], args["model"],
		args["gpu"], args["parallel_cpu"], args["parallel_gpu"], False,
	)


def process_streamed(
		args: dict, face_path: Path, source_path: Path,
		output_path: Path, vid_info: VidInfo,
		src_fps_output_max: int | float | None,  # fps to read from source, so drop frames if src fps is higher, None = use all frames.
		fps_output: int | float,  # fps that will be output, always same as fps_use if that isn't None
):
	# Note: cv2 VideoCapture is much faster (almost 2x) but has no way to easily skip frames,
	# would have to get frame timestamps and implement skipping by hand when using fps_use to always use, or assume constant fps.

	if args["cv2_reader"] or (args["ffmpeg_reader"] is False and src_fps_output_max is None):
		gen = _frame_gen_cv2(source_path)
	else:
		gen = _frame_gen_ffmpeg(args, source_path, vid_info.width, vid_info.height, src_fps_output_max)

	# _swap_gen wants each frame to be (some_identifier_or_ctx, frame) so use enumerate to just get pos
	# and then remove again afterwards for vid_save_gen that just wants frames
	gen = enumerate(gen)
	settings = _swap_settings(face_path, args)
	gen = parallel_process_gen(settings, gen)
	gen = _get_face(gen)
	vid_save_gen(args, source_path, output_path, vid_info, fps_output, gen)


def vid_save_gen(args, source_path: Path, output_path: Path, vid_info: VidInfo, fps_output: int | float, frame_gen):
	output_path_write, audio_source_path, audio_2stage, overwrite = _video_save(args, source_path, output_path, vid_info)

	_tmp_file_ctx = functools.partial(tmp_path_move_ctx, trail_org_ext = True, overwrite = overwrite, overwrite_delete_tmp = overwrite)
	_tmp_file_ctx = no_tmp_ctx if args["no_tmp"] else _tmp_file_ctx
	with _tmp_file_ctx(output_path_write) as tmp_path:
		pos_args = _parse_ffmpeg_args(args["ffmpeg_writer_args"] or [], fill = True)
		ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info"], pos_args = pos_args)
		create_video_from_frame_gen(
			frame_gen, vid_info.width, vid_info.height, fps_output, tmp_path,
			preset = args["preset"], crf = args["crf"], audio_source_path = audio_source_path, audio_shortest = args["audio_shortest"],
			**ffmpeg,
		)

	if audio_2stage:
		ensure(output_path != output_path_write)
		with _tmp_file_ctx(output_path) as tmp_path:
			ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info"])
			add_audio(output_path_write, source_path, tmp_path, shortest = args["audio_shortest"], **ffmpeg)


def _video_save(args, source_path: Path, output_path: Path, vid_info: VidInfo):
	vid_output_audio: bool = args["vid_output_audio"]
	overwrite = args["overwrite"]
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

	return output_path_gen, audio_source_path, audio_2stage, overwrite


def _parse_ffmpeg_args(ffmpeg_arg_list: list[tuple[str, str]], fill: bool | Iterable[int] = None) -> dict[int, list[str]]:
	by_num = collections.defaultdict(list) if fill is True else { }
	if fill and fill is not True:
		for i in fill or []:
			by_num.setdefault(i, [])

	for num, argument in ffmpeg_arg_list:
		if argument.strip():
			by_num.setdefault(int(num), []).extend(shlex.split(argument))

	return by_num


def _frame_gen_ffmpeg(args, source_path: Path, width, height, fps_to_output: int | float | None, ):
	ensure(width and height, c = (width, height))

	img_size = width * height * 3
	ffmpeg = [args["ffmpeg"], "-hide_banner", "-loglevel", "info"]
	fps = ["-filter:v", f"fps=fps={fps_to_output}"] if fps_to_output else []

	pos_args = _parse_ffmpeg_args(args["ffmpeg_reader_args"] or [], fill = True)
	com = [
		*ffmpeg,
		*pos_args[0],
		"-i", str(source_path),
		*pos_args[1],
		*fps,
		"-pix_fmt", "bgr24", "-f", "rawvideo",
		*pos_args[2],
		"pipe:",
		*pos_args[3],
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


def error_exit(message: str):
	print("ERROR:", message)
	exit(1)


def process_image_mode(
		args: dict, face_path: Path, source_path: Path,
		output_path: Path | None, vid_info: VidInfo,
		# fps for frames to take from source, None if using all,
		fps_use: int | float | None,
		fps_swapped: int | float,  # fps that will be output, always same as fps_use if that is given
):
	if args["work_dir"]:
		workdir = Path(args["work_dir"])
	else:
		if output_path:
			_workdir_name = f"{output_path.name}.tmp"
			if args["work_dir_root"]:
				workdir = Path(args["work_dir_root"])
				workdir = workdir / _workdir_name
			else:
				workdir = output_path.with_name(_workdir_name)
		elif args["work_dir_root"]:
			workdir = Path(args["work_dir_root"])
			_workdir_name = f"{source_path.name}.tmp"
			workdir = workdir / _workdir_name
		else:
			workdir = None

	name_suffix_org = args["name_suffix_org"]
	name_suffix_swapped = args["name_suffix_swapped"]
	ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info"])

	if source_path.is_file():
		if args["frames_dir"]:
			in_frames_dir = Path(args["frames_dir"])
		else:
			_root = args["frames_dir_root"] or workdir
			if not _root:
				exit(error_exit(
					"in image-mode without output_path, with video source_path given, "
					"one of work_dir or work_dir_root or frames_dir or frames_dir_root required to extract source_video frames to"
				))
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

	if output_path and not args["no_output"]:
		in_frame_paths = get_framepaths(in_frames_dir, name_suffix_org, ensure_continuous = True)
	else:
		extensions = (".png", ".jpg", ".jpeg", *([name_suffix_org] if name_suffix_org else []))
		in_frame_paths = get_imagepaths(in_frames_dir, extensions)

	status(f"got {len(in_frame_paths)} input frames/images total.")
	print(f"{in_frame_paths=}")

	with Timer("swap took {:.2f} secs"):
		if args["swapped_dir"]:
			swapped_frames_dir = Path(args["swapped_dir"])
			makedir(swapped_frames_dir, exist_ok = True, parents = False)
		else:
			_root = args["swapped_dir_root"] or workdir
			if not _root:
				exit(error_exit(
					"in image-mode without output_path, "
					"one of work_dir or work_dir_root or swapped_dir or swapped_dir_root required to write swapped frames to"
				))
			swapped_frames_dir = _root / f"f_swapped__{source_path.name}__F{fps_use or 'srcfps'}"
			parents = 2 if args["swapped_dir_root"] else 3
			makedir(swapped_frames_dir, exist_ok = True, parents = parents)

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
			try:
				import tqdm
				fp_todo_use = tqdm.tqdm(fp_todo)
			except ImportError:
				fp_todo_use = fp_todo

			settings = _swap_settings(face_path, args)

			gen = enumerate(fp_todo_use)
			gen = parallel_process_gen(settings, gen, True)
			for i in gen:
				pass
		else:
			status("skipping swapping, all finished already")

	if not output_path or args["no_output"]:
		status("swap successful, done!")
		return

	status("swap successful, creating video!")
	vid_save_frames(args, swapped_frames_dir, source_path, output_path, vid_info, fps_swapped)


def vid_save_frames(args, swapped_frames_dir: Path, source_path: Path, output_path: Path, vid_info: VidInfo, fps_output: int | float):
	output_path_write, audio_source_path, audio_2stage, overwrite = _video_save(args, source_path, output_path, vid_info)

	swapped_pat = name_pattern(args["name_suffix_swapped"])
	_tmp_file_ctx = functools.partial(tmp_path_move_ctx, trail_org_ext = True, overwrite = overwrite, overwrite_delete_tmp = overwrite)
	_tmp_file_ctx = no_tmp_ctx if args["no_tmp"] else _tmp_file_ctx
	with _tmp_file_ctx(output_path_write) as tmp_path:
		pos_args = _parse_ffmpeg_args(args["ffmpeg_writer_args"] or [], fill = True)
		ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info", ], pos_args = pos_args)
		create_video_with_audio(
			swapped_frames_dir, fps_output, tmp_path,
			audio_source_path = audio_source_path, audio_shortest = args["audio_shortest"],
			filename_pattern = swapped_pat,
			preset = args["preset"], crf = args["crf"],
			**ffmpeg,
		)

	if audio_2stage:
		ensure(output_path != output_path_write)
		with _tmp_file_ctx(output_path) as tmp_path:
			ffmpeg = dict(ffmpeg = args["ffmpeg"], extra_args = ["-hide_banner", "-loglevel", "info"])
			add_audio(output_path_write, source_path, tmp_path, shortest = args["audio_shortest"], **ffmpeg)


def make_parser():
	def make_add_argument(
			add_dash_to_underscore: bool,  # --what-ever -> --what_ever
			add_underscore_to_dash: bool,  # --what_ever -> --what-ever
			add_no_underscore: bool,  # --what_ever -> --whatever
			add_no_dash: bool,  # --what-ever -> --whatever
			keep_original: bool = True,
	):
		def _custom_add_argument(org_fn, *args, **kwargs):
			new = []
			for i in args:
				if isinstance(i, str) and i.startswith("--"):
					add = (
						i if keep_original else None,
						"--" + i[2:].replace("_", "-") if add_underscore_to_dash else None,
						"--" + i[2:].replace("-", "_") if add_dash_to_underscore else None,
						"--" + i[2:].replace("_", "") if add_no_underscore else None,
						"--" + i[2:].replace("-", "") if add_no_dash else None,
					)
					new.extend(i for i in add if i is not None)
				else:
					new.append(i)

			return org_fn(*new, **kwargs)

		return _custom_add_argument

	add_arg = make_add_argument(
		True, True, True, True, True
	)

	def patch_parser(parser):
		parser.add_argument = functools.partial(add_arg, parser.add_argument)

	def existing_path(p: str):
		path = Path(p)
		if not path.exists():
			raise argparse.ArgumentTypeError(f"path not found: {p!r}")
		return path

	parser = argparse.ArgumentParser()
	patch_parser(parser)
	parser.add_argument("-f", "--face", type = Path, required = True, help = "use this face")
	parser.add_argument("-s", "--source-vid", type = Path, required = True, help = "replace this face")
	parser.add_argument("-o", "--output-vid", type = Path, help = "save output to this file")
	parser.add_argument("-O", "--output-vid-formatted", help = "save output to this file with {} formatting")

	parser.add_argument("-M", "--model", type = existing_path)
	parser.add_argument("-T", "--model-type", choices = ["onnx", "onnx_local", "torch"], default = "onnx")

	parser.add_argument("-v", "--verbose", action = "store_true")
	parser.add_argument("-m", "--multi-face", action = "store_true")

	parser.add_argument("-l", "--local", action = "store_true",
						help = "always use opencv source reader")

	parser.add_argument("--image-mode", action = "store_true",
						help = "work with directories of images")

	parser.add_argument("-V", "--no-output", action = "store_true",
						help = "don't create video in image mode (even when output path given)")

	parser.add_argument("--no-tmp", action = "store_true",
						help = "work with directories of images")

	parser.add_argument("-y", "--overwrite", action = "store_true", help = "save output to this file with {} formatting")
	parser.add_argument("--gpu", action = "store_true",
						help = "use gpu")
	parser.add_argument("--keep-frames", action = "store_true",
						help = "keep frames directory")
	parser.add_argument("-r", "--fps-target", type = str_to_num,
						help = "maximum source fps wanted, will drop frames if source is higher fps, does nothing if source fps is lower")
	parser.add_argument("--fps-source", type = str_to_num,
						help = "source video fps, only needed for png sequence folder sources or maybe weird file formats")

	parser.add_argument("--crf", type = int, default = 15,
						help = "output crf")
	parser.add_argument("--preset", default = "superfast",
						help = "output preset")
	parser.add_argument("--audio-ff-args", action = "append",
						help = "extra audio merging ffmpeg args")

	parser.add_argument("--audio-shortest", action = "store_true",
						help = "shorten audio file to vid length, should only be needed if seeking with -XYZ")

	parser.add_argument("--ffmpeg-reader", action = "store_true",
						help = "always use ffmpeg source reader")
	parser.add_argument("--cv2-reader", action = "store_true",
						help = "always use opencv source reader")

	parser.add_argument("-R", "--ffmpeg-reader-args", nargs = 2, action = "append",
						help = "format: position_idx argument. arguments passed to ffmpeg reader. Note: add space in front if it starts with -")
	parser.add_argument("-W", "--ffmpeg-writer-args", nargs = 2, action = "append",
						help = "arguments passed to ffmpeg reader after input. Note: add space in front if it starts with -")

	formatarg = lambda inp: inp.lstrip(".").strip()
	parser.add_argument("-F", "--format", default = "mp4", type = formatarg, help = "video container to use, default: mp4")
	parser.add_argument("-I", "--img-format", default = "png", type = formatarg, help = "image container to use, default: png")
	parser.add_argument("--plain-format", type = formatarg, help = "video container to use for plain files")

	parser.add_argument("--name-suffix-org", default = DEFAULT_FRAME_SUFFIX_ORG,
						help = "suffix (including extension) of original frame names")
	parser.add_argument("--name-suffix-swapped", default = DEFAULT_FRAME_SUFFIX_SWAPPED,
						help = "suffix (including extension) of original frame names")

	parser.add_argument("--frames-dir", type = existing_path, help = "source frames tmp dir")
	parser.add_argument("--frames-dir-root", type = existing_path, help = "source frames tmp root dir")

	parser.add_argument("--swapped-dir", type = Path, help = "swapped tmp dir")
	parser.add_argument("--swapped-dir-root", type = existing_path, help = "swapped tmp root dir")

	parser.add_argument("--work-dir", type = Path, help = "work tmp dir")
	parser.add_argument("--work-dir-root", type = existing_path, help = "work tmp root dir")

	parser.add_argument("-A", "--no-audio", dest = "vid_output_audio", action = "store_false",
						help = "dont try to copy audio from source")
	parser.add_argument("-P", "--no-plain", dest = "vid_output_plain", action = "store_false",
						help = "dont create plaint output video (.plain.mp4 file without original audio)")
	parser.add_argument("-d", "--direct-audio", action = "store_true",
						help = "add audio directly to output file in one go (instead of plain file and then 2nd separate file with audio merged in)")

	parser.add_argument("-S", "--redo-swapped", action = "store_true",
						help = "always redo any already swapped images")

	parser.add_argument("-C", "--redo-completed-swap", action = "store_true",
						help = "redo swapping if it has been fully completed")

	parser.add_argument("--max-memory", default = 16, type = int, help = "set max memory")
	parser.add_argument("--parallel-cpu", type = int, default = max(psutil.cpu_count(logical = True), 1),
						help = "number of cores to use")
	parser.add_argument("--parallel-gpu", type = int, default = 1,
						help = "number of instance to run in parallel on the GPU, seems faster on some GPU if enough mem")

	parser.add_argument("--ffprobe", default = "ffprobe", help = "ffprobe command/path")
	parser.add_argument("--ffmpeg", default = "ffmpeg", help = "ffprobe command/path")

	def process_args(args):
		pass

	@functools.wraps(parser.parse_args)
	def parse_args(fn, *pargs, **pkwargs):
		oargs = fn(*pargs, **pkwargs)

		args = { }
		for name, value in vars(oargs).items():
			args[name] = value

		process_args(args)
		return args

	parser.parse_args_dict = functools.partial(parse_args, parser.parse_args)

	return parser


if __name__ == "__main__":
	def setup():
		parser = make_parser()
		args = parser.parse_args_dict()
		logging.basicConfig(level = logging.DEBUG if args["verbose"] else logging.INFO)
		if args["verbose"]:
			for k, v in args.items():
				print(f"{k + '':<20} {v!r}")
			print()
		start(args)


	setup()
