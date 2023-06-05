from __future__ import annotations

import contextlib
import errno
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Union, Optional, Any, TypeVar
from collections.abc import Iterable, Collection, Sequence, Callable, Generator, AsyncGenerator


def str_to_num(num: str) -> int | float:
	if "." in num or "e" in num:
		return float(num)
	return int(num)


def to_num(num: str | int | float) -> int | float:
	if isinstance(num, str):
		if "." in num or "e" in num:
			return float(num)
		return int(num)

	if isinstance(num, int) or isinstance(num, float):
		return num

	raise ValueError("not a num", num)


def run_command(command, mode = "silent"):
	if mode == "debug":
		return os.system(command)
	return os.popen(command).read()


@dataclass
class VidInfo():
	width: int
	height: int
	fps: int | float
	has_audio: bool


def get_video_info(vid_path: str | Path, ffprobe = "ffprobe"):
	obj = subprocess.check_output([
		ffprobe, '-v', 'error', '-show_entries', 'stream=width,height,r_frame_rate,codec_type',
		str(vid_path), '-of', 'default=noprint_wrappers=1', '-print_format', 'json'])
	obj = json.loads(obj)
	streams = obj["streams"]
	by_type = { }
	for i in streams:
		by_type.setdefault(i["codec_type"], []).append(i)

	vid_streams = by_type.get("video")
	ensure(vid_streams, c = obj)
	ensure(len(vid_streams) == 1, c = ("expected 1 video stream, got multiple", len(vid_streams), vid_streams))

	vid = vid_streams[0]
	width, height = vid["width"], vid["height"]
	ensure(width and isinstance(width, int), c = repr(width))
	ensure(height and isinstance(height, int), c = repr(height))

	a, b = map(str_to_num, vid["r_frame_rate"].split("/"))
	fps = a / b if b != 1 else a

	audio = by_type.get("audio")
	return VidInfo(width, height, fps, bool(audio))


def make_temp_name():
	return f"{time.time_ns()}.{random.random()}.tmp"


def write_atomic(data: bytes, target: str | Path,
				 random_name_fallback: bool = True,
				 may_exist: bool = False):
	if not may_exist and target.exists():
		raise ValueError("target exists", target)

	target = Path(target)
	try:
		tmp_file = target.with_name(target.name + ".tmp")
		tmp_file.write_bytes(data)
	except OSError as e:
		if not (random_name_fallback and e.errno == errno.ENAMETOOLONG):
			raise
		# name was too long, try with shortened cut + random stuff
		tmp_file = target.with_name(target.name[:70] + make_temp_name())
		tmp_file.write_bytes(data)

	if not may_exist and target.exists():
		raise ValueError("target exists", target)

	os.rename(tmp_file, target)


def extract_frames(input_path, output_dir: Path, target_fps: Optional[int] = None,
				   filename_pattern = "%05d.png", ffmpeg = "ffmpeg", extra_args = None):
	fps = ["-filter:v", f"fps=fps={target_fps}"] if target_fps else []
	subprocess.check_call([ffmpeg, "-n", *(extra_args or []), "-i", str(input_path), *fps, str(output_dir / filename_pattern)])


def create_video_from_frame_gen(
		frame_gen: Iterable[bytes], width: int, height: int, fps: int | float, target: Path,
		audio_source_path: Path | str | None = None, audio_shortest: bool = False,
		crf: int | None = None, preset: str | None = None,
		finish_timeout = 60, check = True,
		ffmpeg = "ffmpeg", extra_args = None, pos_args: dict[int, list[str]] | None = None,
):

	preset = ["-preset", preset] if preset else []
	crf = ["-crf", str(crf)] if crf else []

	pos_args = pos_args or { }
	audio = []
	if audio_source_path is not None:
		audio = [
			"-i", str(audio_source_path),
			*(["-shortest"] if audio_shortest else []),
			*(pos_args.get(2) or []),
			"-map", "0:v:0", "-map", "1:a:0",
		]

	com = [
		ffmpeg, "-n" if str(target) != "/dev/null" else "-y", *(extra_args or []),
		*(pos_args.get(0) or []),
		'-f', 'rawvideo', "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}", "-framerate", str(fps), '-i', '-',
		*(pos_args.get(1) or []),
		*audio,
		*(pos_args.get(3) or []),
		"-c:v", "libx264", *preset, *crf, "-pix_fmt", "yuv420p", "-r", str(fps),
		*(pos_args.get(4) or []),
		str(target),
		*(pos_args.get(5) or []),
	]
	proc = subprocess.Popen(com, stdin = subprocess.PIPE, bufsize = 16 * 1024 ** 2)

	for pos, frame_buf in enumerate(frame_gen):
		proc.stdin.write(frame_buf)

	proc.stdin.close()
	exitcode = proc.wait(finish_timeout)
	if check:
		ensure(exitcode == 0)


def add_audio(video_path: Path, audio_source_path: Path, target_path: Path, ffmpeg = "ffmpeg", extra_args = None, shortest = False):
	# TODI: fix timestamps?
	subprocess.check_output([
		ffmpeg, "-n", *(extra_args or []),
		"-i", str(video_path),
		"-i", str(audio_source_path),
		*(["-shortest"] if shortest else []),
		"-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
		str(target_path),
	])


def create_video_with_audio(
		frames_dir: Path, fps: int | float, target: Path,
		audio_source_path: Path | str = None, audio_shortest: bool = False,
		filename_pattern = "%05d.png",
		crf: int | None = None, preset: str | None = None,
		ffmpeg = "ffmpeg", extra_args = None, pos_args: dict[int, list[str]] | None = None,
):
	preset = ["-preset", preset] if preset else []
	crf = ["-crf", str(crf)] if crf else []

	pos_args = pos_args or { }
	audio = []
	if audio_source_path is not None:
		audio = [
			"-i", str(audio_source_path),
			*(["-shortest"] if audio_shortest else []),
			*(pos_args.get(2) or []),
			"-map", "0:v:0", "-map", "1:a:0",
		]

	com = [
		ffmpeg, "-n", *(extra_args or []),
		*(pos_args.get(0) or []),
		"-framerate", str(fps),
		"-i", str(frames_dir / filename_pattern),
		*(pos_args.get(1) or []),
		*audio,
		*(pos_args.get(3) or []),
		"-c:v", "libx264", *preset, *crf, "-pix_fmt", "yuv420p",
		*(pos_args.get(4) or []),
		str(target),
		*(pos_args.get(5) or []),
	]
	subprocess.check_output(com)


def noop(*args, **kwargs):
	pass


def is_img(path):
	path = str(path)
	return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))


def ensure(val, *, c = None):
	# ensure value is truthy
	if not val:
		raise ValueError("ensure error", type(val), val, c)


def ensure_equal(left, right, c = None):
	# ensure values are == equal
	if left != right:
		raise ValueError("ensure_equal error, left != right",
						 type(left), type(right), left, right, c)


def _with_ext(path: str | Path, ext: str, keep_org_ext: bool = True, trail_org_ext: bool = False):
	# not using with_suffix as that would require ext to start with a "."
	if keep_org_ext:
		if trail_org_ext:
			# name.tmp.ext"
			return path.with_name(path.stem + ext + path.suffix)
		else:
			# name.ext.tmp"
			return path.with_name(path.name + ext)
	else:
		# name.tmp
		return path.with_name(path.stem + ext)


@contextlib.contextmanager
def tmp_path_move_ctx(
		path: str | Path,
		tmp_ext: str = ".tmp",
		keep_org_ext: bool = True,
		trail_org_ext: bool = False,

		overwrite: bool = False,
		overwrite_path: bool = False,
		overwrite_tmp: bool = True,

		overwrite_delete: bool = False,
		overwrite_delete_path: bool = False,
		overwrite_delete_tmp: bool = False,

		move_on_error: bool | str = False,
		ignore_error: bool = False,
):
	ensure(tmp_ext)
	path = Path(path)

	tmp_path = _with_ext(path, tmp_ext, keep_org_ext, trail_org_ext)
	ensure(path != tmp_path)

	if not (overwrite or overwrite_path) and path.exists():
		raise FileExistsError(path)

	if not (overwrite or overwrite_tmp) and tmp_path.exists():
		raise FileExistsError(tmp_path)

	if overwrite and (overwrite_delete or overwrite_delete_path) and path.exists():
		path.unlink(False)

	if overwrite and (overwrite_delete or overwrite_delete_tmp) and tmp_path.exists():
		tmp_path.unlink(False)

	def _move(target: Path):
		if not overwrite and target.exists():
			raise FileExistsError(target)
		os.rename(tmp_path, target)

	try:
		yield tmp_path
	except:
		# TODI: multi exception in case of move error?
		if move_on_error is True:
			_move(path)
		elif isinstance(move_on_error, str):
			err_path = _with_ext(path, tmp_ext, keep_org_ext, trail_org_ext)
			_move(err_path)

		if not ignore_error:
			raise
	else:
		_move(path)


class Timer():
	def __init__(self,
				 print_format: Optional[Union[str, bool]] = None,
				 start: bool = False,
				 print_fn: Callable[[str], Any] = print,
				 time_fn: Callable[[], float] = time.monotonic,
				 ):
		if isinstance(print_format, bool):
			if print_format is True:
				print_format = "{secs}s"
			else:
				print_format = None

		self.print_format = print_format
		self.print_fn = print_fn
		self.time_fn = time_fn

		# start/end timestamps, by default monotonic time unless time_fn is set
		self.start_ts = None
		self.end_ts = None
		self.seconds: Optional[float] = None

		if start:
			self.start()

	def start(self):
		self.start_ts = self.time_fn()

	def end(self, run: bool = True) -> float:
		self.end_ts = self.time_fn()
		self.seconds = self.end_ts - self.start_ts

		if run:
			self._end()

		return self.seconds

	def _end(self):
		if self.print_format is not None:
			self.print_fn(self.print_format.format(
				self.seconds, self.seconds, self.seconds, self.seconds, self.seconds, self.seconds,
				s = self.seconds, secs = self.seconds, seconds = self.seconds,
				start = self.start_ts, end = self.end_ts,
			))

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, *args):
		self.end()
