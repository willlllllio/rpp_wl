import errno
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Optional


def run_command(command, mode = "silent"):
	if mode == "debug":
		return os.system(command)
	return os.popen(command).read()


def detect_fps(input_path, ffprobe = "ffprobe"):
	output = subprocess.check_output([
		ffprobe, "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1",
		"-show_entries", "stream=r_frame_rate", str(input_path)])

	output = output.decode()
	try:
		return int(output.split("/")[0]) // int(output.split("/")[1])
	except:
		raise ValueError("couldn't get fps", output)


def detect_dimensions(input_path: Path, ffprobe = "ffprobe"):
	output = subprocess.check_output([
		ffprobe, "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
		str(input_path)]
	)

	output = output.decode()
	w, h = output.strip().split("x")
	return int(w), int(h)


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


def create_video(frames_dir: Path, fps: int | float, target: Path, filename_pattern = "%05d.png", ffmpeg = "ffmpeg", extra_args = None):
	subprocess.check_output([
		ffmpeg, "-n", *(extra_args or []),
		"-framerate", str(fps),
		"-i", frames_dir / filename_pattern,
		"-c:v", "libx264", "-crf", "7", "-pix_fmt", "yuv420p", str(target)
	])


def create_video_from_frame_gen(frame_gen, width: int, height: int, fps: int | float, target: Path, ffmpeg = "ffmpeg", extra_args = None,
								finish_timeout = 10):
	proc = subprocess.Popen([
		ffmpeg, "-n", *(extra_args or []),
		'-f', 'rawvideo', "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}", "-framerate", str(fps), '-i', '-',
		"-c:v", "libx264", "-crf", "7", "-pix_fmt", "yuv420p", "-r", str(fps), str(target),
	],
		stdin = subprocess.PIPE
	)

	for pos, frame_buf in enumerate(frame_gen):
		proc.stdin.write(frame_buf)

	proc.stdin.close()
	proc.wait(finish_timeout)


def add_audio(video_path: Path, audio_source_path: Path, target_path: Path, ffmpeg = "ffmpeg", extra_args = None):
	subprocess.check_output([
		ffmpeg, "-n", *(extra_args or []),
		"-i", str(video_path), "-i", str(audio_source_path), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
		str(target_path)
	])


def create_video_with_audio(frames_dir: Path, fps: int | float, audio_source_path: Path, target: Path,
							filename_pattern = "%05d.png", ffmpeg = "ffmpeg", extra_args = None):
	subprocess.check_output([
		ffmpeg, "-n", *(extra_args or []),
		"-framerate", str(fps), "-i", frames_dir / filename_pattern,
		"-i", str(audio_source_path),
		"-c:v", "libx264", "-crf", "7", "-pix_fmt", "yuv420p",
		"-map", "0:v:0", "-map", "1:a:0",
		str(target)
	])


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


from typing import Union, Optional, Any, TypeVar
from collections.abc import Iterable, Collection, Sequence, Callable, Generator, AsyncGenerator


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
