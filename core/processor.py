import os
from pathlib import Path

import cv2
import insightface
from core.utils import write_atomic
import core.globals

FACE_SWAPPER = None
FACE_ANALYSER = None


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


def process_video(source_img: Path, frame_paths: list[tuple[Path, Path]], load_own_model: bool, skip_existing: bool):
	source_face = get_face(cv2.imread(str(source_img)))
	if not frame_paths:
		return

	if load_own_model:
		# needed to run multiple at the same time on the GPU, seems to give better utilization on some
		swapper = load_face_swapper()
	else:
		swapper = get_face_swapper()

	for (src_frame_path, target_frame_path) in frame_paths:
		if skip_existing and target_frame_path.exists():
			print("R", end = '', flush = True)
			continue

		frame = cv2.imread(str(src_frame_path))
		try:
			face = get_face(frame)
		except Exception as ex:
			print('F', end = '', flush = True)
			continue

		if face:
			try:
				# result = get_face_swapper().get(frame, face, source_face, paste_back = True)
				result = swapper.get(frame, face, source_face, paste_back = True)
			except Exception as ex:
				print('E', end = '', flush = True)
				continue

			is_ok, buffer = cv2.imencode(".png", result)
			if not is_ok:
				raise ValueError("failed encoding image??")

			write_atomic(buffer, target_frame_path, may_exist = False)
			print('.', end = '', flush = True)
		else:
			print('S', end = '', flush = True)


def process_img(source_img, target_path, output_file):
	frame = cv2.imread(target_path)
	face = get_face(frame)
	source_face = get_face(cv2.imread(source_img))
	result = get_face_swapper().get(frame, face, source_face, paste_back = True)
	cv2.imwrite(output_file, result)
