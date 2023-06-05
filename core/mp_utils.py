import threading
import time


def _make_limited_gen(gen, qlen_max, sleeptime):
	cnt = 0
	lock = threading.RLock()

	def read_gen(gen):
		nonlocal cnt
		for i in gen:
			while True:
				with lock:
					# print("read cnt", cnt)
					if cnt < qlen_max:
						cnt += 1
						break

				# print(f"read cnt max reached, waiting {cnt} >= {qlen_max}", )
				time.sleep(sleeptime)

			# if too many unfinished, wait
			yield i

	def finished_gen(gen):
		nonlocal cnt
		for i in gen:
			# mark finished
			with lock:
				cnt -= 1
			yield i

	return read_gen(gen), finished_gen


def _make_wait_fn(imap_gen, qlen_max, sleeptime):
	def wait_fn():
		while True:
			qlen = len(imap_gen._items)
			# print("qlen", qlen)
			if qlen <= qlen_max:
				break
			# print(f"waiting for queue to get shorter: {qlen} > {qlen_max}, proc {os.getpid()} thread: {threading.currentThread().ident}")
			time.sleep(sleeptime)

	return wait_fn


def imap_backpressure(
		imap, fn, gen,
		read_qlen_max = None,
		read_qsleep = 0.01,

		task_qlen_max = None,
		task_qsleep = 0.01,
):
	wait_fn = None

	if task_qlen_max is not None:
		def wrapper(*args, **kwargs):
			wait_fn()
			return fn(*args, **kwargs)
	else:
		wrapper = fn

	if read_qlen_max is not None:
		gen, finished_gen = _make_limited_gen(gen, read_qlen_max, read_qsleep)
		gen = imap_gen = imap(wrapper, gen)
		gen = finished_gen(gen)
	else:
		gen = imap_gen = imap(fn, gen)

	if task_qlen_max is not None:

		wait_fn = _make_wait_fn(imap_gen, task_qlen_max, task_qsleep)

	return gen
