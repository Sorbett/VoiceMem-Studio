"""Controllable executor for snapshot scheduling without native models or sleeps."""
from concurrent.futures import Future


class ManualExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, operation):
        future = Future()
        self.jobs.append((future, operation))
        return future

    def start_next(self):
        future, operation = self.jobs.pop(0)
        if not future.set_running_or_notify_cancel():
            return lambda: None

        def finish():
            try:
                future.set_result(operation())
            except BaseException as exc:
                future.set_exception(exc)
        return finish

    def run_next(self):
        self.start_next()()
