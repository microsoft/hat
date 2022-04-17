from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
from functools import wraps
from typing import Any, List

from numpy import isin

def _sigsev_guard(fn):
    "Decorator that makes decorated function execute in a forked process"

    @wraps(fn)
    def wrapper(*args, **kwargs):
        q = mp.Queue()
        p = mp.Process(target=lambda q: q.put(fn(*args, **kwargs)), args=(q,))
        p.start()
        p.join()

        exit_code = p.exitcode

        if exit_code == 0:
            return q.get()
        else:
            logging.warning('Process did not exit correctly. Exit code: {}'.format(exit_code))

            result = q.get(block=False)
            if isinstance(result, BaseException):
                raise result

        return None

    return wrapper


class CallableFunc(ABC):

    @_sigsev_guard
    def __call__(self, *args: Any) -> float:
        try:
            self.init_runtime()
            try:
                self.init_main(args=args)
                timings: List[float] = self.main(args=args)
            finally:
                self.cleanup_main(args=args)
        finally:
            self.cleanup_runtime()

        return timings[0]

    @_sigsev_guard
    def benchmark(self, warmup_iters, iters, batch_size, args) -> List[float]:
        try:
            self.init_runtime()
            try:
                self.init_main(warmup_iters=warmup_iters, args=args)
                timings = self.main(iters=iters, batch_size=batch_size, args=args)
            finally:
                self.cleanup_main(args=args)
        finally:
            self.cleanup_runtime()
        return timings

    @abstractmethod
    def init_runtime(self):
        ...

    @abstractmethod
    def cleanup_runtime(self):
        ...

    @abstractmethod
    def init_main(self, warmup_iters=0, *args) -> float:
        ...

    @abstractmethod
    def cleanup_main(self, *args):
        ...

    @abstractmethod
    def main(self, iters=1, *args: Any):
        ...
