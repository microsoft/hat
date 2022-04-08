from abc import ABC, abstractmethod
from typing import Any


class CallableFunc(ABC):

    def __call__(self, *args: Any) -> float:
        time = -1.
        try:
            self.init_runtime()
            try:
                self.init_main(args=args)
                time: float = self.main(args=args)
            finally:
                self.cleanup_main(args=args)
        finally:
            self.cleanup_runtime()

        return time

    def benchmark(self, args):
        time = -1.
        try:
            self.init_runtime()
            try:
                self.init_main(warmup_iters=1, args=args)
                time: float = self.main(iters=100, args=args)
            finally:
                self.cleanup_main(args=args)
        finally:
            self.cleanup_runtime()
        return time

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
