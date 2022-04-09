from abc import ABC, abstractmethod
from typing import Any, List


class CallableFunc(ABC):

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
