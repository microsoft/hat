from abc import ABC, abstractmethod
from typing import Any, List


class CallableFunc(ABC):

    def __call__(self, *args: Any, gpu_id: int = 0) -> float:
        try:
            self.init_runtime(benchmark=False, gpu_id=gpu_id)
            try:
                self.init_main(benchmark=False, args=args, gpu_id=gpu_id)
                timings: List[float] = self.main(benchmark=False, args=args)
            finally:
                self.cleanup_main(benchmark=False, args=args)
        finally:
            self.cleanup_runtime(benchmark=False)

        return timings[0]

    def benchmark(self, warmup_iters, iters, batch_size, args, gpu_id: int) -> List[float]:
        try:
            self.init_runtime(benchmark=True, gpu_id=gpu_id)
            try:
                self.init_main(benchmark=True, warmup_iters=warmup_iters, args=args, gpu_id=gpu_id)
                timings = self.main(benchmark=True, iters=iters, batch_size=batch_size, args=args)
            finally:
                self.cleanup_main(benchmark=True, args=args)
        finally:
            self.cleanup_runtime(benchmark=True)
        return timings

    @abstractmethod
    def init_runtime(self, benchmark: bool, gpu_id: int):
        ...

    @abstractmethod
    def init_main(self, benchmark: bool, warmup_iters=0, *args) -> float:
        ...

    @abstractmethod
    def main(self, benchmark: bool, iters=1, *args: Any):
        ...

    @abstractmethod
    def cleanup_main(self, benchmark: bool, *args):
        ...

    @abstractmethod
    def cleanup_runtime(self, benchmark: bool):
        ...
