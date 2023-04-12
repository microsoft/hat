from abc import ABC, abstractmethod
from typing import Any, List


class CallableFunc(ABC):

    def __call__(self, *args: Any, device_id: int = 0, working_dir: str=None) -> float:
        try:
            self.init_runtime(benchmark=False, device_id=device_id, working_dir=working_dir)
            try:
                self.init_batch(benchmark=False, args=args, device_id=device_id)
                timing = self.run_batch(benchmark=False, iters=1, args=args)
            finally:
                self.cleanup_batch(benchmark=False, args=args)
        finally:
            self.cleanup_runtime(benchmark=False, working_dir=working_dir)

        return timing

    def benchmark(self, warmup_iters, iters, batch_size, min_time_in_sec: int, args, device_id: int, working_dir: str) -> List[float]:
        try:
            self.init_runtime(benchmark=True, device_id=device_id, working_dir=working_dir)
            try:
                self.init_batch(benchmark=True, warmup_iters=warmup_iters, args=args, device_id=device_id)

                # Run multiple batches
                batch_timings_ms: List[float] = []
                min_time_in_ms = min_time_in_sec * 1000
                iterations = 0
                while True:
                    batch_time_ms = self.run_batch(benchmark=True, iters=iters, args=args)
                    batch_timings_ms.append(batch_time_ms)
                    iterations += iters

                    if sum(batch_timings_ms) >= min_time_in_ms and len(batch_timings_ms) >= batch_size:
                        break

                mean_elapsed_time_ms = sum(batch_timings_ms) / iterations
            finally:
                self.cleanup_batch(benchmark=True, args=args)
        finally:
            self.cleanup_runtime(benchmark=True, working_dir=working_dir)
        return mean_elapsed_time_ms, batch_timings_ms

    @abstractmethod
    def init_runtime(self, benchmark: bool, device_id: int, working_dir: str):
        ...

    @abstractmethod
    def init_batch(self, benchmark: bool, warmup_iters=0, device_id: int=0, *args):
        ...

    @abstractmethod
    def run_batch(self, benchmark: bool, iters: int, *args: Any) -> float:
        ...

    @abstractmethod
    def cleanup_batch(self, benchmark: bool, *args):
        ...

    @abstractmethod
    def cleanup_runtime(self, benchmark: bool, working_dir: str):
        ...

    @abstractmethod
    def should_flush_cache(self) -> bool:
        ...