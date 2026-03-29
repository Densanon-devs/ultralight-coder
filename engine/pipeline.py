"""
Async Pipeline — Phase 3 Performance

Thread-based pipeline that parallelizes I/O-bound operations and
provides a generation queue for multi-request support.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  User Prompt                                            │
    │       ↓                                                 │
    │  ┌─────────┐                                            │
    │  │ Router  │ (fast, runs inline)                        │
    │  └────┬────┘                                            │
    │       ↓                                                 │
    │  ┌─────────────── PARALLEL I/O ──────────────────┐      │
    │  │                                                │      │
    │  │  Thread 1: Module Loading + LoRA swap          │      │
    │  │  Thread 2: Memory Retrieval (all tiers)        │      │
    │  │  Thread 3: Predictive Pre-loading              │      │
    │  │                                                │      │
    │  └────────────────────┬───────────────────────────┘      │
    │                       ↓                                  │
    │  ┌─────────────┐                                        │
    │  │ Fusion      │ (assembles prompt from parallel results)│
    │  └──────┬──────┘                                        │
    │         ↓                                                │
    │  ┌─────────────────────────────────────┐                │
    │  │  Generation Queue (thread-safe)     │                │
    │  │  • Accepts requests while generating │                │
    │  │  • FIFO with priority support       │                │
    │  │  • Single worker (model is not      │                │
    │  │    thread-safe)                     │                │
    │  └──────┬──────────────────────────────┘                │
    │         ↓                                                │
    │  Response returned via Future                            │
    └─────────────────────────────────────────────────────────┘

Why threads not asyncio:
    - llama-cpp-python is a C extension that releases the GIL during inference
    - Module I/O (disk reads, YAML parsing) benefits from threading
    - Memory retrieval (JSON search) benefits from threading
    - Generation is truly CPU-bound but GIL-free via C bindings

The pipeline is optional — the engine can still run synchronously.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from queue import Queue, PriorityQueue
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a parallel I/O phase."""
    active_modules: list = field(default_factory=list)
    memory_context: dict = field(default_factory=dict)
    system_knowledge: Optional[str] = None
    lora_swapped: bool = False
    preloaded_modules: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(order=True)
class GenerationRequest:
    """A request in the generation queue."""
    priority: int                       # Lower = higher priority (0 = NPC, 1 = user, 2 = background)
    timestamp: float = field(compare=False, default_factory=time.time)
    prompt: str = field(compare=False, default="")
    max_tokens: Optional[int] = field(compare=False, default=None)
    temperature: Optional[float] = field(compare=False, default=None)
    callback: Optional[Callable] = field(compare=False, default=None)
    request_id: str = field(compare=False, default="")
    extra_kwargs: dict = field(compare=False, default_factory=dict)
    _future: Optional[Future] = field(compare=False, default=None, repr=False)


class ParallelIO:
    """
    Runs module loading, memory retrieval, and pre-loading in parallel.

    Usage:
        parallel = ParallelIO(max_workers=3)
        result = parallel.execute(
            module_loader=lambda: module_manager.get_multiple(['code', 'math']),
            memory_retriever=lambda: memory.recall('user prompt'),
            preloader=lambda: module_manager.preload(['json_format']),
        )
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="pie-io",
        )

    def execute(
        self,
        module_loader: Optional[Callable] = None,
        memory_retriever: Optional[Callable] = None,
        lora_swapper: Optional[Callable] = None,
        preloader: Optional[Callable] = None,
        knowledge_loader: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Execute I/O tasks in parallel and collect results.

        All callables are optional — only submitted if provided.
        Returns PipelineResult with all collected data + timing info.
        """
        result = PipelineResult()
        futures = {}
        start = time.monotonic()

        # Submit tasks
        if module_loader:
            futures["modules"] = self._executor.submit(self._timed_call, "modules", module_loader)
        if memory_retriever:
            futures["memory"] = self._executor.submit(self._timed_call, "memory", memory_retriever)
        if lora_swapper:
            futures["lora"] = self._executor.submit(self._timed_call, "lora", lora_swapper)
        if preloader:
            futures["preload"] = self._executor.submit(self._timed_call, "preload", preloader)
        if knowledge_loader:
            futures["knowledge"] = self._executor.submit(self._timed_call, "knowledge", knowledge_loader)

        # Collect results
        for task_name, future in futures.items():
            try:
                task_result, elapsed = future.result(timeout=10.0)
                result.timings[task_name] = elapsed

                if task_name == "modules":
                    result.active_modules = task_result or []
                elif task_name == "memory":
                    result.memory_context = task_result or {}
                elif task_name == "lora":
                    result.lora_swapped = bool(task_result)
                elif task_name == "preload":
                    result.preloaded_modules = task_result or []
                elif task_name == "knowledge":
                    result.system_knowledge = task_result

            except Exception as e:
                logger.error(f"Parallel task '{task_name}' failed: {e}")
                result.timings[task_name] = -1

        result.timings["total_parallel"] = time.monotonic() - start
        logger.debug(f"Parallel I/O completed: {result.timings}")

        return result

    def _timed_call(self, name: str, func: Callable):
        """Run a callable and return (result, elapsed_seconds)."""
        start = time.monotonic()
        try:
            result = func()
            elapsed = time.monotonic() - start
            logger.debug(f"Task '{name}' completed in {elapsed:.3f}s")
            return result, elapsed
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(f"Task '{name}' failed after {elapsed:.3f}s: {e}")
            raise

    def shutdown(self):
        """Clean shutdown of thread pool."""
        self._executor.shutdown(wait=True)


class GenerationQueue:
    """
    Thread-safe generation queue for multi-request support.

    Supports:
        - Priority queue (NPC requests > user requests > background)
        - Single worker thread (model is not thread-safe)
        - Future-based result retrieval
        - Queue depth monitoring
        - Graceful shutdown

    Usage (NPC game mode with multiple actors):
        queue = GenerationQueue(model=base_model)
        queue.start()

        future1 = queue.submit(prompt="Noah says...", priority=0)
        future2 = queue.submit(prompt="Player asks...", priority=1)

        response1 = future1.result()  # blocks until done
        response2 = future2.result()
    """

    # Priority levels
    PRIORITY_NPC = 0        # Game NPCs — lowest latency
    PRIORITY_USER = 1       # Interactive user prompts
    PRIORITY_BACKGROUND = 2 # Background tasks (memory compression, etc.)

    def __init__(self, generate_fn: Optional[Callable] = None):
        """
        Args:
            generate_fn: The generation function to call.
                         Signature: (prompt, max_tokens, temperature) -> str
        """
        self._generate_fn = generate_fn
        self._queue: PriorityQueue[GenerationRequest] = PriorityQueue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._request_count = 0
        self._lock = threading.Lock()

        # Stats
        self._total_generated = 0
        self._total_generation_time = 0.0

    def start(self):
        """Start the generation worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="pie-gen-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Generation queue worker started")

    def stop(self):
        """Stop the generation worker thread."""
        self._running = False
        # Push a sentinel to unblock the worker
        sentinel = GenerationRequest(priority=999, prompt="__SHUTDOWN__")
        self._queue.put(sentinel)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        logger.info("Generation queue worker stopped")

    def submit(
        self,
        prompt: str,
        priority: int = PRIORITY_USER,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> Future:
        """
        Submit a generation request to the queue.

        Returns a Future that will contain the generated text.
        """
        future = Future()

        with self._lock:
            self._request_count += 1
            request_id = f"req-{self._request_count}"

        request = GenerationRequest(
            priority=priority,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            callback=callback,
            request_id=request_id,
            extra_kwargs=kwargs,
            _future=future,
        )

        self._queue.put(request)
        logger.debug(f"Queued generation request {request_id} (priority={priority}, queue_depth={self._queue.qsize()})")

        return future

    def generate_sync(self, prompt: str, max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None, **kwargs) -> str:
        """
        Synchronous generation — bypasses the queue.
        Used when the queue isn't running or for simple single-request mode.
        """
        if self._generate_fn is None:
            return "[No generation function configured]"

        return self._generate_fn(prompt, max_tokens, temperature, **kwargs)

    def _worker_loop(self):
        """Main worker loop — processes requests from the queue."""
        logger.debug("Generation worker loop started")

        while self._running:
            try:
                request = self._queue.get(timeout=1.0)

                if request.prompt == "__SHUTDOWN__":
                    break

                logger.debug(f"Processing {request.request_id} (priority={request.priority})")
                start = time.monotonic()

                try:
                    if self._generate_fn:
                        response = self._generate_fn(
                            request.prompt,
                            request.max_tokens,
                            request.temperature,
                            **request.extra_kwargs,
                        )
                    else:
                        response = "[No generation function configured]"

                    elapsed = time.monotonic() - start
                    self._total_generated += 1
                    self._total_generation_time += elapsed

                    logger.debug(
                        f"Generated {request.request_id} in {elapsed:.2f}s "
                        f"({len(response)} chars)"
                    )

                    # Set the future result
                    if request._future:
                        request._future.set_result(response)

                    # Call callback if provided
                    if request.callback:
                        try:
                            request.callback(response)
                        except Exception as e:
                            logger.error(f"Callback error for {request.request_id}: {e}")

                except Exception as e:
                    logger.error(f"Generation failed for {request.request_id}: {e}")
                    if request._future:
                        request._future.set_exception(e)

                self._queue.task_done()

            except Exception:
                # Queue.get timeout — just loop
                continue

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict:
        """Get queue status."""
        avg_time = (
            self._total_generation_time / self._total_generated
            if self._total_generated > 0 else 0
        )
        return {
            "running": self._running,
            "queue_depth": self.queue_depth,
            "total_generated": self._total_generated,
            "avg_generation_time": round(avg_time, 3),
            "total_requests": self._request_count,
        }


class Pipeline:
    """
    High-level pipeline orchestrator.

    Combines ParallelIO + GenerationQueue into a single interface
    that the engine can use as a drop-in replacement for the
    synchronous process() loop.
    """

    def __init__(
        self,
        parallel_workers: int = 3,
        enable_queue: bool = True,
    ):
        self.parallel_io = ParallelIO(max_workers=parallel_workers)
        self.generation_queue = GenerationQueue()
        self._queue_enabled = enable_queue
        self._stream_fn = None

    def set_generate_fn(self, fn: Callable):
        """Set the generation function (from BaseModel)."""
        self.generation_queue._generate_fn = fn

    def start(self):
        """Start the pipeline (generation queue worker)."""
        if self._queue_enabled:
            self.generation_queue.start()

    def stop(self):
        """Shutdown the pipeline."""
        if self._queue_enabled:
            self.generation_queue.stop()
        self.parallel_io.shutdown()

    def run_parallel_io(self, **tasks) -> PipelineResult:
        """Run I/O tasks in parallel."""
        return self.parallel_io.execute(**tasks)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response, using the queue if running,
        or synchronously if not.
        """
        if self._queue_enabled and self.generation_queue.is_running:
            future = self.generation_queue.submit(prompt, **kwargs)
            return future.result(timeout=120)  # 2 minute timeout
        else:
            return self.generation_queue.generate_sync(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """
        Phase 5: Streaming generation — yields tokens directly.
        Bypasses the queue (streaming is inherently synchronous).
        """
        if self._stream_fn:
            yield from self._stream_fn(prompt, **kwargs)

    def set_stream_fn(self, fn):
        """Set the streaming generation function."""
        self._stream_fn = fn

    def status(self) -> dict:
        return {
            "parallel_workers": self.parallel_io.max_workers,
            "queue": self.generation_queue.status(),
        }
