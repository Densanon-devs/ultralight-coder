"""
Auto-Tuner — Self-Assessment & Optimization Tool

Profiles the hardware, sweeps key parameters, and finds the optimal
configuration for the current system.

Usage:
    python main.py --tune           # Full auto-tune
    python main.py --tune quick     # Fast sweep (fewer iterations)

What it does:
    1. PROFILE: Detect CPU cores, RAM, GPU, disk speed
    2. SWEEP:   Test each variable in isolation
    3. ANALYZE: Score each setting on quality + speed
    4. APPLY:   Write optimal settings to config
    5. REPORT:  Show what changed and why

Tuned variables:
    - threads        (CPU parallelism)
    - gpu_layers     (GPU offloading)
    - batch_size     (inference batching)
    - temperature    (response quality)
    - chat_format    (prompt template)
    - budget ratios  (token allocation)
    - context_length (memory vs quality tradeoff)
"""

import copy
import json
import logging
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Hardware Profile ──────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    os: str = ""
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    disk_free_gb: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  OS:       {self.os}",
            f"  CPU:      {self.cpu_name}",
            f"  Cores:    {self.cpu_cores_physical} physical, {self.cpu_cores_logical} logical",
            f"  RAM:      {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB free",
        ]
        if self.gpu_available:
            lines.append(f"  GPU:      {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)")
        else:
            lines.append("  GPU:      None detected")
        lines.append(f"  Disk:     {self.disk_free_gb:.1f} GB free")
        return "\n".join(lines)


def profile_hardware() -> HardwareProfile:
    """Detect hardware capabilities."""
    import psutil

    hw = HardwareProfile()
    hw.os = f"{platform.system()} {platform.release()}"

    # CPU
    hw.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
    hw.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
    try:
        if platform.system() == "Windows":
            hw.cpu_name = platform.processor() or "Unknown CPU"
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        hw.cpu_name = line.split(":")[1].strip()
                        break
    except Exception:
        hw.cpu_name = platform.processor() or "Unknown CPU"

    # RAM
    mem = psutil.virtual_memory()
    hw.ram_total_gb = mem.total / (1024 ** 3)
    hw.ram_available_gb = mem.available / (1024 ** 3)

    # Disk
    disk = shutil.disk_usage(Path.cwd())
    hw.disk_free_gb = disk.free / (1024 ** 3)

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            hw.gpu_available = True
            hw.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            hw.gpu_vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
    except ImportError:
        pass

    return hw


# ── Benchmark Prompts ─────────────────────────────────────────

TUNE_PROMPTS = [
    {
        "prompt": "Write a Python function that checks if a number is prime.",
        "category": "code",
        "checks": {
            "has_def": lambda r: "def " in r,
            "has_return": lambda r: "return" in r.lower(),
            "has_code": lambda r: "```" in r or "def " in r,
            "good_length": lambda r: 30 < len(r) < 1500,
        },
    },
    {
        "prompt": "What is 15 times 7?",
        "category": "math",
        "checks": {
            "correct": lambda r: "105" in r,
            "concise": lambda r: len(r) < 500,
        },
    },
    {
        "prompt": "Give me a JSON object with name and age fields.",
        "category": "json",
        "checks": {
            "has_braces": lambda r: "{" in r and "}" in r,
            "has_name": lambda r: '"name"' in r,
            "has_age": lambda r: '"age"' in r,
        },
    },
    {
        "prompt": "Explain what Python is in one sentence.",
        "category": "general",
        "checks": {
            "mentions_lang": lambda r: any(w in r.lower() for w in ["programming", "language", "code"]),
            "reasonable": lambda r: 10 < len(r) < 500,
        },
    },
]


def _run_tuning_prompts(model, fusion, router, modules, memory, **gen_kwargs) -> dict:
    """Run tuning prompts and return aggregated scores."""
    total_quality = 0
    total_checks = 0
    passed_checks = 0
    total_tokens = 0
    total_time = 0

    for pd in TUNE_PROMPTS:
        routing = router.route(pd["prompt"], available_modules=modules.available_modules)
        active = modules.get_multiple(routing.selected_modules)
        ctx = memory.recall(pd["prompt"])

        prompt = fusion.assemble(
            user_input=pd["prompt"],
            active_modules=active,
            memory_context=ctx,
            blend_weights=routing.blend_weights,
        )

        t0 = time.monotonic()
        response = model.generate(prompt, **gen_kwargs)
        elapsed = time.monotonic() - t0

        resp_tokens = model.count_tokens(response)
        total_tokens += resp_tokens
        total_time += elapsed

        for name, check_fn in pd["checks"].items():
            total_checks += 1
            try:
                if check_fn(response):
                    passed_checks += 1
            except Exception:
                pass

    quality = passed_checks / total_checks if total_checks else 0
    speed = total_tokens / total_time if total_time > 0 else 0

    return {
        "quality": round(quality, 3),
        "speed_tps": round(speed, 1),
        "total_time": round(total_time, 2),
        "checks": f"{passed_checks}/{total_checks}",
    }


# ── Sweep Functions ───────────────────────────────────────────

@dataclass
class SweepResult:
    """Result of a single parameter sweep."""
    variable: str
    setting: str
    quality: float
    speed: float
    total_time: float
    checks: str
    # Combined score: weighted quality + speed
    score: float = 0.0

    def compute_score(self, quality_weight: float = 0.7, speed_weight: float = 0.3,
                      max_speed: float = 1.0):
        """Compute combined score (quality-biased by default)."""
        norm_speed = min(self.speed / max_speed, 1.0) if max_speed > 0 else 0
        self.score = self.quality * quality_weight + norm_speed * speed_weight


def _sweep(variable_name, values, make_kwargs, model, fusion_factory, router, modules, memory) -> list[SweepResult]:
    """Run a sweep over a single variable."""
    results = []
    for val in values:
        label = f"{variable_name}={val}"
        kwargs = make_kwargs(val)
        fusion = fusion_factory()
        scores = _run_tuning_prompts(model, fusion, router, modules, memory, **kwargs)
        results.append(SweepResult(
            variable=variable_name,
            setting=str(val),
            quality=scores["quality"],
            speed=scores["speed_tps"],
            total_time=scores["total_time"],
            checks=scores["checks"],
        ))
    # Compute combined scores
    max_speed = max(r.speed for r in results) if results else 1.0
    for r in results:
        r.compute_score(max_speed=max_speed)
    return results


def _print_sweep(results: list[SweepResult], title: str):
    """Print sweep results as a table."""
    print(f"\n  {title}")
    print(f"  {'Setting':<25s} {'Quality':>8s} {'Speed':>8s} {'Time':>7s} {'Score':>7s} {'Checks':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

    best = max(results, key=lambda r: r.score)
    for r in results:
        marker = " <-- best" if r is best else ""
        print(f"  {r.setting:<25s} {r.quality:>7.0%} {r.speed:>6.1f}/s {r.total_time:>6.1f}s {r.score:>6.3f} {r.checks:>8s}{marker}")

    return best


# ── Main Tuner ────────────────────────────────────────────────

class AutoTuner:
    """
    Runs systematic parameter sweeps and determines optimal configuration.
    """

    def __init__(self, config_path: str = "config.yaml", quick: bool = False):
        self.config_path = config_path
        self.quick = quick
        self.config = None
        self.model = None
        self.router = None
        self.modules = None
        self.memory = None
        self.hw = None
        self.optimal = {}
        self.all_results = {}

    def run(self) -> dict:
        """Run the full auto-tuning process."""
        print("=" * 62)
        print("  AUTO-TUNER: Hardware Profile + Parameter Optimization")
        print("=" * 62)

        # Step 1: Profile hardware
        print("\n[1/6] Profiling hardware...")
        self.hw = profile_hardware()
        print(self.hw.summary())

        # Step 2: Initialize
        print("\n[2/6] Loading engine components...")
        self._init_engine()

        # Step 3: Run sweeps
        print("\n[3/6] Running parameter sweeps...")
        self._sweep_threads()
        self._sweep_temperature()
        self._sweep_chat_format()
        self._sweep_budgets()
        self._sweep_batch_size()
        if self.hw.gpu_available:
            self._sweep_gpu_layers()

        # Step 4: Analyze
        print("\n[4/6] Analyzing results...")
        recommendations = self._analyze()

        # Step 5: Report
        print("\n[5/6] Optimization Report")
        self._print_report(recommendations)

        # Step 6: Apply
        print("\n[6/6] Applying optimal settings...")
        self._apply(recommendations)

        self._cleanup()
        return recommendations

    def _init_engine(self):
        """Load all components."""
        from engine.config import Config
        from engine.base_model import BaseModel
        from engine.router import Router
        from engine.module_manager import ModuleManager
        from engine.memory import MemorySystem

        self.config = Config(self.config_path)
        self.model = BaseModel(self.config.base_model)
        self.model.load()
        self.router = Router(self.config.router)
        self.modules = ModuleManager(self.config.modules)
        self.modules.discover()
        self.memory = MemorySystem(self.config.memory)
        self.memory.short_term.clear()

    def _fusion_factory(self, **overrides):
        """Create a fusion layer, optionally overriding config fields."""
        from engine.fusion import FusionLayer
        fc = copy.copy(self.config.fusion)
        for k, v in overrides.items():
            setattr(fc, k, v)
        return lambda: FusionLayer(fc, token_counter=self.model.count_tokens)

    def _sweep_threads(self):
        """Find optimal thread count based on CPU cores."""
        physical = self.hw.cpu_cores_physical
        logical = self.hw.cpu_cores_logical

        # Test: 1, physical/2, physical, physical+2, logical
        candidates = sorted(set([
            1,
            max(1, physical // 2),
            physical,
            min(physical + 2, logical),
            logical,
        ]))
        if self.quick:
            candidates = [max(1, physical // 2), physical, min(physical + 2, logical)]

        print(f"\n  Sweeping threads: {candidates} (physical={physical}, logical={logical})")
        results = []

        for threads in candidates:
            from engine.base_model import BaseModel
            self.model.unload()
            self.config.base_model.threads = threads
            self.model = BaseModel(self.config.base_model)
            self.model.load()

            fusion_fn = self._fusion_factory()
            scores = _run_tuning_prompts(
                self.model, fusion_fn(), self.router, self.modules, self.memory,
                max_tokens=128,
            )
            results.append(SweepResult(
                variable="threads", setting=str(threads),
                quality=scores["quality"], speed=scores["speed_tps"],
                total_time=scores["total_time"], checks=scores["checks"],
            ))

        max_speed = max(r.speed for r in results) if results else 1.0
        for r in results:
            r.compute_score(max_speed=max_speed)

        best = _print_sweep(results, "Thread Count")
        self.all_results["threads"] = results
        self.optimal["threads"] = int(best.setting)

        # Reload with best threads for remaining tests
        from engine.base_model import BaseModel
        self.model.unload()
        self.config.base_model.threads = self.optimal["threads"]
        self.model = BaseModel(self.config.base_model)
        self.model.load()

    def _sweep_temperature(self):
        """Find optimal temperature."""
        temps = [0.1, 0.3, 0.5, 0.7, 1.0]
        if self.quick:
            temps = [0.1, 0.3, 0.7]

        print(f"\n  Sweeping temperature: {temps}")
        results = _sweep(
            "temperature", temps,
            lambda t: {"temperature": t, "max_tokens": 200},
            self.model, self._fusion_factory(), self.router, self.modules, self.memory,
        )
        best = _print_sweep(results, "Temperature")
        self.all_results["temperature"] = results
        self.optimal["temperature"] = float(best.setting)

    def _sweep_chat_format(self):
        """Find optimal chat format."""
        formats = ["chatml", "raw", "alpaca"]
        if self.quick:
            formats = ["chatml", "raw"]

        print(f"\n  Sweeping chat format: {formats}")
        results = []
        for fmt in formats:
            fusion_fn = self._fusion_factory(chat_format=fmt)
            scores = _run_tuning_prompts(
                self.model, fusion_fn(), self.router, self.modules, self.memory,
                max_tokens=200,
            )
            results.append(SweepResult(
                variable="chat_format", setting=fmt,
                quality=scores["quality"], speed=scores["speed_tps"],
                total_time=scores["total_time"], checks=scores["checks"],
            ))

        max_speed = max(r.speed for r in results) if results else 1.0
        for r in results:
            r.compute_score(max_speed=max_speed)

        best = _print_sweep(results, "Chat Format")
        self.all_results["chat_format"] = results
        self.optimal["chat_format"] = best.setting

    def _sweep_budgets(self):
        """Find optimal token budget allocation."""
        configs = {
            "default":       (0.15, 0.25, 0.20, 0.30, 0.10),
            "heavy_modules": (0.10, 0.40, 0.15, 0.25, 0.10),
            "heavy_conv":    (0.10, 0.20, 0.15, 0.45, 0.10),
            "balanced":      (0.10, 0.30, 0.20, 0.30, 0.10),
            "minimal_sys":   (0.05, 0.35, 0.20, 0.30, 0.10),
        }
        if self.quick:
            configs = {k: v for k, v in list(configs.items())[:3]}

        print(f"\n  Sweeping budget ratios: {list(configs.keys())}")
        results = []
        for name, (bs, bm, bmem, bc, br) in configs.items():
            fusion_fn = self._fusion_factory(
                budget_system=bs, budget_modules=bm, budget_memory=bmem,
                budget_conversation=bc, budget_reserve=br,
            )
            scores = _run_tuning_prompts(
                self.model, fusion_fn(), self.router, self.modules, self.memory,
                max_tokens=200,
            )
            results.append(SweepResult(
                variable="budget", setting=name,
                quality=scores["quality"], speed=scores["speed_tps"],
                total_time=scores["total_time"], checks=scores["checks"],
            ))

        max_speed = max(r.speed for r in results) if results else 1.0
        for r in results:
            r.compute_score(max_speed=max_speed)

        best = _print_sweep(results, "Token Budget Allocation")
        self.all_results["budget"] = results
        self.optimal["budget"] = best.setting
        # Store the actual values
        self.optimal["budget_values"] = configs[best.setting]

    def _sweep_batch_size(self):
        """Find optimal batch size."""
        batches = [64, 128, 256, 512, 1024]
        if self.quick:
            batches = [128, 512]

        print(f"\n  Sweeping batch_size: {batches}")
        results = []
        for bs in batches:
            from engine.base_model import BaseModel
            self.model.unload()
            self.config.base_model.batch_size = bs
            self.model = BaseModel(self.config.base_model)
            self.model.load()

            fusion_fn = self._fusion_factory()
            scores = _run_tuning_prompts(
                self.model, fusion_fn(), self.router, self.modules, self.memory,
                max_tokens=128,
            )
            results.append(SweepResult(
                variable="batch_size", setting=str(bs),
                quality=scores["quality"], speed=scores["speed_tps"],
                total_time=scores["total_time"], checks=scores["checks"],
            ))

        max_speed = max(r.speed for r in results) if results else 1.0
        for r in results:
            r.compute_score(max_speed=max_speed)

        best = _print_sweep(results, "Batch Size")
        self.all_results["batch_size"] = results
        self.optimal["batch_size"] = int(best.setting)

        # Reload with best
        from engine.base_model import BaseModel
        self.model.unload()
        self.config.base_model.batch_size = self.optimal["batch_size"]
        self.model = BaseModel(self.config.base_model)
        self.model.load()

    def _sweep_gpu_layers(self):
        """Find optimal GPU offloading level."""
        # Test 0 (CPU), half, and full offload
        candidates = [0, 11, 22, 99]  # TinyLlama has ~22 layers
        if self.quick:
            candidates = [0, 99]

        print(f"\n  Sweeping gpu_layers: {candidates}")
        results = []
        for layers in candidates:
            try:
                from engine.base_model import BaseModel
                self.model.unload()
                self.config.base_model.gpu_layers = layers
                self.model = BaseModel(self.config.base_model)
                self.model.load()

                fusion_fn = self._fusion_factory()
                scores = _run_tuning_prompts(
                    self.model, fusion_fn(), self.router, self.modules, self.memory,
                    max_tokens=128,
                )
                results.append(SweepResult(
                    variable="gpu_layers", setting=str(layers),
                    quality=scores["quality"], speed=scores["speed_tps"],
                    total_time=scores["total_time"], checks=scores["checks"],
                ))
            except Exception as e:
                print(f"    gpu_layers={layers} failed: {e}")

        if results:
            max_speed = max(r.speed for r in results) if results else 1.0
            for r in results:
                r.compute_score(max_speed=max_speed)

            best = _print_sweep(results, "GPU Offloading")
            self.all_results["gpu_layers"] = results
            self.optimal["gpu_layers"] = int(best.setting)
        else:
            self.optimal["gpu_layers"] = 0

        # Reload with best
        from engine.base_model import BaseModel
        self.model.unload()
        self.config.base_model.gpu_layers = self.optimal.get("gpu_layers", 0)
        self.model = BaseModel(self.config.base_model)
        self.model.load()

    def _analyze(self) -> dict:
        """Analyze all sweep results and build recommendations."""
        recs = {
            "threads": self.optimal.get("threads", 4),
            "temperature": self.optimal.get("temperature", 0.3),
            "chat_format": self.optimal.get("chat_format", "chatml"),
            "batch_size": self.optimal.get("batch_size", 512),
            "gpu_layers": self.optimal.get("gpu_layers", 0),
        }

        if "budget_values" in self.optimal:
            bs, bm, bmem, bc, br = self.optimal["budget_values"]
            recs["budget_system"] = bs
            recs["budget_modules"] = bm
            recs["budget_memory"] = bmem
            recs["budget_conversation"] = bc
            recs["budget_reserve"] = br

        return recs

    def _print_report(self, recs: dict):
        """Print the final optimization report."""
        print(f"\n  {'=' * 58}")
        print(f"  OPTIMAL CONFIGURATION FOR THIS SYSTEM")
        print(f"  {'=' * 58}")
        print(f"  Hardware:      {self.hw.cpu_name}")
        print(f"  Cores:         {self.hw.cpu_cores_physical}P/{self.hw.cpu_cores_logical}L")
        if self.hw.gpu_available:
            print(f"  GPU:           {self.hw.gpu_name}")
        print()

        for key, val in recs.items():
            current = self._get_current(key)
            changed = current != val
            marker = " (changed)" if changed else ""
            print(f"  {key:<25s} = {str(val):<15s} was: {str(current):<10s}{marker}")

    def _get_current(self, key: str):
        """Get current config value for comparison."""
        mapping = {
            "threads": self.config.base_model.threads,
            "temperature": self.config.base_model.temperature,
            "chat_format": self.config.fusion.chat_format,
            "batch_size": self.config.base_model.batch_size,
            "gpu_layers": self.config.base_model.gpu_layers,
            "budget_system": self.config.fusion.budget_system,
            "budget_modules": self.config.fusion.budget_modules,
            "budget_memory": self.config.fusion.budget_memory,
            "budget_conversation": self.config.fusion.budget_conversation,
            "budget_reserve": self.config.fusion.budget_reserve,
        }
        return mapping.get(key, "?")

    def _apply(self, recs: dict):
        """Write optimal settings to a tuned config file."""
        tuned_file = Path(self.config_path).parent / "tuned_config.json"
        with open(tuned_file, "w") as f:
            json.dump({
                "hardware": {
                    "os": self.hw.os,
                    "cpu": self.hw.cpu_name,
                    "cores_physical": self.hw.cpu_cores_physical,
                    "cores_logical": self.hw.cpu_cores_logical,
                    "ram_gb": round(self.hw.ram_total_gb, 1),
                    "gpu": self.hw.gpu_name if self.hw.gpu_available else None,
                    "gpu_vram_gb": round(self.hw.gpu_vram_gb, 1) if self.hw.gpu_available else None,
                },
                "optimal": recs,
                "timestamp": time.time(),
            }, f, indent=2)
        print(f"  Saved to: {tuned_file}")
        print(f"  These settings will be loaded on next startup.")

    def _cleanup(self):
        """Clean up resources."""
        if self.model:
            self.model.unload()
