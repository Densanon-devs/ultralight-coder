"""
Module Manager — Plug-in Intelligence System

Phase 3: Smart Caching + Predictive Pre-loading

Manages the lifecycle of skill modules (LoRA adapters + prompt injections).
Each module is defined by a YAML manifest that specifies:
  - What it does (metadata)
  - How to activate it (LoRA path, system prompt injection)
  - When to use it (handled by the Router)

Phase 3 enhancements:
  - Frequency-aware eviction (LFU + LRU hybrid)
  - Co-occurrence tracking: modules used together get pre-loaded together
  - Warm/cold cache tiers: hot modules stay in memory, cold ones on standby
  - Predictive pre-loading: anticipate next modules from routing patterns
  - Thread-safe operations for async pipeline
"""

import json
import logging
import time
import threading
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.config import ModulesConfig

logger = logging.getLogger(__name__)


@dataclass
class ModuleManifest:
    """
    Describes a single plug-in module.
    Loaded from a module's manifest.yaml file.
    """
    name: str
    description: str = ""
    version: str = "0.1.0"

    # LoRA adapter (optional — not all modules need one)
    lora_path: Optional[str] = None

    # Prompt injection: text prepended to system prompt when active
    system_prompt_injection: str = ""

    # Prompt injection: text added to user prompt context
    context_injection: str = ""

    # Output formatting instructions
    output_format: Optional[str] = None

    # Module type: "lora", "prompt", "hybrid", "tool"
    module_type: str = "prompt"

    # Priority hint for the fusion layer
    priority: int = 5

    # Tags for discovery
    tags: list[str] = field(default_factory=list)


@dataclass
class LoadedModule:
    """A module that is currently loaded and cached."""
    manifest: ModuleManifest
    loaded_at: float = 0.0
    use_count: int = 0
    last_used: float = 0.0
    tier: str = "warm"                  # Phase 3: "hot", "warm", "cold"


@dataclass
class ModuleStats:
    """Phase 3: Usage statistics for a module, used by smart cache."""
    total_uses: int = 0
    recent_uses: int = 0                # Uses in the last N interactions
    avg_gap_between_uses: float = 0.0   # Average interactions between activations
    last_used_interaction: int = 0      # Interaction number of last use
    co_used_with: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ModuleManager:
    """
    Manages discovery, loading, caching, and lifecycle of modules.

    Phase 3 cache tiers:
        HOT  — Modules used in the last 3 interactions. Always in memory.
               Eviction: never (until they cool down to warm).
        WARM — Modules used recently but not in the last 3 turns.
               Eviction: LFU (least frequently used) when at capacity.
        COLD — Registered but not in cache. Loaded on demand.

    Predictive pre-loading:
        After each routing decision, the manager checks co-occurrence
        patterns and pre-loads modules likely to be needed next.
    """

    # Tier thresholds
    HOT_RECENCY = 3         # Last N interactions → hot tier
    WARM_RECENCY = 15       # Last N interactions → warm tier (else cold)

    def __init__(self, config: ModulesConfig):
        self.config = config
        self.modules_dir = Path(config.directory)
        self._registry: dict[str, ModuleManifest] = {}
        self._cache: dict[str, LoadedModule] = {}

        # Phase 3: Smart cache state
        self._stats: dict[str, ModuleStats] = {}
        self._interaction_count: int = 0
        self._stats_file = Path(config.directory).parent / "data" / "module_stats.json"
        self._lock = threading.RLock()  # Thread safety for async pipeline

        self._load_stats()

    # ── Discovery ─────────────────────────────────────────────

    def discover(self) -> list[str]:
        """
        Scan the modules directory and register all available modules.
        Returns list of discovered module names.
        """
        with self._lock:
            if not self.modules_dir.exists():
                logger.warning(f"Modules directory not found: {self.modules_dir}")
                self.modules_dir.mkdir(parents=True, exist_ok=True)
                return []

            discovered = []
            for module_dir in sorted(self.modules_dir.iterdir()):
                if not module_dir.is_dir():
                    continue

                manifest_path = module_dir / "manifest.yaml"
                if not manifest_path.exists():
                    logger.debug(f"Skipping {module_dir.name} — no manifest.yaml")
                    continue

                try:
                    manifest = self._load_manifest(manifest_path, module_dir)
                    self._registry[manifest.name] = manifest
                    discovered.append(manifest.name)
                    # Initialize stats if new
                    if manifest.name not in self._stats:
                        self._stats[manifest.name] = ModuleStats()
                    logger.info(
                        f"Discovered module: {manifest.name} "
                        f"({manifest.module_type}) — {manifest.description}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load module {module_dir.name}: {e}")

            logger.info(f"Discovered {len(discovered)} modules: {discovered}")
            return discovered

    def _load_manifest(self, manifest_path: Path, module_dir: Path) -> ModuleManifest:
        """Parse a module's manifest.yaml into a ModuleManifest."""
        with open(manifest_path, "r") as f:
            raw = yaml.safe_load(f)

        lora_path = raw.get("lora_path")
        if lora_path:
            lora_file = module_dir / lora_path
            lora_path = str(lora_file) if lora_file.exists() else None

        return ModuleManifest(
            name=raw.get("name", module_dir.name),
            description=raw.get("description", ""),
            version=raw.get("version", "0.1.0"),
            lora_path=lora_path,
            system_prompt_injection=raw.get("system_prompt_injection", ""),
            context_injection=raw.get("context_injection", ""),
            output_format=raw.get("output_format"),
            module_type=raw.get("module_type", "prompt"),
            priority=raw.get("priority", 5),
            tags=raw.get("tags", []),
        )

    # ── Loading & Caching ─────────────────────────────────────

    def get(self, module_name: str) -> Optional[LoadedModule]:
        """
        Get a module, loading it into cache if needed.
        Thread-safe. Returns None if module doesn't exist.
        """
        with self._lock:
            # Check cache first
            if module_name in self._cache:
                cached = self._cache[module_name]
                cached.use_count += 1
                cached.last_used = time.time()
                return cached

            # Check registry
            if module_name not in self._registry:
                logger.warning(f"Module not found: {module_name}")
                return None

            # Load into cache
            manifest = self._registry[module_name]
            loaded = LoadedModule(
                manifest=manifest,
                loaded_at=time.time(),
                use_count=1,
                last_used=time.time(),
                tier="warm",
            )

            # Evict if needed (smart eviction)
            self._smart_evict()

            self._cache[module_name] = loaded
            logger.info(f"Loaded module into cache: {module_name}")
            return loaded

    def get_multiple(self, module_names: list[str]) -> list[LoadedModule]:
        """Load multiple modules at once. Thread-safe."""
        loaded = []
        for name in module_names:
            module = self.get(name)
            if module:
                loaded.append(module)
        return loaded

    # ── Phase 3: Smart Eviction ───────────────────────────────

    def _smart_evict(self):
        """
        Frequency-aware eviction: LFU + LRU hybrid.

        Eviction priority (evict first):
            1. Cold-tier modules (shouldn't be cached, but just in case)
            2. Warm-tier with lowest use frequency
            3. Warm-tier with oldest last_used (LRU tiebreaker)

        Never evicts hot-tier modules.
        """
        while len(self._cache) >= self.config.max_cached:
            # Find best candidate for eviction
            candidates = []
            for name, mod in self._cache.items():
                if mod.tier == "hot":
                    continue  # Never evict hot modules
                stats = self._stats.get(name, ModuleStats())
                # Score: lower = more evictable
                # Combine frequency (total uses) with recency (last_used)
                freq_score = stats.total_uses
                recency_score = mod.last_used
                candidates.append((name, freq_score, recency_score))

            if not candidates:
                # All modules are hot — force-evict the least-used hot module
                candidates = [
                    (name, self._stats.get(name, ModuleStats()).total_uses, mod.last_used)
                    for name, mod in self._cache.items()
                ]

            if not candidates:
                break

            # Sort: lowest frequency first, then oldest last_used
            candidates.sort(key=lambda x: (x[1], x[2]))
            evict_name = candidates[0][0]
            logger.info(f"Smart eviction: {evict_name} (freq={candidates[0][1]}, tier={self._cache[evict_name].tier})")
            del self._cache[evict_name]

    def _update_tiers(self):
        """
        Reassign cache tiers based on recent usage patterns.
        Called after each interaction.
        """
        with self._lock:
            for name, mod in self._cache.items():
                stats = self._stats.get(name, ModuleStats())
                gap = self._interaction_count - stats.last_used_interaction

                if gap <= self.HOT_RECENCY:
                    mod.tier = "hot"
                elif gap <= self.WARM_RECENCY:
                    mod.tier = "warm"
                else:
                    mod.tier = "cold"

    # ── Phase 3: Usage Tracking & Co-occurrence ───────────────

    def record_usage(self, module_names: list[str]):
        """
        Record which modules were used in this interaction.
        Updates frequency stats and co-occurrence matrix.
        Thread-safe.
        """
        with self._lock:
            self._interaction_count += 1

            for name in module_names:
                if name not in self._stats:
                    self._stats[name] = ModuleStats()
                stats = self._stats[name]

                # Update frequency
                stats.total_uses += 1
                stats.recent_uses += 1

                # Update average gap
                if stats.last_used_interaction > 0:
                    gap = self._interaction_count - stats.last_used_interaction
                    # Exponential moving average
                    stats.avg_gap_between_uses = (
                        0.7 * stats.avg_gap_between_uses + 0.3 * gap
                    )
                stats.last_used_interaction = self._interaction_count

                # Update co-occurrence
                for other_name in module_names:
                    if other_name != name:
                        stats.co_used_with[other_name] += 1

            # Decay recent_uses periodically (every 20 interactions)
            if self._interaction_count % 20 == 0:
                for stats in self._stats.values():
                    stats.recent_uses = max(0, stats.recent_uses - 1)

            # Reassign tiers
            self._update_tiers()

            # Persist stats periodically
            if self._interaction_count % 10 == 0:
                self._save_stats()

    def predict_next_modules(self, current_modules: list[str], top_k: int = 2) -> list[str]:
        """
        Predict which modules are likely needed next based on co-occurrence.

        If 'code' and 'math' are often used together, and you just activated
        'code', this will suggest pre-loading 'math'.

        Returns list of module names to pre-load (not already in current_modules).
        """
        with self._lock:
            co_scores: dict[str, int] = defaultdict(int)

            for name in current_modules:
                stats = self._stats.get(name)
                if stats:
                    for co_mod, count in stats.co_used_with.items():
                        if co_mod not in current_modules and co_mod in self._registry:
                            co_scores[co_mod] += count

            if not co_scores:
                return []

            # Sort by co-occurrence frequency
            sorted_mods = sorted(co_scores.items(), key=lambda x: x[1], reverse=True)
            predictions = [name for name, _ in sorted_mods[:top_k]]

            logger.debug(f"Predicted next modules: {predictions} (from co-occurrence with {current_modules})")
            return predictions

    def preload(self, module_names: list[str]):
        """
        Pre-load modules into cache in the background.
        Used by the async pipeline to anticipate needs.
        Thread-safe.
        """
        for name in module_names:
            if name not in self._cache and name in self._registry:
                self.get(name)  # get() is already thread-safe
                logger.info(f"Pre-loaded module: {name}")

    # ── Stale Cleanup ─────────────────────────────────────────

    def cleanup_stale(self):
        """Remove cold-tier modules that haven't been used within cache_ttl."""
        with self._lock:
            now = time.time()
            stale = [
                name
                for name, mod in self._cache.items()
                if mod.tier != "hot" and (now - mod.last_used) > self.config.cache_ttl
            ]
            for name in stale:
                logger.info(f"Removing stale module from cache: {name} (tier={self._cache[name].tier})")
                del self._cache[name]

    # ── Stats Persistence ─────────────────────────────────────

    def _load_stats(self):
        """Load usage stats from disk."""
        if self._stats_file.exists():
            try:
                with open(self._stats_file, "r") as f:
                    data = json.load(f)
                self._interaction_count = data.get("interaction_count", 0)
                for name, raw in data.get("modules", {}).items():
                    self._stats[name] = ModuleStats(
                        total_uses=raw.get("total_uses", 0),
                        recent_uses=raw.get("recent_uses", 0),
                        avg_gap_between_uses=raw.get("avg_gap", 0.0),
                        last_used_interaction=raw.get("last_used_interaction", 0),
                        co_used_with=defaultdict(int, raw.get("co_used_with", {})),
                    )
                logger.info(f"Loaded module stats ({self._interaction_count} interactions)")
            except Exception as e:
                logger.error(f"Failed to load module stats: {e}")

    def _save_stats(self):
        """Persist usage stats to disk."""
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "interaction_count": self._interaction_count,
                "modules": {},
            }
            for name, stats in self._stats.items():
                data["modules"][name] = {
                    "total_uses": stats.total_uses,
                    "recent_uses": stats.recent_uses,
                    "avg_gap": stats.avg_gap_between_uses,
                    "last_used_interaction": stats.last_used_interaction,
                    "co_used_with": dict(stats.co_used_with),
                }
            with open(self._stats_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save module stats: {e}")

    # ── Properties & Info ─────────────────────────────────────

    @property
    def available_modules(self) -> list[str]:
        """List all discovered module names."""
        return list(self._registry.keys())

    @property
    def cached_modules(self) -> list[str]:
        """List currently cached module names."""
        return list(self._cache.keys())

    def module_info(self, module_name: str) -> Optional[dict]:
        """Get info about a module for display purposes."""
        manifest = self._registry.get(module_name)
        if not manifest:
            return None

        cached = self._cache.get(module_name)
        stats = self._stats.get(module_name, ModuleStats())
        return {
            "name": manifest.name,
            "description": manifest.description,
            "type": manifest.module_type,
            "has_lora": manifest.lora_path is not None,
            "tags": manifest.tags,
            "cached": cached is not None,
            "tier": cached.tier if cached else "cold",
            "use_count": cached.use_count if cached else 0,
            "total_uses": stats.total_uses,
            "co_modules": dict(stats.co_used_with) if stats.co_used_with else {},
        }

    def list_all(self) -> list[dict]:
        """Get info for all registered modules."""
        return [
            self.module_info(name)
            for name in self._registry
            if self.module_info(name) is not None
        ]

    def cache_status(self) -> dict:
        """Phase 3: Detailed cache status."""
        with self._lock:
            tiers = {"hot": [], "warm": [], "cold": []}
            for name, mod in self._cache.items():
                tiers[mod.tier].append(name)
            return {
                "total_cached": len(self._cache),
                "max_cached": self.config.max_cached,
                "tiers": tiers,
                "interaction_count": self._interaction_count,
            }

    def create_module_template(self, name: str, module_type: str = "prompt") -> Path:
        """Create a new module directory with a template manifest."""
        module_dir = self.modules_dir / name
        module_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "name": name,
            "description": f"Description for {name} module",
            "version": "0.1.0",
            "module_type": module_type,
            "priority": 5,
            "tags": [name],
            "system_prompt_injection": f"You are specialized in {name}.",
            "context_injection": "",
            "output_format": None,
        }

        if module_type in ("lora", "hybrid"):
            manifest["lora_path"] = "adapter.gguf"

        manifest_path = module_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created module template: {module_dir}")
        return module_dir
