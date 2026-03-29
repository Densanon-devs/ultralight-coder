"""
Router / Controller — The "Most Important System"

Phase 2: Smart Routing with three operating modes:
    - "rule_based": v1 keyword matching (unchanged, always available)
    - "classifier": learned TF-IDF + logistic regression routing
    - "hybrid": classifier first, rule-based fallback on low confidence

Plus multi-module blending: when multiple modules are selected,
the router now assigns blend weights and resolves conflicts so
the fusion layer knows how to prioritize each module's contribution.

The router is the decision-maker. Get this wrong and everything
downstream suffers. Get it right and a tiny model punches
way above its weight.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from engine.config import RouterConfig, RoutingRule, BlendingConfig
from engine.classifier import IntentClassifier, ClassifierPrediction, create_classifier

logger = logging.getLogger(__name__)


@dataclass
class ModuleWeight:
    """A module with its blend weight for the fusion layer."""
    name: str
    weight: float = 1.0                 # 0.0-1.0 blend weight
    source: str = "rule"                # "rule", "classifier", "default"


@dataclass
class RoutingDecision:
    """The output of the router — what modules to activate and why."""
    selected_modules: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)

    # Phase 2: Blending info
    blend_weights: list[ModuleWeight] = field(default_factory=list)
    routing_mode: str = "rule_based"    # Which mode was actually used
    classifier_confidence: float = 0.0
    used_fallback: bool = False


class Router:
    """
    v2: Hybrid Router with Smart Classification + Multi-Module Blending

    Operating modes:
        "rule_based"  — v1 keyword matching (fast, deterministic, zero training)
        "classifier"  — learned TF-IDF + logistic regression (requires training data)
        "hybrid"      — classifier with automatic rule-based fallback

    Blending strategies:
        "weighted"  — exponential decay by rank (top module gets 1.0, next gets 0.7, etc.)
        "priority"  — weights based on module manifest priority
        "equal"     — all selected modules get equal weight

    The classifier is always collecting training data, even in rule_based mode.
    Switch to hybrid mode once you have enough samples.
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self.rules = config.rules
        self.blending = config.blending
        self._compile_patterns()

        # Phase 2+4: Initialize classifier (tfidf or neural via factory)
        self.classifier = create_classifier(config.classifier)

    def _compile_patterns(self):
        """Pre-compile keyword patterns for faster matching."""
        self._compiled_rules = []
        for rule in self.rules:
            pattern = "|".join(re.escape(kw) for kw in rule.keywords)
            compiled = re.compile(pattern, re.IGNORECASE)
            self._compiled_rules.append((compiled, rule))

    def route(
        self,
        user_prompt: str,
        conversation_history: Optional[list[dict]] = None,
        available_modules: Optional[list[str]] = None,
        system_state: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Analyze user input and decide which modules to activate.

        In hybrid mode:
            1. Ask classifier for prediction
            2. If confident enough → use classifier result
            3. If not → fall back to rule-based
            4. Apply multi-module blending weights
            5. Record interaction for future training

        Args:
            user_prompt: The current user message
            conversation_history: Recent conversation turns
            available_modules: List of module names that are actually installed
            system_state: Current system state

        Returns:
            RoutingDecision with selected modules, blend weights, and metadata
        """
        mode = self.config.mode

        if mode == "classifier":
            decision = self._route_classifier(user_prompt, available_modules)
        elif mode == "hybrid":
            decision = self._route_hybrid(user_prompt, conversation_history, available_modules)
        else:
            decision = self._route_rule_based(user_prompt, conversation_history, available_modules)

        # Apply multi-module blending
        if self.blending.enabled and len(decision.selected_modules) > 1:
            decision.blend_weights = self._compute_blend_weights(decision)
        elif decision.selected_modules:
            # Single module gets full weight
            decision.blend_weights = [
                ModuleWeight(name=decision.selected_modules[0], weight=1.0, source=decision.routing_mode)
            ]

        logger.info(
            f"Router [{decision.routing_mode}]: {decision.selected_modules} "
            f"(confidence={decision.classifier_confidence:.2f}, "
            f"fallback={decision.used_fallback})"
        )

        return decision

    # ── Routing Modes ─────────────────────────────────────────

    def _route_rule_based(
        self,
        user_prompt: str,
        conversation_history: Optional[list[dict]] = None,
        available_modules: Optional[list[str]] = None,
    ) -> RoutingDecision:
        """v1 rule-based routing — keyword matching with priority scoring."""
        decision = RoutingDecision(routing_mode="rule_based")

        # Score each rule against the prompt
        module_scores: dict[str, float] = {}
        module_reasons: dict[str, list[str]] = {}

        for compiled_pattern, rule in self._compiled_rules:
            matches = compiled_pattern.findall(user_prompt)
            if matches:
                score = len(matches) * rule.priority
                module_name = rule.module

                if available_modules and module_name not in available_modules:
                    logger.debug(f"Rule matched module '{module_name}' but it's not available")
                    continue

                if module_name not in module_scores:
                    module_scores[module_name] = 0
                    module_reasons[module_name] = []

                module_scores[module_name] += score
                matched_keywords = list(set(matches))
                module_reasons[module_name].append(
                    f"Matched keywords: {', '.join(matched_keywords)} "
                    f"(priority={rule.priority})"
                )

        # Check conversation history for context signals
        if conversation_history:
            recent_context = self._analyze_history(conversation_history)
            for module_name, boost in recent_context.items():
                if module_name in module_scores:
                    module_scores[module_name] += boost
                    module_reasons.setdefault(module_name, []).append(
                        f"Context boost from conversation history (+{boost:.1f})"
                    )

        # Sort by score and apply max_active limit
        sorted_modules = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_modules[:self.config.max_active_modules]

        # Add default modules
        selected_names = {name for name, _ in selected}
        for default_mod in self.config.default_modules:
            if default_mod not in selected_names:
                selected.append((default_mod, 0))
                module_reasons.setdefault(default_mod, []).append("Default module")

        # Build decision
        decision.selected_modules = [name for name, _ in selected]
        decision.scores = {name: score for name, score in selected}
        decision.execution_order = decision.selected_modules

        for module_name in decision.selected_modules:
            reasons = module_reasons.get(module_name, ["No specific reason"])
            decision.reasoning.extend([f"[{module_name}] {r}" for r in reasons])

        return decision

    def _route_classifier(
        self,
        user_prompt: str,
        available_modules: Optional[list[str]] = None,
    ) -> RoutingDecision:
        """Classifier-only routing — uses trained model, no fallback."""
        decision = RoutingDecision(routing_mode="classifier")

        prediction = self.classifier.predict(user_prompt)

        if prediction.used_fallback:
            decision.used_fallback = True
            decision.reasoning.extend(prediction.reasoning)
            # Add defaults
            decision.selected_modules = list(self.config.default_modules)
            return decision

        # Filter to available modules
        for mod in prediction.predicted_modules:
            if available_modules is None or mod in available_modules:
                decision.selected_modules.append(mod)
                decision.scores[mod] = prediction.confidences.get(mod, 0.0)

        # Limit to max_active
        decision.selected_modules = decision.selected_modules[:self.config.max_active_modules]
        decision.execution_order = decision.selected_modules
        decision.classifier_confidence = prediction.overall_confidence
        decision.reasoning.extend(prediction.reasoning)

        # Add defaults
        selected_set = set(decision.selected_modules)
        for default_mod in self.config.default_modules:
            if default_mod not in selected_set:
                decision.selected_modules.append(default_mod)

        return decision

    def _route_hybrid(
        self,
        user_prompt: str,
        conversation_history: Optional[list[dict]] = None,
        available_modules: Optional[list[str]] = None,
    ) -> RoutingDecision:
        """
        Hybrid routing — classifier first, rule-based fallback.

        Strategy:
            1. Get classifier prediction
            2. If confident → use it
            3. If not confident → use rule-based
            4. If both agree on some modules → boost those (consensus bonus)
        """
        # Get classifier prediction
        cls_pred = self.classifier.predict(user_prompt)

        # Get rule-based prediction (always compute for consensus)
        rule_decision = self._route_rule_based(user_prompt, conversation_history, available_modules)

        # Decide which to trust
        if (not cls_pred.used_fallback
                and cls_pred.overall_confidence >= self.config.classifier.confidence_threshold):
            # Classifier is confident — use it as primary
            decision = RoutingDecision(routing_mode="hybrid:classifier")
            decision.classifier_confidence = cls_pred.overall_confidence

            for mod in cls_pred.predicted_modules:
                if available_modules is None or mod in available_modules:
                    confidence = cls_pred.confidences.get(mod, 0.0)

                    # Consensus bonus: if rule-based also selected this module
                    if mod in rule_decision.scores:
                        confidence = min(confidence * 1.2, 1.0)  # 20% bonus
                        decision.reasoning.append(
                            f"[{mod}] classifier={cls_pred.confidences.get(mod, 0):.3f} "
                            f"+ rule consensus bonus"
                        )
                    else:
                        decision.reasoning.append(
                            f"[{mod}] classifier={confidence:.3f}"
                        )

                    decision.selected_modules.append(mod)
                    decision.scores[mod] = confidence

            # Limit and order
            decision.selected_modules = decision.selected_modules[:self.config.max_active_modules]
            decision.execution_order = decision.selected_modules

        else:
            # Classifier not confident — fall back to rules
            decision = rule_decision
            decision.routing_mode = "hybrid:rules"
            decision.used_fallback = True
            decision.reasoning.insert(0,
                f"Classifier confidence too low ({cls_pred.overall_confidence:.3f} < "
                f"{self.config.classifier.confidence_threshold}) — using rule-based routing"
            )

        # Add defaults
        selected_set = set(decision.selected_modules)
        for default_mod in self.config.default_modules:
            if default_mod not in selected_set:
                decision.selected_modules.append(default_mod)

        return decision

    # ── Multi-Module Blending ─────────────────────────────────

    def _compute_blend_weights(self, decision: RoutingDecision) -> list[ModuleWeight]:
        """
        Compute blend weights for multi-module responses.

        The fusion layer uses these weights to determine how much
        prompt space each module gets and how to resolve conflicts.
        """
        strategy = self.blending.strategy
        modules = decision.selected_modules[:self.blending.max_blend_modules]
        scores = decision.scores

        weights: list[ModuleWeight] = []

        if strategy == "weighted":
            # Exponential decay by rank
            for i, mod in enumerate(modules):
                weight = self.blending.weight_decay ** i
                source = decision.routing_mode
                weights.append(ModuleWeight(name=mod, weight=round(weight, 3), source=source))

        elif strategy == "priority":
            # Weight by the module's routing score (normalized)
            max_score = max(scores.values()) if scores else 1.0
            for mod in modules:
                score = scores.get(mod, 0.0)
                weight = score / max_score if max_score > 0 else 1.0
                weights.append(ModuleWeight(name=mod, weight=round(weight, 3), source="priority"))

        elif strategy == "equal":
            for mod in modules:
                weights.append(ModuleWeight(name=mod, weight=1.0, source="equal"))

        else:
            # Default: equal weights
            for mod in modules:
                weights.append(ModuleWeight(name=mod, weight=1.0, source="default"))

        # Normalize so weights sum to ~1.0
        total_weight = sum(w.weight for w in weights)
        if total_weight > 0:
            for w in weights:
                w.weight = round(w.weight / total_weight, 3)

        logger.debug(
            f"Blend weights ({strategy}): "
            f"{[(w.name, w.weight) for w in weights]}"
        )

        return weights

    # ── Training Integration ──────────────────────────────────

    def record_interaction(self, prompt: str, modules: list[str]):
        """
        Record an interaction for classifier training.
        Called automatically after each routing + generation cycle.
        Always records, regardless of routing mode.
        """
        self.classifier.add_sample(prompt, modules)

    def train_classifier(self) -> dict:
        """Manually trigger classifier training."""
        return self.classifier.train()

    def rate_routing(self, feedback: str):
        """Rate the last routing decision (for active learning)."""
        self.classifier.rate_last(feedback)

    # ── History Analysis ──────────────────────────────────────

    def _analyze_history(self, history: list[dict]) -> dict[str, float]:
        """
        Look at recent conversation history to boost relevant modules.
        """
        boosts: dict[str, float] = {}
        recent = history[-3:]
        for turn in recent:
            content = turn.get("content", "")
            for compiled_pattern, rule in self._compiled_rules:
                if compiled_pattern.search(content):
                    mod = rule.module
                    boosts[mod] = boosts.get(mod, 0) + rule.priority * 0.3
        return boosts

    # ── Dynamic Rule Management ───────────────────────────────

    def add_rule(self, keywords: list[str], module: str, priority: int = 5):
        """Dynamically add a new routing rule at runtime."""
        new_rule = RoutingRule(keywords=keywords, module=module, priority=priority)
        self.rules.append(new_rule)

        pattern = "|".join(re.escape(kw) for kw in keywords)
        compiled = re.compile(pattern, re.IGNORECASE)
        self._compiled_rules.append((compiled, new_rule))

        logger.info(f"Added routing rule: {keywords} -> {module} (priority={priority})")

    def remove_rule(self, module: str):
        """Remove all routing rules for a given module."""
        self.rules = [r for r in self.rules if r.module != module]
        self._compiled_rules = [
            (p, r) for p, r in self._compiled_rules if r.module != module
        ]
        logger.info(f"Removed routing rules for module: {module}")

    # ── Diagnostics ───────────────────────────────────────────

    def explain(self, user_prompt: str) -> str:
        """Human-readable explanation of routing decision (enhanced for Phase 2)."""
        decision = self.route(user_prompt)
        lines = [
            f"Routing Analysis for: \"{user_prompt[:80]}\"",
            f"Mode: {decision.routing_mode}",
            "",
        ]

        if not decision.selected_modules:
            lines.append("No modules selected — base model only.")
        else:
            lines.append("Selected modules:")
            for mod in decision.selected_modules:
                score = decision.scores.get(mod, 0)
                lines.append(f"  [{mod}] score={score:.2f}")

            if decision.blend_weights:
                lines.append("")
                lines.append("Blend weights:")
                for w in decision.blend_weights:
                    lines.append(f"  [{w.name}] weight={w.weight:.3f} (source={w.source})")

            lines.append("")
            lines.append("Reasoning:")
            for reason in decision.reasoning:
                lines.append(f"  {reason}")

        # Classifier status
        lines.append("")
        cls_status = self.classifier.status()
        lines.append(f"Classifier: {'trained' if cls_status['trained'] else 'not trained'}")
        lines.append(f"  Training samples: {cls_status['samples']}")
        if cls_status['trained']:
            lines.append(f"  Known modules: {cls_status['known_modules']}")
            lines.append(f"  Confidence threshold: {cls_status['confidence_threshold']}")

        return "\n".join(lines)

    def status(self) -> dict:
        """Get full router status."""
        return {
            "mode": self.config.mode,
            "rule_count": len(self.rules),
            "blending": {
                "enabled": self.blending.enabled,
                "strategy": self.blending.strategy,
            },
            "classifier": self.classifier.status(),
        }
