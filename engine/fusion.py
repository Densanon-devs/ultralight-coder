"""
Fusion Layer — Prompt Assembly & Output Composition

Phase 2: Structured XML-tagged prompt assembly with:
    - Explicit token budgets per section (configurable percentages)
    - Blend-weight-aware module context sizing
    - XML-tagged sections for cleaner model parsing
    - Priority-based truncation when sections exceed budgets
    - Backward-compatible "simple" mode for v1 behavior

The fusion layer assembles the final prompt:

  <system> System Prompt </system>
  <knowledge> Static Knowledge </knowledge>
  <modules> Module Injections (weighted by blend) </modules>
  <memory> Long-Term Memory </memory>
  <conversation> Recent History </conversation>
  <format> Output Instructions </format>
  <user> Current Input </user>

This layer is the glue. It determines prompt quality, which
directly determines output quality.
"""

import logging
from typing import Optional

from engine.config import FusionConfig
from engine.module_manager import LoadedModule

logger = logging.getLogger(__name__)


class FusionLayer:
    """
    Assembles the final prompt from all system components.
    Manages token budgets to stay within context limits.

    Phase 2 adds:
        - "structured" mode with XML tags
        - Per-section token budgets
        - Blend-weight-aware module sizing
        - Smarter truncation strategies
    """

    def __init__(self, config: FusionConfig, token_counter=None):
        """
        Args:
            config: FusionConfig from system configuration
            token_counter: Callable(str) -> int for counting tokens.
                          Falls back to char/4 estimate if not provided.
        """
        self.config = config
        self._count_tokens = token_counter or (lambda text: len(text) // 4)

    def assemble(
        self,
        user_input: str,
        active_modules: Optional[list[LoadedModule]] = None,
        memory_context: Optional[dict[str, str]] = None,
        npc_profile: Optional[dict] = None,
        system_knowledge: Optional[str] = None,
        blend_weights: Optional[list] = None,
    ) -> str:
        """
        Build the complete prompt for the base model.

        Args:
            user_input: The current user message
            active_modules: Modules selected by the router
            memory_context: Dict from MemorySystem.recall()
            npc_profile: NPC personality data (if in NPC mode)
            system_knowledge: Static knowledge to include
            blend_weights: Phase 2 ModuleWeight list from router

        Returns:
            Fully assembled prompt string
        """
        if self.config.mode == "lean":
            return self._assemble_lean(
                user_input, active_modules, memory_context,
            )
        elif self.config.mode == "structured":
            return self._assemble_structured(
                user_input, active_modules, memory_context,
                npc_profile, system_knowledge, blend_weights,
            )
        else:
            return self._assemble_simple(
                user_input, active_modules, memory_context,
                npc_profile, system_knowledge,
            )

    # ── Lean Assembly (code-optimized) ─────────────────────────

    def _assemble_lean(
        self,
        user_input: str,
        active_modules: Optional[list[LoadedModule]] = None,
        memory_context: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Lean code-optimized assembly. Minimal overhead, maximum room for user code.

        No XML tags. No module bloat. Just:
        1. One-line system prompt
        2. Conversation history (if any)
        3. User input

        The model's first attempt is the quality ceiling. Every extra token
        in the prompt is attention budget stolen from the code generation.
        """
        system = self.config.system_prompt.strip()

        # Only include conversation history — skip modules, memory, adapters
        # for first-turn. Include for multi-turn.
        context_parts = []

        if memory_context and memory_context.get("conversation"):
            conv = memory_context["conversation"]
            budget = int(self.config.max_prompt_tokens * self.config.budget_conversation)
            conv_text = self._fit_conversation(conv, budget)
            if conv_text:
                context_parts.append(conv_text)

        # For multi-turn: include compressed state if conversation is long
        if memory_context and memory_context.get("compressed"):
            compressed = memory_context["compressed"]
            c_budget = int(self.config.max_prompt_tokens * self.config.budget_memory)
            c_text = self._fit_to_budget(compressed, c_budget)
            if c_text:
                context_parts.append(c_text)

        # Assemble into chat format
        if context_parts:
            # Multi-turn: system + context + user
            context_block = "\n\n".join(context_parts)
            full_user = f"{context_block}\n\n{user_input}"
        else:
            # First turn: just system + user (cleanest possible)
            full_user = user_input

        prompt = self._apply_chat_format_lean(system, full_user)

        total_tokens = self._count_tokens(prompt)
        logger.debug(f"Lean prompt: ~{total_tokens} tokens")
        return prompt

    def _apply_chat_format_lean(self, system: str, user: str) -> str:
        """Apply chat format with minimal overhead."""
        fmt = self.config.chat_format

        if fmt == "chatml":
            return (f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        elif fmt == "llama3":
            return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        elif fmt == "phi3":
            return (f"<|system|>\n{system}<|end|>\n"
                    f"<|user|>\n{user}<|end|>\n"
                    f"<|assistant|>\n")
        elif fmt == "alpaca":
            return (f"### System:\n{system}\n\n"
                    f"### Instruction:\n{user}\n\n"
                    f"### Response:\n")
        else:
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    # ── Phase 2: Structured Assembly ──────────────────────────

    def _assemble_structured(
        self,
        user_input: str,
        active_modules: Optional[list[LoadedModule]] = None,
        memory_context: Optional[dict[str, str]] = None,
        npc_profile: Optional[dict] = None,
        system_knowledge: Optional[str] = None,
        blend_weights: Optional[list] = None,
    ) -> str:
        """
        XML-tagged structured assembly with per-section token budgets.

        Each section gets a calculated token budget based on config percentages.
        If a section is empty, its budget is redistributed to others.
        Blend weights control how much space each module gets.
        """
        total_budget = self.config.max_prompt_tokens
        user_tokens = self._count_tokens(user_input) + 20  # +20 for tags/formatting

        # Calculate section budgets
        budgets = self._calculate_budgets(total_budget - user_tokens)
        sections = []
        used_tokens = 0

        # --- Section 1: System Prompt ---
        system_prompt = self._build_system_prompt(active_modules, npc_profile)
        system_text = self._fit_to_budget(system_prompt, budgets["system"])
        if system_text:
            sections.append(f"<system>\n{system_text}\n</system>")
            used_tokens += self._count_tokens(system_text)

        # --- Section 2: System Knowledge ---
        if system_knowledge:
            knowledge_text = self._fit_to_budget(system_knowledge, budgets["modules"])
            if knowledge_text:
                sections.append(f"<knowledge>\n{knowledge_text}\n</knowledge>")
                used_tokens += self._count_tokens(knowledge_text)

        # --- Section 3: Module Context (blend-weight-aware) ---
        if active_modules:
            module_context = self._build_weighted_module_context(
                active_modules, blend_weights, budgets["modules"],
            )
            if module_context:
                sections.append(f"<modules>\n{module_context}\n</modules>")
                used_tokens += self._count_tokens(module_context)

        # --- Section 4: Long-Term Memory ---
        if memory_context and memory_context.get("long_term"):
            lt_text = self._fit_to_budget(
                memory_context["long_term"], budgets["memory"],
            )
            if lt_text:
                sections.append(f"<memory>\n{lt_text}\n</memory>")
                used_tokens += self._count_tokens(lt_text)

        # --- Section 4b: Tool Descriptions (Phase 5) ---
        if memory_context and memory_context.get("tools"):
            tools_text = self._fit_to_budget(
                memory_context["tools"], budgets["modules"] // 2,
            )
            if tools_text:
                sections.append(f"<tools>\n{tools_text}\n</tools>")
                used_tokens += self._count_tokens(tools_text)

        # --- Section 4c: Micro-Adapter Context (Phase 4) ---
        if memory_context and memory_context.get("adapter"):
            adapter_text = self._fit_to_budget(
                memory_context["adapter"], budgets["memory"] // 3,
            )
            if adapter_text:
                sections.append(f"<adapter>\n{adapter_text}\n</adapter>")
                used_tokens += self._count_tokens(adapter_text)

        # --- Section 4c: Compressed State (Phase 4) ---
        if memory_context and memory_context.get("compressed"):
            compressed_text = self._fit_to_budget(
                memory_context["compressed"], budgets["memory"] // 2,
            )
            if compressed_text:
                sections.append(f"<compressed>\n{compressed_text}\n</compressed>")
                used_tokens += self._count_tokens(compressed_text)

        # --- Section 5: Conversation History ---
        if memory_context and memory_context.get("conversation"):
            remaining = total_budget - used_tokens - user_tokens
            conv_budget = min(budgets["conversation"], remaining)
            conv_text = self._fit_conversation(
                memory_context["conversation"], conv_budget,
            )
            if conv_text:
                sections.append(f"<conversation>\n{conv_text}\n</conversation>")
                used_tokens += self._count_tokens(conv_text)

        # --- Section 6: Output Format Instructions ---
        format_instruction = self._get_format_instruction(active_modules, npc_profile)
        if format_instruction:
            sections.append(f"<format>\n{format_instruction}\n</format>")
            used_tokens += self._count_tokens(format_instruction)

        # --- Section 7: User Input --- (added by chat formatter)

        # Apply chat format template
        system_content = sections[0]  # First section is always <system>...</system>
        context_sections = sections[1:]  # Everything else is context

        prompt = self._apply_chat_format(
            system_content=system_content,
            context_sections=context_sections,
            user_input=user_input,
        )

        total_tokens = self._count_tokens(prompt)
        logger.debug(
            f"Structured prompt: ~{total_tokens} tokens "
            f"(budget: {total_budget}, used: {used_tokens})"
        )

        return prompt

    def _calculate_budgets(self, available_tokens: int) -> dict[str, int]:
        """
        Calculate per-section token budgets from config percentages.

        Returns dict mapping section name → token budget.
        """
        return {
            "system": int(available_tokens * self.config.budget_system),
            "modules": int(available_tokens * self.config.budget_modules),
            "memory": int(available_tokens * self.config.budget_memory),
            "conversation": int(available_tokens * self.config.budget_conversation),
            "reserve": int(available_tokens * self.config.budget_reserve),
        }

    def _fit_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        if not text:
            return ""

        tokens = self._count_tokens(text)
        if tokens <= max_tokens:
            return text

        # Truncate by approximate character ratio
        ratio = max_tokens / max(tokens, 1)
        cut_at = int(len(text) * ratio)
        truncated = text[:cut_at].rsplit(" ", 1)[0]  # Don't break mid-word
        return truncated + "\n[...truncated]"

    def _fit_conversation(self, conversation: str, max_tokens: int) -> str:
        """Fit conversation history to budget, keeping most recent turns."""
        tokens = self._count_tokens(conversation)
        if tokens <= max_tokens:
            return conversation

        # Keep most recent turns
        lines = conversation.split("\n")
        kept = []
        used = 0
        for line in reversed(lines):
            line_tokens = self._count_tokens(line)
            if used + line_tokens > max_tokens:
                break
            kept.append(line)
            used += line_tokens

        if not kept:
            return ""

        kept.reverse()
        return "[...earlier conversation omitted...]\n" + "\n".join(kept)

    def _build_weighted_module_context(
        self,
        active_modules: list[LoadedModule],
        blend_weights: Optional[list] = None,
        budget: int = 500,
    ) -> str:
        """
        Build module context with token allocation proportional to blend weights.

        Higher-weighted modules get more prompt space.
        """
        if not active_modules:
            return ""

        # Build weight lookup
        weight_map = {}
        if blend_weights:
            for w in blend_weights:
                weight_map[w.name] = w.weight

        # Sort modules by weight (highest first)
        sorted_modules = sorted(
            active_modules,
            key=lambda m: weight_map.get(m.manifest.name, 0.5),
            reverse=True,
        )

        # Allocate tokens per module proportionally
        total_weight = sum(weight_map.get(m.manifest.name, 0.5) for m in sorted_modules)
        contexts = []

        for module in sorted_modules:
            mod_name = module.manifest.name
            mod_weight = weight_map.get(mod_name, 0.5)
            mod_budget = int(budget * (mod_weight / total_weight)) if total_weight > 0 else budget // len(sorted_modules)

            parts = []

            # System prompt injection
            if module.manifest.system_prompt_injection:
                parts.append(module.manifest.system_prompt_injection)

            # Context injection
            if module.manifest.context_injection:
                parts.append(module.manifest.context_injection)

            if parts:
                combined = "\n".join(parts)
                fitted = self._fit_to_budget(combined, mod_budget)
                weight_str = f" (weight={mod_weight:.2f})" if blend_weights else ""
                contexts.append(f"[{mod_name}{weight_str}]\n{fitted}")

        return "\n\n".join(contexts) if contexts else ""

    # ── Phase 1: Simple Assembly (unchanged) ──────────────────

    def _assemble_simple(
        self,
        user_input: str,
        active_modules: Optional[list[LoadedModule]] = None,
        memory_context: Optional[dict[str, str]] = None,
        npc_profile: Optional[dict] = None,
        system_knowledge: Optional[str] = None,
    ) -> str:
        """v1 simple concatenation assembly — backward compatible."""
        sections = []
        token_budget = self.config.max_prompt_tokens
        used_tokens = 0

        # System Prompt
        system_prompt = self._build_system_prompt(active_modules, npc_profile)
        sys_tokens = self._count_tokens(system_prompt)
        sections.append(system_prompt)
        used_tokens += sys_tokens

        # System Knowledge
        if system_knowledge:
            knowledge_section = f"\n[Knowledge]\n{system_knowledge}\n"
            k_tokens = self._count_tokens(knowledge_section)
            if used_tokens + k_tokens < token_budget * 0.4:
                sections.append(knowledge_section)
                used_tokens += k_tokens

        # Module Context
        if active_modules:
            module_context = self._build_module_context_simple(active_modules)
            if module_context:
                m_tokens = self._count_tokens(module_context)
                if used_tokens + m_tokens < token_budget * 0.6:
                    sections.append(module_context)
                    used_tokens += m_tokens

        # Long-Term Memory
        if memory_context and memory_context.get("long_term"):
            lt_memory = memory_context["long_term"]
            lt_tokens = self._count_tokens(lt_memory)
            if used_tokens + lt_tokens < token_budget * 0.7:
                sections.append(f"\n{lt_memory}\n")
                used_tokens += lt_tokens

        # Tool Descriptions (Phase 5)
        if memory_context and memory_context.get("tools"):
            tools = memory_context["tools"]
            t_tokens = self._count_tokens(tools)
            if used_tokens + t_tokens < token_budget * 0.65:
                sections.append(f"\n{tools}\n")
                used_tokens += t_tokens

        # Micro-Adapter Context (Phase 4)
        if memory_context and memory_context.get("adapter"):
            adapter = memory_context["adapter"]
            a_tokens = self._count_tokens(adapter)
            if used_tokens + a_tokens < token_budget * 0.7:
                sections.append(f"\n{adapter}\n")
                used_tokens += a_tokens

        # Compressed State (Phase 4)
        if memory_context and memory_context.get("compressed"):
            compressed = memory_context["compressed"]
            c_tokens = self._count_tokens(compressed)
            if used_tokens + c_tokens < token_budget * 0.75:
                sections.append(f"\n{compressed}\n")
                used_tokens += c_tokens

        # Conversation History
        if memory_context and memory_context.get("conversation"):
            conversation = memory_context["conversation"]
            conv_tokens = self._count_tokens(conversation)
            remaining = token_budget - used_tokens - self._count_tokens(user_input) - 50
            if conv_tokens <= remaining:
                sections.append(f"\n[Conversation]\n{conversation}\n")
                used_tokens += conv_tokens
            else:
                truncated = self._fit_conversation(conversation, remaining)
                if truncated:
                    sections.append(f"\n[Conversation]\n{truncated}\n")
                    used_tokens += self._count_tokens(truncated)

        # Output Format
        format_instruction = self._get_format_instruction(active_modules, npc_profile)
        if format_instruction:
            sections.append(format_instruction)
            used_tokens += self._count_tokens(format_instruction)

        # User Input
        sections.append(f"\nUser: {user_input}\nAssistant:")

        prompt = "\n".join(sections)
        total_tokens = self._count_tokens(prompt)
        logger.debug(f"Simple prompt: ~{total_tokens} tokens (budget: {token_budget})")
        return prompt

    # ── Chat Format Templates ─────────────────────────────────

    def _apply_chat_format(self, system_content: str, context_sections: list[str],
                           user_input: str) -> str:
        """
        Wrap prompt content in the appropriate chat template.

        Formats:
            chatml:  <|im_start|>system\n...<|im_end|>  (TinyLlama, Qwen, etc.)
            llama2:  [INST] <<SYS>>\n...<</SYS>>\n...[/INST]
            alpaca:  ### Instruction:\n...\n### Response:
            raw:     XML tags only (no chat template, for testing)
        """
        fmt = self.config.chat_format

        # Strip XML tags from system content for chat templates
        system_text = system_content
        for tag in ["<system>", "</system>"]:
            system_text = system_text.replace(tag, "")
        system_text = system_text.strip()

        # Build context block from remaining sections
        context_block = "\n\n".join(context_sections) if context_sections else ""

        if fmt == "chatml":
            # ChatML format (TinyLlama, Qwen, OpenChat, etc.)
            parts = [f"<|im_start|>system\n{system_text}"]
            if context_block:
                parts.append(f"\n{context_block}")
            parts.append("<|im_end|>")
            parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

        elif fmt == "phi3":
            # Phi-3 / Phi-3.5 format
            parts = [f"<|system|>\n{system_text}"]
            if context_block:
                parts.append(f"\n{context_block}")
            parts.append("<|end|>")
            parts.append(f"<|user|>\n{user_input}<|end|>")
            parts.append("<|assistant|>\n")
            return "\n".join(parts)

        elif fmt == "llama3":
            # Llama 3 / 3.2 format
            parts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_block}<|eot_id|>"]
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        elif fmt == "llama2":
            # Llama 2 format
            parts = [f"[INST] <<SYS>>\n{system_text}\n<</SYS>>"]
            if context_block:
                parts.append(f"\n{context_block}\n")
            parts.append(f"\n{user_input} [/INST]")
            return "\n".join(parts)

        elif fmt == "alpaca":
            # Alpaca format
            parts = [f"### System:\n{system_text}\n"]
            if context_block:
                parts.append(f"{context_block}\n")
            parts.append(f"### Instruction:\n{user_input}\n")
            parts.append("### Response:\n")
            return "\n".join(parts)

        else:
            # Raw format (XML tags, no chat template)
            all_sections = [system_content] + context_sections
            all_sections.append(f"<user>\n{user_input}\n</user>\n\nAssistant:")
            return "\n\n".join(all_sections)

    # ── Shared Helpers ────────────────────────────────────────

    def _build_system_prompt(
        self,
        active_modules: Optional[list[LoadedModule]] = None,
        npc_profile: Optional[dict] = None,
    ) -> str:
        """Build the system prompt with module injections."""
        parts = [self.config.system_prompt.strip()]

        # NPC personality override
        if npc_profile:
            name = npc_profile.get("name", "NPC")
            personality = npc_profile.get("personality", "")
            backstory = npc_profile.get("backstory", "")
            parts = [
                f"You are {name}. {personality}",
                backstory,
            ]

        # Module system prompt injections (used in simple mode)
        if active_modules and self.config.mode == "simple":
            injections = []
            for module in sorted(active_modules, key=lambda m: m.manifest.priority, reverse=True):
                if module.manifest.system_prompt_injection:
                    injections.append(module.manifest.system_prompt_injection)

            if injections:
                parts.append("\n[Active capabilities]")
                parts.extend(injections)

        return "\n".join(parts)

    def _build_module_context_simple(self, active_modules: list[LoadedModule]) -> str:
        """Build context injections from active modules (v1 simple mode)."""
        contexts = []
        for module in active_modules:
            if module.manifest.context_injection:
                contexts.append(
                    f"[{module.manifest.name}]\n"
                    f"{module.manifest.context_injection}"
                )

        if not contexts:
            return ""

        return "\n[Module Context]\n" + "\n\n".join(contexts) + "\n"

    def _get_format_instruction(
        self,
        active_modules: Optional[list[LoadedModule]] = None,
        npc_profile: Optional[dict] = None,
    ) -> Optional[str]:
        """Get output format instructions from modules or NPC profile."""
        if npc_profile and npc_profile.get("json_output"):
            schema = npc_profile.get("output_schema", {})
            return (
                f"Respond ONLY with valid JSON matching this schema:\n"
                f"{schema}"
            )

        if active_modules:
            for module in active_modules:
                if module.manifest.output_format:
                    return module.manifest.output_format

        return None

    def estimate_remaining_tokens(
        self,
        user_input: str,
        active_modules: Optional[list[LoadedModule]] = None,
        memory_context: Optional[dict[str, str]] = None,
    ) -> int:
        """Estimate how many tokens are left for generation after prompt assembly."""
        prompt = self.assemble(
            user_input=user_input,
            active_modules=active_modules,
            memory_context=memory_context,
        )
        prompt_tokens = self._count_tokens(prompt)
        return max(0, self.config.max_prompt_tokens - prompt_tokens)
