# Multi-Agent Architecture — Test Results

**Date:** 2026-04-01
**Branch:** `multi-agent-architect`
**Architecture:** Architect (3B) -> Workers (0.5B + augmentors) -> Assembler (3B)
**Hardware:** RTX 3060 12GB, both models loaded simultaneously (~3.5GB VRAM)

---

## Test 1: Task Queue System

**Prompt:** "build a task queue with sqlite backend, worker threads, and a cli interface"

### Timing

| Phase | Time | % of Total |
|-------|-----:|:----------:|
| Architect (plan) | 28.2s | 19% |
| Workers (5 pieces) | 27.9s | 19% |
| Assembler (stitch) | 92.8s | 62% |
| **Total** | **148.9s** | |

### Quality Assessment

| Criteria | Score | Notes |
|----------|:-----:|-------|
| **Decomposition** | 8/10 | 5 sensible components: TaskQueue, Task, WorkerThread, SQLiteBackend, CLI. Dependencies correctly identified. |
| **Individual pieces** | 7/10 | Each worker produced runnable code. SQLiteBackend has parameterized queries. CLI uses argparse properly. |
| **Integration** | 3/10 | Assembler duplicated TaskQueue class. Duplicate imports. Pieces don't wire together — CLI creates its own queue instead of using shared one. |
| **Correctness** | 4/10 | WorkerThread uses deque as context manager (wrong). Task class is a stub. No table creation SQL. Code won't run as-is. |
| **Completeness** | 6/10 | All 5 requested components present. 149 lines. Missing: table creation, proper thread start/join, real task execution. |

### Output Stats
- Lines: 149
- Classes: 5 (TaskQueue x2 duplicate, Task, WorkerThread, SQLiteBackend)
- Functions: 1 (main)
- Imports: 5 unique (threading, sqlite3, queue, argparse, time) — 2 duplicated in output

---

## Test 2: REST API with Auth

**Prompt:** "build a REST API with user registration, password hashing, token-based authentication, and a protected endpoint that returns user profile data"

### Timing

| Phase | Time | % of Total |
|-------|-----:|:----------:|
| Architect (plan) | 31.8s | 19% |
| Workers (5 pieces) | 26.9s | 16% |
| Assembler (stitch) | 105.3s | 65% |
| **Total** | **163.9s** | |

### Quality Assessment

| Criteria | Score | Notes |
|----------|:-----:|-------|
| **Decomposition** | 9/10 | 5 clean components: UserRegistration, PasswordHasher, TokenGenerator, TokenVerifier, UserProfileEndpoint. Correct dep: profile needs token verifier. |
| **Individual pieces** | 5/10 | TokenGenerator uses JWT correctly. UserRegistration uses Pydantic. But PasswordHasher added random dunder methods (__iter__, __next__, __enter__, __exit__, __get__, __set__) that make no sense — augmentor injected the wrong pattern. |
| **Integration** | 5/10 | No duplicated classes this time. Main demo section shows usage flow. Imports mostly clean. But defaultdict not imported (used in UserProfileEndpoint). |
| **Correctness** | 4/10 | PasswordHasher.hash_password calls .hexdigest() on bytes (AttributeError). TokenVerifier is a stub (always returns "user_id"). Missing hashlib import. jose not stdlib. |
| **Completeness** | 7/10 | All 5 components present. 140 lines. Has demo section showing full flow. Missing: actual database, proper password hashing (hashlib/bcrypt), real token verification. |

### Output Stats
- Lines: 140
- Classes: 5 (UserRegistration, PasswordHasher, TokenGenerator, TokenVerifier, UserProfileEndpoint)
- Imports: 6 (pydantic, datetime, jose, typing — but jose is not stdlib, defaultdict missing)

---

## Comparative Analysis

### Speed

| Metric | Test 1 | Test 2 | Average |
|--------|:------:|:------:|:-------:|
| Plan time | 28.2s | 31.8s | **30.0s** |
| Build time (total) | 27.9s | 26.9s | **27.4s** |
| Build per piece | 5.6s | 5.4s | **5.5s** |
| Assemble time | 92.8s | 105.3s | **99.0s** |
| **Total** | **148.9s** | **163.9s** | **156.4s** |

**Bottleneck: Assembly at 62-65% of total time.** The 3B model generating 140-149 lines of combined code takes ~100s. Workers are fast (5.5s/piece average) thanks to the 0.5B + augmentors.

### Quality

| Criteria | Test 1 | Test 2 | Notes |
|----------|:------:|:------:|-------|
| Decomposition | 8/10 | 9/10 | Architect (3B) excels at planning |
| Individual pieces | 7/10 | 5/10 | Workers hit-or-miss; augmentor can inject wrong pattern |
| Integration | 3/10 | 5/10 | Biggest weakness — pieces don't wire together cleanly |
| Correctness | 4/10 | 4/10 | Code has runtime errors (wrong methods, missing imports) |
| Completeness | 6/10 | 7/10 | All components present but stubs/missing logic |
| **Average** | **5.6** | **6.0** | |

### Efficiency

| Metric | Value | Assessment |
|--------|:-----:|------------|
| VRAM usage | ~3.5GB | 29% of 12GB — very efficient |
| Tokens generated (est.) | ~2000 | Across all phases |
| Lines per second | ~1.0 | Total output / total time |
| Worker throughput | ~5.5s/piece | Fast — augmentors keep workers efficient |

---

## Key Findings

### What Works

1. **Architect decomposition is excellent.** The 3B model consistently breaks complex tasks into 4-5 sensible, well-named components with correct dependency ordering. This is where the large model adds clear value.

2. **Workers are fast.** The 0.5B + augmentors pipeline produces code in 3-10s per piece. The augmentor system fires correctly for domain-specific tasks (sqlite3, argparse, threading).

3. **Both models fit on GPU simultaneously.** 3.5GB total VRAM for both models — leaves 8.5GB headroom on a 12GB card.

4. **The pipeline flows end-to-end.** No crashes, no hangs, clean handoffs between architect -> workers -> assembler.

### What Doesn't Work

1. **Assembly is the bottleneck — both in speed and quality.** The 3B model takes ~100s to stitch pieces and produces integration errors (duplicate classes, missing imports, pieces that don't connect). Assembly is 62-65% of total time.

2. **Workers don't know about each other.** Each worker builds in isolation. The CLI worker creates its own TaskQueue instead of using the one from the TaskQueue worker. Workers need the interface specs from other pieces.

3. **Augmentor can inject wrong patterns.** The PasswordHasher got __iter__, __next__, __enter__, __exit__, __get__, __set__ methods because the augmentor matched a descriptor/iterator pattern instead of a hashing pattern. The 0.5B faithfully copied the wrong example.

4. **No execution validation.** The pipeline produces code that looks plausible but has runtime errors (missing imports, wrong method calls). No step verifies the code actually runs.

### Improvement Priorities

| Priority | Fix | Expected Impact |
|:--------:|-----|-----------------|
| 1 | **Give workers interface context** — pass other subtask names/signatures so pieces can reference each other | Integration 3/10 -> 7/10 |
| 2 | **Add execution check after assembly** — try to run the code, feed errors back to assembler for a fix pass | Correctness 4/10 -> 7/10 |
| 3 | **Speed up assembly** — reduce assembler prompt size, or have it just concatenate + fix imports instead of rewriting | Assembly time 100s -> 30s |
| 4 | **Add hashing/auth to failure routing keywords** — prevent wrong augmentor pattern on security-related code | Worker quality on auth tasks |
