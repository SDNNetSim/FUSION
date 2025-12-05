
# Prompt for Claude – Phase 1 Core Domain (v5-final-docs)

You are helping refactor a large open-source optical network simulator called **FUSION**.

You must work **only on Phase 1: Core Domain Model**, as defined in
`.claude/v4-docs/migration/phase_1_core_model.md`.  
Phase 1 consists of micro-phases P1.1–P1.5:

- **P1.1: Domain Scaffolding** (SimulationConfig, domain package skeleton)
- **P1.2: Request Wrapper** (`Request`, enums, legacy adapters)
- **P1.3: Lightpath Wrapper**
- **P1.4: Result Objects**
- **P1.5: StatsCollector Skeleton**

No other phases (orchestrator, pipelines, RL/ML, legacy removal, etc.) are in scope for this task.

---

## Goal

Using the existing v4 docs + `phase_1_core_model.md`, design the **Phase-1 doc set for v5**:

- A new directory tree under **`.claude/v5-final-docs/phase-1-core-domain/`**
- Short, clearly labeled **micro-task markdown files** for each sub-phase P1.1–P1.5
- Each micro-task explicitly declares:
  - Its **ID** (for example, `P1.2.b`)
  - Its **purpose**
  - The **exact context files** to read (small and explicit)
  - The **concrete output** another future Claude call should produce

This run should produce **only documentation structure and task specs**, not actual code changes.

---

## Inputs you may assume

You can assume the human will provide you with:

- `.claude/v4-docs/migration/phase_1_core_model.md` (Phase-1 objectives, constraints, and examples)
- Relevant v4 docs such as:
  - `architecture/domain_model.md`
  - `architecture/result_objects.md`
  - `architecture/stats_and_metrics.md`
- Any other specific files **only when you name them as context** in a micro-task

You must not rely on reading “the whole repo” at once. Design things so a programmer can feed you small, targeted files later.

---

## Constraints / Style

- All new docs must live under:  
  **`.claude/v5-final-docs/phase-1-core-domain/`**
- Keep Phase 1 **strictly additive**, matching `phase_1_core_model.md`:
  - New domain classes (`SimulationConfig`, `Request`, `Lightpath`, result types, `StatsCollector` skeleton)
  - No changes to existing function signatures
  - `run_comparison.py` and existing tests must keep passing
- Prefer many short files over a few long ones.
- Each micro-task file should:
  - Be short enough for a single Claude message (a few screens max)
  - Reference at most **3–5 context files**
  - Be **self-contained**: a programmer can run that task without opening any other micro-task

---

## Sub-phase and micro-task labeling

For Phase 1 you must:

1. Use sub-phase directories:

   - `phase-1-core-domain/P1.1_domain_scaffolding/`
   - `phase-1-core-domain/P1.2_request_wrapper/`
   - `phase-1-core-domain/P1.3_lightpath_wrapper/`
   - `phase-1-core-domain/P1.4_result_objects/`
   - `phase-1-core-domain/P1.5_stats_collector/`

2. Inside each sub-phase directory, define:

   - A small **index file** named `P1.X.index.md` describing:
     - Goals of that sub-phase
     - High-level constraints
     - List of its micro-tasks (with IDs)

   - Several **micro-task files**, named:

     - `P1.X.a_*.md`
     - `P1.X.b_*.md`
     - `P1.X.c_*.md`
     - …

   Each micro-task file must start with a header block:

   ```markdown
   # Task ID: P1.X.a – <short name>

   **Sub-phase:** P1.X  
   **Scope:** Phase 1 – Core Domain Model only  
   **Task type:** (choose one: context-extraction | design | refactor-plan | verification-plan)
   ```

---

## Context handling rules

For every micro-task file, include a **“Context to load”** section:

- List **exact filenames** and optionally line ranges to be provided to Claude when running this task.
- If several micro-tasks share context, define a **small shared context file** in the same sub-phase, such as:

  - `P1.2.shared_context_request_legacy.md`

  and have each task reference that instead of the large v4 docs.

Examples of context sections:

```markdown
## Context to load before running this task

- `.claude/v4-docs/migration/phase_1_core_model.md` (P1.2 section only)
- `.claude/v4-docs/architecture/domain_model.md` (Request-related content)
- `.claude/v5-final-docs/phase-1-core-domain/P1.2_request_wrapper/P1.2.shared_context_request_legacy.md`
```

You are responsible for deciding **which shared context files to create** for each sub-phase so that later tasks stay small and focused.

---

## What to cover for each sub-phase

Your micro-task design must fully cover the Phase-1 objectives from `phase_1_core_model.md` and nothing beyond.

### P1.1 – Domain Scaffolding

Design tasks such that:

- `fusion/domain/__init__.py` and `fusion/domain/config.py` are planned and documented.
- `SimulationConfig` dataclass is fully specified, including:
  - Fields, types, and immutability
  - `from_engine_props` and `to_engine_props` behavior
- There is a micro-task to:
  - Identify all current config sources (e.g., `engine_props` usage points)
  - Plan how and where `SimulationConfig` will be constructed and used **without changing signatures yet**

### P1.2 – Request Wrapper

Tasks must:

- Specify the `Request` dataclass, `RequestStatus`, and any enums/flags (blocked, routed, protected, sliced, groomed, etc.).
- Define the exact shape of `from_legacy_dict` and `to_legacy_dict`, based on current request dicts.
- Include at least:
  - A **context-extraction task** to summarise how “request dicts” are currently structured across the codebase.
  - A **design task** that finalizes the Request API based on that summary.
  - A **verification-plan task** that defines tests for requests, including edge cases (partial grooming, slicing, protection flags).

### P1.3 – Lightpath Wrapper

Tasks must:

- Specify the `Lightpath` dataclass fields, properties (utilization, num_slots, num_hops), and legacy adapters.
- Include micro-tasks to:
  - Identify all current lightpath representations and where they live.
  - Align those representations with the new Lightpath object.
  - Define tests for Lightpath creation, utilization logic, and legacy conversions.

### P1.4 – Result Objects

Tasks must:

- Design concrete result dataclasses: `RouteResult`, `SpectrumResult`, `GroomingResult`, `SlicingResult`, `SNRResult`, `AllocationResult` (names can be adjusted if Phase-1 doc specifies).
- Carefully standardize:
  - `success` flags
  - `block_reason` (and relation to `BlockReason` enum from P1.2)
  - Any metrics required by the comparison scripts and StatsCollector.
- Include micro-tasks for:
  - Mapping existing return structures (tuples/dicts) to the new result objects.
  - Defining minimal interfaces so that higher-level components can consume results later without knowing internals.

### P1.5 – StatsCollector Skeleton

Tasks must:

- Plan the location and API of `StatsCollector` (for example, `fusion/stats/collector.py`).
- Decide exactly **what it tracks in Phase 1** (blocking probability, reasons, grooming/slicing/protection counts, modulation use, etc.) consistent with `phase_1_core_model.md`.
- Clarify how `StatsCollector` consumes:
  - `SimulationConfig`
  - `Request`
  - `AllocationResult`
- Include micro-tasks for:
  - Designing fields and helper methods (e.g., `record_arrival`, `record_snr`, `blocking_probability`).
  - Planning test cases.
  - Ensuring its output format stays compatible with `run_comparison.py`.

---

## Deliverables for this prompt

Produce:

1. A **directory tree** for `.claude/v5-final-docs/phase-1-core-domain/`, showing:
   - Sub-phase folders P1.1–P1.5
   - Index files
   - All micro-task files with their IDs in filenames

2. For **each micro-task file**, a short spec block containing:
   - **Task ID** (e.g., `P1.3.b`)
   - **Filename**
   - **One-sentence purpose**
   - **Context to load before running the task** (list of files; include any shared context files you propose)
   - **Outputs** – a bullet list of what a future Claude call should produce

Keep the answer concise, but make the structure complete enough that a programmer can immediately start creating these `.md` files and then drive the code changes by feeding you one task at a time.
