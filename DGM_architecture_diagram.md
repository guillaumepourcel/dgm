# DGM Architecture Diagram

This document gives a visual map of how Darwin Godel Machine (DGM) runs end-to-end.

## High-level flow

```mermaid
flowchart TD
    A[Run DGM_outer.py] --> B[Initialize run archive and output]
    B --> C[Generation loop]
    C --> D[Choose parents and targets]
    D --> E[Launch parallel self_improve workers]

    E --> F[Worker setup container]
    F --> G[Apply ancestor patches]
    G --> H[Run coding_agent.py to generate patch]
    H --> I[Evaluate patch in benchmark harness]
    I --> J[Write child metadata]

    J --> K[Filter compiled and valid children]
    K --> L[Update archive]
    L --> M[Append dgm_metadata.jsonl]
    M --> C
```

## High-level workflow elements explained

- **A. Run DGM_outer.py**  
  Starts the outer evolutionary process and parses run options (generation count, worker count, archive update mode, SWE vs Polyglot mode).

- **B. Initialize run archive and output**  
  Creates the run directory under `output_dgm/`, loads or seeds the archive, and prepares generation state.

- **C. Generation loop**  
  Main iteration loop. Each pass produces a new batch of candidate descendants from the current archive.

- **D. Choose parents and targets**  
  Selects which archived node(s) to mutate and what to improve (e.g., unresolved tasks, empty-patch handling, context-length issues).

- **E. Launch parallel self_improve workers**  
  Spawns multiple independent self-improvement attempts concurrently, each trying to create one child patch.

- **F. Worker setup container**  
  Inside each worker, builds/reuses Docker image and starts an isolated container to safely run code edits and evaluation.

- **G. Apply ancestor patches**  
  Reconstructs the selected parent state by applying lineage patches in order so the child starts from that exact inherited version.

- **H. Run coding_agent.py to generate patch**  
  Calls the coding agent in-container; the agent uses tool-calling LLM logic to modify code and emit `model_patch.diff`.

- **I. Evaluate patch in benchmark harness**  
  Runs SWE-bench or Polyglot harness to measure task resolution and produce objective fitness metrics.

- **J. Write child metadata**  
  Stores per-child outputs (`metadata.json`, patch, chat/eval logs) for traceability and later archive decisions.

- **K. Filter compiled and valid children**  
  Discards runs that are invalid (e.g., empty/non-compiling outcomes or incomplete eval structure).

- **L. Update archive**  
  Adds accepted children to the archive according to policy (`keep_all` or `keep_better`).

- **M. Append dgm_metadata.jsonl**  
  Logs generation-level summary (entries tried, children produced, accepted children, resulting archive), then loops back to **C**.

## Component map

```mermaid
flowchart LR
    subgraph Outer[Outer Evolution Layer]
        O1[DGM_outer.py]
    end

    subgraph Improve[Self Improve Worker Layer]
        S1[self_improve_step.py]
        S2[Docker container lifecycle]
        S3[Patch lineage apply]
        S4[Evaluation trigger]
    end

    subgraph Agent[Coding Agent Layer]
        A1[coding_agent.py and coding_agent_polyglot.py]
        A2[llm_withtools.py]
        A3[llm.py]
        A4[tools folder]
    end

    subgraph Eval[Benchmark Layer]
        E1[swe_bench harness]
        E2[polyglot harness]
        E3[reports and metrics]
    end

    subgraph Shared[Shared Infra]
        U1[utils docker_utils.py]
        U2[utils evo_utils.py]
        U3[utils eval_utils.py]
        U4[prompts templates]
    end

    O1 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> A1
    A1 --> A2
    A2 --> A3
    A2 --> A4
    S1 --> S4
    S4 --> E1
    S4 --> E2
    E1 --> E3
    E2 --> E3

    O1 -.uses.-> U2
    S1 -.uses.-> U1
    S1 -.uses.-> U2
    A1 -.uses.-> U3
    S1 -.prompts.-> U4
```

## Output structure (what gets written)

```mermaid
flowchart TD
    R[output_dgm run_id] --> R1[dgm_outer.log]
    R --> R2[dgm_metadata.jsonl]
    R --> C[child_run_id folder]

    C --> C1[metadata.json]
    C --> C2[model_patch.diff]
    C --> C3[self_evo.md]
    C --> C4[predictions and eval artifacts]
```

## Reading guide

- `DGM_outer.py` decides **which descendants to try next**.
- `self_improve_step.py` creates one descendant and measures it.
- `coding_agent*.py` + `llm_withtools.py` perform **actual code modification**.
- Harnesses (`swe_bench/`, `polyglot/`) provide **fitness signals** used for archive updates.
