<h1 align="center">
    Darwin Gödel Machine:<br/>Open-Ended Evolution of Self-Improving Agents
</h1>

<p align="center">
  <a href="https://github.com/jennyzzt/dgm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2505.22954"><img src="https://img.shields.io/badge/arXiv-2505.22954-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://sakana.ai/dgm/"><img src="https://img.shields.io/badge/-Blog-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://x.com/SakanaAILabs/status/1928272612431646943"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
  <a href="https://drive.google.com/drive/folders/1Kcu9TbIa9Z50pJ7S6hH9omzzD1pxIYZC?usp=sharing"><img src="https://img.shields.io/badge/Experiment%20Logs-4285F4?style=for-the-badge&logo=googledrive&logoColor=white"></a>
</p>


Repository for **Darwin Gödel Machine (DGM)**, a novel self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks.

<p align="center">
  <img src="./misc/overview.gif" width="100%" height="auto" />
</p>
<!-- <p align="center">
<img src="./misc/conceptual.svg"/></a><br>
</p> -->


## Setup
```bash
# API keys, add to ~/.bashrc
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

```bash
# Verify that Docker is properly configured in your environment.
docker run hello-world
 
# If a permission error occurs, add the user to the Docker group
sudo usermod -aG docker $USER
newgrp docker
```

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: for running analysis
sudo apt-get install graphviz graphviz-dev
pip install -r requirements_dev.txt
```

```bash
# Clone SWE-bench
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36
pip install -e .
cd ../../

# Prepare Polyglot
# Make sure git is properly configured in your environment with username and email
python -m polyglot.prepare_polyglot_dataset
```

## Running the DGM
```bash
python DGM_outer.py
```
By default, outputs will be saved in the `output_dgm/` directory.

## File Structure
- `analysis/` scripts used for plotting and analysis
- `initial/` SWE-bench logs and performance of the initial agent
- `initial_polyglot/` Polyglot logs and performance of the initial agent
- `swe_bench/` code needed for SWE-bench evaluation
- `polyglot/` code needed for Polyglot evaluation
- `prompts/` prompts used for foundation models
- `tests/` tests for the DGM system
- `tools/` tools available to the foundation models
- `coding_agent.py` main implementation of the initial coding agent
- `DGM_outer.py` entry point for running the DGM algorithm

## DGM Code Structure (Detailed)

This section explains how the code is organized and how data flows through the system during a run.

### 1) Top-level execution pipeline

The DGM runtime has three main layers:

1. **Outer evolutionary loop**
   - `DGM_outer.py`
   - Chooses parent versions from the archive, launches multiple self-improvement attempts in parallel, and updates the archive generation by generation.

2. **One self-improvement attempt**
   - `self_improve_step.py`
   - Creates an isolated Docker container, applies parent patches, asks an LLM agent to modify DGM itself, evaluates the patch, and stores metadata.

3. **LLM coding agent + tools**
   - `coding_agent.py`, `llm_withtools.py`, `llm.py`, `tools/`
   - The agent receives a problem statement, edits code via tools, and emits `model_patch.diff`.

### 2) What each important file does

- `DGM_outer.py`
  - Main entrypoint (`python DGM_outer.py`).
  - Maintains the **archive** of accepted descendants.
  - Parent selection strategies: random / score-based / score-and-children.
  - Runs many `self_improve(...)` calls concurrently with `ThreadPoolExecutor`.
  - Saves generation-level state in `output_dgm/<run_id>/dgm_metadata.jsonl`.

- `self_improve_step.py`
  - Handles one child creation and evaluation.
  - Loads benchmark subsets (SWE-bench or Polyglot mode).
  - Builds/starts container, applies ancestor patches in order.
  - Runs `coding_agent.py` to produce a new patch.
  - Evaluates with `swe_bench.harness` or `polyglot.harness`.
  - Writes run metadata to `metadata.json` for archive filtering.

- `coding_agent.py`
  - Wraps `AgenticSystem`.
  - Sends the task to the tool-using LLM agent (`chat_with_agent`).
  - Diffs current repo vs base commit and writes `model_patch.diff`.

- `coding_agent_polyglot.py`
  - Polyglot variant for non-Python tasks/languages.

- `llm_withtools.py`
  - Tool-calling orchestration for Claude/OpenAI-style APIs.
  - Loads tools dynamically from `tools/`.
  - Executes iterative loop: LLM response -> tool call -> tool result -> next LLM step.

- `llm.py`
  - Model client factory and LLM response helpers.
  - Supports Anthropic, OpenAI, Bedrock, Vertex, and others listed in `AVAILABLE_LLMS`.

### 3) Supporting directories

- `tools/`
  - Agent tools (currently shell execution and full-file editing).
  - Tool specs are exposed to the LLM through JSON schemas.

- `swe_bench/`
  - SWE-bench evaluation harness and report generation.
  - Runs issue-level containers, applies patches, executes tests, aggregates JSON reports.

- `polyglot/`
  - Polyglot benchmark harness and language-specific evaluation logic.

- `prompts/`
  - Prompt templates for coding, self-improvement diagnosis, and tool usage behavior.

- `utils/`
  - Shared helpers:
    - `docker_utils.py`: container lifecycle, copy in/out, logging
    - `evo_utils.py`: patch ancestry + archive/evaluation helpers
    - `eval_utils.py`: parse evaluation logs and score reports
    - `git_utils.py`, `common_utils.py`, etc.

- `analysis/`
  - Plotting/visualization scripts for experiment trajectories and comparisons.

- `tests/`
  - Unit tests for tools and harness-facing components.

- `initial/`, `initial_polyglot/`
  - Cached baseline run artifacts used as generation 0 seeds.

### 4) Per-generation flow inside DGM

For each generation in `DGM_outer.py`:

1. Select parents from current archive.
2. Pick an improvement target (e.g., unresolved issue, empty-patch handling, stochasticity, context-length failure).
3. Spawn parallel `self_improve` workers.
4. Each worker creates a candidate child run with `model_patch.diff`.
5. Evaluate child on benchmark subsets.
6. Keep only compiled/valid children, then update archive.
7. Append generation state to `dgm_metadata.jsonl`.

### 5) Output layout

Default output root: `output_dgm/<run_id>/`

- `dgm_outer.log`: outer-loop logs
- `dgm_metadata.jsonl`: archive state per generation
- `<child_run_id>/`
  - `metadata.json`: evaluation summary and diagnostics
  - `model_patch.diff`: patch produced by the agent
  - `self_evo.md`: coding-agent interaction trace
  - benchmark prediction/report artifacts

This structure lets you reconstruct lineage (parent -> child), inspect why a child was accepted/rejected, and analyze performance across generations.

## Logs from Experiments
This [google drive folder](https://drive.google.com/drive/folders/1Kcu9TbIa9Z50pJ7S6hH9omzzD1pxIYZC?usp=sharing) contains all the foundation model output logs from the experiments shown in the paper.

## Safety Consideration
> [!WARNING]  
> This repository involves executing untrusted, model-generated code. We strongly advise users to be aware of the associated safety risks. While it is highly unlikely that such code will perform overtly malicious actions under our current settings and with the models we use, it may still behave destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.

## Acknowledgement

The evaluation framework implementations are based on the [SWE-bench](https://github.com/swe-bench/SWE-bench) and [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) repositories.

## Citing
If you find this project useful, please consider citing:
```bibtex
@article{zhang2025darwin,
  title={Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents},
  author={Zhang, Jenny and Hu, Shengran and Lu, Cong and Lange, Robert and Clune, Jeff},
  journal={arXiv preprint arXiv:2505.22954},
  year={2025}
}
```
