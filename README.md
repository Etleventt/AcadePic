# AcadePic

A prompt-centered studio for generating and refining academic figures from paper content.

This repository is a fork and adaptation of [PaperBanana](https://github.com/dwzhu-pku/PaperBanana), focused on a more practical workflow for paper authors:

- parse paper sections from LaTeX
- draft figure prompts with `Planner + Stylist`
- generate multiple candidate images
- run manual `Critic` refinement round by round
- keep persistent history for later comparison

The current primary interface is **Prompt Studio**, a standalone FastAPI app.  
The original Streamlit-based PaperBanana demo is still present in the repo, but it is no longer the recommended entry point for this fork.

## Relationship to PaperBanana

AcadePic is built on top of the open-source PaperBanana codebase and still reuses part of its agent/provider stack, including:

- Planner
- Stylist
- Visualizer
- Critic
- provider/runtime utilities

This means AcadePic is currently an **independent app inside the forked repository**, not yet a fully separate standalone codebase.

## Main Entry: Prompt Studio

Prompt Studio is the recommended way to use this fork.

It provides:

- full-paper paste / file loading
- LaTeX section tree parsing (`\\section`, `\\subsection`, `\\subsubsection`)
- direct section-to-method-text selection
- editable Planner / Stylist prompt outputs
- single-image trial generation
- batch candidate generation with configurable concurrency
- streaming status updates
- manual Critic workflow
- persistent history with archive support

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Etleventt/AcadePic.git
cd AcadePic
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare configuration

This repository ignores `configs/model_config.yaml`, so you should create your own local config from the template:

```bash
cp configs/model_config.template.yaml configs/model_config.yaml
```

Then edit:

- API keys
- base URLs
- default text/image models
- optional default paper file path

Alternatively, you can leave the file mostly empty and fill values directly in the Prompt Studio UI.

### 5. Run Prompt Studio

```bash
./scripts/run_prompt_studio.sh
```

Then open:

- [http://127.0.0.1:8610/studio](http://127.0.0.1:8610/studio)

## Prompt Studio Workflow

### Paper parsing

You can:

- paste the full paper into the text box
- or configure a default local file path and load it directly

Prompt Studio will parse:

- `\\section`
- `\\subsection`
- `\\subsubsection`

and render them as a collapsible tree.  
Selecting a section uses the **entire subtree content**, not just the local heading block.

### Prompt generation

Prompt generation uses:

- `Planner`
- `Stylist`

The retrieval stage from the original PaperBanana pipeline is intentionally skipped in this workflow.

Outputs are:

- editable
- stream-updated in the UI
- stored into history incrementally

### Candidate generation

You can:

- try one image first
- then generate multiple candidates in batch

New generations are appended as new candidates instead of replacing old ones, so different prompt revisions can be compared side by side.

### Critic refinement

Critic is a **manual step**:

1. choose a generated candidate
2. run Critic
3. inspect:
   - raw Critic output
   - critique suggestions
   - revised prompt
4. optionally:
   - apply the revised prompt
   - generate a new single candidate from the revised prompt

Critic does **not** overwrite the original Planner / Stylist outputs unless you explicitly apply the revised prompt.

### History

Prompt Studio stores history persistently and supports:

- reload
- append new candidates
- archive (soft delete)

Old `results/demo/*.json` records from the original Streamlit workflow are also recognized and can be loaded into Prompt Studio as legacy history entries.

## Configuration Notes

Prompt Studio supports both:

- `openai_compatible`
- `google_compatible`

The UI can:

- load defaults from `configs/model_config.yaml`
- save current settings back to the config
- fetch supported text/image models from the configured provider

## Legacy Interfaces

This repo still contains the original or earlier interfaces, including:

- `demo.py`
- `main.py`
- visualization helpers under `visualize/`

Those are not the main focus of this fork anymore.  
If you are only interested in AcadePic, use Prompt Studio.

## What Is Not Part of the Main Prompt Studio Path

Some files in this repository are personal experiments, local paper materials, or auxiliary tools and are not required for the main Prompt Studio workflow, for example:

- `mypaper/`
- `prompt_studio_history/`
- `prompt_studio_archive/`
- `tmp/`
- some `nanobanana_*` configs/scripts/docs
- standalone HTML tools under `tools/`

These should generally not be treated as part of the core AcadePic application.

## License

This repository remains under the original upstream license:

- [Apache License 2.0](LICENSE)

Please retain upstream attribution and license notices when redistributing or modifying the codebase.

## Acknowledgement

AcadePic is based on the excellent open-source work:

- [PaperBanana](https://github.com/dwzhu-pku/PaperBanana)
- original paper: [PaperBanana: Automating Academic Illustration for AI Scientists](https://huggingface.co/papers/2601.23265)

This fork focuses on a more hands-on, prompt-driven workflow for real paper writing and figure iteration.
