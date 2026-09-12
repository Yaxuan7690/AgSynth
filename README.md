# AgSynth

AgSynth is a multi-agent reverse-synthesis framework for generating visually grounded mathematical-reasoning data. It implements the pipeline described in *AgSynth: A Multi-Agent Reverse Synthesis Framework for Multimodal Mathematical Reasoning Data Generation*.

The generator follows an image-first workflow:

1. Agent 1 extracts a seed problem's mathematical kernel and designs a divergent concept.
2. Agent 2 converts the concept and seed image into a visual blueprint.
3. Agent 3 writes and executes constrained Matplotlib code to render the image.
4. Agent 4 inspects the rendered image and generates a problem, solution, and answer.
5. Agent 5 audits every stage, enforces figure dependency, and applies a terminal two-model veto.

## Scope

This repository contains the data-generation pipeline only. It does not include seed datasets, generated AgSynth-10K data, model weights, or the GRPO training/evaluation implementation reported in the paper. Do not claim to reproduce the paper's benchmark results from this repository alone.

## Requirements

- Python 3.10 or newer
- Access to an OpenAI-compatible chat-completions service with vision support
- Two distinct vision-capable models for the terminal verification gate

Install the runtime dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Configure credentials and models

Copy the template and fill in your local values:

```bash
copy .env.example .env
```

The runtime loads `.env` automatically and never stores credentials, endpoints, or model identifiers in source code. The following variables are required for a real generation run:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Credential for the compatible API service |
| `AGENT_DEFAULT_MODEL` | Default model for Agents 1-5 |
| `AGENT_FINAL_VERIFIER_A_MODEL` | First terminal verifier |
| `AGENT_FINAL_VERIFIER_B_MODEL` | Second terminal verifier; must differ from verifier A |

`OPENAI_BASE_URL` is optional. Per-agent and text-masking model overrides are documented in [`.env.example`](.env.example).

Never commit `.env`, seed data that you cannot redistribute, generated private data, or credentials.

## Prepare a demo seed

The repository includes a minimal seed record and a script that creates its image:

```bash
python examples/create_demo_seed.py
```

This creates `examples/images/demo_seed.png`. The seed record is [`examples/seed_questions.json`](examples/seed_questions.json).

## Run

Validate paths and seed records without contacting a model:

```bash
python main.py --seed-json examples/seed_questions.json --image-dir examples/images --dry-run
```

Generate one accepted example:

```bash
python main.py --seed-json examples/seed_questions.json --image-dir examples/images --output-dir output --count 1 --workers 1 --difficulty easy
```

The public defaults are deliberately conservative: one requested sample and one worker. Increase `--count` and `--workers` only after estimating model cost and validating a small run.

## Input and output

Seed records may be a JSON array or JSONL. The recommended fields are:

```json
{
  "id": "demo_001",
  "image_path": "demo_seed.png",
  "question": "Seed problem text",
  "answer": "Seed answer",
  "solution": "Optional seed solution"
}
```

Each accepted output is appended to `generated_questions.jsonl` using:

```json
{
  "image": "output/generated_images/generated_question_1.png",
  "problem": "Generated problem text",
  "standard_answer": "Generated answer",
  "detailed_solution": ["Step 1", "Step 2"]
}
```

## Quality and safety boundaries

The pipeline uses stage-wise audits, a maximum of three stage retries, Stage 4.5 figure-dependency validation, and unanimous terminal approval from two independent verifiers. A sample failing any terminal check is rejected.

Painter code is restricted to an allowlist and runs in an isolated subprocess with a timeout and sanitized environment. This is risk reduction, not a complete security sandbox. Run the generator only in an isolated environment and review generated artifacts before public redistribution. See [SECURITY.md](SECURITY.md).

## Development checks

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
ruff check .
```

## Citation

See [CITATION.cff](CITATION.cff). If you use this code, cite the accompanying AgSynth paper.

## License

This project is licensed under the [Apache License 2.0](LICENSE). You may use,
modify, and redistribute the code under its terms. The license does not grant
rights to project names, trademarks, seed data, generated datasets, or external
model services.
