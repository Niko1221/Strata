# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. The second family, GLM-5.3
(glm-dsa), runs on its own engine `strata-glm` (`src/glm/`: CPU, optional CUDA) behind the same server - see
[docs/GLM53.md](docs/GLM53.md). It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- GLM-5.3 (glm-dsa), the second model family: how to build and run `strata-glm`, the container format, the
  checks against colibri, the measured speed and what is missing: [docs/GLM53.md](docs/GLM53.md). Its tests:
  `glm_kernels_test`, `glm_kv_test` and `glm_cuda_test` (CMake), `python -m unittest serve.test_glm`,
  `python tools/test_setup_glm.py`, and with the model `tests/glm/test_glm_tokenizer.py` and `tools/glm_validate.py`.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
