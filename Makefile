# Strata: run the model and the usual chores.  `make` lists the targets.
#
# Written for GNU Make 3.81, the one macOS ships in /usr/bin (Apple stays on it: 3.82 moved to GPLv3), so it uses no
# .ONESHELL, no `!=` and no `::=`: every recipe line is its own shell.  It works the same on Linux.
#
#   make run                       start the model in this terminal (Ctrl-C stops it)
#   make start / status / stop     the same in the background, with its log in strata-run.log
#   make chat PROMPT="Hi"          one request through the OpenAI API

# your own defaults (PORT := 8090, API_KEY := ...), kept out of git: Makefile.local
-include Makefile.local

PORT        ?= 8080
HOST        ?= 127.0.0.1
API_KEY     ?=
MODEL       ?=
CONTEXT     ?=
SETUP_ARGS  ?=
PROMPT      ?= Say hello in one short sentence.
EFFORT      ?= none
MAX_TOKENS  ?= 200
REASONING_BUDGET ?=
WAIT_S      ?= 600
PY          := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CMAKE       := $(if $(wildcard .venv/bin/cmake),.venv/bin/cmake,cmake)
URL         := http://$(if $(filter 0.0.0.0,$(HOST)),127.0.0.1,$(HOST)):$(PORT)
AUTH        := $(if $(API_KEY),-H "Authorization: Bearer $(API_KEY)",)
RUN_ARGS    := --port $(PORT) $(if $(filter-out 127.0.0.1,$(HOST)),--host $(HOST),) $(if $(API_KEY),--api-key $(API_KEY),)
SERVER_PAT  := serve/server.py --engine strata .*--port $(PORT)

# fails when another program listens on PORT: names it, and how to pick another port
PORT_FREE   = @who=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN 2> /dev/null | awk 'NR==2 {print $$1 " (pid " $$2 ")"}'); \
	if [ -n "$$who" ]; then \
	  if pgrep -f "$(SERVER_PAT)" > /dev/null; then echo "Strata already runs on port $(PORT): make status / make stop"; \
	  else echo "port $(PORT) is used by $$who - start Strata on another one: make $@ PORT=8090 (or put PORT := 8090 in Makefile.local)"; fi; \
	  exit 1; \
	fi

# make run CONTEXT=131072: sets --max-context in the installed config first (tools/set_context.py; it stays set)
SET_CONTEXT = $(if $(CONTEXT),$(PY) tools/set_context.py $(CONTEXT),@true)

# the prompt reaches `make chat` through the environment, so quotes and apostrophes in it need no shell quoting
export PROMPT MAX_TOKENS EFFORT REASONING_BUDGET

# A/B of two engine builds: with and without metal/patches/ (docs/MACOS.md, "For developers")
AB_A        ?= build-metal-a/metal/strata-metal
AB_B        ?= build-metal-b/metal/strata-metal
AB_GGUF     ?=
AB_PROMPTS  ?=
AB_ROUNDS   ?= 7

.PHONY: help setup pull check update run start stop status wait chat models build build-ab test test-engine ab clean

help:
	@echo "Strata - make targets (variables: PORT=$(PORT) HOST=$(HOST) API_KEY=... MODEL=... SETUP_ARGS=...;"
	@echo "                       defaults of your own in Makefile.local, e.g. PORT := 8090)"
	@echo ""
	@echo "  First time: make check, make pull MODEL=Q2_0, make run.  On a Mac: docs/MACOS.md"
	@echo ""
	@echo "  make setup        install or change a model (asks; SETUP_ARGS=\"--model Q2_0 --yes\" for no questions)"
	@echo "  make pull          build the engine and download a model, without starting it: make pull MODEL=Q2_0"
	@echo "                     (from Hugging Face into ../Strata-data, resumable: run it again if it stops)"
	@echo "  make check         only check what this computer can run"
	@echo "  make update        update the engine and settings without starting"
	@echo "  make run           start the model here, in the foreground (Ctrl-C stops it); CONTEXT=131072 sets the"
	@echo "                     context window first (4096-262144 tokens, it stays set; a bigger one needs more memory)"
	@echo "  make start         start it in the background (log: strata-run.log) and wait until it answers"
	@echo "  make status        is it up?  (GET /health)"
	@echo "  make stop          stop the background server"
	@echo "  make chat          one chat request: make chat PROMPT=\"Write a haiku\" [MAX_TOKENS=100 EFFORT=none|low|medium|high]"
	@echo "                     REASONING_BUDGET=N: at most N tokens of thinking, then it answers (0: no cap; it counts"
	@echo "                     toward MAX_TOKENS, so keep it 64+ below; EFFORT=none turns thinking off)"
	@echo "  make models        GET /v1/models"
	@echo "  make build         compile the engine here (macOS: the Metal engine; elsewhere setup builds it)"
	@echo "  make test          the tests that need no GPU and no model"
	@echo "  make test-engine   the Metal engine protocol tests: TEST_GGUF=<a small .gguf with <|im_start|>>"
	@echo "  make build-ab      build the plain (A) and patched (B) Metal engines for an A/B"
	@echo "  make ab            A/B two engines: AB_GGUF=<shard 1> AB_PROMPTS=<ids.json> [AB_A=... AB_B=...]"
	@echo "  make clean         remove the build folders"

setup:
	./setup.sh --setup $(if $(MODEL),--model $(MODEL),) $(SETUP_ARGS)

# setup's own download (Hugging Face, resumable; HF_ENDPOINT for a mirror, SETUP_ARGS="--data-dir DIR" for elsewhere)
pull:
	./setup.sh --setup --no-start $(if $(MODEL),--model $(MODEL),) $(SETUP_ARGS)

check:
	./setup.sh --check

update:
	./update.sh

run:
	$(PORT_FREE)
	$(SET_CONTEXT)
	./setup.sh $(RUN_ARGS) $(SETUP_ARGS)

start:
	$(PORT_FREE)
	$(SET_CONTEXT)
	@echo "starting Strata on $(URL) (log: strata-run.log) ..."
	@nohup ./setup.sh $(RUN_ARGS) --no-browser $(SETUP_ARGS) > strata-run.log 2>&1 &
	@$(MAKE) --no-print-directory wait

# /health says 503 while the model loads, 200 once it is ready
wait:
	@i=0; while [ $$i -lt $(WAIT_S) ]; do \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' -m 2 $(URL)/health); \
	  if [ "$$code" = "200" ]; then echo "ready: $(URL)/v1 (OpenAI), $(URL)/v1/messages (Anthropic)"; exit 0; fi; \
	  if ! pgrep -f "$(SERVER_PAT)" > /dev/null && [ $$i -gt 20 ]; then echo "the server is not running: see strata-run.log"; tail -5 strata-run.log; exit 1; fi; \
	  sleep 2; i=$$((i + 2)); \
	done; echo "not ready after $(WAIT_S) s: see strata-run.log"; exit 1

status:
	@if ! pgrep -f "$(SERVER_PAT)" > /dev/null; then \
	  who=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN 2> /dev/null | awk 'NR==2 {print $$1 " (pid " $$2 ")"}'); \
	  echo "Strata is not running on port $(PORT)$${who:+ - $$who listens there}"; exit 1; \
	fi; \
	code=$$(curl -s -o /dev/null -w '%{http_code}' -m 5 $(URL)/health); \
	case "$$code" in 200) echo "up: $(URL)";; 503) echo "loading: $(URL)";; *) echo "starting (no answer yet): $(URL)";; esac

# SIGTERM, not SIGINT: a job started with `&` by a non-interactive shell has SIGINT ignored (POSIX), and Python then
# installs no Ctrl+C handler; the server maps SIGTERM to Ctrl+C's path (QUIT to the engine, #96).  A second SIGTERM
# ends the engine at once; SIGKILL only when even that does not end it.
stop:
	@pids=$$(pgrep -f "$(SERVER_PAT)"); \
	if [ -z "$$pids" ]; then echo "no Strata server on port $(PORT)"; exit 0; fi; \
	kill -TERM $$pids; \
	i=0; while kill -0 $$pids 2> /dev/null && [ $$i -lt 30 ]; do sleep 1; i=$$((i + 1)); done; \
	if kill -0 $$pids 2> /dev/null; then kill -TERM $$pids; sleep 5; fi; \
	if kill -0 $$pids 2> /dev/null; then kill -KILL $$pids; sleep 1; fi; \
	if kill -0 $$pids 2> /dev/null; then echo "still running: $$pids"; exit 1; fi; \
	echo "stopped"

# One request (tools/chat.py: it checks MAX_TOKENS and REASONING_BUDGET before sending, and exits 1 without an
# answer).  REASONING_BUDGET=N: at most N tokens of thinking (the server's reasoning_budget_tokens), then it answers;
# 0 = no cap (EFFORT=none turns thinking off).  The answer is on stdout, the token count on stderr.
chat:
	@URL="$(URL)" API_KEY="$(API_KEY)" $(PY) tools/chat.py

models:
	@curl -sS -m 10 $(AUTH) $(URL)/v1/models; echo

build:
ifeq ($(shell uname -s),Darwin)
	$(CMAKE) -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON
	$(CMAKE) --build build-metal --target strata-metal strata-vision -j
else
	@echo "on Linux / Windows setup builds or downloads the engine: make setup (or ./setup.sh --setup --build)"
endif

# tools/ and serve/ have no __init__.py, so `unittest discover` will not start in them: the modules are named instead
TEST_MODULES := $(subst /,.,$(patsubst %.py,%,$(wildcard tools/test_setup_*.py serve/test_*.py))) \
                $(if $(filter Darwin,$(shell uname -s)),metal.test_setup_mac,)

test:
	$(PY) -m unittest $(TEST_MODULES)

test-engine:
	@test -n "$(TEST_GGUF)" || { echo "TEST_GGUF=<a small .gguf, e.g. LiquidAI LFM2-350M Q8_0>"; exit 1; }
	STRATA_METAL_EXE=$(or $(TEST_EXE),build-metal/metal/strata-metal) STRATA_METAL_GGUF=$(TEST_GGUF) $(PY) -m unittest metal.test_strata_metal -v

build-ab:
	$(CMAKE) -S . -B build-metal-a -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DSTRATA_METAL_KERNEL_PATCHES=OFF
	$(CMAKE) --build build-metal-a --target strata-metal -j
	$(CMAKE) -S . -B build-metal-b -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DSTRATA_METAL_KERNEL_PATCHES=ON
	$(CMAKE) --build build-metal-b --target strata-metal -j

ab:
	@test -n "$(AB_GGUF)" -a -n "$(AB_PROMPTS)" || { echo "AB_GGUF=<model shard 1> AB_PROMPTS=<json list of token-id lists>"; exit 1; }
	$(PY) metal/bench/ab.py --a $(AB_A) --b $(AB_B) --gguf $(AB_GGUF) --prompts $(AB_PROMPTS) --rounds $(AB_ROUNDS) --log strata-ab.log

clean:
	rm -rf build-metal build-metal-a build-metal-b
