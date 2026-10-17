# Contributing this bundle upstream

This directory (`docker/`) is a Docker deployment for Strata: engine image +
management console + gateway. This file is aimed at contributors who want to open
a pull request against [Niko1221/Strata](https://github.com/Niko1221/Strata).

## Scope of this directory

The upstream repository already ships a root-level `Dockerfile` and
`docker-entrypoint.sh` (a single inference container). This directory provides a
**complete stack**: `nginx` (TLS + authorization entry point), `manager`
(monitoring and management console), and `strata` (the engine), wired together
with Docker Compose.

The upstream root `Dockerfile` is left untouched. The `Dockerfile` here is derived
from it (same build logic, plus China-friendly mirror defaults) so that upstream
changes stay easy to merge back.

## Checklist before submitting

- [ ] `docker compose config` produces no warnings
- [ ] `node tests/i18n-coverage.js` reports no untranslated strings
- [ ] `./scripts/gen-cert.sh` generates a certificate (Linux / macOS / Git Bash)
- [ ] `docker compose up -d --build` brings all three containers to `healthy`
- [ ] Authorization works: an unauthenticated request to `/api/*` returns 401
- [ ] No secrets committed — `.gitignore` excludes `*.key`, `*.crt` and `.env`
- [ ] Commands in the README match actual behaviour
- [ ] Sources are in sync: `git diff upstream/main -- <file>` shows nothing missed

## Code layout

| Path | Language | Purpose |
| --- | --- | --- |
| `manager/app.py` | Python (FastAPI) | Metrics collection, API key store, user sessions, container control, WebSocket shell, inference proxy |
| `manager/static/index.html` | HTML/CSS/JS | Console single page (no build step; libraries are vendored) |
| `nginx.conf` | nginx | TLS, authorization via `auth_request`, SSE/WebSocket forwarding |
| `scripts/*.sh` | Bash | Certificate generation, API key bootstrap, build context preparation |

### Frontend conventions

- No bundler. CSS and JS are inline in `index.html`; libraries live in `manager/static/`
- Charts use uPlot (lower memory than Chart.js with many points); the shell uses xterm.js
- Authentication state is centralized in `ME` (current user) and `checkAuth()`. **No data
  request is issued while logged out**, which keeps the console free of 401 noise
- **i18n**: English is the source language, Chinese is the fallback text carried in the
  HTML. Add user-facing strings to the `I18N` dictionary and wrap runtime-generated text
  in `t()`. Run `node tests/i18n-coverage.js` after changes — it must report
  `still Chinese after applyLang: 0`.

### Backend conventions

- Metrics and keys live in SQLite (`/data/manager.db`, a Docker volume). Tables are
  declared with `CREATE TABLE IF NOT EXISTS` in `init_db()`, so upgrades need no migration
- Guard administrative endpoints with `_is_admin(request)` (session cookie or admin API key)
- New endpoints should follow the `/api/<resource>` naming and reuse `_is_admin`, rather
  than adding more rules to nginx

## Known limitations (PRs welcome)

1. **The web shell runs as root.** It requires a click on *Connect* in the Terminal tab.
   Hardening options (command auditing, session timeouts, a non-root user) are not
   implemented yet.
2. **The engine is single-slot.** Concurrent requests queue. This comes from the engine
   design, not from this bundle.
3. **Metric sampling granularity.** System metrics are sampled every 10 seconds.
   Lowering `SAMPLE_INTERVAL` increases precision but also load on the manager and the
   Docker API.
4. **WebSocket authorization.** `/ws/` deliberately bypasses nginx `auth_request`:
   the subrequest does not inherit the original query string, so `?api_key=` cannot be
   verified there. The manager's WebSocket endpoints check the session cookie themselves.
5. **Self-signed TLS.** Clients must trust the certificate manually. ACME/Let's Encrypt
   or mounting an externally issued certificate would be a natural improvement.

## Debugging

```bash
# rebuild a single component
docker compose build manager && docker compose up -d manager

# detailed console log
docker compose logs -f manager

# shell inside the engine container
docker compose exec strata bash

# check the authorization chain
curl -sk -o /dev/null -w "%{http_code}\n" https://127.0.0.1:38888/api/sys/latest   # expect 401
curl -sk https://127.0.0.1:38888/api/sys/latest -H "Authorization: Bearer $(cat certs/admin.key)"
```

> On Windows, connect to `127.0.0.1` rather than `localhost`. `localhost` resolves to
> IPv6 `::1` first, while Docker's port forwarding listens on IPv4 only, so clients may
> fall back and show a multi-second delay on the first request. (`localhost` does work,
> it is just slower.)

## Syncing upstream

Upstream moves fast; re-sync periodically:

```bash
# 1. Fetch the upstream tree
curl -L -o strata-main.tar.gz https://github.com/Niko1221/Strata/archive/refs/heads/main.tar.gz

# 2. Copy the files that this directory shares with upstream
#    Dockerfile / docker-entrypoint.sh / .dockerignore
#    Keep what is unique to this directory:
#    docker-compose.yml, Dockerfile.nginx, nginx.conf, manager/, scripts/,
#    tests/, README.md, CONTRIBUTING.md

# 3. Re-apply the local Dockerfile adjustments (China mirrors, removal of the
#    `syntax` directive). See the commit history, or README → "docker.io pull timeouts".
```
