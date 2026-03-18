# Claude Code Instructions

## Git / GitHub Workflow

Automatically commit and push to GitHub after every meaningful change — no need to ask for confirmation.

Follow this pattern every time:
1. `git add` the relevant files (never add `.env`, secrets, or large data files)
2. `git commit` with a clear conventional commit message
3. `git push` to `origin main`

Use conventional commit prefixes: `feat:`, `fix:`, `ci:`, `docs:`, `refactor:`, `test:`

**Never add `Co-Authored-By` lines to commit messages.**
