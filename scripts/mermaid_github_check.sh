#!/usr/bin/env bash
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
#
# mermaid_github_check.sh — thin wrapper around mermaid_github_check.mjs.
#
# Validates every ```mermaid block against the SAME mermaid core version
# GitHub renders with (pinned in scripts/mermaid-github-check/package.json),
# so "green here" implies "renders on GitHub" — unlike the mmdc gate, whose
# bleeding-edge mermaid accepts syntax GitHub's older core rejects.
#
# The wrapper's only jobs: locate node, make sure the pinned deps are present
# (one-time `npm ci`, reused after), then hand off to the .mjs.
#
# Usage:
#   scripts/mermaid_github_check.sh                 # tracked *.md in CWD's repo
#   scripts/mermaid_github_check.sh --all           # parent + all 4 submodules
#   scripts/mermaid_github_check.sh FILE.md ...     # just these files
#   scripts/mermaid_github_check.sh --require ...    # node/deps absent = FAIL
#
# Exit 0 = all blocks render on GitHub (or node absent without --require),
#        1 = a block would fail on GitHub, 2 = bad setup / --require unmet.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/mermaid-github-check"
MJS="$SCRIPT_DIR/mermaid_github_check.mjs"

REQUIRE=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--require" ]; then REQUIRE=1; else ARGS+=("$a"); fi
done

# 1. node present? -----------------------------------------------------------
NODE="$(command -v node || true)"
if [ -z "$NODE" ]; then
    echo "mermaid_github_check: node not found — GitHub-mermaid gate SKIPPED" >&2
    [ "$REQUIRE" -eq 1 ] && { echo "  (--require: treating as failure)" >&2; exit 2; }
    exit 0
fi

# 2. pinned deps present? install once from the lockfile if not ---------------
if [ ! -f "$DEPS_DIR/node_modules/mermaid/package.json" ]; then
    echo "mermaid_github_check: installing pinned mermaid+jsdom (one-time)…" >&2
    if command -v npm >/dev/null 2>&1; then
        # `npm ci` is reproducible (installs EXACTLY the lockfile) and needs no
        # network beyond the registry fetch; it is the CI-safe install path.
        ( cd "$DEPS_DIR" && npm ci --no-audit --no-fund ) >/tmp/mermaid_github_deps.log 2>&1 \
            || ( cd "$DEPS_DIR" && npm install --no-audit --no-fund ) >>/tmp/mermaid_github_deps.log 2>&1
    fi
    if [ ! -f "$DEPS_DIR/node_modules/mermaid/package.json" ]; then
        echo "mermaid_github_check: could not install pinned deps (see /tmp/mermaid_github_deps.log)" >&2
        [ "$REQUIRE" -eq 1 ] && exit 2
        exit 0
    fi
fi

# 3. run the faithful checker ------------------------------------------------
exec "$NODE" "$MJS" "${ARGS[@]}"
