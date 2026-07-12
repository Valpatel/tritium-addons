# GitHub-faithful mermaid check

A validation gate that fails **exactly when GitHub would fail to render** a
`​```mermaid`​` block — closing the gap left by the mmdc-based gate.

## Why this exists

`scripts/mermaid_validate.py` renders every block through **mmdc**
(`@mermaid-js/mermaid-cli`), which bundles the *bleeding-edge* mermaid core
(**11.16.0** on this box). GitHub renders markdown mermaid with an **older,
pinned** mermaid core. So a diagram can pass mmdc yet show the red
"Unable to render rich display… Parse error in text" box on GitHub — the gate is
green while the README is broken.

This checker parses each block with the **same mermaid core GitHub deploys**,
under GitHub's config (`securityLevel:'strict'`, default theme, no external CDN).
Green here ⇒ GitHub renders it.

## Which mermaid version does GitHub use? (and how to re-check it)

GitHub renders markdown mermaid in its internal *Viewscreen* iframe, loading a
**pinned** mermaid bundle (not the latest). The authoritative, first-hand way to
read the **live deployed** version — from GitHub community discussion
[#37498](https://github.com/orgs/community/discussions/37498) (user *@cbornet*,
Sep 2023) — is to paste this into **any** GitHub markdown surface (a gist works):

```mermaid
info
```

GitHub's own renderer draws **"mermaid version X.Y.Z"** — that string *is* the
version GitHub is running right now. Reported history (discussions #37498 +
[#70672](https://github.com/orgs/community/discussions/70672)):

| Date | GitHub mermaid core |
|------|---------------------|
| 2022 | 9.1.6 |
| 2023 | 9.3.0 → 10.0.2 → 10.6.1 |
| Feb 2024 | 10.8.0 |
| Apr 2025 | 11.4.1 |
| last confirmed | **11.15.0** |

mmdc on this box bundles **11.16.0** — exactly one minor *ahead* of GitHub. That
one-minor drift is what breaks READMEs.

## Why we pin 11.15.0

`package.json` pins `mermaid` **exactly** (not `^`) at **11.15.0**, the last
version GitHub was confirmed to deploy. GitHub only ever moves **forward**, so:

```
pinned(11.15.0)  <=  github(>= 11.15.0)     always holds
```

Everything this gate accepts, GitHub's equal-or-newer renderer also accepts. The
failure direction is **safe**: at worst it rejects a brand-new feature GitHub
just adopted (a false FAIL you notice and fix by bumping the pin) — it never lets
a GitHub-broken diagram through. **When the `info` trick shows GitHub has moved
on, bump the single `mermaid` pin in `package.json`, run `npm install` to refresh
the lockfile, and commit.**

## Engine: parse, not render

We call `mermaid.parse()` (mermaid's own grammar + diagram-type detection) at the
pinned version, inside a **jsdom** DOM (mermaid touches `document` for its
DOMPurify hooks even to *parse* some types, e.g. gantt). `parse()` throws on the
exact inputs GitHub's parser rejects, and — unlike mmdc — is **not fooled by
mermaid's "Syntax error" placeholder SVG** (mmdc renders that bomb graphic and
still exits 0). No browser, no chromium, **no network at run time**. ~0.5 s for
the whole parent corpus.

## Usage

```bash
scripts/mermaid_github_check.sh                # tracked *.md in CWD's repo
scripts/mermaid_github_check.sh --all          # parent + all 4 submodules
scripts/mermaid_github_check.sh FILE.md ...    # just these files
scripts/mermaid_github_check.sh --require ...   # node/deps absent = FAIL (CI)
node  scripts/mermaid_github_check.mjs --pinned-version    # print the pin

node  scripts/mermaid-github-check/selftest.mjs # prove the gate still discriminates
```

Wired into `scripts/md_lint.sh` (hence `scripts/check.sh` and the pre-push hook)
as a third gate beside the mmdc render check, and into
`.github/workflows/ci.yml`. Exit `0` all render on GitHub · `1` a block would
break on GitHub · `2` bad setup / `--require` unmet.

## Fidelity proof (mmdc GREEN, this gate RED)

Two diagrams mmdc 11.16.0 renders as **genuine SVGs** but GitHub's 11.15.0 core
**cannot** render:

**1. `cynefin-beta` — an entire diagram *type* added in mermaid 11.16.0:**

```
cynefin-beta
  complex
    "Item one"
  clear
    "Item two"
```

**2. ER `string?` — optional-attribute suffix added in 11.16.0:**

```
erDiagram
  CUSTOMER {
    string firstName
    string? middleName
  }
```

| Gate | cynefin-beta | ER `string?` |
|------|--------------|--------------|
| `mermaid_validate.py` (mmdc 11.16.0) | **ok** (real cynefin SVG) | **ok** (real ER SVG) |
| `mermaid_github_check` (11.15.0 = GitHub) | **FAIL** — `No diagram type detected … cynefin-beta` | **FAIL** — `Expecting 'ATTRIBUTE_WORD', got '?'` |

`selftest.mjs` runs both through the real checker on every CI run, so the fidelity
never silently regresses.

## Governance — branch protection (owner runs these)

The mermaid gate is one required status check in the val-ark governance model
(main = high bar, dev = decent; both PR-only, required checks, no force-push).
These are **outward-facing `gh api` calls the repo owner runs** — they are *not*
applied by this branch. Owner = `Valpatel`, repo = `tritium`; the check contexts
are the workflow **job names** (`mermaid-github-fidelity`, `markdown`).

```bash
# dev — decent bar: PR-only, required checks, no force-push
gh api -X PUT repos/Valpatel/tritium/branches/dev/protection \
  -H "Accept: application/vnd.github+json" \
  -f 'required_status_checks[strict]=true' \
  -f 'required_status_checks[contexts][]=mermaid-github-fidelity' \
  -f 'required_status_checks[contexts][]=markdown' \
  -F 'enforce_admins=false' \
  -f 'required_pull_request_reviews[required_approving_review_count]=1' \
  -F 'restrictions=' -F 'allow_force_pushes=false' -F 'allow_deletions=false'

# main — high bar: dev->main release PRs only, stricter review, admins enforced
gh api -X PUT repos/Valpatel/tritium/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  -f 'required_status_checks[strict]=true' \
  -f 'required_status_checks[contexts][]=mermaid-github-fidelity' \
  -f 'required_status_checks[contexts][]=markdown' \
  -F 'enforce_admins=true' \
  -f 'required_pull_request_reviews[required_approving_review_count]=1' \
  -f 'required_pull_request_reviews[dismiss_stale_reviews]=true' \
  -F 'restrictions=' -F 'allow_force_pushes=false' -F 'allow_deletions=false'
```

**Never push to `main`** — it advances only via a reviewed `dev` → `main` release
PR + tag. Fork PRs run with a read-only token and no secrets (`ci.yml` uses
`pull_request`, never `pull_request_target`) and are never auto-merged
(reviewer ≠ author, adversarial review).
