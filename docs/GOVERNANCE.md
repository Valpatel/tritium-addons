# Governance — CI/CD & branch protection (tritium-addons)

`tritium-addons` is a **submodule** of [`Valpatel/tritium`](https://github.com/Valpatel/tritium).
The full governance strategy (review model, fork safety, release rules) lives in
the parent's [`docs/GOVERNANCE.md`](https://github.com/Valpatel/tritium/blob/dev/docs/GOVERNANCE.md).
This file records the pieces specific to **this repo**.

**Copyright** Matthew Valancy / Valpatel Software LLC — AGPL-3.0.

## CI (`.github/workflows/ci.yml`) — fork-safe

`pull_request` (never `pull_request_target`) on `[dev, main]` + `push` on `[dev]`,
`permissions: contents: read`, `concurrency … cancel-in-progress`. Two jobs, both
required status-check contexts:

| Job (status context) | Gates |
|-----------------------|-------|
| `mermaid-github-fidelity` | Every ` ```mermaid ` block renders on GitHub — pinned to GitHub's EXACT mermaid core (`scripts/mermaid-github-check/`). Green here = green on GitHub. `selftest.mjs` proves the mmdc-vs-GitHub fidelity gap first. |
| `python-syntax` | `python -m compileall .` — syntax gate, no service/hardware deps. |

Markdown + privacy are gated **locally** via the parent's shared
`core.hooksPath = scripts/githooks` (pre-push) + `scripts/md_lint.sh`.

## Branch protection — exact `gh api` commands

> **OUTWARD-FACING — run by the repository owner (`@mvalancy`), not an agent.**
> Required status-check contexts are the CI job names below.

### `main` — high bar (release branch)

```bash
gh api -X PUT repos/Valpatel/tritium-addons/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": { "strict": true, "contexts": ["mermaid-github-fidelity", "python-syntax"] },
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 1, "dismiss_stale_reviews": true, "require_code_owner_reviews": true },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true,
  "required_conversation_resolution": true
}
JSON
```

### `dev` — decent bar (integration branch)

```bash
gh api -X PUT repos/Valpatel/tritium-addons/branches/dev/protection --input - <<'JSON'
{
  "required_status_checks": { "strict": true, "contexts": ["mermaid-github-fidelity", "python-syntax"] },
  "enforce_admins": false,
  "required_pull_request_reviews": { "required_approving_review_count": 1, "dismiss_stale_reviews": false, "require_code_owner_reviews": false },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

Verify: `gh api repos/Valpatel/tritium-addons/branches/main/protection | jq '.required_status_checks.contexts'` — should list both job names. Never push to `main`; advance it only via a reviewed `dev -> main` PR.
