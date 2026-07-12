#!/usr/bin/env node
// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
//
// selftest.mjs — proves the GitHub-faithful gate actually adds fidelity.
//
// It runs the REAL checker (../mermaid_github_check.mjs) end-to-end against two
// kinds of temp fixtures and asserts the discrimination that matters:
//
//   GOOD  — standard diagrams GitHub (mermaid 11.15.0) renders  -> gate exit 0
//   DRIFT — syntax/types that ONLY exist in a newer mermaid core -> gate exit 1
//
// The DRIFT fixtures are the exact diagrams mmdc's bleeding-edge core
// (11.16.0) renders as real SVGs — so the current mmdc gate is GREEN on them
// while GitHub shows the red "Parse error" box. This gate must be RED on them.
// Kept as inline strings (never tracked .md) so no GitHub-hostile mermaid ever
// lands in the repo's own docs.
//
// Usage:  node scripts/mermaid-github-check/selftest.mjs   (exit 0 = fidelity intact)

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CHECKER = path.join(path.dirname(HERE), "mermaid_github_check.mjs");

// Standard diagrams — every one renders on GitHub's 11.15.0 core.
const GOOD = `# good
\`\`\`mermaid
graph TD
  A[Start] --> B{OK?}
  B -->|yes| C[Ship]
  B -->|no| A
\`\`\`

\`\`\`mermaid
sequenceDiagram
  Operator->>Fleet: dispatch
  Fleet-->>Operator: telemetry
\`\`\`

\`\`\`mermaid
gantt
  title Wave
  section Build
  Edge :a1, 2026-01-01, 7d
\`\`\`
`;

// DRIFT — real, renderable diagrams in mermaid >= 11.16.0 that GitHub's 11.15.0
// core CANNOT render (verified: mmdc draws both as genuine SVGs).
//   1. cynefin-beta  : an entire diagram TYPE added in 11.16.0 (11.15.0 says
//                      "No diagram type detected").
//   2. ER `string?`  : optional-attribute suffix added in 11.16.0 (11.15.0 says
//                      "Expecting 'ATTRIBUTE_WORD', got '?'").
const DRIFT_CYNEFIN = `# drift-cynefin
\`\`\`mermaid
cynefin-beta
  complex
    "Item one"
  clear
    "Item two"
\`\`\`
`;
const DRIFT_ER = `# drift-er
\`\`\`mermaid
erDiagram
  CUSTOMER {
    string firstName
    string? middleName
  }
\`\`\`
`;

function runChecker(file) {
    try {
        const out = execFileSync(process.execPath, [CHECKER, file], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        });
        return { code: 0, out };
    } catch (e) {
        return { code: e.status ?? 1, out: `${e.stdout || ""}${e.stderr || ""}` };
    }
}

function main() {
    const dir = mkdtempSync(path.join(tmpdir(), "mgc-selftest-"));
    const good = path.join(dir, "good.md");
    const cyn = path.join(dir, "drift_cynefin.md");
    const er = path.join(dir, "drift_er.md");
    writeFileSync(good, GOOD);
    writeFileSync(cyn, DRIFT_CYNEFIN);
    writeFileSync(er, DRIFT_ER);

    const checks = [];
    const good_r = runChecker(good);
    checks.push(["GOOD diagrams accepted (exit 0)", good_r.code === 0, good_r]);

    const cyn_r = runChecker(cyn);
    checks.push([
        "DRIFT cynefin-beta rejected (exit 1, cause named)",
        cyn_r.code === 1 && /cynefin-beta/.test(cyn_r.out),
        cyn_r,
    ]);

    const er_r = runChecker(er);
    checks.push([
        "DRIFT ER string? rejected (exit 1, cause named)",
        er_r.code === 1 && /got '\?'|ATTRIBUTE_WORD/.test(er_r.out),
        er_r,
    ]);

    let failed = 0;
    for (const [label, pass, r] of checks) {
        process.stdout.write(`  [${pass ? "PASS" : "FAIL"}] ${label}\n`);
        if (!pass) {
            failed++;
            process.stdout.write(`         got exit ${r.code}; output:\n`);
            for (const ln of r.out.split("\n").slice(0, 8)) process.stdout.write(`         ${ln}\n`);
        }
    }
    if (failed) {
        process.stdout.write(`\nselftest: ${failed} check(s) FAILED — fidelity NOT proven\n`);
        return 1;
    }
    process.stdout.write(
        "\nselftest: OK — gate accepts GitHub-renderable diagrams and rejects newer-core drift\n",
    );
    return 0;
}

process.exit(main());
