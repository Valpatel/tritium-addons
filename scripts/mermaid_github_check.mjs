#!/usr/bin/env node
// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
//
// mermaid_github_check.mjs — GitHub-FAITHFUL mermaid validation.
//
// The problem this solves
// -----------------------
// scripts/mermaid_validate.py renders every ```mermaid block through mmdc
// (@mermaid-js/mermaid-cli), which bundles the BLEEDING-EDGE mermaid core
// (11.16.0 on this box). GitHub renders markdown mermaid with an OLDER,
// pinned mermaid core. So a diagram can render clean under mmdc yet show the
// red "Unable to render rich display / Parse error in text" box on GitHub —
// the mmdc gate is GREEN while the actual README is BROKEN.
//
// This checker closes that gap: it parses each block with the SAME mermaid
// core version GitHub runs, under GitHub's config (securityLevel:'strict',
// default theme, no external CDN). "Passes this check" therefore implies
// "GitHub renders it".
//
// Which mermaid version does GitHub use, and how do we know
// --------------------------------------------------------
// GitHub renders markdown mermaid inside its internal "Viewscreen" iframe
// service, loading a pinned mermaid bundle (NOT the latest). The authoritative,
// first-hand way to read the LIVE deployed version — discovered by community
// user @cbornet (github/orgs/community discussion #37498, Sep 2023) — is to
// put a mermaid block whose entire body is the word `info` into any GitHub
// markdown surface (a gist works):
//
//     ```mermaid
//     info
//     ```
//
// GitHub's own renderer draws "mermaid version X.Y.Z" — that string IS the
// version GitHub is running. Reported history from that thread + #70672:
//   9.1.6 (2022) -> 9.3.0 -> 10.0.2 -> 10.6.1 -> 10.8.0 -> 11.4.1 (Apr 2025)
//   -> 11.15.0 (last confirmed deploy). mmdc here bundles core 11.16.0 — exactly
//   one minor AHEAD of GitHub, which is the drift that bites.
//
// Why we pin 11.15.0 (conservative + faithful)
// --------------------------------------------
// scripts/mermaid-github-check/package.json pins mermaid EXACTLY at 11.15.0 —
// the last version GitHub was confirmed to deploy. GitHub only ever moves
// FORWARD, so pinned(11.15.0) <= github(>=11.15.0) always holds: anything this
// gate accepts, GitHub's equal-or-newer renderer also accepts. The failure
// direction is safe — at worst this gate rejects a brand-new feature GitHub
// just adopted (a false FAIL you notice and fix by bumping the pin), and it
// NEVER lets a GitHub-broken diagram through. When you confirm (via the `info`
// trick) that GitHub has moved on, bump the single pin in package.json.
//
// Engine: parse, not mmdc
// -----------------------
// We call mermaid.parse() (mermaid's own grammar + diagram-type detection) at
// the pinned version, inside a jsdom DOM (mermaid needs a document for its
// DOMPurify hooks even to PARSE some types, e.g. gantt). parse() THROWS on the
// exact inputs GitHub's parser rejects — and, unlike mmdc, it is NOT fooled by
// mermaid's "Syntax error" placeholder SVG (mmdc renders that bomb graphic and
// still exits 0, so the mmdc gate silently passes diagrams broken on BOTH mmdc
// and GitHub). No browser, no chromium, no network at run time.
//
// Usage
// -----
//   scripts/mermaid_github_check.sh                 # wrapper: installs pinned
//                                                   # deps once, then runs this
//   node scripts/mermaid_github_check.mjs           # tracked *.md in CWD's repo
//   node scripts/mermaid_github_check.mjs --all     # parent + all 4 submodules
//   node scripts/mermaid_github_check.mjs A.md B.md # just these files
//   node scripts/mermaid_github_check.mjs --pinned-version   # print pin & exit
//
// Exit 0 = every block parses at the pinned version (or none found),
//        2 = bad invocation / pinned deps missing,
//        1 = at least one block would fail to render on GitHub.

import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { pathToFileURL, fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.dirname(SCRIPT_DIR);
const DEPS_DIR = path.join(SCRIPT_DIR, "mermaid-github-check");
const SUBMODULES = ["tritium-sc", "tritium-lib", "tritium-edge", "tritium-addons"];

// GitHub's markdown mermaid config: strict sandboxing, default theme, no auto
// run. securityLevel:'strict' is mermaid's own default too, but we set it
// explicitly so the gate never drifts if a future mermaid changes its default.
const GITHUB_CONFIG = {
    startOnLoad: false,
    securityLevel: "strict",
    theme: "default",
    logLevel: 5, // 'fatal' — keep mermaid's own logger off our stdout
};

const FENCE_RE = /^(\s*)(`{3,}|~{3,})(.*)$/;

function fail(msg, code = 2) {
    process.stderr.write(`mermaid_github_check: ${msg}\n`);
    process.exit(code);
}

// Fence-aware extraction, matched to scripts/mermaid_validate.py + md_lint.py:
// only a closing fence of the same marker ends a block; non-mermaid fences are
// skipped wholesale so their contents never leak into a mermaid block.
function extractBlocks(file) {
    let text;
    try {
        text = readFileSync(file, "utf8");
    } catch {
        return [];
    }
    const lines = text.split("\n");
    const blocks = [];
    let inFence = false;
    let marker = "";
    let isMermaid = false;
    let startLine = 0;
    let buf = [];
    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        const m = FENCE_RE.exec(raw);
        if (m) {
            const mk = m[2].slice(0, 3);
            if (!inFence) {
                inFence = true;
                marker = mk;
                isMermaid = m[3].trim().toLowerCase().startsWith("mermaid");
                startLine = i + 1;
                buf = [];
            } else if (mk === marker) {
                if (isMermaid && buf.length) {
                    blocks.push({ line: startLine, text: buf.join("\n") + "\n" });
                }
                inFence = false;
                isMermaid = false;
            }
            continue;
        }
        if (inFence && isMermaid) buf.push(raw);
    }
    return blocks;
}

function discoverMd(dir) {
    try {
        const out = execFileSync("git", ["-C", dir, "ls-files", "*.md"], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "ignore"],
        });
        return out.split("\n").filter(Boolean).map((p) => path.join(dir, p));
    } catch {
        return [];
    }
}

// jsdom must be installed and globals wired BEFORE mermaid is imported: mermaid
// touches document/window at module-eval time to set up its DOMPurify hooks.
async function loadMermaid() {
    const mermaidEntry = path.join(DEPS_DIR, "node_modules", "mermaid", "dist", "mermaid.esm.mjs");
    const jsdomEntry = path.join(DEPS_DIR, "node_modules", "jsdom");
    if (!fs.existsSync(mermaidEntry)) {
        fail(
            `pinned deps missing (${path.relative(REPO_ROOT, mermaidEntry)}). ` +
                `Run: (cd scripts/mermaid-github-check && npm ci)  — or use the ` +
                `scripts/mermaid_github_check.sh wrapper, which does it for you.`,
            2,
        );
    }
    const { JSDOM } = await import(pathToFileURL(path.join(jsdomEntry, "lib", "api.js")).href).catch(
        () => import(jsdomEntry),
    );
    const w = new JSDOM("<!DOCTYPE html><body></body>", { pretendToBeVisual: true }).window;
    for (const k of [
        "window", "document", "DOMParser", "Node", "Element", "SVGElement",
        "HTMLElement", "getComputedStyle", "MutationObserver", "requestAnimationFrame",
    ]) {
        try {
            globalThis[k] = w[k];
        } catch {
            // some globals (e.g. navigator on Node >=21) are read-only; mermaid.parse
            // does not need them, so a failed assignment is fine to swallow.
        }
    }
    const mermaid = (await import(pathToFileURL(mermaidEntry).href)).default;
    const version = JSON.parse(
        readFileSync(path.join(DEPS_DIR, "node_modules", "mermaid", "package.json"), "utf8"),
    ).version;
    mermaid.initialize(GITHUB_CONFIG);
    return { mermaid, version };
}

// Trim mermaid's (sometimes multi-line, caret-annotated) error to something a
// human can act on, while keeping the useful "line N" + caret context.
function cleanError(err) {
    const msg = String((err && err.message) || err || "parse failed");
    const kept = msg
        .split("\n")
        .map((s) => s.replace(/\s+$/, ""))
        .filter((s) => s.trim().length)
        .slice(0, 4);
    return kept.join(" | ").slice(0, 400);
}

async function main() {
    const argv = process.argv.slice(2);
    if (argv.includes("--help") || argv.includes("-h")) {
        process.stdout.write(
            "usage: mermaid_github_check.mjs [--all] [--pinned-version] [FILE.md ...]\n",
        );
        return 0;
    }
    if (argv.includes("--pinned-version")) {
        const v = JSON.parse(
            readFileSync(path.join(DEPS_DIR, "node_modules", "mermaid", "package.json"), "utf8"),
        ).version;
        process.stdout.write(`${v}\n`);
        return 0;
    }

    const all = argv.includes("--all");
    const fileArgs = argv.filter((a) => !a.startsWith("-"));

    let files;
    if (all) {
        files = discoverMd(REPO_ROOT);
        for (const sub of SUBMODULES) {
            const d = path.join(REPO_ROOT, sub);
            if (fs.existsSync(path.join(d, ".git"))) files.push(...discoverMd(d));
        }
    } else if (fileArgs.length) {
        files = fileArgs;
    } else {
        files = discoverMd(process.cwd());
    }

    // (file, line, text) work items
    const work = [];
    for (const f of files) {
        for (const b of extractBlocks(f)) work.push({ file: f, line: b.line, text: b.text });
    }
    if (!work.length) {
        process.stderr.write("mermaid_github_check: no mermaid blocks found\n");
        return 0;
    }

    const { mermaid, version } = await loadMermaid();

    const failures = [];
    const perFile = new Map(); // file -> {ok, fail}
    for (const w of work) {
        const rec = perFile.get(w.file) || { ok: 0, fail: 0 };
        try {
            // parse() runs GitHub's grammar + diagram-type detection at the pinned
            // version and THROWS on exactly what GitHub's renderer rejects.
            await mermaid.parse(w.text);
            rec.ok++;
        } catch (e) {
            rec.fail++;
            failures.push({ file: w.file, line: w.line, err: cleanError(e), text: w.text });
        }
        perFile.set(w.file, rec);
    }

    // Per-file "N ok, M FAILED" line (only files that HAVE blocks).
    for (const [file, rec] of perFile) {
        const rel = path.isAbsolute(file) ? path.relative(process.cwd(), file) : file;
        const flag = rec.fail ? "  <<<" : "";
        process.stderr.write(`  ${rel}: ${rec.ok} ok, ${rec.fail} FAILED${flag}\n`);
    }

    // Exact failing block + error, clickable file:line, so a broken diagram is
    // findable without hunting through an anonymous temp file.
    for (const f of failures.sort((a, b) => (a.file + a.line).localeCompare(b.file + b.line))) {
        const rel = path.isAbsolute(f.file) ? path.relative(process.cwd(), f.file) : f.file;
        process.stdout.write(`\n${rel}:${f.line}: [MERMAID-GITHUB] ${f.err}\n`);
        const preview = f.text.replace(/\n+$/, "").split("\n").slice(0, 12);
        for (const ln of preview) process.stdout.write(`    | ${ln}\n`);
    }

    process.stderr.write(
        `\nmermaid_github_check: ${work.length} block(s) in ${files.length} file(s) — ` +
            `${work.length - failures.length} ok, ${failures.length} FAILED ` +
            `[mermaid ${version} == GitHub's pinned core; securityLevel:strict]\n`,
    );
    return failures.length ? 1 : 0;
}

main().then(
    (code) => process.exit(code),
    (e) => fail(`unexpected: ${(e && e.stack) || e}`, 2),
);
