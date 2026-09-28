/**
 * Check install ordering and failure cleanup with a scripted command runner.
 * Supply GPU detection responses, write uv.lock, and create .venv/bin/python:
 * returning exit code 0 alone cannot exercise the installer's filesystem checks.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import { existsSync } from "node:fs"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { install, listRuntimes, runtimeDir, verify, weightsDir } from "../src/install.ts"
import { findModel } from "../src/models.ts"
import type { RunOptions, RunResult, Runner } from "../src/subprocess.ts"

let data: string
let saved: string | undefined

beforeEach(async () => {
  data = await mkdtemp(join(tmpdir(), "onesystem-install-"))
  saved = process.env.ONESYSTEM_DATA_DIR
  process.env.ONESYSTEM_DATA_DIR = data
})

afterEach(async () => {
  if (saved === undefined) delete process.env.ONESYSTEM_DATA_DIR
  else process.env.ONESYSTEM_DATA_DIR = saved
  await rm(data, { recursive: true, force: true })
})

/** A lock file that passes the accelerator check, so the test is about order not resolution. */
const GOOD_LOCK =
  '[[package]]\nname = "torch"\nversion = "2.14.0+rocm7.2"\n\n' +
  '[[package]]\nname = "triton-rocm"\nversion = "3.8.0"\n'

const ROCM_TORCH = { hip: "6.2.0", cuda: null, available: true, arch: ["gfx1201"] }
const CUDA_TORCH = { hip: null, cuda: "13.0", available: true, arch: ["sm_90"] }

type Step =
  | "have:uv"
  | "have:lspci"
  | "lspci"
  | "have:rocm-smi"
  | "rocm-smi"
  | "fetch:source"
  | "fetch:weights"
  | "uv lock"
  | "uv sync"
  | "verify"
  | "unclassified"

/** What each subprocess is *for*, which is the only way to read the order meaningfully. */
function classify(cmd: string, args: string[]): Step {
  if (cmd === "sh") {
    const c = args[1] ?? ""
    if (c === "command -v uv") return "have:uv"
    if (c === "command -v lspci") return "have:lspci"
    if (c === "command -v rocm-smi") return "have:rocm-smi"
    if (c.startsWith("lspci")) return "lspci"
  }
  if (cmd === "rocm-smi") return "rocm-smi"
  if (cmd === "uv") {
    if (args[0] === "lock") return "uv lock"
    if (args[0] === "sync") return "uv sync"
    if (args[0] === "run") {
      // Both fetches are `uv run --with huggingface_hub python -c <script> <into>`. The
      // destination is what tells them apart: a source is fetched into staging, weights
      // into their own directory that outlives the install.
      return (args[args.length - 1] ?? "").includes(".partial") ? "fetch:source" : "fetch:weights"
    }
  }
  if (cmd.endsWith("/.venv/bin/python")) return "verify"
  return "unclassified"
}

interface Plan {
  gpu?: "amd" | "nvidia" | "intel" | "unknown"
  uv?: "ok" | "absent"
  lock?: "ok" | "fail" | "hang"
  sync?: "ok" | "fail" | "hang"
  source?: "ok" | "fail" | "hang"
  weights?: "ok" | "fail" | "hang"
  torch?: Record<string, unknown> | "fail" | "garbage"
}

const ok = (extra: Partial<RunResult> = {}): RunResult => ({
  code: 0,
  stdout: "",
  stderr: "",
  timedOut: false,
  ...extra,
})
const fail = (stderr: string): RunResult => ({ code: 1, stdout: "", stderr, timedOut: false })
const hang = (): RunResult => ({ code: 124, stdout: "", stderr: "", timedOut: true })

function scripted(plan: Plan = {}): { runner: Runner; steps: Step[]; cwds: (string | undefined)[] } {
  const steps: Step[] = []
  const cwds: (string | undefined)[] = []

  const runner: Runner = async (cmd, args, opts?: RunOptions): Promise<RunResult> => {
    const step = classify(cmd, args)
    steps.push(step)
    cwds.push(opts?.cwd)
    const into = args[args.length - 1] ?? ""

    switch (step) {
      case "have:uv":
        return plan.uv === "absent" ? fail("uv: not found") : ok()
      case "have:lspci":
      case "have:rocm-smi":
        return ok()
      case "lspci":
        // The real output carries the vendor in a bracket tag -- "[AMD/ATI]", "[NVIDIA
        // Corporation]" -- and the detection reads that, so the fixture has to as well. A
        // friendlier fake string here would test a machine that does not exist.
        if (plan.gpu === "nvidia") return ok({ stdout: "01:00.0 VGA: NVIDIA Corporation Device 2684 [NVIDIA Corporation]" })
        if (plan.gpu === "intel") {
          // The regression case, and the reason it is a real lspci line rather than a
          // made-up one: "Corporation" contains "ati", so the unanchored `/AMD|ATI/i` this
          // replaced read this as an AMD card. The script would then have answered
          // `rocm-smi` happily and the install would have gone through.
          return ok({ stdout: "00:02.0 VGA: Intel Corporation Device 9bc4 [Intel Corporation]" })
        }
        if (plan.gpu === "unknown") return ok({ stdout: "00:02.0 VGA: Virtio GPU 1af4 [Device 1af4]" })
        return ok({
          stdout: "0c:00.0 VGA: Advanced Micro Devices, Inc. Device 0x744c [AMD/ATI] (rev c1)",
        })
      case "rocm-smi":
        return ok({ stdout: "         GPU: 0\n          GFX Version:  gfx1201\n" })
      case "fetch:source": {
        if (plan.source === "hang") return hang()
        if (plan.source === "fail") return fail("404 Client Error: pyproject.toml not found")
        await mkdir(into, { recursive: true })
        await writeFile(join(into, "pyproject.toml"), '[project]\nname = "supersonic-julia"\n')
        return ok({ stdout: `{"path": ${JSON.stringify(into)}}\n` })
      }
      case "fetch:weights": {
        if (plan.weights === "hang") return hang()
        if (plan.weights === "fail") return fail("Connection reset by peer")
        await mkdir(into, { recursive: true })
        return ok()
      }
      case "uv lock": {
        if (plan.lock === "hang") return hang()
        if (plan.lock === "fail") return fail("error: Because laya==0.3.21 depends on nothing")
        // uv writes its output into the working directory, and `install` reads it back.
        await writeFile(join(opts!.cwd!, "uv.lock"), GOOD_LOCK)
        return ok()
      }
      case "uv sync": {
        if (plan.sync === "hang") return hang()
        if (plan.sync === "fail") return fail("error: Failed to install:torch")
        // `verify` refuses to report anything without an interpreter at this path.
        await mkdir(join(opts!.cwd!, ".venv", "bin"), { recursive: true })
        await writeFile(join(opts!.cwd!, ".venv", "bin", "python"), "#!/bin/sh\n")
        return ok()
      }
      case "verify": {
        if (plan.torch === "fail") return fail("ModuleNotFoundError: No module named 'torch'")
        if (plan.torch === "garbage") return ok({ stdout: "warning: something\nnot json\n" })
        // The default build has to follow the simulated vendor. A ROCm torch on an
        // NVIDIA card is refused by `verify`, so defaulting to ROCM_TORCH would make
        // every NVIDIA case a failure test whatever it meant to assert -- and that
        // mismatch is why the NVIDIA branch of `verify` had no coverage of its own.
        const fallback = plan.gpu === "nvidia" ? CUDA_TORCH : ROCM_TORCH
        return ok({ stdout: JSON.stringify(plan.torch ?? fallback) })
      }
      case "unclassified":
        return fail(`the script was asked for something it does not model: ${cmd} ${args.join(" ")}`)
    }
  }

  return { runner, steps, cwds }
}

/** Everything install() must never leave behind after a failure past staging. */
async function assertNothingPublished(model: string): Promise<void> {
  expect(existsSync(runtimeDir(model))).toBe(false)
  expect(existsSync(`${runtimeDir(model)}.partial`)).toBe(false)
  expect((await listRuntimes()).filter((r) => r.installed)).toEqual([])
}

describe("the order of the steps", () => {
  test("a plain wheel install detects, locks, syncs, verifies, then publishes", async () => {
    // The module's real subject. `laya` is a wheel with nothing to fetch, so this is the
    // shortest complete path through it -- and it is only this short because the fetches
    // are conditional, which is a fact about the order worth pinning.
    const { runner, steps } = scripted()
    const rt = await install(findModel("laya"), { runner })

    expect(steps).toEqual([
      "have:uv",
      "have:lspci",
      "lspci",
      "have:rocm-smi",
      "rocm-smi",
      "uv lock",
      "uv sync",
      "verify",
    ])
    expect(steps).not.toContain("unclassified")
    expect(rt.installed).toBe(true)
  })

  test("the lock is checked before anything is downloaded, and the sync before verification", async () => {
    // The two orderings that carry the module's argument. Checking the lock first is the
    // difference between refusing before 6 GB and refusing after; verifying before the
    // rename is the difference between a broken runtime and a rejected one.
    const { runner, steps } = scripted()
    await install(findModel("laya"), { runner })
    expect(steps.indexOf("uv lock")).toBeLessThan(steps.indexOf("uv sync"))
    expect(steps.indexOf("uv sync")).toBeLessThan(steps.indexOf("verify"))
  })

  test("a model with a source and weights fetches both, and the weights land outside staging", async () => {
    // Weights are deliberately fetched to their own directory because they outlive the
    // install; the source is deliberately fetched into staging because it does not. Both
    // facts are in the argument list of the subprocess, so they are assertable.
    const { runner, steps, cwds } = scripted()
    const rt = await install(findModel("julia"), { runner })

    expect(steps).toEqual([
      "have:uv",
      "have:lspci",
      "lspci",
      "have:rocm-smi",
      "rocm-smi",
      "fetch:source",
      "fetch:weights",
      "uv lock",
      "uv sync",
      "verify",
    ])

    // The weights directory is a sibling of the runtime, not inside it.
    expect(existsSync(join(weightsDir("julia"), ".complete"))).toBe(true)
    expect(existsSync(join(rt.dir, "julia-src", "pyproject.toml"))).toBe(true)
    // And nothing was fetched while the manifest was being written.
    expect(cwds.indexOf(undefined)).toBeLessThan(cwds.lastIndexOf(undefined))
  })

  test("the published directory carries a meta.json, and there is no staging beside it", async () => {
    // `meta.json` is the whole definition of "installed": `listRuntimes` requires it, and
    // it is written immediately before the rename so that a directory either has it
    // completely or is not the published one at all.
    const { runner } = scripted()
    const rt = await install(findModel("laya"), { runner })

    const meta = JSON.parse(await readFile(join(rt.dir, "meta.json"), "utf8"))
    expect(meta).toMatchObject({ model: "laya", gfx: "gfx1201" })
    expect(typeof meta.created).toBe("string")
    expect(existsSync(`${rt.dir}.partial`)).toBe(false)

    const listed = await listRuntimes()
    expect(listed.map((r) => [r.name, r.installed])).toEqual([["laya", true]])
  })

  test("--lock-only resolves and publishes nothing at all", async () => {
    // No sync, no verify, and no directory. A directory that exists without a `meta.json`
    // is what `runtimes` reports as a half-install, so the resolve-only path has to leave
    // nothing rather than leave something incomplete.
    const { runner, steps } = scripted()
    const rt = await install(findModel("laya"), { runner, lockOnly: true })

    expect(steps).toContain("uv lock")
    expect(steps).not.toContain("uv sync")
    expect(steps).not.toContain("verify")
    expect(rt.installed).toBe(false)
    expect(await listRuntimes()).toEqual([])
  })
})

describe("failure leaves nothing that looks installed", () => {
  // The invariant the module's whole argument rests on, stated in a comment and checked by
  // nothing: no directory with a `meta.json`, and no `<name>.partial`, after any failure
  // past staging. It used to hold on the `verify` path only -- `uv lock` failing, `uv sync`
  // failing and a fetch failing all left staging behind, harmlessly, because `listRuntimes`
  // skips `.partial` and the next install removes it. True of one path out of five is not
  // an invariant.

  const failures: [string, Plan][] = [
    ["uv lock fails", { lock: "fail" }],
    ["uv lock hangs", { lock: "hang" }],
    ["uv sync fails", { sync: "fail" }],
    ["uv sync hangs", { sync: "hang" }],
    ["verify finds no torch", { torch: "fail" }],
    ["verify cannot read torch", { torch: "garbage" }],
    ["verify finds a CUDA build", { torch: CUDA_TORCH }],
    ["fetching the source fails", { source: "fail" }],
    ["fetching the weights fails", { weights: "fail" }],
  ]

  for (const [name, plan] of failures) {
    test(name, async () => {
      const { runner } = scripted(plan)
      await expect(install(findModel("julia"), { runner })).rejects.toThrow()
      await assertNothingPublished("julia")
    })
  }

  test("a hung step says which step gave up, and after how long", async () => {
    // "timed out" alone is not actionable. `uv lock` and `uv sync` hang for entirely
    // different reasons, and a user staring at an unresponsive `onesystem install` needs
    // to know which one to go and look at.
    const { runner } = scripted({ sync: "hang" })
    await expect(install(findModel("laya"), { runner, timeoutMs: 1234 })).rejects.toThrow(
      /uv sync did not finish within 1234ms/,
    )
  })

  test("a failed install keeps the weights it already fetched", async () => {
    // Weights are 550 MB and are fetched outside staging on purpose. A run that then fails
    // must not throw them away, or every retry re-downloads them.
    const { runner } = scripted({ lock: "fail" })
    await expect(install(findModel("julia"), { runner })).rejects.toThrow()
    expect(existsSync(join(weightsDir("julia"), ".complete"))).toBe(true)
  })

  test("a failed reinstall does not destroy the working runtime it was replacing", async () => {
    // The one that matters most in practice, and the reason `rm(dir)` sits after `verify`
    // rather than next to `mkdir(staging)`. A user reinstalling to fix a broken model
    // should not be left with no model at all.
    const first = scripted()
    await install(findModel("laya"), { runner: first.runner })
    const metaBefore = await readFile(join(runtimeDir("laya"), "meta.json"), "utf8")

    const second = scripted({ sync: "fail" })
    await expect(install(findModel("laya"), { runner: second.runner })).rejects.toThrow(/uv sync failed/)

    expect(existsSync(join(runtimeDir("laya"), "meta.json"))).toBe(true)
    expect(await readFile(join(runtimeDir("laya"), "meta.json"), "utf8")).toBe(metaBefore)
    expect((await listRuntimes()).map((r) => r.installed)).toEqual([true])
  })

  test("the failure message carries uv's own stderr, because the exit code does not", async () => {
    const { runner } = scripted({ lock: "fail" })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(
      /Because laya==0\.3\.21 depends on nothing/,
    )
  })
})

describe("an install on an NVIDIA card", () => {
  test("it pins no accelerator and still installs a fetched model from its local copy", async () => {
    // The end-to-end half of the branch that had no coverage. The manifest assertions live
    // in `install.test.ts`; this is the same branch through the whole sequence, which is
    // what caught the missing `[tool.uv.sources]` -- the manifest was asked for PyPI's
    // `supersonic-julia`, a package that is not on PyPI.
    const { runner, steps } = scripted({ gpu: "nvidia" })
    const rt = await install(findModel("julia"), { runner })

    expect(steps).not.toContain("unclassified")
    expect(steps).toEqual([
      "have:uv",
      "have:lspci",
      "lspci",
      "fetch:source",
      "fetch:weights",
      "uv lock",
      "uv sync",
      "verify",
    ])
    // No `rocm-smi`: an NVIDIA card has no gfx target to read, and asking for one is what
    // the ladder used to do by checking for the binary before it knew the vendor.
    expect(steps).not.toContain("rocm-smi")
    const manifest = await readFile(join(rt.dir, "pyproject.toml"), "utf8")
    expect(manifest).toContain(`supersonic-julia = { path = "julia-src" }`)
    expect(manifest).not.toMatch(/rocm/i)
  })

  test("a CUDA build for the right vendor is accepted", async () => {
    // The check that `verify` did not make before. Asking a CUDA build for a HIP version
    // rejects the correct wheel on every NVIDIA machine, so `install laya` could not
    // complete on NVIDIA at all.
    const { runner } = scripted({ gpu: "nvidia" })
    const rt = await install(findModel("laya"), { runner })
    const check = await verify(rt.dir, { vendor: "nvidia" }, runner)
    expect(check).toEqual({ ok: true, problems: [], torch: "13.0" })
  })

  test("a ROCm build on an NVIDIA card is named as the problem it is", async () => {
    // The mirror of the AMD case, which the suite already covered.
    const { runner } = scripted({ gpu: "nvidia", torch: ROCM_TORCH })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(/ROCm build/)
  })
})

describe("the checks on an installed environment", () => {
  test("a ROCm build for the right arch passes", async () => {
    const { runner } = scripted()
    const rt = await install(findModel("laya"), { runner })
    const check = await verify(rt.dir, { vendor: "amd", gfx: "gfx1201" }, runner)
    expect(check).toEqual({ ok: true, problems: [], torch: "6.2.0" })
  })

  // These three assert through `install`, not through `verify` on a finished directory,
  // because a torch with the wrong properties is exactly the case where install refuses
  // to publish: there is no finished directory to run `verify` against. Asserting the
  // refusal is also the version a user meets.

  test("a CUDA build on an AMD card is named as the problem it is", async () => {
    const { runner } = scripted({ torch: CUDA_TORCH })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(/CUDA build/)
  })

  test("a build for the wrong arch says which arch it knows", async () => {
    // A wrong arch is a runtime crash rather than a slow load, so it has to be caught
    // here. The message names what torch actually has, because "wrong arch" alone leaves
    // the user to go and look it up.
    const { runner } = scripted({ torch: { hip: "6.2.0", cuda: null, available: true, arch: ["gfx1100"] } })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(
      /not built for gfx1201; it knows gfx1100/,
    )
  })

  test("a torch that cannot see the GPU is caught even when the build is right", async () => {
    const { runner } = scripted({ torch: { ...ROCM_TORCH, available: false } })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(/no GPU visible/)
  })

  test("a directory with no interpreter is a failed check, not a crash", async () => {
    const { runner } = scripted()
    const rt = await install(findModel("laya"), { runner })
    await rm(join(rt.dir, ".venv"), { recursive: true, force: true })
    const check = await verify(rt.dir, { vendor: "amd", gfx: "gfx1201" }, runner)
    expect(check.ok).toBe(false)
    expect(check.problems[0]).toMatch(/no interpreter at/)
  })
})

describe("refusing to guess the hardware", () => {
  test("uv missing is refused before anything is created", async () => {
    const { runner, steps } = scripted({ uv: "absent" })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(/uv is not on PATH/)
    // Nothing was detected, staged or fetched, and the message says where to get it.
    expect(steps).toEqual(["have:uv"])
  })

  test("an unidentifiable vendor is refused rather than guessed", async () => {
    // The module's stated principle: a CPU-only install works fine until someone measures
    // a 40x slowdown, so detection fails closed.
    const { runner } = scripted({ gpu: "unknown" })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(/refusing to install/)
    await assertNothingPublished("laya")
  })

  test("an Intel card is refused, not read as AMD", async () => {
    // A machine with an Intel iGPU and no AMD card. `lspci` says
    // "Intel Corporation", and "Corporation" contains "ati" — so the unanchored
    // `/AMD|ATI/i` this replaced classified it as AMD, asked for `rocm-smi`, and the script
    // (like a machine that happened to have ROCm's tools installed) would have said yes and
    // produced an install against a card that is not there.
    //
    // It is the one detection bug found during this work that was not self-limiting, and it
    // was found by writing a fixture for a machine that is not this one. Asserted through
    // `install` because that is where the decision is visible.
    const { runner, steps } = scripted({ gpu: "intel" })
    await expect(install(findModel("laya"), { runner })).rejects.toThrow(/refusing to install/)
    // And it never got as far as asking about ROCm, which is the tell: the old code reached
    // `rocm-smi` on a machine that has no AMD GPU.
    expect(steps).not.toContain("rocm-smi")
    await assertNothingPublished("laya")
  })
})

describe("resuming an interrupted weights download", () => {
  test("a weights directory without .complete is fetched again, however much is in it", async () => {
    // `.complete` is the entire resume story, and it is one file's existence. 550 MB is
    // re-fetched when it is absent, which is expensive and correct: there is no state in
    // which a half-downloaded checkpoint looks whole.
    const dir = weightsDir("julia")
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, "model.safetensors"), "x".repeat(64))

    const { runner, steps } = scripted()
    await install(findModel("julia"), { runner })
    expect(steps).toContain("fetch:weights")
  })

  test("a weights directory with .complete is left alone", async () => {
    const dir = weightsDir("julia")
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, ".complete"), "2026-01-01T00:00:00.000Z")

    const { runner, steps } = scripted()
    await install(findModel("julia"), { runner })
    expect(steps).not.toContain("fetch:weights")
  })

  test("a weights download that fails leaves no .complete behind", async () => {
    // The marker is written after the download, never before. Written before, a failed
    // fetch would leave a checkpoint that is a directory and a lie.
    const { runner } = scripted({ weights: "fail" })
    await expect(install(findModel("julia"), { runner })).rejects.toThrow(/could not fetch weights/)
    expect(existsSync(join(weightsDir("julia"), ".complete"))).toBe(false)
  })
})
