/**
 * Daemon and model commands. Concurrent `start` calls wait for the same healthy
 * daemon; a child that loses the lock exits with LOCK_BUSY_EXIT.
 * `status` emits the DaemonStatus contract from health.ts for plugin discovery.
 */

import { spawn } from "node:child_process"
import { closeSync, existsSync, openSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { loadConfig, type Config } from "./config.ts"
import { configDir, configPath, daemonUrl, lockPath, stateDir } from "./paths.ts"
import { registrations } from "./naming.ts"
import type { DaemonStatus } from "./health.ts"
import { inspect } from "./lock.ts"
import { runDaemon, probe, LOCK_BUSY_EXIT } from "./daemon.ts"
import { describeError } from "./async.ts"
import { pluginAutoloadFiles, switchToBackend, writeBackend } from "./config-edit.ts"
import { findModel, MODELS } from "./models.ts"
import {
  backendFor,
  configHint,
  detectGpu,
  install,
  listRuntimes,
  needsManualCommand,
  runtimeDir,
  runtimesRoot,
  uninstall,
  verify,
} from "./install.ts"
import { logger } from "./log.ts"
import { run } from "./subprocess.ts"

const log = logger("cli")

const USAGE = `usage: onesystem <serve|start|stop|status|config-path|install|uninstall|use|runtimes|register-plugin|doctor>`

function parseArgs(argv: string[]): { command: string } {
  return { command: argv[0] ?? "status" }
}


async function cmdServe(config: Config): Promise<void> {
  await runDaemon(config)
  // runDaemon resolves once the server is listening; hold the process open until a
  // signal or the idle sweep ends it.
  await new Promise(() => {})
}

async function cmdStart(config: Config): Promise<number> {
  const url = daemonUrl(config.host, config.port)

  const existing = await probe(url)
  if (existing) {
    const serving = existing.backends.map((b) => b.name).sort()
    const wanted = registrations(config.backends).map((r) => r.backend).sort()
    if (serving.join() === wanted.join()) {
      log.info("already running", { url, pid: existing.pid })
      return 0
    }
    // Not ours to adopt. A daemon on this port from a different config answers health
    // checks fine, so without this check the wrong backends are served and nothing says so.
    process.stderr.write(
      `a daemon on ${url} (pid ${existing.pid}) serves [${serving.join(", ")}],\n` +
        `but this config wants [${wanted.join(", ")}].\n` +
        `It was started from a different config. Run 'onesystem stop', then start again.\n`,
    )
    return 1
  }

  // Log detached stderr to a file. An inherited pipe would stay open until the
  // daemon exits, blocking command substitutions and captured output.
  const logPath = join(stateDir(), "daemon.log")
  await mkdir(stateDir(), { recursive: true })
  const logFd = openSync(logPath, "a")

  const child = spawn(process.execPath, [import.meta.filename ?? "src/cli.ts", "serve"], {
    detached: true,
    stdio: ["ignore", "ignore", logFd],
    env: process.env,
  })
  child.unref()
  closeSync(logFd)
  log.info("spawned daemon", { pid: child.pid, url, log: logPath })

  // Health is available before any model loads.
  const deadline = Date.now() + 20_000
  while (Date.now() < deadline) {
    await Bun.sleep(100)
    const health = await probe(url)
    if (health) {
      log.info("daemon healthy", { url, pid: health.pid })
      return 0
    }
    if (child.exitCode !== null) {
      // LOCK_BUSY is exit 3: we lost the race to a sibling, and that sibling is now
      // serving. Wait for its health rather than reporting failure.
      if (child.exitCode === LOCK_BUSY_EXIT) continue
      log.error("daemon exited during startup", { code: child.exitCode })
      return child.exitCode ?? 1
    }
  }

  log.error("daemon did not become healthy in time", { url })
  return 1
}

async function cmdStop(config: Config): Promise<number> {
  const url = daemonUrl(config.host, config.port)
  const health = await probe(url)
  if (!health) {
    log.info("no daemon reachable", { url })
    return 0
  }
  const pid = typeof health.pid === "number" ? health.pid : null
  if (pid === null) {
    log.error("daemon did not report a pid")
    return 1
  }
  try {
    process.kill(pid, "SIGTERM")
    log.info("sent SIGTERM", { pid })
  } catch (err) {
    log.error("kill failed", { pid, error: String(err) })
    return 1
  }

  for (let i = 0; i < 100; i++) {
    await Bun.sleep(100)
    if (!(await probe(url))) {
      log.info("daemon exited")
      return 0
    }
  }
  log.error("daemon did not exit within 10s", { pid })
  return 1
}

async function cmdStatus(config: Config, resolved: string): Promise<number> {
  const url = daemonUrl(config.host, config.port)
  const health = await probe(url)
  const holder = await inspect(lockPath()).catch(() => null)

  const report: DaemonStatus = {
    config: resolved,
    configDir: configDir(),
    // Plugins use this configured URL instead of deriving their own.
    url,
    running: health !== null,
    daemon: health,
    lock: holder,
    // Share the naming rules with the plugin.
    registrations: registrations(config.backends),
    idleShutdownSecs: config.idleShutdownSecs,
    ...(config.routing ? { routing: config.routing } : {}),
  }
  process.stdout.write(JSON.stringify(report, null, 2) + "\n")
  return 0
}

async function cmdConfigPath(): Promise<number> {
  const file = configPath()
  process.stdout.write(existsSync(file) ? file + "\n" : `${file}\n(no config exists)\n`)
  return 0
}

/**
 * Install and configure a model on request. Plugin setup must not trigger downloads.
 */
async function cmdInstall(args: string[]): Promise<number> {
  const [name, ...rest] = args
  if (!name) {
    process.stderr.write(`usage: onesystem install <model>\nmodels: ${MODELS.map((m) => m.name).join(", ")}\n`)
    return 2
  }
  const spec = findModel(name)
  const lockOnly = rest.includes("--lock-only")
  try {
    const runtime = await install(spec, { runner: run, lockOnly, onProgress: (m) => log.info(m) })
    if (lockOnly) {
      log.info("resolved only; nothing downloaded", { model: name })
      return 0
    }
    const gpu = await detectGpu(run)
    const check = await verify(runtime.dir, gpu, run)
    process.stdout.write(`installed ${name}\n  interpreter: ${runtime.python}\n`)

    const backend = await backendFor(spec, runtime)
    if (rest.includes("--no-config")) {
      process.stdout.write(`\nnot written to any config (--no-config). The block is:\n${configHint(spec, runtime, backend)}\n`)
    } else {
      // Write the resolved backend so the user does not need to copy a config block.
      try {
        const { config, path } = await loadConfig()
        await writeBackend(path, name, backend, { enabled: true })
        const switched = await switchToBackend(path, config, name)
        process.stdout.write(`\nconfigured ${path}\n  enabled: ${name}\n`)
        // Report other backends disabled by the switch.
        for (const other of switched.off) {
          if (other !== name) {
            process.stdout.write(`  turned off: ${other} (onesystem use ${other} to switch back)\n`)
          }
        }
        if (needsManualCommand(backend)) {
          process.stdout.write(
            `\nSet \`command\` in ${path} to the ${name} MCP server binary.\n`,
          )
        } else {
          process.stdout.write(`\nRun \`onesystem start\`, then make a tool call.\n`)
        }
      } catch (err) {
        // The runtime remains usable if the config write fails.
        process.stdout.write(`\ninstalled, but the config was not updated: ${describeError(err)}\n`)
        process.stdout.write(`\nIt is on disk and usable. Add this yourself:\n${configHint(spec, runtime, backend)}\n`)
      }
    }
    if (spec.interpreterEnv) {
      process.stdout.write(
        `\nSet ${spec.interpreterEnv} in backend env so all sessions use the same runtime.\n`,
      )
    }
    return check.ok ? 0 : 1
  } catch (err) {
    process.stderr.write(`install failed: ${describeError(err)}\n`)
    return 1
  }
}

async function cmdRuntimes(): Promise<number> {
  const found = await listRuntimes()
  if (found.length === 0) {
    process.stdout.write(`no runtimes installed (looked in ${runtimesRoot()})\n`)
    return 0
  }
  for (const r of found) {
    const state = r.installed ? "ok" : "INCOMPLETE (no meta.json — safe to delete)"
    process.stdout.write(`${r.name.padEnd(10)} ${state.padEnd(34)} ${r.python}\n`)
  }
  return 0
}

async function cmdUninstall(args: string[]): Promise<number> {
  const [name] = args
  if (!name) {
    process.stderr.write("usage: onesystem uninstall <model>\n")
    return 2
  }
  const removed = await uninstall(name)
  process.stdout.write(removed ? `removed ${name}\n` : `${name} was not installed\n`)
  return 0
}

/**
 * Check the installed interpreter, PyTorch build, and GPU visibility.
 */
async function cmdDoctor(args: string[]): Promise<number> {
  const gpu = await detectGpu(run)
  process.stdout.write(`gpu: ${gpu.vendor}${gpu.gfx ? ` (${gpu.gfx})` : ""}\n`)
  const targets = args.length > 0 ? [args[0]!] : (await listRuntimes()).map((r) => r.name)
  if (targets.length === 0) {
    process.stdout.write("no runtimes to check\n")
    return 0
  }
  let bad = 0
  for (const name of targets) {
    const check = await verify(runtimeDir(name), gpu, run)
    if (check.ok) {
      process.stdout.write(`${name}: ok${check.torch ? ` (torch ${check.torch})` : ""}\n`)
    } else {
      bad++
      process.stdout.write(`${name}: FAILED\n`)
      for (const p of check.problems) process.stdout.write(`  - ${p}\n`)
    }
  }
  return bad === 0 ? 0 : 1
}

/**
 * Enable one configured model while preserving comments and detecting concurrent edits.
 */
async function cmdUse(args: string[]): Promise<number> {
  const [name] = args
  if (!name) {
    process.stderr.write("usage: onesystem use <model>\n")
    return 2
  }
  const { config, path } = await loadConfig()

  let switched: Awaited<ReturnType<typeof switchToBackend>>
  try {
    switched = await switchToBackend(path, config, name)
  } catch (err) {
    process.stderr.write(`${describeError(err)}\n`)
    return 1
  }

  if (!switched.changed) {
    process.stdout.write(`${name} is already the only enabled backend\n`)
    return 0
  }
  for (const n of switched.on) process.stdout.write(`on   ${n}\n`)
  for (const n of switched.off) process.stdout.write(`off  ${n}\n`)
  process.stdout.write(`\n${path} updated. Run \`onesystem start\` to pick the change up.\n`)
  return 0
}

/**
 * Write the plugin autoload file and remove its old plugins-array entry to avoid
 * duplicate loading. JSONC edits preserve unrelated config and comments.
 */
async function cmdRegisterPlugin(): Promise<number> {
  const { dir, files } = pluginAutoloadFiles()
  let wrote = 0
  try {
    await mkdir(dir, { recursive: true })
    for (const f of files) {
      // Counted before writing, so a re-run reports "already" and does not trip a watcher.
      if ((await readFile(f.path, "utf8").catch(() => "")) !== f.contents) wrote++
      await writeFile(f.path, f.contents)
    }
  } catch (err) {
    process.stderr.write(`could not write ${dir}: ${describeError(err)}\n`)
    return 1
  }

  const names = files.map((f) => f.path.slice(dir.length + 1)).join(", ")
  process.stdout.write(`${wrote === 0 ? "already registered" : "registered"}: ${dir}\n`)
  process.stdout.write(`  ${names}\n`)
  process.stdout.write("  restart OpenCode, then check the footer for the TUI half\n")
  return 0
}


export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  const { command } = parseArgs(argv)

  // Before config loading, so it still works with no config file present at all.
  if (command === "config-path") return cmdConfigPath()

  // Installation and diagnostics must work before a valid config exists.
  if (
    command === "install" ||
    command === "uninstall" ||
    command === "runtimes" ||
    command === "register-plugin" ||
    command === "doctor"
  ) {
    switch (command) {
      case "install":
        return cmdInstall(argv.slice(1))
      case "uninstall":
        return cmdUninstall(argv.slice(1))
      case "runtimes":
        return cmdRuntimes()
      case "register-plugin":
        return cmdRegisterPlugin()
      default:
        return cmdDoctor(argv.slice(1))
    }
  }

  let loaded: { config: Config; path: string }
  try {
    loaded = await loadConfig()
  } catch (err) {
    process.stderr.write(`${(err as Error).message}\n`)
    return 2
  }
  const { config, path } = loaded

  switch (command) {
    case "use":
      return cmdUse(argv.slice(1))
    case "serve":
      await cmdServe(config)
      return 0
    case "start":
      return cmdStart(config)
    case "stop":
      return cmdStop(config)
    case "status":
      return cmdStatus(config, path)
    default:
      process.stderr.write(`unknown command: ${command}\n${USAGE}\n`)
      return 2
  }
}

if (import.meta.main) {
  main().then(
    (code) => {
      if (code !== 0) process.exit(code)
    },
    (err) => {
      process.stderr.write(`onesystem: ${describeError(err)}\n`)
      // Preserve LOCK_BUSY_EXIT so start can wait for the winning daemon.
      process.exit(process.exitCode ?? 1)
    },
  )
}
