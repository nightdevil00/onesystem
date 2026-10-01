# onesystem

Share one GPU decision service across OpenCode sessions. Models load on the first
request and stop when idle.

> **This is a fork.** Upstream is [micahn/onesystem](https://github.com/micahn/onesystem),
> which is where issues and releases belong. The one local change is the NVIDIA
> vendor check described in [My change](#my-change) below; everything else is
> upstream's. The install commands still point at upstream's `install.sh`, which is
> what you want unless you want this fork's checkout.

## My change

Upstream's `verify()` demanded a ROCm build unconditionally. It rejected any
`torch.version.cuda` and required a non-null `torch.version.hip`, so on an NVIDIA
card it rejected the correct CUDA wheel: `install <model>` and `doctor` could never
pass, and no runtime could be installed.

The guard already existed in the neighbouring `assertNoAcceleratorMixups`, which
returns early when the vendor is not AMD. This applies the same shape to `verify()`,
with the checks mirrored for NVIDIA, where the failure is a ROCm build or a non-null
CUDA version. `torch.cuda.is_available()` stays a shared gate for both vendors.

Two smaller things came with it:

- `doctor`'s ok line printed the AMD-flavoured label `torch hip 13.0` on an NVIDIA
  card. It now prints `torch <version>`.
- The install test fixture defaulted its fake torch to a ROCm build regardless of
  the simulated card, so `scripted({ gpu: "nvidia" })` described an NVIDIA card with
  a ROCm build and passed against the unguarded check. The fixture now follows the
  simulated vendor, and two tests cover the NVIDIA branch.

Measured on an NVIDIA GTX 1650 Ti (sm_75, 3.64 GiB), torch 2.14.0+cu130: laya 2570
MiB and julia 668 MiB resident, 3284 MiB of 4096 with both loaded, no OOM. Cold calls
26.5s and 22.4s, warm 0.21s and 0.05s.

## Install

Requires OpenCode V2, Git, curl, mise, and an AMD or NVIDIA GPU. The installer
uses mise to install Bun and uv if they are missing. See
[requirements](#requirements) for GPU setup.

```sh
curl -fsSL https://raw.githubusercontent.com/micahn/onesystem/master/install.sh | bash
```

The installer clones to `~/.local/share/onesystem/repo`, writes the config, and registers
the plugin. It then asks which model runtimes to install — a several-GB download each — and
defaults to none:

```
  1) laya   2) julia   a) all   n) none
  ? models, e.g. "1 2" or "a" [n]:
```

`ONESYSTEM_MODEL` takes the same answers (`laya`, `julia`, `1,2`, `all`, `none`) for when
there is no terminal to ask on. Install both if you like; only one is enabled at a time,
switchable with `onesystem use`. You can run the installer again to update the checkout; it
skips installed runtimes and asks before running `sudo`.

Restart OpenCode, then run `opencode plugin list` to check that `onesystem` loaded, and
look for the status line to confirm the TUI half. Without a model the plugin registers but
has no tools. The first tool call loads the model, taking ~10-15s; warm calls take tens of
milliseconds.

[Manual install](#manual-install) · [Julia or a custom path](#installer-options) ·
[Existing laya setup](#existing-laya-setup)

## Use

The plugin starts the shared daemon as needed. To run commands yourself, open the
checkout:

```sh
cd ~/.local/share/onesystem/repo
bun run src/cli.ts status
```

Use `bun run src/cli.ts <command>` for each command below. If you have added the
package's CLI to `PATH`, you can use `onesystem <command>` instead.

| Command | Action |
| --- | --- |
| `start` | Start the daemon if needed. Safe across concurrent sessions. |
| `status` | Show status as JSON without loading a model. |
| `stop` | Stop the daemon and its local backends. |
| `serve` | Run the daemon in the foreground. |
| `runtimes` | List installed runtimes. |
| `doctor [model]` | Check an installed runtime against the GPU. |
| `install <model>` | Install laya or julia and enable it. |
| `use <model>` | Enable one configured backend and disable the others. |
| `uninstall <model>` | Remove its runtime. Keep downloaded weights. |
| `config-path` | Show the config path. |
| `register-plugin` | Write this checkout into OpenCode's plugins directory. |

Installing or selecting a model disables the other backends. After a config change,
stop the daemon and restart OpenCode to reload the config and tool names.

Logs: `~/.local/state/onesystem/daemon.log`.

### From inside OpenCode

`/onesystem` opens a menu, and is also in the command palette. One flat list, grouped:

| | |
| --- | --- |
| Show status | The same snapshot the sidebar card renders. |
| Restart daemon | Reloads models. Drops warm ones, so the next call pays the load. |
| Enable / Disable `<model>` | Per model, independently. Restarts the daemon after. |
| Install `<model>` | Confirms first, since it is a several-GB download. |
| Daemon port, Idle window | Written to the config. Restarts the daemon after. |
| Poll interval, Sidebar card | OpenCode plugin state, applied without a restart. |

The daemon reads its backend list once at startup, so a toggle or a port change has no
effect until it restarts. The menu restarts it for you, which unloads whatever is warm.

Plugin settings live in OpenCode's own store rather than `onesystem.jsonc`, because they
are not the daemon's: `port` and `idleShutdownSecs` change daemon behaviour, `pollMs` and
`showCard` change what the footer does.

## How it works

```text
OpenCode sessions -> plugin -> shared HTTP daemon -> local MCP process -> GPU
                                                -> existing HTTP service
```

- The plugin registers native tools from `GET /catalog` and sends calls to `POST /call`.
- Startup and status checks load no model. The first backend request starts the process.
- An atomic file lock and the listening port prevent duplicate daemons.
- After 600 seconds idle, local backends stop. Once all are cold, a previously used
  daemon exits. The plugin starts it again before the next tool call.
- Closing one session leaves the shared daemon running for the others.

Other clients can use MCP Streamable HTTP at `/mcp/<backend>`.

## Configuration

Edit `~/.config/onesystem/onesystem.jsonc`. There is one path: a missing file is an error
naming it, not a fallback to some other config. See the
[annotated template](onesystem.config.jsonc) for backend examples.
Its paths are placeholders; `install <model>` writes the real paths.

Most of it is editable from inside OpenCode with the `/onesystem` menu.

| Field | Default | Meaning |
| --- | --- | --- |
| `host` | `127.0.0.1` | Loopback only; the service has no authentication. |
| `port` | `7331` | HTTP port. |
| `idleShutdownSecs` | `600` | Stop an idle local backend after this many seconds. |
| `idleSweepSecs` | `5` | Seconds between idle checks. Must not exceed the idle window. |
| `requestTimeoutSecs` | `120` | Maximum seconds for a forwarded call. |
| `backends` | `{}` | Backend definitions. |

### Backends and tools

- `stdio-mcp`: onesystem starts and stops a local MCP process. Used by laya and julia.
- `systemone-http`: forwards a `systemone` tool to an existing `POST /v1/systemone`
  service. Add one to your own config if you have such a service; it is not in the shipped
  template, since a disabled entry in everyone's config is not a feature. Start the service
  separately. The adapter passes the body through; its schema has not been checked against
  a live service.

Each `stdio-mcp` backend must declare `tools` without its `toolPrefix`. The daemon
serves this list without starting the model. Update it when a backend adds or renames
tools. MCP `tools/list` returns the backend's actual schemas but starts the process.

The plugin strips `toolPrefix` from displayed names. With one backend, tools use
names such as `predict`. With several, they use `laya_predict` and `julia_predict`.
Enable `routing` and set its `default` to let one backend keep unqualified names.
Routing only applies with multiple backends; `routing.tasks` provides logged guidance,
not automatic dispatch. `serverName` overrides the server name in status reports.

For laya, keep `LAYA_PRELOAD: "0"` to load weights only on use. Put `LAYA_PYTHON`
in the backend's `env` when its command needs an explicit interpreter. The installed
`laya-mcp-server` uses its own virtual environment's interpreter.

## Install options

### Requirements

- OpenCode V2. Check `opencode --version`. If mise still selects V1, put the V2
  binary first on `PATH`. This plugin uses the V2 API.
- Git, curl, and mise. Bun and uv are installed with `mise use -g bun uv` when
  they are missing, so the installer stops if mise is not on `PATH`.
- `lspci` from `pciutils` to detect the GPU vendor.
- For AMD, ROCm and `rocm-smi` on `PATH`. NVIDIA uses the CUDA install path and
  does not need ROCm.

On Arch or Omarchy, install the AMD detection tools:

```sh
sudo pacman -S pciutils rocm-core
export PATH=/opt/rocm/bin:$PATH
rocm-smi --showproductname
```

The last command must print a GFX Version. The installer stops if it cannot identify
the GPU or the AMD target. The first runtime install takes several minutes, mostly
to download PyTorch.

### Installer options

Set these variables before running the curl command:

| Variable | Default | Use |
| --- | --- | --- |
| `ONESYSTEM_MODEL` | asks | Which runtimes to install: `laya`, `julia`, `1,2`, `all`, or `none`. |
| `ONESYSTEM_DIR` | `~/.local/share/onesystem/repo` | Choose the checkout path for a curl install. |

For example, `export ONESYSTEM_MODEL=all`, then run the install command.
Running `bash install.sh` from a checkout uses that checkout.

### Manual install

Check the [requirements](#requirements), then:

```sh
git clone https://github.com/micahn/onesystem.git
cd onesystem
bun install
mkdir -p ~/.config/onesystem
cp -n onesystem.config.jsonc ~/.config/onesystem/onesystem.jsonc
bun run src/cli.ts register-plugin
```

That is the whole install. The plugin registers with no model; it has no tools until you
install one or more:

```sh
bun run src/cli.ts install laya      # add julia the same way for both
bun run src/cli.ts start
bun run src/cli.ts status
```

Add `--no-config` to print the config block instead of writing it, or `--lock-only` to
resolve dependencies without installing the runtime. Backends should show `cold` until the
first call.

Registration writes `~/.config/opencode/plugins/onesystem/`, one file per entrypoint,
each re-exporting this checkout:

```ts
// index.ts — server plugin: the laya_* tools
export { default } from "/absolute/path/to/onesystem/src/plugin/index.ts"

// tui.ts — status line and install menu
export { default } from "/absolute/path/to/onesystem/src/plugin/tui.ts"
```

OpenCode discovers that directory on its own. `opencode.json` is not read or written.

A directory rather than a single file because the plugin has two entrypoints and OpenCode
finds the TUI one only when it sits beside the server one. A lone `onesystem.ts` loads the
server half and silently drops the TUI half, which looks like a working install with no
footer.

Re-exports rather than copies, because autodetection skips symlinks and a copy of
`src/plugin/` cannot resolve its own imports or find its CLI at `../cli.ts`. A copy of the
whole tree does resolve, and is worse: a second copy of the code the plugin runs, stale after
`git pull`. Keep the checkout at the path above. Updates take effect when OpenCode reloads
the plugin.

Restart OpenCode, then run `opencode plugin list` for the server half and look for the status
line for the TUI half. The list does not report TUI plugins.

### Existing laya setup

Remove the old local `laya-mcp` entry from `mcp.servers` in `opencode.json`.
Otherwise, sessions still start separate laya processes alongside the shared daemon.

## Development

```sh
bun test
bun run typecheck
```

Tests use fake backends and need no GPU. For an AMD/ROCm check, install the laya
runtime and run `./test/e2e-laya.sh`. It stops and starts the daemon on port 7331,
so run it when no active session needs the service.

The plugin logs nothing by default, because the TUI shares the terminal with OpenCode's
interface and renders whatever a plugin writes. Set `ONESYSTEM_PLUGIN_LOG=1` to see its
diagnostics.

### Measured performance

AMD RX 9070 XT (gfx1201), ROCm 6.4 driver, ROCm 7.2 PyTorch wheel, laya 0.3.21:

| Operation | Time or memory |
| --- | --- |
| Install laya, warm uv cache | ~10 s |
| Install julia, including 585 MB of weights | ~46 s |
| Start daemon, no model loaded | ~190 ms |
| First tool call | laya ~13.7 s; julia ~15.7 s |
| Warm tool call | Tens of milliseconds |
| Both models loaded | 6.6 GB VRAM |

Cold calls spend most of their time importing `transformers`.

On NVIDIA, the numbers are smaller because the card is smaller. GTX 1650 Ti
(sm_75, 3.64 GiB), torch 2.14.0+cu130:

| Operation | Time or memory |
| --- | --- |
| First tool call | laya 26.5 s; julia 22.4 s |
| Warm tool call | laya 0.21 s; julia 0.05 s |
| laya loaded | 2570 MiB resident |
| julia loaded | 668 MiB resident |
| Both models loaded | 3284 MiB of 4096, no OOM |
