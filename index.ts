import { Plugin } from "@opencode/plugin"
import { spawn, type ChildProcess } from "node:child_process"
import { existsSync } from "node:fs"
import { appendFile, mkdir, rm, stat, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"

/**
 * Local Windows text-to-speech plugin for OpenCode.
 *
 * Without `command` it uses the bundled Windows SAPI fallback (sapi.ps1).
 * With `command` it runs any local TTS CLI:
 *   - `{out}`      output audio file, created by the plugin
 *   - `{textfile}` UTF-8 file with the text chunk
 *   - `{text}`     the text chunk itself (be careful with command-line length)
 *   - `{voice}`    the configured voice
 *   - `{lang}`     the configured language
 *   - `{rate}`     the configured rate
 *   - `{plugin}`   this plugin's directory
 *
 * If the command contains neither `{textfile}` nor `{text}`, the text is
 * written to stdin. Set `output: "stdout"` when the engine writes audio to
 * stdout instead of `{out}`.
 */

const PLUGIN_DIR = import.meta.dir
const IS_WINDOWS = process.platform === "win32"
const IS_MAC = process.platform === "darwin"
const POWERSHELL = path.join(
  process.env.SystemRoot ?? "C:\\Windows",
  "System32",
  "WindowsPowerShell",
  "v1.0",
  "powershell.exe",
)
const TEMP_DIR = path.join(os.tmpdir(), "opencode-voice-tts")

type SvcOptions = {
  enabled?: boolean
  python?: string
  script?: string
  model?: string
  config?: string
  speaker?: string
  transpose?: number
  device?: string
  half?: boolean
  f0Method?: string
  autoPredictF0?: boolean
  maxChunkSeconds?: number
  timeoutMs?: number
}

type Options = {
  enabled?: boolean
  speakAnswers?: boolean
  events?: string[]
  mainSessionOnly?: boolean
  command?: string[]
  svc?: SvcOptions
  output?: "file" | "stdout"
  format?: string
  voice?: string
  lang?: string
  rate?: number
  maxChars?: number
  maxTotalChars?: number
  timeoutMs?: number
  playTimeoutMs?: number
  prefix?: string
  cwd?: string
  keepAudio?: boolean
  debug?: boolean
}

type RunResult = {
  code: number | null
  stdout: Buffer
  stderr: string
}

let current: ChildProcess | null = null
let speechToken = 0
const activeChildren = new Set<ChildProcess>()

function numberOption(value: unknown, fallback: number, min: number, max: number) {
  const parsed = typeof value === "number" ? value : Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.min(max, Math.max(min, Math.trunc(parsed)))
}

function log(opts: Options, ...args: unknown[]) {
  const line = args
    .map((value) => (value instanceof Error ? value.stack ?? value.message : String(value)))
    .join(" ")
  console.error(`[voice-tts] ${line}`)
  if (opts.debug) {
    const entry = `${new Date().toISOString()} ${line}\n`
    void mkdir(TEMP_DIR, { recursive: true })
      .then(() => appendFile(path.join(TEMP_DIR, "plugin.log"), entry, "utf8"))
      .catch(() => {})
  }
}

function killTree(child: ChildProcess | null) {
  if (!child?.pid) return
  try {
    child.kill()
  } catch {
    // ignore
  }
  if (process.platform === "win32") {
    try {
      const killer = spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
        windowsHide: true,
        stdio: "ignore",
      })
      killer.unref()
    } catch {
      // ignore
    }
  }
}

function stopCurrent() {
  current = null
  for (const child of activeChildren) killTree(child)
  activeChildren.clear()
}

function playbackAttempts(file: string): Array<[string, string[]]> {
  if (IS_WINDOWS) {
    return [
      [
        POWERSHELL,
        [
          "-NoProfile",
          "-NonInteractive",
          "-ExecutionPolicy",
          "Bypass",
          "-File",
          path.join(PLUGIN_DIR, "play.ps1"),
          "-Path",
          file,
        ],
      ],
    ]
  }
  if (IS_MAC) return [["afplay", [file]]]
  return [
    ["ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", file]],
    ["paplay", [file]],
    ["aplay", ["-q", file]],
  ]
}

async function playAudio(file: string, cwd: string, timeoutMs: number): Promise<RunResult> {
  let lastError: unknown
  let lastResult: RunResult | undefined
  for (const [command, args] of playbackAttempts(file)) {
    try {
      const result = await run(command, args, { cwd, timeoutMs })
      if (result.code === 0) return result
      lastResult = result
    } catch (error) {
      lastError = error
    }
  }
  if (lastResult) return lastResult
  throw lastError instanceof Error
    ? lastError
    : new Error("no audio player found; install ffmpeg (ffplay) or alsa-utils (aplay)")
}

function run(
  command: string,
  args: string[],
  options: {
    input?: string
    cwd?: string
    timeoutMs: number
    captureStdout?: boolean
  },
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const captureStdout = options.captureStdout === true
    let child: ChildProcess
    try {
      child = spawn(command, args, {
        cwd: options.cwd,
        windowsHide: true,
        stdio: ["pipe", captureStdout ? "pipe" : "ignore", "pipe"],
      })
    } catch (error) {
      reject(error)
      return
    }

    current = child
    activeChildren.add(child)
    const stdout: Buffer[] = []
    let stderr = ""
    let settled = false

    const finish = (error: Error | undefined, result?: RunResult) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      activeChildren.delete(child)
      if (current === child) current = null
      if (error) reject(error)
      else resolve(result!)
    }

    const timer = setTimeout(() => {
      killTree(child)
      finish(new Error(`process timed out after ${options.timeoutMs} ms`))
    }, options.timeoutMs)

    child.on("error", (error) => finish(error))
    child.stdout?.on("data", (chunk: Buffer) => stdout.push(chunk))
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8")
      if (stderr.length > 8000) stderr = stderr.slice(-8000)
    })
    child.on("close", (code) => {
      finish(undefined, { code, stdout: Buffer.concat(stdout), stderr: stderr.trim() })
    })

    if (options.input === undefined) child.stdin?.end()
    else child.stdin?.end(options.input, "utf8")
  })
}

function expand(
  template: string,
  values: { out: string; textfile: string; text: string; voice: string; lang: string; rate: string },
) {
  return template
    .split("{plugin}")
    .join(PLUGIN_DIR)
    .split("{out}")
    .join(values.out)
    .split("{textfile}")
    .join(values.textfile)
    .split("{text}")
    .join(values.text)
    .split("{voice}")
    .join(values.voice)
    .split("{lang}")
    .join(values.lang)
    .split("{rate}")
    .join(values.rate)
}

function cleanText(input: string, maxTotalChars: number) {
  let text = input

  // Fenced code blocks are noise when spoken.
  text = text.replace(/```[\s\S]*?```/g, " ")
  text = text.replace(/~~~[\s\S]*?~~~/g, " ")

  // Markdown links, images and bare URLs.
  text = text.replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
  text = text.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
  text = text.replace(/https?:\/\/\S+/g, " ")

  // Headings, list markers, quotes and table pipes.
  text = text.replace(/^\s{0,3}#{1,6}\s+/gm, "")
  text = text.replace(/^\s*>\s?/gm, "")
  text = text.replace(/^\s*[-*+]\s+/gm, "")
  text = text.replace(/^\s*\d+[.)]\s+/gm, "")
  text = text.replace(/\|/g, " ")

  // Inline formatting.
  text = text.replace(/`([^`]+)`/g, "$1")
  text = text.replace(/\*\*|__|~~|\*/g, "")
  text = text.replace(/[ \t]+/g, " ")
  text = text.replace(/\n{2,}/g, "\n")
  text = text.trim()

  if (maxTotalChars && text.length > maxTotalChars) {
    text = text.slice(0, maxTotalChars)
    const lastStop = Math.max(text.lastIndexOf("."), text.lastIndexOf("!"), text.lastIndexOf("?"), text.lastIndexOf("…"))
    if (lastStop > maxTotalChars * 0.5) text = text.slice(0, lastStop + 1)
  }
  return text
}

function splitText(text: string, maxChars: number) {
  if (text.length <= maxChars) return [text]

  const sentences = text.match(/[^.!?…\n]+[.!?…]*\s*/g) ?? [text]
  const chunks: string[] = []
  let currentChunk = ""

  const push = (value: string) => {
    const trimmed = value.trim()
    if (trimmed) chunks.push(trimmed)
  }

  for (const sentence of sentences) {
    if ((currentChunk + sentence).length > maxChars && currentChunk) {
      push(currentChunk)
      currentChunk = sentence
    } else {
      currentChunk += sentence
    }
    while (currentChunk.length > maxChars) {
      push(currentChunk.slice(0, maxChars))
      currentChunk = currentChunk.slice(maxChars)
    }
  }
  push(currentChunk)
  return chunks
}

export default Plugin.define({
  id: "local.voice-tts",
  async setup(ctx) {
    const opts = (ctx.options ?? {}) as Options
    if (opts.enabled === false) return

    const lang = String(opts.lang ?? "ru")
    const voice = String(opts.voice ?? "")
    const rate = numberOption(opts.rate, 0, -10, 10)
    const maxChars = numberOption(opts.maxChars, 900, 120, 4000)
    const maxTotalChars = numberOption(opts.maxTotalChars, 6000, 0, 200_000)
    const timeoutMs = numberOption(opts.timeoutMs, 120_000, 5_000, 600_000)
    const playTimeoutMs = numberOption(opts.playTimeoutMs, 180_000, 5_000, 600_000)
    const prefix = String(opts.prefix ?? "")
    const command = Array.isArray(opts.command) ? opts.command.map(String).filter(Boolean) : []
    const outputMode = opts.output === "stdout" ? "stdout" : "file"
    const format = String(opts.format ?? "wav").replace(/[^a-z0-9]/gi, "") || "wav"
    const cwd = typeof opts.cwd === "string" && opts.cwd ? opts.cwd : PLUGIN_DIR
    const mainSessionOnly = opts.mainSessionOnly !== false
    const keepAudio = opts.keepAudio === true

    const svc = opts.svc ?? {}
    const svcEnabled = svc.enabled === true
    const svcPython = String(svc.python ?? "")
    const svcScript = String(svc.script ?? path.join(PLUGIN_DIR, "svc_worker.py"))
    const svcModel = String(svc.model ?? "")
    const svcConfig = String(svc.config ?? "")
    const svcClusterModel = String(svc.clusterModelPath ?? "")
    const svcClusterRatio = numberOption(svc.clusterInferRatio, 0, 0, 1)
    const svcSpeaker = String(svc.speaker ?? "")
    const svcTranspose = numberOption(svc.transpose, -7, -24, 24)
    const svcDevice = String(svc.device ?? "cuda")
    const svcHalf = svc.half === true
    const svcF0Method = String(svc.f0Method ?? "dio")
    const svcAutoPredictF0 = svc.autoPredictF0 === true
    const svcMaxChunkSeconds = numberOption(svc.maxChunkSeconds, 15, 1, 40)
    const svcTimeoutMs = numberOption(svc.timeoutMs, 120_000, 10_000, 600_000)

    const enabledEvents = new Set(
      (Array.isArray(opts.events) ? opts.events.map(String) : ["answer", "permission", "question", "error"]),
    )

    const textParts = new Map<string, Map<number, string>>()
    const spokenPermissions = new Set<string>()

    const svcActive =
      svcEnabled &&
      svcPython !== "" &&
      svcModel !== "" &&
      svcConfig !== "" &&
      existsSync(svcScript)

    if (svcEnabled && !svcActive) {
      log(opts, "SVC is enabled but python/model/config/script is missing; using the unconverted audio")
    }

    let svcProcess: ChildProcess | null = null
    let svcReadyPromise: Promise<void> | null = null
    let svcSequence = 0
    let svcBuffer = ""
    let svcStderr = ""
    const svcPending = new Map<
      number,
      { resolve: () => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }
    >()

    function rejectSvcPending(error: Error) {
      for (const pending of svcPending.values()) {
        clearTimeout(pending.timer)
        pending.reject(error)
      }
      svcPending.clear()
    }

    function startSvcWorker() {
      if (svcProcess && svcProcess.exitCode === null && svcReadyPromise) return svcReadyPromise

      rejectSvcPending(new Error("SVC worker restarted"))
      svcBuffer = ""
      svcStderr = ""

      const args = [
        "-u",
        svcScript,
        "--model",
        svcModel,
        "--config",
        svcConfig,
        "--cluster-model",
        svcClusterModel,
        "--speaker",
        svcSpeaker,
        "--device",
        svcDevice,
      ]
      if (svcHalf) args.push("--half")

      let child: ChildProcess
      try {
        child = spawn(svcPython, args, {
          cwd: PLUGIN_DIR,
          windowsHide: true,
          stdio: ["pipe", "pipe", "pipe"],
        })
      } catch (error) {
        return Promise.reject(error)
      }

      svcProcess = child
      let resolveReady!: () => void
      let rejectReady!: (error: Error) => void
      svcReadyPromise = new Promise<void>((resolve, reject) => {
        resolveReady = resolve
        rejectReady = reject
      })

      const readyTimer = setTimeout(() => {
        rejectReady(new Error(`SVC worker did not become ready in ${svcTimeoutMs} ms`))
        killTree(child)
      }, svcTimeoutMs)

      child.stdout?.on("data", (chunk: Buffer) => {
        svcBuffer += chunk.toString("utf8")
        let newline = svcBuffer.indexOf("\n")
        while (newline !== -1) {
          const line = svcBuffer.slice(0, newline).trim()
          svcBuffer = svcBuffer.slice(newline + 1)
          newline = svcBuffer.indexOf("\n")
          if (!line) continue
          try {
            const message = JSON.parse(line)
            if (message.ready === true) {
              clearTimeout(readyTimer)
              log(opts, `SVC worker ready (${message.sampleRate ?? "?"} Hz)`)
              resolveReady()
            } else if (message.ready === false) {
              clearTimeout(readyTimer)
              rejectReady(new Error(message.error ?? "SVC worker failed to start"))
            } else if (typeof message.id === "number") {
              const pending = svcPending.get(message.id)
              if (pending) {
                svcPending.delete(message.id)
                clearTimeout(pending.timer)
                if (message.ok) pending.resolve()
                else pending.reject(new Error(message.error ?? "SVC conversion failed"))
              }
            }
          } catch (error) {
            log(opts, `unexpected SVC output: ${line}`)
          }
        }
      })

      child.stderr?.on("data", (chunk: Buffer) => {
        svcStderr += chunk.toString("utf8")
        if (svcStderr.length > 6000) svcStderr = svcStderr.slice(-6000)
      })

      child.on("error", (error) => {
        clearTimeout(readyTimer)
        rejectReady(error)
      })

      child.on("exit", (code) => {
        clearTimeout(readyTimer)
        if (svcProcess === child) svcProcess = null
        svcReadyPromise = null
        const error = new Error(`SVC worker exited with code ${code}: ${svcStderr.slice(-1500)}`)
        rejectReady(error)
        rejectSvcPending(error)
      })

      return svcReadyPromise
    }

    async function convertSvc(input: string, output: string, token: number) {
      if (!svcActive) return false
      await startSvcWorker()
      if (token !== speechToken) return false
      if (!svcProcess?.stdin) throw new Error("SVC worker is not running")

      const id = ++svcSequence
      const response = new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => {
          svcPending.delete(id)
          reject(new Error(`SVC conversion timed out after ${svcTimeoutMs} ms`))
        }, svcTimeoutMs)
        svcPending.set(id, {
          resolve: () => {
            clearTimeout(timer)
            resolve()
          },
          reject: (error) => {
            clearTimeout(timer)
            reject(error)
          },
          timer,
        })
      })

      svcProcess.stdin.write(
        JSON.stringify({
          id,
          input,
          output,
          transpose: svcTranspose,
          f0Method: svcF0Method,
          autoPredictF0: svcAutoPredictF0,
          maxChunkSeconds: svcMaxChunkSeconds,
          clusterInferRatio: svcClusterRatio,
          speaker: svcSpeaker || undefined,
        }) + "\n",
      )
      await response
      return true
    }

    function stopSvcWorker() {
      const child = svcProcess
      svcProcess = null
      svcReadyPromise = null
      rejectSvcPending(new Error("SVC worker stopped"))
      if (child) killTree(child)
    }

    type Rendered = { file: string; cleanup: () => Promise<void> }

    async function renderChunk(chunk: string, token: number): Promise<Rendered | null> {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
      await mkdir(TEMP_DIR, { recursive: true })
      const textFile = path.join(TEMP_DIR, `${id}.txt`)
      const outFile = path.join(TEMP_DIR, `${id}.${format}`)
      let svcFile: string | null = null
      await writeFile(textFile, chunk, "utf8")

      const cleanup = async () => {
        await rm(textFile, { force: true }).catch(() => {})
        if (!keepAudio) await rm(outFile, { force: true }).catch(() => {})
        if (!keepAudio && svcFile) await rm(svcFile, { force: true }).catch(() => {})
      }

      try {
        if (command.length > 0) {
          const args = command
            .slice(1)
            .map((arg) =>
              expand(arg, { out: outFile, textfile: textFile, text: chunk, voice, lang, rate: String(rate) }),
            )
          const usesText = command.some((arg) => arg.includes("{textfile}") || arg.includes("{text}"))
          const result = await run(command[0], args, {
            cwd,
            input: usesText ? undefined : chunk,
            timeoutMs,
            captureStdout: outputMode === "stdout",
          })
          if (token !== speechToken) {
            await cleanup()
            return null
          }
          if (result.code !== 0) {
            throw new Error(`engine exited with code ${result.code}: ${result.stderr || "(no stderr)"}`)
          }
          if (outputMode === "stdout") await writeFile(outFile, result.stdout)
        } else if (!IS_WINDOWS) {
          throw new Error(
            "no TTS command configured: set options.command in opencode.json (this fallback is Windows-only)",
          )
        } else {
          const sapiArgs = [
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            path.join(PLUGIN_DIR, "sapi.ps1"),
            "-Out",
            outFile,
            "-TextFile",
            textFile,
            "-Rate",
            String(rate),
          ]
          if (voice) sapiArgs.push("-Voice", voice)
          const result = await run(POWERSHELL, sapiArgs, { cwd, timeoutMs })
          if (token !== speechToken) {
            await cleanup()
            return null
          }
          if (result.code !== 0) {
            throw new Error(`SAPI exited with code ${result.code}: ${result.stderr || "(no stderr)"}`)
          }
        }

        if (token !== speechToken) {
          await cleanup()
          return null
        }
        if (!existsSync(outFile)) throw new Error(`engine did not create ${outFile}`)
        const info = await stat(outFile)
        if (!info.size) throw new Error(`engine created an empty file ${outFile}`)

        let playFile = outFile
        if (svcActive) {
          svcFile = path.join(TEMP_DIR, `${id}.svc.wav`)
          const started = Date.now()
          try {
            const converted = await convertSvc(outFile, svcFile, token)
            if (token !== speechToken) {
              await cleanup()
              return null
            }
            if (converted && existsSync(svcFile) && (await stat(svcFile)).size > 0) {
              playFile = svcFile
              log(opts, `SVC converted in ${Date.now() - started} ms`)
            }
          } catch (error) {
            log(opts, "SVC failed; playing the unconverted audio:", error)
          }
        }

        if (token !== speechToken) {
          await cleanup()
          return null
        }
        return { file: playFile, cleanup }
      } catch (error) {
        await cleanup()
        throw error
      }
    }

    async function playChunk(file: string, token: number) {
      if (token !== speechToken) return
      const played = await playAudio(file, cwd, playTimeoutMs)
      if (token !== speechToken) return
      if (played.code !== 0) {
        throw new Error(`playback exited with code ${played.code}: ${played.stderr || "(no stderr)"}`)
      }
    }

    async function speakWithToken(text: string, kind: string, token: number) {
      const cleaned = cleanText(text, maxTotalChars)
      if (!cleaned) return
      log(opts, `speaking ${kind}, ${cleaned.length} chars`)
      const chunks = splitText(prefix + cleaned, maxChars)
      if (chunks.length === 0) return

      // Synthesize the next chunk while the current one is playing.
      const start = (index: number): Promise<Rendered | null> =>
        index < chunks.length
          ? renderChunk(chunks[index], token).catch((error) => {
              log(opts, "render failed:", error)
              return null
            })
          : Promise.resolve(null)

      let pending = start(0)
      for (let index = 0; index < chunks.length; index++) {
        const rendered = await pending
        if (token !== speechToken) {
          if (rendered) await rendered.cleanup()
          return
        }
        if (!rendered) return
        pending = start(index + 1)
        try {
          await playChunk(rendered.file, token)
        } finally {
          await rendered.cleanup()
        }
        if (token !== speechToken) {
          void pending.then((extra) => extra?.cleanup())
          return
        }
      }
    }

    let speechChain: Promise<void> = Promise.resolve()

    function say(text: string, kind: string) {
      const token = ++speechToken
      stopCurrent()
      speechChain = speechChain
        .then(() => (token === speechToken ? speakWithToken(text, kind, token) : undefined))
        .catch((error) => log(opts, `${kind} failed:`, error))
    }

    if (command.length > 1 && command.some((arg) => arg.endsWith("tts_f5.py"))) {
      void run(command[0], [command[1], "--warm"], { cwd, timeoutMs: Math.max(timeoutMs, 240_000) })
        .then((result) => {
          if (result.code === 0) log(opts, "worker prewarmed")
          else log(opts, `prewarm failed (code ${result.code}): ${result.stderr}`)
        })
        .catch((error) => log(opts, "prewarm failed:", error))
    }

    async function claim(key: string) {
      try {
        await mkdir(TEMP_DIR, { recursive: true })
        const safe = key.replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 140)
        await writeFile(path.join(TEMP_DIR, `claim-${safe}`), String(process.pid), { flag: "wx" })
        return true
      } catch (error: any) {
        if (error?.code === "EEXIST") return false
        return true
      }
    }

    async function finishStep(sessionID: string, messageID: string, finish: string) {
      const key = `${sessionID}:${messageID}`
      const parts = textParts.get(key)
      textParts.delete(key)
      if (!parts || parts.size === 0) return
      if (finish === "tool-calls" || finish === "error" || finish === "content-filter") return

      // Let the last text event settle before reading.
      await new Promise((resolve) => setTimeout(resolve, 200))

      try {
        if (mainSessionOnly) {
          const session = await ctx.session.get({ sessionID })
          if (session?.parentID) return
        }

        const text = [...parts.entries()]
          .sort(([left], [right]) => left - right)
          .map(([, value]) => value)
          .join("\n\n")
        const cleaned = cleanText(text, maxTotalChars)
        if (!cleaned) return

        if (!(await claim(`msg-${sessionID}-${messageID}`))) return
        say(cleaned, "answer")
      } catch (error) {
        log(opts, "answer handler failed:", error)
      }
    }

    const wanted: string[] = []
    if (enabledEvents.has("answer") && opts.speakAnswers !== false) {
      wanted.push("session.text.ended", "session.step.ended")
    }
    if (enabledEvents.has("permission")) wanted.push("permission.asked")
    if (enabledEvents.has("question")) wanted.push("form.created")
    if (enabledEvents.has("error")) wanted.push("session.execution.failed")

    if (wanted.length === 0) return

    const controller = new AbortController()
    void (async () => {
      try {
        for await (const event of ctx.event.subscribe({ types: wanted as any, signal: controller.signal })) {
          try {
            if (event.type === "session.text.ended") {
              const key = `${event.data.sessionID}:${event.data.assistantMessageID}`
              let parts = textParts.get(key)
              if (!parts) {
                parts = new Map()
                textParts.set(key, parts)
              }
              parts.set(event.data.ordinal, event.data.text)
            } else if (event.type === "session.step.ended") {
              void finishStep(event.data.sessionID, event.data.assistantMessageID, event.data.finish)
            } else if (event.type === "permission.asked") {
              const id = event.data.id
              if (spokenPermissions.has(id)) continue
              spokenPermissions.add(id)
              if (!(await claim(`perm-${id}`))) continue
              const action = event.data.action
              const resource = event.data.resources?.[0]
              const message = event.data.message
              say(
                message ||
                  `Требуется разрешение: ${action}${resource ? `, ${path.basename(resource)}` : ""}`,
                "permission",
              )
            } else if (event.type === "form.created") {
              if (!(await claim(`form-${event.id}`))) continue
              const title = event.data.form?.title
              say(title ? `Агент задал вопрос: ${title}` : "Агент задал вопрос", "question")
            } else if (event.type === "session.execution.failed") {
              if (mainSessionOnly) {
                const session = await ctx.session.get({ sessionID: event.data.sessionID })
                if (session?.parentID) continue
              }
              if (!(await claim(`error-${event.id}`))) continue
              say("Ошибка выполнения задачи", "error")
            }
          } catch (error) {
            log(opts, `event ${event.type} failed:`, error)
          }
        }
      } catch (error) {
        if (!controller.signal.aborted) log(opts, "event stream stopped:", error)
      }
    })()

    log(
      opts,
      `loaded; engine=${command.length ? command[0] : "Windows SAPI"}; svc=${svcActive ? path.basename(svcModel) : "off"}; events=${[...enabledEvents].join(",")}`,
    )

    return () => {
      controller.abort()
      stopCurrent()
      stopSvcWorker()
    }
  },
})
