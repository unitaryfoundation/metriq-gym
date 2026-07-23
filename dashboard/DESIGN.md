# Metriq-Gym Jobs Dashboard — Design Instructions

A lightweight, single-page, local dashboard for tracking metriq-gym benchmark jobs
through their full lifecycle: **dispatch → queued/running on hardware → results ready →
polled → uploaded to metriq-data**.

The target user story: an operator dispatches many jobs (typically the benchmark
configurations from the Metriq paper's suite spec) across several devices, then wants to
see at a glance which jobs have come back, which devices still have gaps in their
Metriq-score coverage, and which completed results still need to be uploaded to
`metriq-data`.

---

## 1. Architecture (keep it small)

**One HTML file + one small Python server file. No build step, no framework, no database.**

```
dashboard/
├── server.py      # ~150 lines: stdlib http.server or FastAPI; serves index.html + JSON API
├── index.html     # markup + CSS + vanilla JS in one file
└── state.json     # sidecar: dashboard-owned state (upload records, last-poll cache)
```

### Data sources

| Source | What it provides | Access |
|---|---|---|
| `~/Library/Application Support/metriq-gym/localdb.jsonl` | One JSON line per dispatched job: `id`, `job_type`, `params`, `data.provider_job_ids`, `provider_name`, `device_name`, `dispatch_time`, `suite_id`, `suite_name`, `result_data`, `runtime_seconds` | Read-only. Resolve via `metriq_gym.paths.get_data_db_path()` so `MGYM_LOCAL_DB_DIR` is honored |
| `mgym job poll <id>` | Live provider status for pending jobs (QUEUED w/ position, RUNNING, COMPLETED, FAILED); populates `result_data` in the localdb on completion | Subprocess, on demand |
| `mgym job upload <id>` | Opens the metriq-data PR | Subprocess, on demand |
| `state.json` (sidecar) | Upload status (`{job_id: {pr_url, uploaded_at}}`) and cached poll results | Dashboard-owned |

> **Important:** metriq-gym does **not** record upload state in the localdb —
> `upload_job()` opens a PR and prints the URL but writes nothing back. The sidecar
> `state.json` is therefore required; the server records the PR URL there after a
> successful upload action.

### API surface (all local)

- `GET /api/jobs` — parse localdb.jsonl, merge sidecar state, return the wire model below
- `POST /api/poll/{job_id}` — run `mgym job poll`, parse stdout, cache + return status
- `POST /api/poll-all` — poll every non-terminal job, serialized with a small delay between calls
- `POST /api/upload/{job_id}` — run `mgym job upload`, capture PR URL into `state.json`

The page re-fetches `/api/jobs` every 60 s (plain `setInterval` + `fetch`; no websockets).
`/api/jobs` must never hit provider APIs — reading the page is always instant; only
explicit poll actions talk to the cloud.

### Wire model (what `/api/jobs` returns per job)

```json
{
  "id": "7ba5ee0e-2c1a-4b1c-a24c-04ae34827837",
  "benchmark": "Linear Ramp QAOA",
  "provider": "braket",
  "device": "Cepheus-1-108Q",
  "num_qubits": 10,
  "params": { "...": "full params dict for the detail drawer" },
  "suite_id": null,
  "suite_name": null,
  "dispatch_time": "2026-07-23T14:02:11",
  "runtime_seconds": null,
  "state": "queued | running | ready_to_upload | uploaded | failed | unknown",
  "queue_position": 4,
  "pr_url": null,
  "spec_match": true
}
```

### State derivation

| State | Rule |
|---|---|
| `ready_to_upload` | `result_data` present in localdb AND no upload record in sidecar |
| `uploaded` | upload record exists in sidecar |
| `queued` / `running` | `result_data` null; last cached poll says QUEUED/RUNNING (show min queue position across tasks) |
| `failed` | last cached poll reported FAILED/CANCELLED for any task |
| `unknown` | `result_data` null and never polled this session — render as neutral "not polled yet", with the Poll action as the affordance |

---

## 2. Page layout

Single page, `max-width: 1120px`, centered, generous whitespace — mirroring
metriq.info's container. Four sections top to bottom:

```
┌──────────────────────────────────────────────────────────────────┐
│  [metriq-gym logo/wordmark]  jobs           ⟳ Poll all pending   │  ← nav
├──────────────────────────────────────────────────────────────────┤
│  ⚠ 3 completed jobs are ready to upload to metriq-data           │  ← action banner
├──────────────────────────────────────────────────────────────────┤
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │
│  │   4    │ │   1    │ │   3    │ │   12   │ │   0    │          │  ← stat tiles
│  │ Queued │ │Running │ │ Ready  │ │Uploaded│ │ Failed │          │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘          │
├──────────────────────────────────────────────────────────────────┤
│  Metriq-score coverage                                           │
│  device ↓ / benchmark →   BSEQ CLOPS EPLG Mirror QMLK LRQAOA …   │  ← coverage matrix
│  ibm_torino                ✓    ✓    ✓    ✓    2/4   4/4  …      │
│  Cepheus-1-108Q            —    —    —    —    —     1/4  …      │
├──────────────────────────────────────────────────────────────────┤
│  Jobs                       [All ▾] [Device ▾] [Benchmark ▾]     │
│  ● state | benchmark | n | device | dispatched | runtime | ⋮     │  ← jobs table
│  …rows…                                                          │
└──────────────────────────────────────────────────────────────────┘
│  (click row → slide-in detail drawer with params, task IDs,      │
│   result summary, raw JSON, and per-job actions)                 │
```

### 2a. Nav

Brand lockup on the left (reuse the metriq wordmark treatment: logo + a small
lowercase accent-colored suffix, e.g. `jobs` where metriq.info shows `beta`).
Right side: a subtle **⟳ Poll all pending** button and a muted "last refreshed 12 s ago"
timestamp.

### 2b. Action banner (conditional)

Only rendered when `ready_to_upload > 0` or `failed > 0`. Styled exactly like
metriq.info's beta banner: amber gradient background (`#fff7ed → #fff5d7`), `1px solid
#fde68a` border, text `#92400e`, radius 10px. Content: "**3 jobs** ready to upload to
metriq-data" with an inline "Upload all" text button. Failed jobs get a second line in
the red status color. This is the page's single call-to-action surface — nothing else
competes for attention.

### 2c. Stat tiles (the five lifecycle counts)

Five equal tiles in a row (wrap to 2+3 under 720px). Per the hero-number pattern:
big numeral (32–40px, weight 600), small muted label underneath with the status icon.
Tiles are **filters**: clicking "Ready" filters the jobs table to that state (and sets an
active style using metriq.info's filter tokens: bg `#eef4ff`, border `#93bbfd`).
No trend sparklines — counts only. A tile at zero renders at 45% opacity.

### 2d. Metriq-score coverage matrix (the distinctive feature)

Answers: *"for each device, how much of the paper's Metriq-score suite have I collected?"*

- **Rows:** devices (any device appearing in the localdb).
- **Columns:** the 8 suite benchmarks — BSEQ, CLOPS, EPLG, Mirror Circuits, QML Kernel,
  LR-QAOA, WIT, QFT.
- **Cell contents:** completion against the paper's spec widths, defined as data (see
  spec table below): single-config benchmarks show a state glyph (✓ uploaded, ● ready,
  ◌ in flight, — never dispatched); multi-width benchmarks show a fraction (`2/4`) with a
  thin horizontal progress underline, colored by the *worst* pending state.
- A job counts toward a cell only if its params match the spec config (`spec_match:
  true`); matching is a subset comparison of the job's params against the spec entry
  (ignore `benchmark_name`, compare the rest). Non-spec jobs still appear in the jobs
  table, marked with a muted "custom" chip.
- Cells are clickable → filters the table to that device + benchmark.

**Spec configs (embed as a JSON constant; source: paper §IV + repo suite files):**

| Benchmark | Spec instances counted |
|---|---|
| LR-QAOA | 1D chain, p=[10], Δβ=0.3, Δγ=0.6, shots 1000, trials 10 at N ∈ {10, 20, 50, 100} (`metriq_gym/suites/lr_qaoa_scale.json`) |
| QML Kernel | n ∈ {10, 20, 30, 50}, shots 1000 |
| QFT | sweep min 4 → max 20, skip 4 (i.e. widths {4,8,12,20}), method 1 |
| EPLG | one chain-length run (lengths panel per paper; IBM/OriginQ vs reduced IQM/Rigetti variants both count) |
| Mirror Circuits | fixed panel (w,d) ∈ {(8,64),(16,32),(24,16),(32,8),(64,4),(128,2)}, p2q=0.5, 10 circuits |
| BSEQ | device-wide, shots 1000 |
| CLOPS | N=100, L=100, M=1000, S=100 |
| WIT | n=7, shots 8192 |

### 2e. Jobs table

One row per job, newest first. Columns:

| status chip | benchmark | n | device | provider | suite | dispatched | runtime | actions |

- Status chip: tinted background + icon + word (never color alone — see §4).
- `dispatched` shows relative time ("2 h ago") with the ISO timestamp on hover.
- Queued rows append the queue position to the chip ("Queued · #4").
- Actions column (icon buttons, visible on hover): **Poll** (non-terminal states),
  **Upload** (ready state), **PR ↗** (uploaded state, links to the PR).
- Filter row above the table: state pills, device dropdown, benchmark dropdown —
  metriq.info filter-token styling. Filters compose with tile/matrix clicks.
- Suite-dispatched jobs get a thin left border in `--accent-2` teal and group together
  under a collapsible suite header row showing aggregate progress ("lr_qaoa_1d_scale_suite
  on ibm_fez — 3/4 complete").

### 2f. Detail drawer

Clicking a row slides a 480px drawer from the right (page never navigates):
full params as pretty-printed JSON, provider task IDs (each linking to the provider
console where a URL pattern is known — Braket: task detail page in the AWS console),
dispatch/app-version metadata, result summary when present (for LR-QAOA: approximation
ratios and the beats-random flag per depth), and the raw localdb record behind a
"copy JSON" button. Per-job Poll / Upload buttons repeat here at full size.

---

## 3. Visual identity (lifted from metriq.info's styles.css)

Use metriq.info's actual tokens verbatim so the dashboard reads as a sibling of the site:

```css
:root {
  --bg: #ffffff;
  --bg-2: #f8fafc;
  --panel: rgba(17, 27, 44, 0.05);
  --text: #1e2a3b;
  --muted: #6c778a;
  --accent: #2563eb;        /* metriq blue — links, active states, running */
  --accent-2: #1fb4c2;      /* teal — suite accents, secondary highlights */
  --border: rgba(15, 23, 42, 0.08);
  --radius: 12px;
  --shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
  --font: "Iowan Old Style", "Palatino LT STD", "Book Antiqua", Palatino, serif;
}
```

- **Typography is the signature:** metriq.info sets *everything* — headings and body — in
  the Iowan Old Style / Palatino serif stack. Do the same. Body 16px/1.65, weight 400;
  headings weight 600 with tight letter-spacing. Exception: numerals in stat tiles, the
  coverage matrix, and table numeric columns use `font-variant-numeric: tabular-nums`,
  and job/task IDs use the system mono stack at 12–13px.
- **Surfaces:** white page, no heavy cards. Sections separated by `1px solid var(--border)`
  rules and whitespace, not boxes (metriq.info sets `.card { background: transparent;
  border: none }`). Reserve `--shadow` + radius for the detail drawer and dropdowns only.
- **Buttons:** metriq.info's `.btn` recipe — `#f8fafc` bg, 1px `rgba(15,23,42,.14)`
  border, radius 6px, weight 600, 13px; hover tints toward blue (`#eef2fd`). Primary
  action (`Upload`) uses `.btn--accent`: solid `--accent`, white text.
- **Light theme only** — metriq.info ships no dark mode; matching its look means
  committing to light. (If dark mode is ever added, re-derive status tints against the
  dark surface rather than inverting.)

---

## 4. Status color system

Status colors are reserved for job state and used nowhere else. Every status renders as
**icon + word + color** — never color alone. Chips: 12.5px, weight 600, 4px 10px padding,
999px radius, tinted background with dark same-hue text (all pairs ≥ 4.5:1 on their tint):

| State | Icon | Text/icon color | Chip background |
|---|---|---|---|
| Queued | ◌ (hollow) | `#6c778a` | `#f1f5f9` |
| Running | ● (2s pulse) | `#2563eb` | `#eef4ff` |
| Ready to upload | ↑ | `#b45309` | `#fff7ed` |
| Uploaded | ✓ | `#15803d` | `#f0fdf4` |
| Failed | ✕ | `#b91c1c` | `#fef2f2` |

Semantics: **amber = your action needed** (results sitting locally, not yet contributed);
**green = done** (in metriq-data); blue = machine is working; gray = waiting on the queue;
red = needs investigation. The same five colors drive the stat tiles, matrix glyphs, and
banner accents. "Running" is deliberately the same blue as `--accent`: an active link
color for an active state.

No categorical chart palette is needed anywhere — the page has no multi-series charts
(counts, a matrix, and a table cover the use case; that's by design, keep it that way).

---

## 5. Interaction details

- **Auto-refresh:** `/api/jobs` every 60 s; pause the timer when the tab is hidden
  (`visibilitychange`). A manual ⟳ always works.
- **Poll all pending:** disabled while running; streams per-job results into the UI as
  each subprocess finishes (server can respond with NDJSON lines). Rate-limit to ≥ 2 s
  between provider calls.
- **Upload:** confirm dialog naming the job and target repo, then run; on success swap
  the chip to Uploaded and surface the PR link. On failure show the subprocess stderr in
  the drawer verbatim.
- **Empty states:** no jobs at all → one-line hint with the dispatch command to copy.
  Matrix with one device → still render (the grid is useful from the first job).
- **Row hover:** `--bg-2` wash; all click targets ≥ 32px tall.
- **No pagination** below 200 jobs; beyond that, lazy-render table rows (the matrix and
  tiles always aggregate everything).

---

## 6. Build checklist

1. `server.py`: jobs endpoint reading localdb.jsonl via `get_data_db_path()`; sidecar
   read/write; subprocess wrappers for poll/upload with parsed stdout.
2. `index.html`: tokens from §3, layout from §2, chips from §4; vanilla JS render
   functions (`renderTiles`, `renderMatrix`, `renderTable`, `renderDrawer`) off one
   in-memory `jobs[]` array; filters as pure functions of a single `filterState` object.
3. Spec-config JSON constant (§2d table) + subset param matcher.
4. Manual pass: long device names (`arn:aws:...` never shown — always `device_name`),
   50+ job render time, a failed-poll job, a suite group, banner with 0/1/many.
