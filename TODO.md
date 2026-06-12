# TODO — Gemma 4 12B unified (PR #327) adoption / local validation

Working notes for validating **PR #327 "Add Gemma 4 12B unified"** in this checkout.
Research project that tracks this: `~/Projects/Research/gemma-4-12b/` (see `wiki/RECON.md`).

## Current state (2026-06-08) — Tasks 1–4 DONE, text + text+image generation validated
- Branch **`model/gemma-4-12b-unified`** (PR #327 head) checked out @ `75c77d5`.
- Remotes: `origin` = `ml-explore/mlx-swift-lm`, `fork` = `mdstaff/mlx-swift-lm`.
- ✅ **Task 1 — all 3 `Gemma4UnifiedTests` pass under `xcodebuild`** (`-destination 'platform=macOS,arch=arm64'`,
  `-only-testing:MLXLMTests/Gemma4UnifiedTests`), incl. the Metal `modelVisionForward`. The metallib
  blocker was a **missing Metal Toolchain** — fixed with `xcodebuild -downloadComponent MetalToolchain`
  (~700 MB). Not a PR bug. (Old `swift test`/CLI metallib limitation still stands; use Xcode.)
- ✅ **Task 2 — added `gemma4_12B_it_4bit` `ModelConfiguration`** (VLMRegistry + `all()`,
  `Libraries/MLXVLM/VLMModelFactory.swift`); `swift build` clean.
- ✅ **Task 3 — text-only generation passes, KV-cache clean.** Added `gemma4_unified_12B()` to
  `IntegrationTesting/IntegrationTestingTests/CoherenceIntegrationTests.swift` (`vlmContainer` +
  `planetsCoherence`, `import MLXVLM`). Model pre-staged via `hf download` (6.3 GB in
  `~/.cache/huggingface/hub`). Run: **1 test passed, 7.68 s**, coherent planet list, **NO SIGTRAP** —
  the #282 Gap-3 KV-cache round-trip on the real 48-layer 5:1 interleave is clean (`num_kv_shared_layers: 0`).
  ⚠️ Gotcha: Swift Testing `-only-testing` needs the trailing `\(\)`; without it xcodebuild runs **0 tests**
  but still prints `** TEST SUCCEEDED **` (false green). Real run shows `Test run with 1 test … passed`.
- ✅ **Task 4 — text+image generation passes.** Added `gemma4_unified_12B_vision()` (reuses the existing
  `ChatSessionTests.visionModel` helper: red `CIImage` → "what color?"). Run: **1 test passed, 4.18 s**,
  model answered **"Red"** — full encoder-free vision path end-to-end (patchify → `VisionEmbedder` →
  `MultimodalEmbedder` → masked scatter → generation), no crash. Model loads from the same HF cache.

### Prior state (2026-06-04, superseded)
- `swift test --filter Gemma4UnifiedTests`: `configDecoding` passed; `processorPatchifiesImages` +
  `modelVisionForward` failed with `Failed to load the default metallib` (CLI limitation, now resolved via Xcode + Metal Toolchain).

## Decisions (from project owner)
- **Run via Xcode / `xcodebuild`** (supported path that bundles the metallib) — not the CLI. Owner runs
  xcodebuild commands themselves; work sequentially.
- **Download done**: `mlx-community/gemma-4-12B-it-4bit` (6.3 GB) now cached in `~/.cache/huggingface/hub`.

## Tasks 1–4 — DONE (2026-06-08). Exact commands for reproduction:
- Unit tests (no download): `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MLXLMTests/Gemma4UnifiedTests`
- Pre-stage model: `hf download mlx-community/gemma-4-12B-it-4bit` (6.3 GB → `~/.cache/huggingface/hub`)
- Text-only: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -only-testing:IntegrationTestingTests/CoherenceIntegrationTests/gemma4_unified_12B\(\)`
- Text+image: same as above with `…/gemma4_unified_12B_vision\(\)`
- ⚠️ The trailing `\(\)` is **required** (Swift Testing); without it → 0 tests run but `** TEST SUCCEEDED **`.

## Local changes — committed `46b71fb4`, pushed to `fork`
Branch **`gemma-4-12b-local-validation`** (off PR #327 @ `75c77d5`), pushed to `fork` (mdstaff). Contains:
1. `Libraries/MLXVLM/VLMModelFactory.swift` — `gemma4_12B_it_4bit` config + `all()`.
2. `IntegrationTesting/IntegrationTestingTests/CoherenceIntegrationTests.swift` — 2 tests + `import MLXVLM`.
3. `TODO.md` (this file).
→ Owner verdict: keep on the fork, don't upstream / no PR. (`origin`=ml-explore upstream — do not push.)

## Phase 3 — drafter / MTP: DONE 2026-06-12 (#308 adopted + extended for unified, validated on-device)
Full analysis: `~/Projects/Research/gemma-4-12b/wiki/RECON.md` ("Phase 3 resolution — 2026-06-09"
for the adopt-vs-build call and gap table; "Phase 3 implementation — 2026-06-12" for results).
- **#308 MERGED upstream 2026-06-11** (`e145aca`); this branch merged `origin/main` @ `e3cb1e1`
  as **`d07374e`** — 6 conflict hunks vs our #327 layer hand-resolved (signature unions of
  `tokenTypeIds` × `emitDrafterState`; kept un-chunked image prefill over #337's chunk loop —
  the bidirectional vision overlay can't span chunk boundaries).
- **Unified extension (G2–G4):** `Gemma4Unified.callAsFunction(_:cache:state:)` reading
  `mtpEmitFlagKey`; `draftBlock` target cast widened to `Gemma4 | Gemma4Unified`;
  `"gemma4_unified_assistant"` registered. Config decode + centroids-off worked by construction.
- **Validation:** `Gemma4UnifiedMTPIntegrationTests` (3 tests, all pass) pairing
  `gemma4_12B_it_4bit` target + `mlx-community/gemma-4-12B-it-assistant-bf16` drafter.
  Prompt-entropy bracket (4 tests after adding the coding prompt): coding **78/144 = 0.542
  accept, 1.21× — net win**; factual 40/67 = 0.597, ~1.08×; creative 62/193 = 0.32,
  **0.83× (net slowdown)**. No sticky passthrough, coherent greedy output. Verdict:
  workload-dependent — the 12B-4bit target is fast enough that the drafter's per-token cost
  (262k-vocab head) needs ~0.40–0.45 acceptance to break even; low-entropy generation (code)
  clears it, high-entropy prose doesn't. 31B-8bit (#308's 1.586×) remains uniformly favorable.
- ⚠️ Test-file gotcha: must `import Tokenizers` (the `#huggingFaceTokenizerLoader()` macro
  expansion references it) AND qualify `any MLXLMCommon.Tokenizer` (repo vendors its own
  `Tokenizer` protocol; bare name is ambiguous).
- Unexplored follow-ups: blockSize sweep (2–3 at low acceptance), 8-bit drafter, longer runs.
- Still out of scope unless asked: audio/video processor paths (model-side only in #327);
  upstream follow-up PR (needs owner sign-off; #327 itself is now CONFLICTING vs main).

## Notes
- `IntegrationTesting/IntegrationTesting.xcodeproj` uses its **own** DerivedData → first build re-resolves
  the whole SwiftPM graph (one-time). Needs the **Metal Toolchain** (`xcodebuild -downloadComponent MetalToolchain`).
- mlx-swift is pinned `.upToNextMinor(from: "0.31.4")` (`Package.swift:39`); no core mlx-swift changes
  needed for this work.
- Benign noise to ignore: mlx-swift Metal `-Wunused-const-variable` warnings; `CoreData`/`NSXPCConnection`/
  `com.apple.contactsd` errors (headless xctest).
