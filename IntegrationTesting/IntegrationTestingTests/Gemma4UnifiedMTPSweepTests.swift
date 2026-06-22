// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
@_spi(Testing) import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - Gemma 4 Unified (12B) MTP performance exploration
//
// This file is the *authoritative* perf source for the unified 12B MTP pair
// (the integration throughput tests in `Gemma4UnifiedMTPIntegrationTests` are
// short smoke tests — see RECON "Ordering-fix re-measurement"). Two sweeps:
//
//   1. blockSize ∈ {2, 3, 4} against the bf16 drafter, over two prompts
//      engineered to *sustain* generation past the warmup regime (the
//      smoke-test prompts hit EOS early). RECON hypothesis: blockSize 2–3
//      may beat 4 in the high-entropy regime (break-even accept ≈0.40–0.45).
//   2. The same sweep against the 8-bit drafter. #267's quant study showed
//      int4 breaks the drafter's top-1; 8-bit is untested — compare its
//      acceptance against bf16.
//
// Cost-tuned defaults: `sweepMaxTokens=256`, `sweepRepetitions=1`. At greedy
// (temperature=0) the **accept rate is exactly deterministic** — the core
// scientific output (per-cell accept rate, blockSize ranking, bf16-vs-8bit)
// needs no repeats; repetitions only smooth wall-clock tok/s (a single sample
// carries ~±1–2% timing noise, noted in the table). The full matrix at these
// defaults is ~18 generations of ≤256 tokens + two 12B target loads (a few
// minutes). Bump the two constants to 512 / 2–3 for a heavier, tok/s-stable
// run. A discarded warmup absorbs one-time Metal costs; speedup is vs a
// once-per-prompt non-speculative baseline at the same length.
//
// Opt-in: this multi-cell sweep takes minutes, so it only runs
// when the marker file `~/.run_mtp_sweep` exists. Toggle from the CLI:
//   touch ~/.run_mtp_sweep   # enable, then run the -only-testing command
//   rm ~/.run_mtp_sweep      # disable
//
// A *file* gate (not an env var) is deliberate: `xcodebuild` does not forward
// the invoking shell's environment to the swift-testing host that evaluates
// `.enabled(if:)`, so `TEST_MTP_SWEEP=1 xcodebuild …` silently skips. The
// home-directory marker is visible to that process regardless.

// MARK: - Model IDs

private let unifiedTargetModelId = "mlx-community/gemma-4-12B-it-4bit"
private let unifiedDrafterBf16ModelId = "mlx-community/gemma-4-12B-it-assistant-bf16"
private let unifiedDrafter8bitModelId = "mlx-community/gemma-4-12B-it-assistant-8bit"

/// True when the opt-in marker `~/.run_mtp_sweep` exists. Filesystem state is
/// visible to the swift-testing host process; the invoking shell's
/// environment is not (see the file header).
private var mtpSweepEnabled: Bool {
    let marker = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".run_mtp_sweep")
    return FileManager.default.fileExists(atPath: marker.path)
}

private let sweepBlockSizes = [2, 3, 4]
private let sweepMaxTokens = 256
private let sweepRepetitions = 1

/// Prompts engineered to sustain generation past the warmup regime rather
/// than natural-stopping early (the smoke-test story/CSV/sky prompts EOS well
/// before 128). One low-entropy expository, one high-entropy creative.
private let sustainedPrompts: [(label: String, text: String)] = [
    (
        "tcp",
        "Write a detailed, 800-word technical explanation of how TCP congestion "
            + "control works. Cover slow start, congestion avoidance, fast retransmit, "
            + "fast recovery, and how the congestion window evolves in each phase."
    ),
    (
        "story",
        "Write a detailed short story of at least 800 words about a lighthouse keeper "
            + "who discovers a mysterious hand-drawn map hidden inside the lens room. "
            + "Develop the setting, the keeper's history, and what the map leads to."
    ),
]

// MARK: - Measurement

private struct StreamMeasurement {
    let tokPerSec: Double
    let proposed: Int
    let accepted: Int
    let generated: Int

    var acceptRate: Double { proposed > 0 ? Double(accepted) / Double(proposed) : 0 }
}

/// Run a single `generate` stream to completion and reduce its `.info` event
/// to a `StreamMeasurement`. `blockSize == nil` is the non-speculative
/// baseline (no drafter); a non-nil value runs MTP at that block size.
private func runStream(
    loaded: MTPLoadedPair,
    lmInput: LMInput,
    maxTokens: Int,
    blockSize: Int?
) async throws -> StreamMeasurement? {
    let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    let stream: AsyncStream<Generation>
    if let blockSize {
        stream = try generate(
            input: lmInput, parameters: parameters, context: loaded.context,
            mtpDrafter: loaded.drafter, blockSize: blockSize)
    } else {
        stream = try generate(input: lmInput, parameters: parameters, context: loaded.context)
    }

    var info: GenerateCompletionInfo?
    for await event in stream {
        if case .info(let completionInfo) = event { info = completionInfo }
    }
    guard let info else { return nil }

    let tokPerSec =
        info.generateTime > 0 ? Double(info.generationTokenCount) / info.generateTime : 0
    return StreamMeasurement(
        tokPerSec: tokPerSec,
        proposed: info.proposedDraftTokens ?? 0,
        accepted: info.acceptedDraftTokens ?? 0,
        generated: info.generationTokenCount)
}

// MARK: - Tests

@Suite(.serialized)
struct Gemma4UnifiedMTPSweepTests {

    /// blockSize sweep against the production bf16 drafter.
    @Test(.enabled(if: mtpSweepEnabled))
    func testBlockSizeSweepBf16Drafter() async throws {
        try await runBlockSizeSweep(
            drafterId: unifiedDrafterBf16ModelId, drafterLabel: "bf16")
    }

    /// Same sweep against the 8-bit drafter — acceptance comparison vs bf16.
    @Test(.enabled(if: mtpSweepEnabled))
    func testBlockSizeSweep8bitDrafter() async throws {
        try await runBlockSizeSweep(
            drafterId: unifiedDrafter8bitModelId, drafterLabel: "8bit")
    }

    /// Load the target + drafter once, warm up once, then for each sustained
    /// prompt measure a once-per-prompt baseline and a `sweepRepetitions`-mean
    /// MTP cell at each block size. One table row per (prompt, blockSize).
    private func runBlockSizeSweep(drafterId: String, drafterLabel: String) async throws {
        guard
            let loaded = try await loadMTPPair(
                targetId: unifiedTargetModelId,
                drafterId: drafterId,
                targetTokenizerLoader: #huggingFaceTokenizerLoader())
        else {
            Issue.record(
                "required checkpoint not in HF cache (12B 4-bit target or \(drafterLabel) drafter); skipping sweep"
            )
            return
        }

        // One discarded warmup absorbs Metal kernel compilation and memory-pool
        // growth so no measured stream pays those one-time costs.
        let warmupInput = try await loaded.context.processor.prepare(
            input: UserInput(chat: [.user(sustainedPrompts[0].text)]))
        _ = try await runStream(
            loaded: loaded, lmInput: warmupInput, maxTokens: 16, blockSize: nil)

        print(
            "[MTP sweep \(drafterLabel)] prompt | blockSize | mtp tok/s (n=\(sweepRepetitions)) | baseline tok/s | speedup | accept rate | gen tokens"
        )

        var sawAcceptedDraft = false

        for prompt in sustainedPrompts {
            let lmInput = try await loaded.context.processor.prepare(
                input: UserInput(chat: [.user(prompt.text)]))

            guard
                let baseline = try await runStream(
                    loaded: loaded, lmInput: lmInput, maxTokens: sweepMaxTokens, blockSize: nil)
            else {
                Issue.record("baseline stream for \(prompt.label) emitted no .info event")
                return
            }

            for blockSize in sweepBlockSizes {
                var samples: [StreamMeasurement] = []
                for _ in 0 ..< sweepRepetitions {
                    guard
                        let sample = try await runStream(
                            loaded: loaded, lmInput: lmInput, maxTokens: sweepMaxTokens,
                            blockSize: blockSize)
                    else {
                        Issue.record(
                            "MTP stream (\(prompt.label), bs=\(blockSize)) emitted no .info event")
                        return
                    }
                    samples.append(sample)
                }

                let meanTokPerSec = samples.map(\.tokPerSec).reduce(0, +) / Double(samples.count)
                let meanProposed = samples.map(\.proposed).reduce(0, +) / samples.count
                let meanAccepted = samples.map(\.accepted).reduce(0, +) / samples.count
                let acceptRate = meanProposed > 0 ? Double(meanAccepted) / Double(meanProposed) : 0
                let speedup = baseline.tokPerSec > 0 ? meanTokPerSec / baseline.tokPerSec : 0
                if meanAccepted > 0 { sawAcceptedDraft = true }

                print(
                    "[MTP sweep \(drafterLabel)] \(prompt.label) | bs=\(blockSize) | "
                        + "\(String(format: "%.2f", meanTokPerSec)) | "
                        + "\(String(format: "%.2f", baseline.tokPerSec)) | "
                        + "\(String(format: "%.2fx", speedup)) | "
                        + "\(String(format: "%.1f%%", acceptRate * 100)) (\(meanAccepted)/\(meanProposed)) | "
                        + "\(samples[0].generated)"
                )
            }
        }

        // Sanity: the drafter must produce accepted drafts somewhere in the
        // sweep. A flat zero across every cell means the \(drafterLabel)
        // drafter regressed (e.g. int-quant breaking top-1 — the open question
        // for the 8-bit trial).
        #expect(
            sawAcceptedDraft,
            "\(drafterLabel) drafter accepted zero drafts across the entire sweep — drafter quality regression"
        )
    }
}
