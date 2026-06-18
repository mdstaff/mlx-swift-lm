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

// MARK: - Gemma 4 Unified (12B) MTP speculative decoding
//
// End-to-end validation that the MTP iterator works with the unified
// (encoder-free) family: target `gemma4_unified` (12B 4-bit) + drafter
// `gemma4_unified_assistant` (12B assistant, bf16). The drafter shares the
// old family's draft architecture — same `Gemma4AssistantDraftModel` — but
// its `text_config.model_type` is `gemma4_unified_text` and the top-level
// model type is `gemma4_unified_assistant`, exercising the unified registry
// key and the unified-aware target cast in `draftBlock`.
//
// Expectations are calibrated against the 31B suite
// (`MTPIteratorEndToEndDiagnosticTests`): accepted drafts > 0 with no
// sticky passthrough. Byte-identity with baseline is NOT asserted — the
// 4-bit target's quantization noise breaks it for high-entropy content
// (documented #308 INT4-target caveat); greedy semantics are preserved.

// MARK: - Helpers

private let unifiedTargetModelId = "mlx-community/gemma-4-12B-it-4bit"
private let unifiedDrafterModelId = "mlx-community/gemma-4-12B-it-assistant-bf16"

private func hfSnapshotDir(modelId: String) -> URL? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let hub = home.appendingPathComponent(".cache/huggingface/hub")
    let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    let snapshots = hub.appendingPathComponent(folderName).appendingPathComponent("snapshots")
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first
}

private struct LoadedPair {
    let context: ModelContext
    let drafter: any MTPDrafterModel
}

private func loadUnifiedTargetAndDrafter() async throws -> LoadedPair? {
    guard let targetDir = hfSnapshotDir(modelId: unifiedTargetModelId) else { return nil }
    guard let drafterDir = hfSnapshotDir(modelId: unifiedDrafterModelId) else { return nil }

    let context = try await VLMModelFactory.shared.load(
        from: targetDir,
        using: #huggingFaceTokenizerLoader()
    )

    // Load the drafter through the registry + factory (not by direct
    // construction) so the `gemma4_unified_assistant` creator key is on the
    // tested path.
    await Gemma4AssistantRegistration.register()
    let container = try await MTPDrafterModelFactory.shared.loadContainer(
        from: drafterDir, using: NoOpTokenizerLoader()
    )
    let drafter = await container.perform { ctx in
        ctx.model as? any MTPDrafterModel
    }
    guard let drafter else { return nil }

    return LoadedPair(context: context, drafter: drafter)
}

/// `MTPDrafterModelFactory` ignores the loader (drafters borrow their
/// target's tokenizer), but the protocol requires a non-optional argument.
/// Throws (rather than traps) so an invariant break fails the test instead
/// of killing the process.
private struct UnexpectedTokenizerLoad: Error {}

private final class NoOpTokenizerLoader: TokenizerLoader {
    func load(from url: URL) async throws -> any MLXLMCommon.Tokenizer {
        throw UnexpectedTokenizerLoad()
    }
}

private struct CollectedRun {
    var text = ""
    var info: GenerateCompletionInfo?
}

private func collect(_ stream: AsyncStream<Generation>) async -> CollectedRun {
    var run = CollectedRun()
    for await event in stream {
        switch event {
        case .chunk(let chunk):
            run.text += chunk
        case .toolCall:
            break
        case .info(let completionInfo):
            run.info = completionInfo
        }
    }
    return run
}

// MARK: - Tests

@Suite(.serialized)
struct Gemma4UnifiedMTPIntegrationTests {

    /// Registry + factory path: the 12B assistant's `config.json` carries
    /// `model_type: gemma4_unified_assistant`; loading it through
    /// `MTPDrafterModelFactory` validates the unified creator key and the
    /// unified `text_config` decode (`gemma4_unified_text` auto-detection).
    @Test
    func testUnifiedAssistantFactoryLoad() async throws {
        guard let drafterDir = hfSnapshotDir(modelId: unifiedDrafterModelId) else {
            Issue.record("12B-assistant-bf16 checkpoint not in HF cache; skipping")
            return
        }

        await Gemma4AssistantRegistration.register()
        let container = try await MTPDrafterModelFactory.shared.loadContainer(
            from: drafterDir, using: NoOpTokenizerLoader()
        )
        let isDrafter = await container.perform { ctx in
            ctx.model is Gemma4AssistantDraftModel
        }
        #expect(isDrafter)
    }

    /// Full MTP cycle on the unified 12B pair at production blockSize=4:
    /// speculation must run, at least one draft must be accepted, and the
    /// iterator must not engage sticky-passthrough. No acceptance-rate floor
    /// yet — the first successful run logs the observed rate (yardstick:
    /// 0.658 accept / ~2x from the QAT-MTP llama.cpp benchmark; realistic
    /// MLX expectation ~1.5-1.6x per the 31B-8bit numbers).
    @Test
    func testMTP12BUnifiedPairProducesAcceptedDrafts() async throws {
        guard let loaded = try await loadUnifiedTargetAndDrafter() else {
            Issue.record(
                "required checkpoint not in HF cache (12B 4-bit target or 12B drafter); skipping"
            )
            return
        }

        let userInput = UserInput(chat: [
            .user("Why is the sky blue? Explain in one paragraph.")
        ])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)

        let stream = try generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: 64, temperature: 0),
            context: loaded.context,
            mtpDrafter: loaded.drafter,
            blockSize: 4
        )
        let run = await collect(stream)

        guard let info = run.info else {
            Issue.record("stream completed without emitting an .info event")
            return
        }

        let proposed = info.proposedDraftTokens ?? -1
        let accepted = info.acceptedDraftTokens ?? -1
        let rate = proposed > 0 ? "\(accepted)/\(proposed)" : "n/a (proposed=0)"
        print(
            "[Gemma4UnifiedMTP 12B] proposed=\(proposed), accepted=\(accepted), rate=\(rate), passthrough=\(info.passthroughReason ?? "nil"), generated=\(info.generationTokenCount) tokens in \(info.generateTime.formatted())s"
        )
        print("[Gemma4UnifiedMTP 12B] text: \(run.text)")

        #expect(
            info.proposedDraftTokens != nil, "proposedDraftTokens nil — stats plumbing broken")
        #expect(
            info.acceptedDraftTokens != nil, "acceptedDraftTokens nil — stats plumbing broken")
        #expect(
            info.passthroughReason == nil,
            "iterator engaged sticky-passthrough: \(info.passthroughReason ?? "")")
        #expect(proposed > 0, "speculation never ran (proposed=0)")
        #expect(accepted > 0, "no drafts accepted (accepted=0 of \(proposed))")
        #expect(!run.text.isEmpty, "MTP generated text is empty")
    }

    /// Throughput comparison on the same loaded pair: one baseline (no
    /// drafter) stream and one MTP stream over an identical prompt, tok/s
    /// printed for the writeup. No speedup floor asserted — wall-clock
    /// numbers vary across machines; the diagnostic value is in the logged
    /// ratio.
    ///
    /// High-entropy creative prompt — measured floor for acceptance on this
    /// pair (32%; 0.83–0.84x, a net slowdown — measured before the warmup
    /// run was added, with MTP first on a cold pair).
    @Test
    func testMTP12BUnifiedVsBaselineThroughput() async throws {
        try await runThroughputComparison(
            label: "creative",
            prompt: "Write a short story about a lighthouse keeper who discovers a map.")
    }

    /// Low-entropy counterpart: code generation is dominated by predictable
    /// syntax and boilerplate, the regime where the drafter's top-1 should
    /// match the target most often. Brackets the acceptance range together
    /// with the creative prompt above and the factual prompt in the
    /// accepted-drafts test (59.7%).
    @Test
    func testMTP12BUnifiedCodingPromptThroughput() async throws {
        try await runThroughputComparison(
            label: "coding",
            prompt: "Write a Swift function that parses a CSV line, handling quoted fields.")
    }

    /// Mid-entropy factual prompt — same prompt as the accepted-drafts test
    /// (59.7% acceptance there). Gives the factual class a within-run
    /// throughput number; the ~1.08x previously in RECON was estimated
    /// across separate runs.
    @Test
    func testMTP12BUnifiedFactualPromptThroughput() async throws {
        try await runThroughputComparison(
            label: "factual",
            prompt: "Why is the sky blue? Explain in one paragraph.")
    }

    private func runThroughputComparison(label: String, prompt: String) async throws {
        guard let loaded = try await loadUnifiedTargetAndDrafter() else {
            Issue.record(
                "required checkpoint not in HF cache (12B 4-bit target or 12B drafter); skipping"
            )
            return
        }

        let userInput = UserInput(chat: [.user(prompt)])
        let lmInput = try await loaded.context.processor.prepare(input: userInput)
        let parameters = GenerateParameters(maxTokens: 128, temperature: 0)

        // The first stream on a freshly loaded pair pays one-time costs
        // (Metal kernel compilation, memory-pool growth); discard a short
        // warmup so neither timed stream absorbs them.
        _ = await collect(
            try generate(
                input: lmInput,
                parameters: GenerateParameters(maxTokens: 16, temperature: 0),
                context: loaded.context
            ))

        let baselineRun = await collect(
            try generate(
                input: lmInput,
                parameters: parameters,
                context: loaded.context
            ))
        let mtpRun = await collect(
            try generate(
                input: lmInput,
                parameters: parameters,
                context: loaded.context,
                mtpDrafter: loaded.drafter,
                blockSize: 4
            ))

        guard let mtpInfo = mtpRun.info, let baselineInfo = baselineRun.info else {
            Issue.record(
                "missing .info event (mtp=\(mtpRun.info != nil), baseline=\(baselineRun.info != nil))"
            )
            return
        }

        let mtpTokPerSec =
            mtpInfo.generateTime > 0
            ? Double(mtpInfo.generationTokenCount) / mtpInfo.generateTime : 0
        let baselineTokPerSec =
            baselineInfo.generateTime > 0
            ? Double(baselineInfo.generationTokenCount) / baselineInfo.generateTime : 0
        let speedup = baselineTokPerSec > 0 ? mtpTokPerSec / baselineTokPerSec : 0
        let accepted = mtpInfo.acceptedDraftTokens ?? -1
        let proposed = mtpInfo.proposedDraftTokens ?? -1

        print(
            "[Gemma4UnifiedMTP 12B throughput \(label)] mtp=\(String(format: "%.2f", mtpTokPerSec)) tok/s, baseline=\(String(format: "%.2f", baselineTokPerSec)) tok/s, speedup=\(String(format: "%.2f", speedup))x, accepted=\(accepted)/\(proposed)"
        )
        print("[Gemma4UnifiedMTP 12B throughput \(label)] text: \(mtpRun.text)")

        #expect(!mtpRun.text.isEmpty, "MTP generated text is empty")
        #expect(!baselineRun.text.isEmpty, "baseline generated text is empty")
    }
}
