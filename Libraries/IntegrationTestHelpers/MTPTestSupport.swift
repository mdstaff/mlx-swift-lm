// Copyright © 2026 Apple Inc.

// Shared support for the MTP (multi-token-prediction) speculative-decoding
// integration tests. These helpers were copy-pasted across the MTP test
// files in `IntegrationTesting/IntegrationTestingTests`; hoisting them here
// keeps the locally-cached-checkpoint lookup, the no-op tokenizer loader, and
// the registry+factory drafter-load path in one place.

import Foundation
import MLXLMCommon
import MLXVLM

// MARK: - HF cache lookup

/// Resolve the local snapshot directory for a Hugging Face model id in the
/// shared hub cache (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>`),
/// returning `nil` when the checkpoint is not cached. MTP tests use this to
/// skip gracefully rather than trigger a multi-gigabyte download.
public func hfSnapshotDir(modelId: String) -> URL? {
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

// MARK: - No-op tokenizer loader

/// `MTPDrafterModelFactory` ignores the tokenizer loader (drafters borrow
/// their target's tokenizer), but the factory's protocol requires a
/// non-optional argument. This loader throws rather than traps so an
/// invariant break fails the calling test instead of killing the process.
public struct UnexpectedTokenizerLoad: Error {
    public init() {}
}

public final class NoOpTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from url: URL) async throws -> any MLXLMCommon.Tokenizer {
        throw UnexpectedTokenizerLoad()
    }
}

// MARK: - Target + drafter pair

/// A loaded target `ModelContext` paired with its MTP drafter, ready for the
/// `generate(..., mtpDrafter:)` overload.
public struct MTPLoadedPair {
    public let context: ModelContext
    public let drafter: any MTPDrafterModel

    public init(context: ModelContext, drafter: any MTPDrafterModel) {
        self.context = context
        self.drafter = drafter
    }
}

/// Load a target + MTP drafter pair from the local HF cache via the
/// registry + factory path — the production code path, so the drafter's
/// `model_type` creator key (`gemma4_assistant` / `gemma4_unified_assistant`)
/// is exercised rather than bypassed by direct construction.
///
/// Returns `nil` if either checkpoint is absent from the cache so callers can
/// skip. The target's tokenizer loader is injected (callers pass
/// `#huggingFaceTokenizerLoader()`); the drafter borrows the target's
/// tokenizer, so its loader is the throwing `NoOpTokenizerLoader`.
public func loadMTPPair(
    targetId: String,
    drafterId: String,
    targetTokenizerLoader: any TokenizerLoader
) async throws -> MTPLoadedPair? {
    guard let targetDir = hfSnapshotDir(modelId: targetId) else { return nil }
    guard let drafterDir = hfSnapshotDir(modelId: drafterId) else { return nil }

    let context = try await VLMModelFactory.shared.load(
        from: targetDir,
        using: targetTokenizerLoader
    )

    await Gemma4AssistantRegistration.register()
    // `load(from:using:)` returns the `MTPDrafterContext` directly (via
    // `sending`), keeping the model out of the container's `perform`, whose
    // `R: Sendable` constraint a non-Sendable `any MTPDrafterModel` can't
    // satisfy. Same registry path as `loadContainer` — `_load` resolves the
    // `gemma4_assistant` / `gemma4_unified_assistant` creator key either way.
    let drafterContext = try await MTPDrafterModelFactory.shared.load(
        from: drafterDir, using: NoOpTokenizerLoader()
    )

    return MTPLoadedPair(context: context, drafter: drafterContext.model)
}
