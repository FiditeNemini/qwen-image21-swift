import XCTest
@testable import QwenImage21

final class SchedulerTests: XCTestCase {
    func testShiftAndTerminal() {
        // mu at 4096 target tokens with the 2.1 config (256/8192, 0.5/0.9)
        let mu = QwenImage21Scheduler.calculateShift(imageSeqLen: 4096)
        XCTAssertEqual(mu, 0.5 + (0.9 - 0.5) / Float(8192 - 256) * Float(4096 - 256), accuracy: 1e-6)
        let s = QwenImage21Scheduler.sigmas(steps: 4, mu: mu)
        XCTAssertEqual(s.count, 5)
        XCTAssertEqual(s[0], 1, accuracy: 1e-7)
        XCTAssertEqual(s[3], 0.02, accuracy: 1e-6)  // shift_terminal
        XCTAssertEqual(s[4], 0)
        XCTAssertTrue(zip(s, s.dropFirst()).allSatisfy { $0 > $1 })
    }

    /// Noise replay (diffusers#14824): an edit must never draw the text-to-image noise for the
    /// same seed, or it re-runs the generation instead of editing.
    func testEditNoiseIsDomainSeparatedFromTextToImage() {
        for seed: UInt64 in [0, 1, 42, 4242, UInt64.max] {
            XCTAssertEqual(QwenImage21Latents.noiseSeed(seed, isEdit: false), seed)
            XCTAssertNotEqual(QwenImage21Latents.noiseSeed(seed, isEdit: true), seed)
            // deterministic: same inputs, same draw
            XCTAssertEqual(QwenImage21Latents.noiseSeed(seed, isEdit: true),
                           QwenImage21Latents.noiseSeed(seed, isEdit: true))
        }
        // and no edit seed collides with the T2I seed of another ordinary seed nearby
        let edits = Set((0..<64).map { QwenImage21Latents.noiseSeed(UInt64($0), isEdit: true) })
        XCTAssertTrue(edits.isDisjoint(with: Set((0..<64).map { UInt64($0) })))
    }

    func testCalculateDimensions() {
        let (w, h) = QwenImage21Latents.calculateDimensions(targetArea: 1024 * 1024, ratio: 1)
        XCTAssertEqual(w, 1024); XCTAssertEqual(h, 1024)
        let (w2, h2) = QwenImage21Latents.calculateDimensions(targetArea: 320 * 320, ratio: 0.75)
        XCTAssertEqual(w2, 288); XCTAssertEqual(h2, 384)
    }
}
