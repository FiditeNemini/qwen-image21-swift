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

    func testCalculateDimensions() {
        let (w, h) = QwenImage21Latents.calculateDimensions(targetArea: 1024 * 1024, ratio: 1)
        XCTAssertEqual(w, 1024); XCTAssertEqual(h, 1024)
        let (w2, h2) = QwenImage21Latents.calculateDimensions(targetArea: 320 * 320, ratio: 0.75)
        XCTAssertEqual(w2, 288); XCTAssertEqual(h2, 384)
    }
}
