// Offline manifest / configuration tests for the Qwen-Image-2.1 engine package (no weights).
import Foundation
import MLXToolKit
import XCTest

@testable import MLXQwenImage21

final class QwenImage21PackageTests: XCTestCase {
    func testManifestDeclaresTheResearchLicenceOutsideTheAllowlist() {
        let m = QwenImage21Package.manifest
        XCTAssertEqual(m.license.weightLicense, .qwenResearch)
        XCTAssertEqual(m.license.weightLicense.identifier, "LicenseRef-Qwen-Research")
        // The whole point of the package-local declaration: `.permissiveOnly` must NOT admit it.
        XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(.qwenResearch))
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.contractVersion, ContractVersion.current)
    }

    func testTwoSurfacesOneModel() {
        let m = QwenImage21Package.manifest
        XCTAssertEqual(Set(m.surfaces.map(\.capability)), [.textToImage, .imageEdit])
        XCTAssertEqual(Set(m.surfaces.map(\.name)), ["qwen-image-2.1"])
        XCTAssertEqual(m.requirements.footprints.map(\.quant), [.bf16])
        XCTAssertTrue(m.requirements.footprints.allSatisfy { $0.residentBytes > 0 && $0.peakActivationBytes > 0 })
    }

    func testConfigurationRoundTripsAndDefaults() throws {
        let cfg = QwenImage21Configuration(snapshotPath: "/tmp/qi21", textEncoderPath: "/tmp/qwen3vl", defaultSteps: 20)
        let data = try JSONEncoder().encode(cfg)
        let back = try JSONDecoder().decode(QwenImage21Configuration.self, from: data)
        XCTAssertEqual(back.snapshotPath, "/tmp/qi21")
        XCTAssertEqual(back.textEncoderPath, "/tmp/qwen3vl")
        XCTAssertEqual(back.defaultSteps, 20)
        XCTAssertEqual(back.defaultTrueCFGScale, 1.0)
        XCTAssertEqual(back.defaultOutputResolution, 1024)
        XCTAssertEqual(back.defaultEditOutputResolution, 768)  // reference degrades at 1024² edits (AB-R-0261)
        XCTAssertTrue(back.useKVCache)
        XCTAssertEqual(QwenImage21Configuration().quant, .bf16)
    }
}
