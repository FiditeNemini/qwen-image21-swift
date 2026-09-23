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

    /// The declared footprint is the MEASURED split (AB-R-0290), not the old estimate.
    func testManifestCarriesTheMeasuredSplit() {
        let fp = QwenImage21Package.manifest.requirements.footprints
        XCTAssertEqual(fp.count, 1)
        XCTAssertEqual(fp[0].residentBytes, 15_600_000_000)       // measured floor 15.58 GB
        XCTAssertEqual(fp[0].peakActivationBytes, 43_200_000_000)  // 10-ref edit 35.98 GB × 1.2
    }

    /// run() must refuse anything the footprint was not measured at, or the declaration lies.
    func testEnvelopeGuard() {
        typealias E = QwenImage21Envelope
        // T2I inside: 1024² and EVERY size on the model card (the cap is their max area, 2400×1792)
        XCTAssertEqual(E.maxTextToImagePixels, 2400 * 1792)
        XCTAssertNil(E.violation(targetWidth: 1024, targetHeight: 1024, referenceCount: 0, outputResolution: 1024))
        for (w, h) in E.documentedTextToImageSizes {
            XCTAssertNil(E.violation(targetWidth: w, targetHeight: h, referenceCount: 0, outputResolution: 1024), "\(w)x\(h)")
        }
        // edits inside: 1024² with up to 10 references
        XCTAssertNil(E.violation(targetWidth: 1024, targetHeight: 1024, referenceCount: 10, outputResolution: 1024))
        // outside: T2I beyond the largest documented size; edits above 1024²; >10 refs; edit resolution > 1024
        XCTAssertNotNil(E.violation(targetWidth: 2432, targetHeight: 1792, referenceCount: 0, outputResolution: 1024))
        XCTAssertNotNil(E.violation(targetWidth: 4096, targetHeight: 4096, referenceCount: 0, outputResolution: 1024))
        XCTAssertNotNil(E.violation(targetWidth: 2048, targetHeight: 2048, referenceCount: 1, outputResolution: 1024))
        XCTAssertNotNil(E.violation(targetWidth: 1024, targetHeight: 1024, referenceCount: 11, outputResolution: 1024))
        XCTAssertNotNil(E.violation(targetWidth: 768, targetHeight: 768, referenceCount: 1, outputResolution: 1536))
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
        XCTAssertEqual(back.defaultEditOutputResolution, 1024)  // 768 cap lifted once noise replay was the cause
        XCTAssertTrue(back.useKVCache)
        XCTAssertEqual(QwenImage21Configuration().quant, .bf16)
    }
}
