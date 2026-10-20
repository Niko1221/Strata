import XCTest
@testable import StrataMobile

final class PairingTests: XCTestCase {
    private let pin = String(repeating: "a1", count: 32)

    func testQRInvitationPreservesExactCertificateAndEndpoint() throws {
        let invitation = try PairingInvitation.parse("strata://pair?url=https%3A%2F%2F192.168.1.4%3A8787&code=012345&fingerprint=\(pin)")
        XCTAssertEqual(invitation.url, "https://192.168.1.4:8787")
        XCTAssertEqual(invitation.code, "012345")
        XCTAssertEqual(invitation.fingerprint, pin)
    }

    func testLegacyHostPortInvitationAndIPv6() throws {
        let invitation = try PairingInvitation.parse("strata://pair?host=mac.local&port=8787&code=123456&fingerprint=\(pin)")
        XCTAssertEqual(try PairingInvitation.validatedURL(invitation.url).host, "mac.local")
        XCTAssertNoThrow(try PairingInvitation.validatedURL("https://[::1]:8787"))
    }

    func testRejectsDowngradeCredentialEmbeddingAndAmbiguousQR() {
        for address in ["http://192.168.1.4:8787", "https://user:secret@mac.local", "https://mac.local/path", "https://mac.local?redirect=evil", "https://mac.local/#fragment", "https://mac.local:99999"] {
            XCTAssertThrowsError(try PairingInvitation.validatedURL(address), address)
        }
        XCTAssertThrowsError(try PairingInvitation.parse("strata://pair?url=https%3A%2F%2Fmac.local&url=https%3A%2F%2Fevil.local&code=123456&fingerprint=\(pin)"))
        XCTAssertThrowsError(try PairingInvitation.parse("other://pair?host=mac.local&port=8787&code=123456&fingerprint=\(pin)"))
        XCTAssertThrowsError(try PairingInvitation.parse("strata://pair?host=mac.local&port=8787&code=１２３４５６&fingerprint=\(pin)"))
        XCTAssertThrowsError(try PairingInvitation.normalizedFingerprint(String(repeating: "g", count: 64)))
    }

    func testCertificatePinUsesSHA256OfExactDERBytes() {
        XCTAssertEqual(PinnedSessionDelegate.digest(Data("abc".utf8)), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        XCTAssertNotEqual(PinnedSessionDelegate.digest(Data("abc".utf8)), PinnedSessionDelegate.digest(Data("abcd".utf8)))
    }

    func testBearerIsOnlyAddedToBoundCompanionEndpoint() throws {
        let client = CompanionClient(baseURL: URL(string: "https://mac.local:8787")!, fingerprint: pin, token: "test-token")
        let request = try client.request("/v1/status")
        XCTAssertEqual(request.url?.absoluteString, "https://mac.local:8787/v1/status")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-token")
        XCTAssertThrowsError(try client.request("https://other.local/v1/status"))
        XCTAssertThrowsError(try client.request("/v1/../api/secrets"))
        XCTAssertThrowsError(try client.request("/v1/status?redirect=other"))
    }

    func testLiveTLSRefusesAnUnexpectedCertificate() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let value = environment["STRATA_TEST_INVITATION"] ?? environment["TEST_RUNNER_STRATA_TEST_INVITATION"] else {
            throw XCTSkip("Requires a running Mac companion. No mock TLS acceptance claim.")
        }
        let invitation = try PairingInvitation.parse(value)
        let address = try PairingInvitation.validatedURL(invitation.url)
        let correct = CompanionClient(baseURL: address, fingerprint: invitation.fingerprint)
        do {
            try await correct.discard("/v1/status", method: "GET")
            XCTFail("Unpaired status should be refused")
        } catch let error as CompanionError {
            guard case .server(let message) = error else { XCTFail("Unexpected response: \(error)"); return }
            XCTAssertTrue(message.contains("Pair")) // Reached HTTP only after the correct pin was accepted.
        }
        let incorrect = CompanionClient(baseURL: address, fingerprint: String(repeating: "0", count: 64))
        do {
            try await incorrect.discard("/v1/status", method: "GET")
            XCTFail("A different leaf certificate must never be trusted")
        } catch {
            XCTAssertTrue(error is URLError, "The wrong pin must fail before any HTTP response")
        }
    }
}
