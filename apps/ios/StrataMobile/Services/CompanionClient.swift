import Foundation
import Security
import CryptoKit

final class PinnedSessionDelegate: NSObject, URLSessionDelegate, URLSessionTaskDelegate {
    let fingerprint: String
    init(fingerprint: String) { self.fingerprint = fingerprint }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let trust = challenge.protectionSpace.serverTrust,
              let certificates = SecTrustCopyCertificateChain(trust) as? [SecCertificate],
              let certificate = certificates.first,
              Self.digest(SecCertificateCopyData(certificate) as Data) == fingerprint else {
            completionHandler(.cancelAuthenticationChallenge, nil); return
        }
        // The exact certificate was supplied out of band by the Mac's QR code.
        // Only this leaf is trusted; arbitrary self-signed certificates are refused.
        completionHandler(.useCredential, URLCredential(trust: trust))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        // A redirect must never move a pairing code or bearer token to another server.
        completionHandler(nil)
    }
}

final class CompanionClient {
    let baseURL: URL
    let token: String?
    private let session: URLSession

    init(baseURL: URL, fingerprint: String, token: String? = nil, session: URLSession? = nil) {
        self.baseURL = baseURL; self.token = token
        if let session { self.session = session; return }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 20
        config.timeoutIntervalForResource = 600
        config.httpCookieStorage = nil
        config.urlCache = nil
        self.session = URLSession(configuration: config, delegate: PinnedSessionDelegate(fingerprint: fingerprint), delegateQueue: nil)
    }

    deinit { session.invalidateAndCancel() }

    func request(_ path: String, method: String = "GET", body: [String: String]? = nil) throws -> URLRequest {
        guard path.hasPrefix("/v1/"), !path.contains(".."), !path.contains("?") else {
            throw CompanionError.server("Invalid companion request.")
        }
        var req = URLRequest(url: baseURL.appendingPathComponent(String(path.dropFirst())))
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let body { req.httpBody = try JSONEncoder().encode(body) }
        return req
    }

    func perform<T: Decodable>(_ path: String, method: String = "GET", body: [String: String]? = nil) async throws -> T {
        let (data, response) = try await session.data(for: request(path, method: method, body: body))
        try Self.check(response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    func discard(_ path: String, method: String = "POST", body: [String: String]? = nil) async throws {
        let (data, response) = try await session.data(for: request(path, method: method, body: body))
        try Self.check(response, data: data)
    }

    func stream(message: String, chatId: String?, onFrame: @escaping @MainActor (ChatFrame) -> Void) async throws {
        var body = ["message": message]
        if let chatId { body["chatId"] = chatId }
        var streamRequest = try request("/v1/chat", method: "POST", body: body)
        streamRequest.timeoutInterval = 600 // Model prefill can exceed the dashboard's short timeout.
        let (bytes, response) = try await session.bytes(for: streamRequest)
        try Self.check(response)
        var completed = false
        for try await line in bytes.lines {
            try Task.checkCancellation()
            let raw: String
            if line.hasPrefix("data:") { raw = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces) }
            else if line.hasPrefix("{") { raw = line } // NDJSON is also understood.
            else { continue }
            guard raw != "[DONE]", let data = raw.data(using: .utf8) else { continue }
            let frame = try JSONDecoder().decode(ChatFrame.self, from: data)
            await onFrame(frame)
            if frame.type == "error" { throw CompanionError.server(frame.message ?? "The Mac could not finish this reply.") }
            if frame.type == "done" { completed = true }
        }
        if !completed { throw CompanionError.server("The connection ended before the reply finished. Check your Mac before trying again.") }
    }

    private static func check(_ response: URLResponse, data: Data = Data()) throws {
        guard let http = response as? HTTPURLResponse else { throw CompanionError.server("The Mac returned an unreadable response.") }
        guard (200..<300).contains(http.statusCode) else {
            let error = try? JSONDecoder().decode(ServerError.self, from: data)
            let message = error?.message ?? error?.error
            throw CompanionError.server(message ?? (http.statusCode == 401 ? "This pairing has expired or was removed on the Mac. Pair again." : "The Mac returned an error (\(http.statusCode))."))
        }
    }
}

struct ChatFrame: Decodable {
    var type: String
    var text: String?
    var message: String?
    var chatId: String?
}
private struct ServerError: Decodable { var error: String?; var message: String? }

enum ConnectionKeychain {
    private static let service = "com.strata.companion.connection"
    private static let account = "paired-mac"
    private static var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service, kSecAttrAccount as String: account]
    }
    static func load() -> PairedConnection? {
        var read = query; read[kSecReturnData as String] = true; read[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        guard SecItemCopyMatching(read as CFDictionary, &result) == errSecSuccess, let data = result as? Data else { return nil }
        return try? JSONDecoder().decode(PairedConnection.self, from: data)
    }
    static func save(_ connection: PairedConnection) throws {
        let data = try JSONEncoder().encode(connection)
        let values: [String: Any] = [kSecValueData as String: data,
                                    kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]
        let status = SecItemUpdate(query as CFDictionary, values as CFDictionary)
        if status == errSecItemNotFound {
            var insert = query; values.forEach { insert[$0.key] = $0.value }
            let result = SecItemAdd(insert as CFDictionary, nil)
            guard result == errSecSuccess else { throw CompanionError.keychain(result) }
        } else if status != errSecSuccess { throw CompanionError.keychain(status) }
    }
    static func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw CompanionError.keychain(status) }
    }
}
