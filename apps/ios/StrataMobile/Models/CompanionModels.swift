import Foundation

struct CompanionStatus: Codable, Equatable {
    var macName: String
    var hostConnected: Bool
    var modelName: String?
    var engineState: String
    var tokensPerSecond: Double?
    var memoryUsedGB: Double?
    var memoryTotalGB: Double?
    var activeRequests: Int
}

struct CompanionModel: Codable, Identifiable, Equatable {
    var id: String
    var name: String
    var size: String?
    var state: String
    var detail: String?
    var selected: Bool? = nil
}

struct CompanionChat: Codable, Identifiable, Equatable {
    var id: String
    var title: String
}

struct ChatMessage: Codable, Identifiable, Equatable {
    var id: String
    var role: String
    var content: String
}

// Shared Mac conversations retain their identity, workspace, and permission mode.
struct MacConversationSummary: Decodable, Identifiable, Equatable {
    var id: String
    var title: String
    var updatedAt: String?
    var workspace: String?
    var modelName: String?
    var mode: String?
    var active: Bool
    var preview: String?

    enum CodingKeys: String, CodingKey { case id, title, updatedAt, workspace, modelName, mode, active, preview }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        title = try values.decode(String.self, forKey: .title)
        if let text = try? values.decode(String.self, forKey: .updatedAt) { updatedAt = text }
        else if let number = try? values.decode(Double.self, forKey: .updatedAt) { updatedAt = String(number) }
        else { updatedAt = nil }
        workspace = try values.decodeIfPresent(String.self, forKey: .workspace)
        modelName = try values.decodeIfPresent(String.self, forKey: .modelName)
        mode = try values.decodeIfPresent(String.self, forKey: .mode)
        active = try values.decodeIfPresent(Bool.self, forKey: .active) ?? false
        preview = try values.decodeIfPresent(String.self, forKey: .preview)
    }

    var workspaceName: String? { workspace.map { URL(fileURLWithPath: $0).lastPathComponent }.flatMap { $0.isEmpty ? nil : $0 } }
}

struct MacConversation: Decodable, Identifiable, Equatable {
    var id: String
    var title: String
    var workspace: String?
    var modelName: String?
    var mode: String?
    var active: Bool
    var status: String?
    var messages: [MacChatMessage]

    var permissionLabel: String {
        switch mode {
        case "plan": return "Plan mode"
        case "bypass": return "Full access on Mac"
        case "accept-edits": return "Accept edits"
        case "auto": return "Automatic permissions"
        default: return "Ask before actions"
        }
    }
}

struct MacChatMessage: Decodable, Identifiable, Equatable {
    var id: String
    var role: String
    var content: String
    var at: String?
    var streaming: Bool?
    var error: String?
    var tools: [MacToolActivity]?
    var reasoning: String?
    var reasoningMs: Double?
    var blocks: [MacMessageBlock]?

    var thinkingText: String? {
        guard let value = reasoning?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else { return nil }
        return value
    }
    var thinkingPreview: String? {
        thinkingText?.split(whereSeparator: \.isNewline).last.map { String($0).trimmingCharacters(in: .whitespaces) }
    }
    var isThinkingInProgress: Bool {
        streaming == true && reasoningMs == nil && content.isEmpty && thinkingText != nil
    }
    func thinkingTitle(isLive: Bool) -> String {
        guard !isLive, let duration = reasoningMs, duration.isFinite, duration >= 0 else { return "Thinking" }
        return "Thought for " + String(format: "%.0f", (duration / 1000).rounded()) + "s"
    }

    /// Ordered Mac text/activity blocks are authoritative. Tool membership is
    /// explicit; old groups without IDs keep their actions in the flat remainder.
    var timelineItems: [MacMessageTimelineItem] {
        let usable = (blocks ?? []).filter { block in
            switch block {
            case .text(let text): return !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            case .group: return true
            case .unsupported: return false
            }
        }
        var result: [MacMessageTimelineItem] = []
        var seenTools = Set<String>()
        var seenGroups = Set<String>()
        let actions = tools ?? []
        if usable.isEmpty {
            if !content.isEmpty { result.append(.text(id: "fallback-text", text: content)) }
        } else {
            var hasText = false
            for (index, block) in usable.enumerated() {
                switch block {
                case .text(let text):
                    hasText = true
                    result.append(.text(id: "text-\(index)", text: text))
                case .group(let group):
                    guard seenGroups.insert(group.id).inserted else { continue }
                    let ids = Set(group.toolIds ?? [])
                    let members = actions.filter { ids.contains($0.id) && seenTools.insert($0.id).inserted }
                    result.append(.group(group, tools: members))
                case .unsupported: break
                }
            }
            // Older snapshots can have group summaries without text blocks.
            if !hasText, !content.isEmpty { result.append(.text(id: "fallback-text", text: content)) }
        }
        for tool in actions where seenTools.insert(tool.id).inserted { result.append(.tool(tool)) }
        return result
    }
}

struct MacActivityGroup: Decodable, Identifiable, Equatable {
    var id: String
    var line: String
    var closed: Bool
    var added: Double?
    var removed: Double?
    var toolIds: [String]?

    enum CodingKeys: String, CodingKey { case id, line, closed, added, removed, toolIds }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(String.self, forKey: .id)
        line = try values.decodeIfPresent(String.self, forKey: .line) ?? ""
        closed = try values.decodeIfPresent(Bool.self, forKey: .closed) ?? false
        added = try values.decodeIfPresent(Double.self, forKey: .added)
        removed = try values.decodeIfPresent(Double.self, forKey: .removed)
        toolIds = try values.decodeIfPresent([String].self, forKey: .toolIds)
    }
    var title: String { line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Mac activity" : line }
    var displayTitle: String {
        guard added != nil, removed != nil else { return title }
        let sentence = title.replacingOccurrences(of: #"\s*\+\d+\s*[−-]\s*\d+\s*$"#, with: "", options: .regularExpression)
        return sentence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Mac activity" : sentence
    }
}

enum MacMessageBlock: Decodable, Equatable {
    case text(String)
    case group(MacActivityGroup)
    case unsupported

    private enum CodingKeys: String, CodingKey { case kind, text }
    init(from decoder: Decoder) throws {
        guard let values = try? decoder.container(keyedBy: CodingKeys.self),
              let kind = try? values.decode(String.self, forKey: .kind) else { self = .unsupported; return }
        switch kind {
        case "text":
            if let text = try? values.decode(String.self, forKey: .text) { self = .text(text) }
            else { self = .unsupported }
        case "group":
            if let group = try? MacActivityGroup(from: decoder), !group.id.isEmpty { self = .group(group) }
            else { self = .unsupported }
        default: self = .unsupported
        }
    }
}

enum MacMessageTimelineItem: Identifiable, Equatable {
    case text(id: String, text: String)
    case group(MacActivityGroup, tools: [MacToolActivity])
    case tool(MacToolActivity)

    var id: String {
        switch self {
        case .text(let id, _): return id
        case .group(let group, _): return "group-" + group.id
        case .tool(let tool): return "tool-" + tool.id
        }
    }
}

struct MacToolActivity: Decodable, Identifiable, Equatable {
    struct Ask: Decodable, Equatable { var why: String?; var detail: String? }
    var id: String
    var name: String
    var preview: String?
    var effect: String?
    var scope: String?
    var verb: String?
    var target: String?
    var ask: Ask?
    var state: String
    var result: String?
    var live: Bool?
    var approved: Bool?
    var durationMs: Double?

    var isBrowser: Bool { name.lowercased().contains("browser") }
    var isRunning: Bool { state == "running" || state == "approved" }
    var title: String {
        if let verb, !verb.isEmpty { return verb }
        if isBrowser { return "Browser activity" }
        if name.lowercased().contains("shell") || scope == "shell" { return "Run a command" }
        return name.replacingOccurrences(of: "_", with: " ")
    }
    var statusLabel: String {
        switch state {
        case "proposed": return "Needs approval"
        case "approved", "running": return "Working"
        case "done": return "Completed"
        case "denied": return "Denied"
        case "error": return "Failed"
        default: return state.capitalized
        }
    }
    var icon: String {
        if isBrowser { return "globe" }
        if scope == "shell" { return "terminal" }
        if scope == "write" { return "doc.badge.gearshape" }
        return "wrench.and.screwdriver"
    }
    var browserURL: URL? {
        guard isBrowser, let target, let url = URL(string: target), ["http", "https"].contains(url.scheme?.lowercased() ?? "") else { return nil }
        return url
    }
}

struct MacConversationList: Decodable { var conversations: [MacConversationSummary]; var activeChatId: String? }
struct MacConversationDetail: Decodable { var conversation: MacConversation; var pending: [MacToolActivity] }

struct PairedConnection: Codable, Equatable {
    var baseURL: URL
    var macName: String
    var fingerprint: String
    var token: String
}

enum CompanionError: LocalizedError {
    case invalidInvitation(String)
    case server(String)
    case unpaired
    case keychain(Int32)
    var errorDescription: String? {
        switch self {
        case .invalidInvitation(let message), .server(let message): return message
        case .unpaired: return "Pair with your Mac to continue."
        case .keychain: return "Your connection could not be saved securely. Please try pairing again."
        }
    }
}

struct PairingInvitation: Equatable {
    var url: String
    var code: String
    var fingerprint: String

    static func validatedURL(_ value: String) throws -> URL {
        guard let url = URL(string: value), let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              parts.scheme == "https", let host = parts.host, !host.isEmpty,
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
              (parts.path.isEmpty || parts.path == "/"), (parts.port ?? 443) > 0, (parts.port ?? 443) <= 65535 else {
            throw CompanionError.invalidInvitation("Enter the HTTPS address shown on your Mac, including its port.")
        }
        return url
    }

    static func normalizedFingerprint(_ value: String) throws -> String {
        let fingerprint = value.replacingOccurrences(of: ":", with: "").lowercased()
        guard fingerprint.count == 64, fingerprint.allSatisfy({ $0.isASCII && $0.isHexDigit }) else {
            throw CompanionError.invalidInvitation("The Mac certificate fingerprint must contain 64 hexadecimal characters. Scan the QR code to fill it in.")
        }
        return fingerprint
    }

    static func parse(_ value: String) throws -> PairingInvitation {
        guard let parts = URLComponents(string: value), parts.scheme == "strata", parts.host == "pair" else {
            throw CompanionError.invalidInvitation("This is not a Strata pairing code. Choose Pair iPhone in Strata on your Mac and scan its QR code.")
        }
        let items = parts.queryItems ?? []
        guard Set(items.map(\.name)).count == items.count else {
            throw CompanionError.invalidInvitation("The pairing code contains duplicate fields.")
        }
        func item(_ key: String) -> String? { items.first { $0.name == key }?.value }
        let address: String
        if let url = item("url") { address = url }
        else if let host = item("host"), let port = item("port"), let number = Int(port), (1...65535).contains(number) {
            var endpoint = URLComponents(); endpoint.scheme = "https"; endpoint.host = host; endpoint.port = number
            guard let url = endpoint.url else { throw CompanionError.invalidInvitation("The Mac address is invalid.") }
            address = url.absoluteString
        } else { throw CompanionError.invalidInvitation("The pairing code is missing the Mac address.") }
        _ = try validatedURL(address)
        guard let code = item("code"), code.count == 6, code.allSatisfy({ $0 >= "0" && $0 <= "9" }),
              let pin = item("fingerprint") else {
            throw CompanionError.invalidInvitation("The pairing code is incomplete. Generate a new code on your Mac.")
        }
        return PairingInvitation(url: address, code: code, fingerprint: try normalizedFingerprint(pin))
    }
}
