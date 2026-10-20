import XCTest
import Combine
@testable import StrataMobile

@MainActor
final class StoreRaceTests: XCTestCase {
    func testPendingHistoryCannotReplaceNewConversation() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        let store = fixture.makeStore()
        let history = fixture.hold("/v1/chats/old")
        let selection = Task { await store.selectChat(CompanionChat(id: "old", title: "Old conversation")) }
        await fulfillment(of: [history.requested], timeout: 2)

        store.newChat()
        history.finish(["messages": [["id": "old-message", "role": "assistant", "content": "Old content"]]])
        await selection.value

        XCTAssertNil(store.currentChatId)
        XCTAssertTrue(store.messages.isEmpty)
        XCTAssertNil(store.error)
    }

    func testOlderHistoryCannotReplaceMoreRecentSelection() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        let store = fixture.makeStore()
        let old = fixture.hold("/v1/chats/old")
        let newer = fixture.hold("/v1/chats/new")
        let first = Task { await store.selectChat(CompanionChat(id: "old", title: "Old")) }
        await fulfillment(of: [old.requested], timeout: 2)
        let second = Task { await store.selectChat(CompanionChat(id: "new", title: "New")) }
        await fulfillment(of: [newer.requested], timeout: 2)

        newer.finish(["messages": [["id": "new-message", "role": "assistant", "content": "New content"]]])
        await second.value
        old.finish(["messages": [["id": "old-message", "role": "assistant", "content": "Old content"]]])
        await first.value

        XCTAssertEqual(store.currentChatId, "new")
        XCTAssertEqual(store.messages.map(\.content), ["New content"])
    }

    func testRefreshCompletingAfterForgetDoesNotRestoreOldMacState() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        let store = fixture.makeStore()
        let status = fixture.hold("/v1/status")
        let models = fixture.hold("/v1/models")
        let chats = fixture.hold("/v1/chats")
        let refreshing = Task { await store.refresh() }
        await fulfillment(of: [status.requested, models.requested, chats.requested], timeout: 2)
        XCTAssertTrue(store.isLoading)

        store.forgetMac()
        XCTAssertEqual(fixture.deletedConnections, 1)
        XCTAssertFalse(store.isLoading)
        status.finish(StoreTransport.status)
        models.finish(["models": [["id": "old-model", "name": "Old Mac model", "state": "loaded"]]])
        chats.finish(["chats": [["id": "old-chat", "title": "Old Mac chat"]]])
        await refreshing.value

        XCTAssertFalse(store.isPaired)
        XCTAssertNil(store.status)
        XCTAssertTrue(store.models.isEmpty)
        XCTAssertTrue(store.chats.isEmpty)
        XCTAssertFalse(store.isLoading)
        XCTAssertNil(store.error)
    }

    func testPairingCannotReplaceConnectionWhileReplyIsSending() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        let store = fixture.makeStore()
        let original = store.connection
        let stream = fixture.hold("/v1/chat")
        let sending = Task { await store.sendLegacyMessage("Synthetic fixture") }
        await fulfillment(of: [stream.requested], timeout: 2)
        XCTAssertTrue(store.isSending)

        await store.pair(url: "https://unexpected.invalid:8787", code: "123456", fingerprint: String(repeating: "a", count: 64))
        XCTAssertEqual(store.connection, original)
        XCTAssertEqual(fixture.savedConnections, 0)
        stream.finishSSE("event: frame\ndata: {\"type\":\"session\",\"chatId\":\"fixture-chat\"}\n\nevent: frame\ndata: {\"type\":\"delta\",\"text\":\"Fixture reply\"}\n\nevent: frame\ndata: {\"type\":\"done\"}\n\n")
        await sending.value

        XCTAssertFalse(store.isSending)
        XCTAssertEqual(store.connection, original)
        XCTAssertEqual(store.messages.last?.content, "Fixture reply")
        XCTAssertNil(store.error)
    }

    func testSharedHistoryContinuesTheSameMacConversation() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in
            switch request.url!.path {
            case "/v1/conversations/mac-existing": return StoreTransport.detail(active: false)
            case "/v1/conversations/mac-existing/messages": return StoreTransport.detail(active: true, content: "Continuing the existing work")
            default: return nil
            }
        }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        XCTAssertEqual(store.currentConversation?.messages.first?.content, "Earlier Mac history")
        XCTAssertEqual(store.currentConversation?.permissionLabel, "Ask before actions")
        await store.sendMessage("Continue that work")
        XCTAssertEqual(store.currentConversation?.id, "mac-existing")
        XCTAssertTrue(store.isWorking)
        let posts = fixture.requests.filter { $0.method == "POST" }
        XCTAssertEqual(posts.map(\.path), ["/v1/conversations/mac-existing/messages"])
        XCTAssertEqual(posts.first?.body?["message"], "Continue that work")
        XCTAssertNil(posts.first?.body?["mode"], "The phone must preserve the Mac conversation's permissions.")
        XCTAssertNil(store.error)
    }

    func testMacStartedBrowserActivityUpdatesWithoutPhoneSending() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? StoreTransport.detail(active: true, toolState: "running") : nil }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        let initialReads = fixture.requests.count
        store.setChatVisible(false)
        await store.refreshLiveState(refreshOverview: false)
        XCTAssertEqual(fixture.requests.count, initialReads, "An offscreen Chat view must stop its fast polling.")
        store.setChatVisible(true)
        await store.refreshLiveState(refreshOverview: false)
        XCTAssertEqual(fixture.requests.count, initialReads + 1)
        let browser = try XCTUnwrap(store.currentConversation?.messages.first?.tools?.first)
        XCTAssertTrue(browser.isBrowser)
        XCTAssertTrue(browser.isRunning)
        XCTAssertEqual(browser.browserURL?.absoluteString, "https://example.com/research")
        XCTAssertEqual(browser.title, "Opened a page")
        XCTAssertTrue(fixture.requests.allSatisfy { $0.method == "GET" })
    }

    func testNewPhoneWorkCreatesASharedMacConversation() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in
            if request.url!.path == "/v1/conversations", request.httpMethod == "POST" { return StoreTransport.detail(active: false, content: "") }
            if request.url!.path == "/v1/conversations/mac-existing/messages" { return StoreTransport.detail(active: true, content: "Started on Mac") }
            return nil
        }
        let store = fixture.makeStore()
        store.newChat()
        await store.sendMessage("New shared work")
        XCTAssertEqual(fixture.requests.filter { $0.method == "POST" }.map(\.path), ["/v1/conversations", "/v1/conversations/mac-existing/messages"])
        XCTAssertNil(store.currentChatId)
        XCTAssertEqual(store.currentConversation?.id, "mac-existing")
        XCTAssertEqual(store.currentConversation?.mode, "manual")
        XCTAssertNil(store.error)
    }

    func testDelayedLiveSnapshotCannotReopenConversationAfterNewChat() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? StoreTransport.detail(active: true) : nil }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        let snapshot = fixture.hold("/v1/conversations/mac-existing")
        let polling = Task { await store.refreshCurrentConversation() }
        await fulfillment(of: [snapshot.requested], timeout: 2)
        store.newChat()
        snapshot.finish(StoreTransport.detail(active: true, content: "A late Mac update"))
        await polling.value
        XCTAssertNil(store.currentConversation)
        XCTAssertTrue(store.pendingApprovals.isEmpty)
        XCTAssertFalse(fixture.requests.contains { $0.path.hasSuffix("/stop") }, "Changing views must leave Mac work running.")
    }

    func testApprovalUsesOpaqueToolIdAndStopTargetsSameConversation() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in
            switch request.url!.path {
            case "/v1/conversations/mac-existing": return StoreTransport.detail(active: true, toolState: "proposed")
            case "/v1/conversations/mac-existing/approval": return StoreTransport.detail(active: true, toolState: "running")
            case "/v1/conversations/mac-existing/stop": return StoreTransport.detail(active: false)
            default: return nil
            }
        }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        let tool = try XCTUnwrap(store.pendingApprovals.first)
        await store.decideApproval(tool, approve: true)
        XCTAssertTrue(store.pendingApprovals.isEmpty)
        await store.stopConversation()
        XCTAssertFalse(store.isWorking)
        let posts = fixture.requests.filter { $0.method == "POST" }
        XCTAssertEqual(posts.map(\.path), ["/v1/conversations/mac-existing/approval", "/v1/conversations/mac-existing/stop"])
        XCTAssertEqual(posts.first?.body?["toolId"], "helper-session:browser-call")
        XCTAssertEqual(posts.first?.body?["decision"], "approve")
        XCTAssertNil(store.error)
    }

    func testUnchangedLiveSnapshotDoesNotPublishNewViewState() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? StoreTransport.detail(active: true, toolState: "running") : nil }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        var publications = 0
        let observation = store.objectWillChange.sink { publications += 1 }
        defer { observation.cancel() }

        await store.refreshCurrentConversation()
        await store.refreshCurrentConversation()
        XCTAssertEqual(publications, 0, "Unchanged polling must not invalidate the conversation's scroll controls.")

        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? StoreTransport.detail(active: true, toolState: "done") : nil }
        await store.refreshCurrentConversation()
        XCTAssertEqual(publications, 1, "A changed action must still update the visible conversation.")
        XCTAssertEqual(store.currentConversation?.messages.first?.tools?.first?.state, "done")
    }

    func testReasoningAndProgressUpdatesArriveWhenAnswerTextIsUnchanged() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        func snapshot(reasoning: String, line: String) -> [String: Any] {
            var detail = StoreTransport.detail(active: true)
            var conversation = detail["conversation"] as! [String: Any]
            var messages = conversation["messages"] as! [[String: Any]]
            messages[0]["reasoning"] = reasoning
            messages[0]["blocks"] = [["kind": "text", "text": "Earlier Mac history"], ["kind": "group", "id": "progress", "line": line, "closed": false, "toolIds": []]]
            conversation["messages"] = messages; detail["conversation"] = conversation
            return detail
        }
        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? snapshot(reasoning: "First step", line: "Ran 3 commands") : nil }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? snapshot(reasoning: "First step\nNext step", line: "Ran 3 commands, edited a file") : nil }
        await store.refreshCurrentConversation()
        let reply = try XCTUnwrap(store.currentConversation?.messages.first)
        XCTAssertEqual(reply.content, "Earlier Mac history")
        XCTAssertEqual(reply.thinkingPreview, "Next step")
        guard case .group(let progress, _) = reply.timelineItems.last else { return XCTFail("The updated progress must follow the same answer text.") }
        XCTAssertEqual(progress.line, "Ran 3 commands, edited a file")
    }

    func testSharedChatRejectsSecondMessageWhileMacIsWorking() async throws {
        let fixture = StoreTransport()
        defer { fixture.close() }
        fixture.respond { request in request.url!.path == "/v1/conversations/mac-existing" ? StoreTransport.detail(active: true) : nil }
        let store = fixture.makeStore()
        await store.selectConversation(try StoreTransport.summary())
        await store.sendMessage("This must not interrupt the Mac")
        XCTAssertTrue(fixture.requests.allSatisfy { $0.method == "GET" })
    }
}

/// Delayed responses exercise the real HTTP decoding path without a live bridge,
/// simulator changes, or access to the app's saved Keychain entry.
private final class StoreTransport {
    static let status: [String: Any] = ["macName": "Fixture Mac", "hostConnected": true, "engineState": "idle", "activeRequests": 0]
    let host = "fixture-\(UUID().uuidString.lowercased()).invalid"
    private let lock = NSLock()
    private var held: [String: HeldStoreResponse] = [:]
    struct RecordedRequest { var path: String; var method: String; var body: [String: String]? }
    private var recordedRequests: [RecordedRequest] = []
    var requests: [RecordedRequest] { lock.lock(); defer { lock.unlock() }; return recordedRequests }
    private var responder: ((URLRequest) -> [String: Any]?)?
    func respond(_ value: @escaping (URLRequest) -> [String: Any]?) { lock.lock(); responder = value; lock.unlock() }
    private var session: URLSession?
    private(set) var deletedConnections = 0
    private(set) var savedConnections = 0

    init() { StoreURLProtocol.register(self, host: host) }

    @MainActor func makeStore() -> CompanionStore {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StoreURLProtocol.self]
        let session = URLSession(configuration: config)
        self.session = session
        let connection = PairedConnection(baseURL: URL(string: "https://\(host):8787")!, macName: "Fixture Mac", fingerprint: String(repeating: "a", count: 64), token: "fixture-token")
        let client = CompanionClient(baseURL: connection.baseURL, fingerprint: connection.fingerprint, token: connection.token, session: session)
        return CompanionStore(connection: connection, client: client,
                              saveConnection: { [weak self] _ in self?.savedConnections += 1 },
                              deleteConnection: { [weak self] in self?.deletedConnections += 1 })
    }

    func hold(_ path: String) -> HeldStoreResponse {
        let response = HeldStoreResponse()
        lock.lock(); held[path] = response; lock.unlock()
        return response
    }

    func receive(_ request: StoreURLProtocol) {
        let path = request.request.url!.path
        let payload: Data?
        if let data = request.request.httpBody { payload = data }
        else if let stream = request.request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                data.append(buffer, count: count)
            }
            payload = data
        } else { payload = nil }
        let body = payload.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: String] }
        lock.lock()
        recordedRequests.append(RecordedRequest(path: path, method: request.request.httpMethod ?? "GET", body: body))
        let response = held[path]
        let responder = responder
        lock.unlock()
        if let response { response.receive(request); return }
        if let json = responder?(request.request) { request.finishJSON(json); return }
        switch path {
        case "/v1/status": request.finishJSON(Self.status)
        case "/v1/models": request.finishJSON(["models": []])
        case "/v1/chats": request.finishJSON(["chats": []])
        case "/v1/conversations": request.finishJSON(["conversations": []])
        default: request.finishJSON(["error": "Unexpected fixture request"], status: 404)
        }
    }

    static func summary() throws -> MacConversationSummary {
        try JSONDecoder().decode(MacConversationSummary.self, from: JSONSerialization.data(withJSONObject: ["id": "mac-existing", "title": "Mac research", "active": true, "updatedAt": 12345]))
    }

    static func detail(active: Bool, content: String = "Earlier Mac history", toolState: String? = nil) -> [String: Any] {
        var message: [String: Any] = ["id": "mac-message", "role": "assistant", "content": content]
        var pending: [[String: Any]] = []
        if let toolState {
            let tool: [String: Any] = ["id": "helper-session:browser-call", "name": "mcp__browser__browser_navigate", "state": toolState, "verb": "Opened a page", "target": "https://example.com/research", "scope": "connected", "result": "Page title: Example", "live": true]
            message["tools"] = [tool]
            if toolState == "proposed" { pending = [tool] }
        }
        return ["conversation": ["id": "mac-existing", "title": "Mac research", "active": active, "mode": "manual", "workspace": "/tmp/fixture-project", "messages": [message]], "pending": pending]
    }

    func close() {
        StoreURLProtocol.unregister(host: host)
        session?.invalidateAndCancel()
    }
}

private final class HeldStoreResponse {
    let requested = XCTestExpectation(description: "Delayed HTTP request started")
    private let lock = NSLock()
    private var request: StoreURLProtocol?
    func receive(_ request: StoreURLProtocol) {
        lock.lock(); self.request = request; lock.unlock()
        requested.fulfill()
    }
    func finish(_ json: [String: Any]) {
        lock.lock(); let request = self.request; self.request = nil; lock.unlock()
        request?.finishJSON(json)
    }
    func finishSSE(_ value: String) {
        lock.lock(); let request = self.request; self.request = nil; lock.unlock()
        request?.finish(Data(value.utf8), contentType: "text/event-stream")
    }
}

private final class StoreURLProtocol: URLProtocol {
    private static let registryLock = NSLock()
    private static var registry: [String: StoreTransport] = [:]
    static func register(_ transport: StoreTransport, host: String) {
        registryLock.lock(); registry[host] = transport; registryLock.unlock()
    }
    static func unregister(host: String) {
        registryLock.lock(); registry.removeValue(forKey: host); registryLock.unlock()
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.registryLock.lock(); let transport = Self.registry[request.url!.host!]; Self.registryLock.unlock()
        transport?.receive(self)
    }
    override func stopLoading() { }
    func finishJSON(_ json: [String: Any], status: Int = 200) {
        finish(try! JSONSerialization.data(withJSONObject: json), contentType: "application/json", status: status)
    }
    func finish(_ data: Data, contentType: String, status: Int = 200) {
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: ["content-type": contentType])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
}
