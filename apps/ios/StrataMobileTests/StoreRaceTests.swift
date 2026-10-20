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
@MainActor
final class ModelManagementTests: XCTestCase {
    func testDownloadPlanReportsStorageVisionAndExpiresInMilliseconds() throws {
        let now = Date(timeIntervalSince1970: 1000)
        var json = StoreTransport.downloadPlan(expiresAt: 1_001_000)
        let plan = try decode(ModelDownloadPlan.self, json)
        XCTAssertTrue(plan.canStart(at: now))
        XCTAssertFalse(plan.canStart(at: Date(timeIntervalSince1970: 1001)))
        XCTAssertEqual(plan.downloadGb, 12)
        XCTAssertEqual(plan.diskFreeGb, 100)
        XCTAssertEqual(plan.diskNeededGb, 14)
        XCTAssertTrue(plan.readPictures)
        XCTAssertFalse(plan.warnings.isEmpty)
        json["diskShort"] = "Not enough free storage"
        XCTAssertFalse(try decode(ModelDownloadPlan.self, json).canStart(at: now))
        XCTAssertEqual(ModelDownloadFormat.gigabytes(nil), "Not reported")
    }

    func testDownloadProgressClampsValuesAndMissingOwnershipCannotCancel() throws {
        var job = try decode(ModelDownloadJob.self, StoreTransport.downloadJob(state: "queued", canCancel: nil))
        XCTAssertTrue(job.isRunning)
        XCTAssertNil(job.canCancel)
        job.percent = 120; XCTAssertEqual(job.progress, 1)
        job.percent = -10; XCTAssertEqual(job.progress, 0)
        job.percent = .nan; XCTAssertNil(job.progress)
        job.state = "cancelled"; XCTAssertTrue(job.canRetry)
        XCTAssertFalse(job.isRunning)
        job.modelId = nil; XCTAssertFalse(job.canRetry)
        XCTAssertNotNil(job.counterLabel)
        job.doneGb = .infinity; XCTAssertNil(job.counterLabel)
    }

    func testOldMacDownloadCapabilityFallbackPreservesTheLibrary() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respond { request in
            request.url!.path == "/v1/models" ? ["models": [["id": "api-fixture", "name": "API model", "state": "available"]]] : nil
        }
        let store = fixture.makeStore()
        await store.refresh()
        let unsupported = fixture.hold("/v1/model-management/targets")
        let refreshing = Task { await store.refreshModelManagement() }
        await fulfillment(of: [unsupported.requested], timeout: 2)
        unsupported.finish(["error": "Unknown endpoint"], status: 404)
        await refreshing.value
        XCTAssertEqual(store.models.map(\.id), ["api-fixture"])
        XCTAssertTrue(store.modelTargets.isEmpty)
        XCTAssertTrue(store.modelManagementMessage?.contains("Update Strata") == true)
        XCTAssertNil(store.error)
    }

    func testReviewCannotStartUntilExplicitConfirmationAndPostsOnlyThePlanToken() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement()
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        let model = try XCTUnwrap(store.modelCatalog?.models.first)
        await store.reviewModelDownload(model, readPictures: false)
        XCTAssertEqual(store.modelDownloadPlan?.planId, "fresh-plan")
        let planRequest = try XCTUnwrap(fixture.requests.last(where: { $0.path.hasSuffix("/plan") }))
        XCTAssertEqual(planRequest.jsonBody?["readPictures"] as? Bool, false)
        XCTAssertNil(planRequest.jsonBody?["readPictures"] as? String)
        XCTAssertFalse(fixture.requests.contains(where: { $0.path.hasSuffix("/downloads") }))

        let started = fixture.hold("/v1/model-management/targets/local/downloads")
        let starting = Task { await store.startModelDownload() }
        await fulfillment(of: [started.requested], timeout: 2)
        fixture.respondManagement(jobs: [StoreTransport.downloadJob()])
        started.finish(["job": StoreTransport.downloadJob()])
        await starting.value
        let request = try XCTUnwrap(fixture.requests.last(where: { $0.path.hasSuffix("/downloads") }))
        XCTAssertEqual(request.body, ["planId": "fresh-plan"])
        XCTAssertEqual(store.modelCatalog?.jobs.first?.id, "own-download")
        XCTAssertNil(store.reviewingModelDownload)
        XCTAssertFalse(store.isStartingModelDownload)
    }

    func testTargetChangeRejectsLateCatalogAndReviewResponses() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement()
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        let model = try XCTUnwrap(store.modelCatalog?.models.first)
        let oldCatalog = fixture.hold("/v1/model-management/targets/local")
        let oldPlan = fixture.hold("/v1/model-management/targets/local/plan")
        let polling = Task { await store.refreshModelManagement() }
        let reviewing = Task { await store.reviewModelDownload(model, readPictures: true) }
        await fulfillment(of: [oldCatalog.requested, oldPlan.requested], timeout: 2)
        await store.selectModelTarget(StoreTransport.remoteTargetId)
        oldCatalog.finish(StoreTransport.managementCatalog(targetName: "Old Mac"))
        oldPlan.finish(StoreTransport.downloadPlan())
        await polling.value; await reviewing.value
        XCTAssertEqual(store.selectedModelTargetId, StoreTransport.remoteTargetId)
        XCTAssertEqual(store.modelCatalog?.targetName, "Remote Strata")
        XCTAssertNil(store.modelDownloadPlan)
        XCTAssertNil(store.reviewingModelDownload)
        XCTAssertFalse(store.isPlanningModelDownload)
        XCTAssertFalse(fixture.requests.contains(where: { $0.path.hasSuffix("/downloads") }))
    }

    func testForgetDuringReviewCannotRestoreOldMacDownloadState() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement()
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        let model = try XCTUnwrap(store.modelCatalog?.models.first)
        let plan = fixture.hold("/v1/model-management/targets/local/plan")
        let reviewing = Task { await store.reviewModelDownload(model, readPictures: true) }
        await fulfillment(of: [plan.requested], timeout: 2)
        store.forgetMac()
        plan.finish(StoreTransport.downloadPlan())
        await reviewing.value
        XCTAssertFalse(store.isPaired)
        XCTAssertTrue(store.modelTargets.isEmpty)
        XCTAssertNil(store.modelCatalog)
        XCTAssertNil(store.modelDownloadPlan)
        XCTAssertNil(store.reviewingModelDownload)
        XCTAssertFalse(store.isPlanningModelDownload)
    }

    func testExpiredReviewCannotPostADownload() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement(expiresAt: 1)
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        await store.reviewModelDownload(try XCTUnwrap(store.modelCatalog?.models.first), readPictures: true)
        await store.startModelDownload()
        XCTAssertTrue(store.modelDownloadReviewError?.contains("expired") == true)
        XCTAssertFalse(fixture.requests.contains(where: { $0.path.hasSuffix("/downloads") }))
    }

    func testMacStartedDownloadPollsOnlyInModelsAndCannotCancelAnotherDevice() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement(jobs: [StoreTransport.downloadJob(canCancel: false)])
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        let reads = fixture.requests.count
        await store.refreshLiveState(refreshOverview: false)
        XCTAssertEqual(fixture.requests.count, reads)
        store.setModelsVisible(true)
        var changed = StoreTransport.downloadJob(canCancel: false); changed["percent"] = 45
        fixture.respondManagement(jobs: [changed])
        await store.refreshLiveState(refreshOverview: false)
        XCTAssertEqual(fixture.requests.count, reads + 1)
        XCTAssertEqual(store.modelCatalog?.jobs.first?.progress, 0.45)
        var forged = try XCTUnwrap(store.modelCatalog?.jobs.first); forged.canCancel = true
        await store.cancelModelDownload(forged)
        XCTAssertFalse(fixture.requests.contains(where: { $0.method == "POST" }))
        var publications = 0
        let observation = store.objectWillChange.sink { publications += 1 }
        defer { observation.cancel() }
        await store.refreshLiveState(refreshOverview: false)
        XCTAssertEqual(publications, 0, "An unchanged progress snapshot should not rebuild the Models screen.")
    }

    func testStopIsScopedAndAPreStopPollCannotReviveTheJob() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement(jobs: [StoreTransport.downloadJob()])
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        let job = try XCTUnwrap(store.modelCatalog?.jobs.first)
        let oldCatalog = fixture.hold("/v1/model-management/targets/local")
        let polling = Task { await store.refreshModelManagement() }
        await fulfillment(of: [oldCatalog.requested], timeout: 2)
        let cancelled = fixture.hold("/v1/model-management/targets/local/downloads/own-download/cancel")
        let cancelling = Task { await store.cancelModelDownload(job) }
        await fulfillment(of: [cancelled.requested], timeout: 2)
        fixture.release("/v1/model-management/targets/local")
        let stopped = StoreTransport.downloadJob(state: "cancelled", canCancel: false)
        fixture.respondManagement(jobs: [stopped])
        cancelled.finish(["job": stopped])
        await cancelling.value
        oldCatalog.finish(StoreTransport.managementCatalog(jobs: [StoreTransport.downloadJob()]))
        await polling.value
        XCTAssertEqual(store.modelCatalog?.jobs.first?.state, "cancelled")
        XCTAssertEqual(fixture.requests.filter { $0.method == "POST" }.map(\.path), ["/v1/model-management/targets/local/downloads/own-download/cancel"])
        XCTAssertNil(store.cancellingModelDownloadId)
        await store.reviewModelDownload(try XCTUnwrap(store.modelCatalog?.models.first), readPictures: true)
        XCTAssertEqual(store.modelDownloadPlan?.planId, "fresh-plan", "Retry must obtain a fresh review instead of restarting the old job.")
        XCTAssertFalse(fixture.requests.contains(where: { $0.path.hasSuffix("/downloads") }))
    }

    func testAPIOnlyModelCannotBeSentToTheDownloadPlanner() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement()
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        let apiModel = try decode(DownloadableModel.self, ["id": "api-provider-model", "name": "API only", "state": "available", "readPictures": false])
        await store.reviewModelDownload(apiModel, readPictures: false)
        XCTAssertNil(store.reviewingModelDownload)
        XCTAssertFalse(fixture.requests.contains(where: { $0.method == "POST" }))
    }

    func testDisappearingTargetsLeaveThePickerBindingSafe() async throws {
        let fixture = StoreTransport(); defer { fixture.close() }
        fixture.respondManagement()
        let store = fixture.makeStore()
        await store.refreshModelManagement()
        XCTAssertEqual(store.modelTargetSelection, "local")
        fixture.respond { request in
            request.url!.path == "/v1/model-management/targets" ? ["targets": [], "reason": "Update Strata"] : nil
        }
        await store.refreshModelManagement(refreshTargets: true)
        XCTAssertTrue(store.modelTargets.isEmpty)
        XCTAssertEqual(store.modelTargetSelection, "", "SwiftUI may read the old Picker binding after its targets disappear.")
        XCTAssertNil(store.modelCatalog)
        XCTAssertEqual(store.modelManagementMessage, "Update Strata")
    }

    private func decode<T: Decodable>(_ type: T.Type, _ json: [String: Any]) throws -> T {
        try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: json))
    }
}

private final class StoreTransport {
    static let status: [String: Any] = ["macName": "Fixture Mac", "hostConnected": true, "engineState": "idle", "activeRequests": 0]
    let host = "fixture-\(UUID().uuidString.lowercased()).invalid"
    private let lock = NSLock()
    private var held: [String: HeldStoreResponse] = [:]
    struct RecordedRequest { var path: String; var method: String; var body: [String: String]?; var jsonBody: [String: Any]? }
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

    func release(_ path: String) {
        lock.lock(); held.removeValue(forKey: path); lock.unlock()
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
        let jsonBody = payload.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        lock.lock()
        recordedRequests.append(RecordedRequest(path: path, method: request.request.httpMethod ?? "GET", body: body, jsonBody: jsonBody))
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
    func finish(_ json: [String: Any], status: Int = 200) {
        lock.lock(); let request = self.request; self.request = nil; lock.unlock()
        request?.finishJSON(json, status: status)
    }
    func finishSSE(_ value: String) {
        lock.lock(); let request = self.request; self.request = nil; lock.unlock()
        request?.finish(Data(value.utf8), contentType: "text/event-stream")
    }
}

private extension StoreTransport {
    static var remoteTargetId: String { "remote-aaaaaaaaaaaaaaaaaaaaaaaa" }
    static func downloadJob(state: String = "running", canCancel: Bool? = true) -> [String: Any] {
        var job: [String: Any] = ["id": "own-download", "modelId": "catalog-native", "name": "Native fixture", "kind": "download", "state": state,
            "summary": "Downloading model files", "percent": 25, "doneGb": 3, "totalGb": 12]
        if let canCancel { job["canCancel"] = canCancel }
        return job
    }
    static func managementCatalog(targetName: String = "Fixture Mac", jobs: [[String: Any]] = []) -> [String: Any] {
        ["supported": true, "targetName": targetName, "capabilities": ["downloads": true, "imports": false],
         "models": [["id": "catalog-native", "name": "Native fixture", "sizeGb": 12, "state": "available", "readPictures": true]], "jobs": jobs]
    }
    static func downloadPlan(expiresAt: Double = Date().timeIntervalSince1970 * 1000 + 60_000) -> [String: Any] {
        ["planId": "fresh-plan", "targetName": "Fixture Mac", "modelName": "Native fixture", "downloadGb": 12, "diskFreeGb": 100, "diskNeededGb": 14,
         "readPictures": true, "expiresAt": expiresAt,
         "warnings": ["Running native models will be unloaded before preparing files.", "The engine may be downloaded or updated.", "The model will not be loaded automatically."]]
    }
    func respondManagement(jobs: [[String: Any]] = [], expiresAt: Double = Date().timeIntervalSince1970 * 1000 + 60_000) {
        respond { request in
            switch request.url!.path {
            case "/v1/model-management/targets":
                return ["targets": [["id": "local", "name": "Fixture Mac", "kind": "local"], ["id": Self.remoteTargetId, "name": "Remote Strata", "kind": "remote"]]]
            case "/v1/model-management/targets/local": return Self.managementCatalog(jobs: jobs)
            case "/v1/model-management/targets/" + Self.remoteTargetId: return Self.managementCatalog(targetName: "Remote Strata")
            case "/v1/model-management/targets/local/plan": return Self.downloadPlan(expiresAt: expiresAt)
            default: return nil
            }
        }
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
