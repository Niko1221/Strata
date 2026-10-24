import SwiftUI

@main
struct StrataMobileApp: App {
    @StateObject private var store = CompanionStore()
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            ContentView(store: store)
                .task(id: scenePhase) {
                    guard scenePhase == .active else { return }
                    var tick = 0
                    while !Task.isCancelled {
                        await store.refreshLiveState(refreshOverview: tick % 5 == 0)
                        tick += 1
                        do { try await Task.sleep(for: .seconds(1)) }
                        catch { return }
                    }
                }
                .onOpenURL { url in
                    do { store.pendingInvitation = try PairingInvitation.parse(url.absoluteString) }
                    catch { store.error = error.localizedDescription }
                }
        }
    }
}
