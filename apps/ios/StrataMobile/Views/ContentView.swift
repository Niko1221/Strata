import SwiftUI

enum CompanionTab: Hashable {
    case overview, chat, models, connection
}

struct ContentView: View {
    @ObservedObject var store: CompanionStore
    @State private var selection: CompanionTab = .overview
    @State private var showingPairing = false

    var body: some View {
        TabView(selection: $selection) {
            NavigationStack {
                OverviewView(store: store, pair: { showingPairing = true }, open: { selection = $0 })
            }
            .tabItem { Label("Home", systemImage: "square.grid.2x2") }
            .tag(CompanionTab.overview)
            .accessibilityIdentifier("homeTab")

            NavigationStack {
                ChatView(store: store, pair: { showingPairing = true })
            }
            .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
            .tag(CompanionTab.chat)
            .accessibilityIdentifier("chatTab")

            NavigationStack {
                ModelsView(store: store, pair: { showingPairing = true })
            }
            .tabItem { Label("Models", systemImage: "square.stack.3d.up") }
            .tag(CompanionTab.models)
            .accessibilityIdentifier("modelsTab")

            NavigationStack {
                ConnectionView(store: store, pair: { showingPairing = true })
            }
            .tabItem { Label("Connection", systemImage: "link") }
            .tag(CompanionTab.connection)
            .accessibilityIdentifier("connectionTab")
        }
        .tint(StrataTheme.accentText)
        .sheet(isPresented: $showingPairing) {
            PairMacView(store: store)
        }
        .onChange(of: store.pendingInvitation, initial: true) { _, invitation in
            if invitation != nil { showingPairing = true }
        }
        .alert("Couldn’t complete the request", isPresented: Binding(
            get: { store.error != nil && !showingPairing },
            set: { if !$0 { store.error = nil } }
        )) {
            Button("OK") { store.error = nil }
        } message: {
            Text(store.error ?? "Please try again.")
        }
        .onChange(of: selection, initial: true) { _, tab in
            store.setChatVisible(tab == .chat)
            store.setModelsVisible(tab == .models)
        }
        .task { if store.isPaired { await store.refresh() } }
    }
}
