import SwiftUI

struct ChatView: View {
    @ObservedObject var store: CompanionStore
    let pair: () -> Void
    @State private var draft = ""
    @State private var showingHistory = false
    @State private var historyQuery = ""
    @State private var followsLatest = true
    @State private var nearBottom = true
    @State private var userScrolling = false
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            if !store.isPaired {
                ScrollView {
                    ConnectionNeededView(icon: "bubble.left.and.bubble.right", title: "An idea starts here", message: "Pair your Mac to pick up your Strata conversations and follow its work.", pair: pair)
                        .padding(.horizontal, 24).padding(.top, 24)
                }
            } else {
                if let conversation = store.currentConversation { contextBar(conversation) }
                conversation
                composer
            }
        }
        .background(StrataTheme.background)
        .toolbar(.hidden, for: .navigationBar)
        .sheet(isPresented: $showingHistory) { history }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Chat").font(.system(.largeTitle, design: .rounded, weight: .bold)).tracking(-1.3).foregroundStyle(StrataTheme.ink)
                Text(chatSubtitle).font(.caption).foregroundStyle(StrataTheme.secondaryInk).lineLimit(1)
            }
            Spacer(minLength: 12)
            if store.isPaired {
                Button { showingHistory = true; Task { await store.refresh(silently: true) } } label: {
                    Image(systemName: "sidebar.left").font(.system(size: 19)).frame(width: 40, height: 44)
                }
                .disabled(store.isSending)
                .accessibilityLabel("Conversation history").accessibilityIdentifier("chatHistoryButton")
                Button { store.newChat(); composerFocused = false } label: {
                    Image(systemName: "square.and.pencil").font(.system(size: 20)).frame(width: 40, height: 44)
                }
                .disabled(store.isSubmittingMessage || store.isSending)
                .accessibilityLabel("New conversation").accessibilityIdentifier("newChatButton")
            }
        }
        .foregroundStyle(StrataTheme.ink)
        .padding(.horizontal, 24).padding(.top, 27).padding(.bottom, 16)
    }

    private func contextBar(_ value: MacConversation) -> some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "desktopcomputer").foregroundStyle(StrataTheme.accentText).padding(.top, 2)
            VStack(alignment: .leading, spacing: 3) {
                Text(value.title).font(.subheadline.weight(.semibold)).foregroundStyle(StrataTheme.ink).lineLimit(2)
                    .accessibilityIdentifier("macConversationTitle")
                Text(value.permissionLabel + (value.workspace.map { " · " + URL(fileURLWithPath: $0).lastPathComponent } ?? ""))
                    .font(.caption2).foregroundStyle(StrataTheme.secondaryInk).lineLimit(2)
            }
            Spacer(minLength: 5)
            if value.active {
                Label("Live", systemImage: "circle.fill").font(.caption2.weight(.semibold)).foregroundStyle(StrataTheme.accentText)
                    .accessibilityElement(children: .combine).accessibilityIdentifier("macConversationLiveBadge")
            }
        }
        .padding(.horizontal, 18).padding(.vertical, 12)
        .background(StrataTheme.accentTint)
    }

    private var visibleMessages: [MacChatMessage] {
        if let current = store.currentConversation { return current.messages }
        return store.messages.map { MacChatMessage(id: $0.id, role: $0.role, content: $0.content) }
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            observeConversationScroll(conversationScroll)
                .scrollDismissesKeyboard(.interactively)
                .refreshable { await store.refreshCurrentConversation() }
                .overlay(alignment: .bottomTrailing) {
                    if showsLatestButton {
                        Button { followsLatest = true; scrollToBottom(proxy) } label: {
                            Label("Latest", systemImage: "arrow.down").font(.caption.weight(.semibold))
                                .padding(.horizontal, 13).padding(.vertical, 9)
                                .foregroundStyle(StrataTheme.accentInk).background(StrataTheme.accent, in: Capsule())
                        }
                        .padding(14).accessibilityIdentifier("followLatestMessagesButton")
                    }
                }
                .onChange(of: store.currentConversation?.id) { _, _ in
                    followsLatest = true; scrollToBottom(proxy)
                }
                .onChange(of: store.currentChatId) { _, _ in
                    followsLatest = true; scrollToBottom(proxy)
                }
                .onChange(of: store.isSubmittingMessage) { old, new in
                    if old, !new, followsLatest { scrollToBottom(proxy) }
                }
        }
    }

    private var conversationScroll: some View {
        ScrollView {
            if store.isSelectingConversation {
                ProgressView("Opening Mac conversation…").padding(.top, 40)
            } else if visibleMessages.isEmpty {
                chatWelcome
            } else {
                VStack(alignment: .leading, spacing: 23) {
                    ForEach(visibleMessages) { message in messageView(message).id(message.id) }
                    ForEach(additionalPendingApprovals) { tool in activity(tool) }
                    if store.isWorking { workingIndicator }
                    Color.clear.frame(height: 1).id("conversationBottom")
                }
                .padding(.horizontal, 24).padding(.top, 20).padding(.bottom, 22)
            }
        }
    }

    @ViewBuilder private func observeConversationScroll<Content: View>(_ content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content
                .defaultScrollAnchor(.bottom, for: .initialOffset)
                // Native anchoring handles text and tool-card height changes
                // without re-entering lazy layout through proxy.scrollTo.
                .defaultScrollAnchor(followsLatest ? .bottom : nil, for: .sizeChanges)
                .onScrollGeometryChange(for: Bool.self, of: { geometry in
                    geometry.contentOffset.y + geometry.containerSize.height >= geometry.contentSize.height - 70
                }, action: { _, newValue in
                    nearBottom = newValue
                    if userScrolling { followsLatest = newValue }
                    else if newValue { followsLatest = true }
                })
                .onScrollPhaseChange { _, newPhase, context in
                    let isUserPhase = newPhase == .tracking || newPhase == .interacting || newPhase == .decelerating
                    if isUserPhase || userScrolling {
                        followsLatest = context.geometry.contentOffset.y + context.geometry.containerSize.height >= context.geometry.contentSize.height - 70
                    }
                    userScrolling = isUserPhase
                }
        } else {
            // iOS 17 keeps native scrolling and offers an explicit Latest action.
            // Live updates never pull an older-history reader back to the bottom.
            content
        }
    }

    private var showsLatestButton: Bool {
        if #available(iOS 18.0, *) { return !followsLatest }
        return !visibleMessages.isEmpty
    }

    private var additionalPendingApprovals: [MacToolActivity] {
        let shown = Set(visibleMessages.flatMap { ($0.tools ?? []).map(\.id) })
        return store.pendingApprovals.filter { !shown.contains($0.id) }
    }

    private var workingIndicator: some View {
        HStack(spacing: 9) {
            ProgressView().controlSize(.small).tint(StrataTheme.accentText)
            Text(liveStatus).font(.caption).foregroundStyle(StrataTheme.secondaryInk)
        }
        .accessibilityIdentifier("chatThinkingIndicator")
    }

    private var liveStatus: String {
        if !store.pendingApprovals.isEmpty { return "Waiting for your approval" }
        if let tool = store.currentConversation?.messages.flatMap({ $0.tools ?? [] }).last(where: \.isRunning) {
            return tool.isBrowser ? "Using the browser on your Mac…" : tool.title + "…"
        }
        return store.currentConversation?.status?.isEmpty == false ? store.currentConversation!.status! : "Your Mac is working…"
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        // Explicit navigation jumps after the content has a deterministic height.
        // Animating a lazy footer lookup can repeatedly re-enter layout.
        proxy.scrollTo("conversationBottom", anchor: .bottom)
    }

    private var chatWelcome: some View {
        VStack(alignment: .leading, spacing: 26) {
            StrataMark(size: 52).padding(.bottom, 5)
            Text("Pick up where\nyou left off.")
                .font(.system(.largeTitle, design: .rounded, weight: .bold)).tracking(-1).foregroundStyle(StrataTheme.ink)
            Text("Open a Mac conversation to see its history, continue the work, and follow browser and tool activity as it happens.")
                .font(.subheadline).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(4)
            if !store.conversations.isEmpty {
                Button { showingHistory = true } label: { Label("Open Mac conversations", systemImage: "desktopcomputer") }
                    .buttonStyle(StrataPrimaryButtonStyle())
                    .accessibilityIdentifier("openMacConversationsButton")
            }
            Text("Or send a message below to start a new conversation on your Mac.")
                .font(.caption).foregroundStyle(StrataTheme.secondaryInk)
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 24).padding(.top, 34).padding(.bottom, 26)
    }

    private func messageView(_ message: MacChatMessage) -> some View {
        let isUser = message.role == "user"
        return HStack(alignment: .top, spacing: 0) {
            if isUser { Spacer(minLength: 34) }
            VStack(alignment: .leading, spacing: 12) {
                if !isUser {
                    HStack(spacing: 6) {
                        StrataMark(size: 22)
                        Text("STRATA").font(.system(.caption2, weight: .bold)).tracking(1.5).foregroundStyle(StrataTheme.secondaryInk)
                    }
                }
                if message.role == "assistant", message.thinkingText != nil {
                    MacThinkingView(message: message, isLive: isLiveMessage(message) && message.isThinkingInProgress)
                }
                ForEach(message.timelineItems) { item in
                    switch item {
                    case .text(_, let text):
                        messageText(text, isUser: isUser)
                    case .group(let group, let tools):
                        MacActivityGroupView(group: group, tools: tools,
                            isLive: isLiveMessage(message),
                            pendingToolIds: Set(store.pendingApprovals.map(\.id)),
                            activity: { tool in activity(tool) })
                    case .tool(let tool):
                        activity(tool)
                    }
                }
                if let error = message.error, !error.isEmpty {
                    Label(error, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(StrataTheme.warning)
                }
            }
            if !isUser { Spacer(minLength: 0) }
        }
    }

    private func isLiveMessage(_ message: MacChatMessage) -> Bool {
        message.streaming == true && store.currentConversation?.active == true
    }

    private func messageText(_ text: String, isUser: Bool) -> some View {
        Text(.init(text)).font(.body).foregroundStyle(StrataTheme.ink).lineSpacing(5).textSelection(.enabled)
            .padding(isUser ? 15 : 0).frame(maxWidth: .infinity, alignment: .leading)
            .background(isUser ? StrataTheme.accentTint : .clear, in: RoundedRectangle(cornerRadius: 19))
            .accessibilityIdentifier(isUser ? "userMessage" : "assistantMessage")
    }

    private func activity(_ tool: MacToolActivity) -> some View {
        MacActivityCard(tool: tool,
                        awaitingApproval: store.pendingApprovals.contains(where: { $0.id == tool.id }),
                        deciding: store.decidingToolId != nil,
                        decide: { approve in Task { await store.decideApproval(tool, approve: approve) } })
    }

    private var composer: some View {
        VStack(spacing: 8) {
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Message your Mac…", text: $draft, axis: .vertical)
                    .lineLimit(1...6).font(.body).foregroundStyle(StrataTheme.ink).focused($composerFocused)
                    .padding(.vertical, 12).padding(.leading, 16).accessibilityIdentifier("messageInput")
                if store.isWorking {
                    Button { Task { await store.cancelMessage() } } label: {
                        Image(systemName: "stop.fill").font(.system(size: 13, weight: .bold)).foregroundStyle(StrataTheme.accentInk)
                            .frame(width: 37, height: 37).background(StrataTheme.accent, in: Circle())
                    }
                    .padding(6).disabled(store.isSubmittingMessage)
                    .accessibilityLabel("Stop response").accessibilityIdentifier("stopMessageButton")
                } else {
                    Button(action: send) {
                        Group {
                            if store.isSubmittingMessage { ProgressView().controlSize(.small) }
                            else { Image(systemName: "arrow.up").font(.system(size: 18, weight: .semibold)) }
                        }
                        .foregroundStyle(canSend ? StrataTheme.accentInk : StrataTheme.secondaryInk)
                        .frame(width: 37, height: 37).background(canSend ? StrataTheme.accent : StrataTheme.surfaceSecondary, in: Circle())
                    }
                    .padding(6).disabled(!canSend).accessibilityLabel("Send message").accessibilityIdentifier("sendMessageButton")
                }
            }
            .background(StrataTheme.surface, in: RoundedRectangle(cornerRadius: 25))
            .overlay(RoundedRectangle(cornerRadius: 25).strokeBorder(StrataTheme.line, lineWidth: 1))
            Text(composerNote).font(.system(size: 10)).foregroundStyle(StrataTheme.secondaryInk)
        }
        .padding(.horizontal, 20).padding(.top, 10).padding(.bottom, 10).background(StrataTheme.background)
    }

    private var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && store.isPaired &&
        store.status?.hostConnected == true && !store.isWorking && !store.isSubmittingMessage && !store.isSelectingConversation
    }
    private var composerNote: String {
        if store.status?.hostConnected != true { return "Your Mac is unavailable. Check Connection to reconnect." }
        return store.isLegacyConversation ? "Earlier iPhone conversation · saved on your Mac" : "Same Mac conversation · updates live while this screen is open"
    }
    private var chatSubtitle: String {
        guard store.isPaired else { return "Your Mac’s conversations, within reach." }
        guard store.status?.hostConnected == true else { return "Mac unavailable · check Connection" }
        return store.currentConversation?.modelName ?? store.status?.modelName ?? "Connected to your Mac"
    }
    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend else { return }
        draft = ""; composerFocused = false; followsLatest = true
        Task {
            await store.sendMessage(text)
            if store.error != nil, draft.isEmpty { draft = text }
        }
    }

    private var filteredConversations: [MacConversationSummary] {
        store.conversations.filter { historyQuery.isEmpty || $0.title.localizedCaseInsensitiveContains(historyQuery) || ($0.workspace?.localizedCaseInsensitiveContains(historyQuery) == true) }
    }
    private var filteredLegacyChats: [CompanionChat] {
        store.chats.filter { historyQuery.isEmpty || $0.title.localizedCaseInsensitiveContains(historyQuery) }
    }
    private var history: some View {
        NavigationStack {
            List {
                Section("On your Mac") {
                    if filteredConversations.isEmpty {
                        Text(historyQuery.isEmpty ? "Mac conversations will appear here." : "No matching conversations.")
                            .font(.subheadline).foregroundStyle(StrataTheme.secondaryInk)
                    }
                    ForEach(filteredConversations) { chat in
                        Button {
                            showingHistory = false
                            Task { await store.selectConversation(chat) }
                        } label: {
                            HStack(spacing: 13) {
                                Image(systemName: chat.active ? "circle.fill" : "bubble.left").font(chat.active ? .caption2 : .body)
                                    .foregroundStyle(StrataTheme.accentText).frame(width: 20)
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(chat.title).foregroundStyle(StrataTheme.ink).lineLimit(2)
                                    Text([chat.active ? "Working now" : nil, chat.workspaceName].compactMap { $0 }.joined(separator: " · "))
                                        .font(.caption).foregroundStyle(chat.active ? StrataTheme.accentText : StrataTheme.secondaryInk)
                                }
                                Spacer(minLength: 0)
                                if store.currentConversation?.id == chat.id { Image(systemName: "checkmark").foregroundStyle(StrataTheme.accentText) }
                            }
                            .padding(.vertical, 6)
                        }
                        .accessibilityIdentifier("macConversation-" + chat.id)
                    }
                }
                if !filteredLegacyChats.isEmpty {
                    Section("Earlier iPhone conversations") {
                        ForEach(filteredLegacyChats) { chat in
                            Button {
                                showingHistory = false
                                Task { await store.selectChat(chat) }
                            } label: {
                                Label(chat.title, systemImage: "iphone").foregroundStyle(StrataTheme.ink).lineLimit(2).padding(.vertical, 6)
                            }
                        }
                    }
                }
            }
            .refreshable { await store.refresh(silently: true) }
            .searchable(text: $historyQuery, prompt: "Search conversations")
            .scrollContentBackground(.hidden).background(StrataTheme.background)
            .navigationTitle("Conversations").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button("Done") { showingHistory = false } } }
        }
        .tint(StrataTheme.accentText)
    }
}
