import SwiftUI

struct OverviewView: View {
    @ObservedObject var store: CompanionStore
    let pair: () -> Void
    let open: (CompanionTab) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                StrataHeader(isPaired: store.isPaired, isOnline: store.status?.hostConnected == true) {
                    open(.connection)
                }
                if store.isPaired {
                    pairedContent
                } else {
                    welcomeContent
                }
            }
            .padding(.horizontal, 24)
            .padding(.top, 16)
            .padding(.bottom, 30)
        }
        .background(StrataTheme.background)
        .toolbar(.hidden, for: .navigationBar)
        .refreshable { if store.isPaired { await store.refresh() } }
    }

    private var welcomeContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            PageHeading(eyebrow: "Meet your Mac’s companion", title: "Your workspace.\nWithin reach.",
                        subtitle: "Your Strata workspace, a little closer. Chat and control your Mac from your iPhone.")

            VStack(spacing: 4) {
                PairingIllustration()
                VStack(spacing: 8) {
                    Text("A powerful pair.")
                        .font(.system(.title3, design: .rounded, weight: .bold))
                        .foregroundStyle(StrataTheme.ink)
                    Text("Your Mac brings it together.\nYour iPhone keeps you connected.")
                        .font(.subheadline)
                        .foregroundStyle(StrataTheme.secondaryInk)
                        .multilineTextAlignment(.center)
                        .lineSpacing(4)
                }
                .padding(.bottom, 23)
                Button(action: pair) {
                    HStack(spacing: 9) {
                        Image(systemName: "qrcode.viewfinder")
                        Text("Pair your Mac")
                        Spacer()
                        Image(systemName: "arrow.right").font(.subheadline.weight(.semibold))
                    }
                    .padding(.horizontal, 20)
                }
                .buttonStyle(StrataPrimaryButtonStyle())
                .accessibilityIdentifier("pairMacButton")
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 18)
            .strataCard(padding: 0)

            HStack(alignment: .top, spacing: 11) {
                Image(systemName: "lock.shield")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(StrataTheme.accentText)
                VStack(alignment: .leading, spacing: 5) {
                    Text("Made for your own Mac")
                        .font(.system(.subheadline, weight: .semibold))
                        .foregroundStyle(StrataTheme.ink)
                    Text("Pair once on your local network. Your Mac stays in control of its models.")
                        .font(.caption)
                        .foregroundStyle(StrataTheme.secondaryInk)
                        .lineSpacing(3)
                }
            }
            .padding(.horizontal, 3)
        }
    }

    private var pairedContent: some View {
        VStack(alignment: .leading, spacing: 23) {
            PageHeading(eyebrow: "Your workspace", title: "Your Mac.\nIn your hands.",
                        subtitle: "\(store.status?.macName ?? store.connection?.macName ?? "Your Mac") · \(store.status?.hostConnected == true ? "Connected to Strata" : "Waiting for a connection")")

            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Image(systemName: "desktopcomputer")
                        .font(.system(size: 23, weight: .regular))
                        .foregroundStyle(StrataTheme.accentText)
                        .frame(width: 49, height: 49)
                        .background(StrataTheme.accentTint, in: RoundedRectangle(cornerRadius: 15))
                    Spacer()
                    Text(engineLabel)
                        .font(.system(.caption, weight: .semibold))
                        .foregroundStyle(StrataTheme.accentText)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 7)
                        .background(StrataTheme.accentTint, in: Capsule())
                }
                VStack(alignment: .leading, spacing: 5) {
                    Text("ACTIVE MODEL")
                        .font(.system(.caption2, weight: .bold))
                        .tracking(1.6)
                        .foregroundStyle(StrataTheme.secondaryInk)
                    Text(store.status?.modelName ?? "No model loaded")
                        .font(.system(.title2, design: .rounded, weight: .bold))
                        .foregroundStyle(StrataTheme.ink)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let status = store.status, status.hostConnected {
                    Divider().overlay(StrataTheme.line)
                    HStack(alignment: .top, spacing: 24) {
                        if let speed = status.tokensPerSecond {
                            metric(value: speed.formatted(.number.precision(.fractionLength(1))), label: "tokens / sec")
                        }
                        if let used = status.memoryUsedGB {
                            metric(value: used.formatted(.number.precision(.fractionLength(1))) + " GB", label: "memory in use")
                        }
                        metric(value: "\(status.activeRequests)", label: "active requests")
                    }
                }
                Button { open(.models) } label: {
                    HStack { Text("Manage models"); Spacer(); Image(systemName: "arrow.up.right") }
                        .font(.system(.subheadline, weight: .semibold))
                        .foregroundStyle(StrataTheme.accentText)
                }
                .padding(.top, 2)
            }
            .strataCard()

            if !store.conversations.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("ON YOUR MAC").font(.caption2.weight(.bold)).tracking(1.6).foregroundStyle(StrataTheme.secondaryInk)
                        Spacer()
                        Button("See all") { open(.chat) }.font(.caption.weight(.semibold)).foregroundStyle(StrataTheme.accentText)
                    }
                    ForEach(store.conversations.prefix(3)) { conversation in
                        Button {
                            Task { await store.selectConversation(conversation); open(.chat) }
                        } label: {
                            HStack(spacing: 13) {
                                Image(systemName: conversation.active ? "circle.fill" : "bubble.left")
                                    .font(conversation.active ? .caption2 : .body).foregroundStyle(StrataTheme.accentText).frame(width: 22)
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(conversation.title).font(.subheadline.weight(.semibold)).foregroundStyle(StrataTheme.ink).lineLimit(2)
                                    Text(conversation.active ? "Working now · follow live" : (conversation.workspaceName ?? "Continue conversation"))
                                        .font(.caption).foregroundStyle(conversation.active ? StrataTheme.accentText : StrataTheme.secondaryInk)
                                }
                                Spacer(minLength: 0)
                                Image(systemName: "chevron.right").font(.caption).foregroundStyle(StrataTheme.secondaryInk)
                            }
                            .strataCard(padding: 16)
                        }
                        .buttonStyle(.plain).accessibilityIdentifier("homeMacConversation-" + conversation.id)
                    }
                }
            }

            Text("Make room for a good idea.")
                .font(.system(.headline, design: .rounded))
                .foregroundStyle(StrataTheme.ink)
                .padding(.top, 4)

            Button { store.newChat(); open(.chat) } label: {
                HStack(spacing: 16) {
                    Image(systemName: "bubble.left.and.bubble.right")
                        .font(.system(size: 26, weight: .light))
                        .foregroundStyle(StrataTheme.accentText)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Start a conversation").font(.headline).foregroundStyle(StrataTheme.ink)
                        Text("Start work on your Mac from here.")
                            .font(.caption).foregroundStyle(StrataTheme.secondaryInk)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "arrow.right").foregroundStyle(StrataTheme.secondaryInk)
                }
                .strataCard()
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("startChatButton")

            if store.status?.hostConnected != true {
                Button { Task { await store.refresh() } } label: {
                    Label(store.isLoading ? "Reconnecting…" : "Reconnect to your Mac", systemImage: "arrow.clockwise")
                }
                .buttonStyle(StrataSecondaryButtonStyle())
                .disabled(store.isLoading)
            }
        }
    }

    private var engineLabel: String {
        guard let status = store.status, status.hostConnected else { return "Offline" }
        switch status.engineState.lowercased() {
        case "ready", "running", "serving": return "Ready"
        case "loading", "starting": return "Starting"
        case "stopped", "idle": return "Standby"
        default: return status.engineState.capitalized
        }
    }

    private func metric(value: String, label: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value).font(.system(.title3, design: .rounded, weight: .semibold)).foregroundStyle(StrataTheme.ink)
            Text(label).font(.caption2).foregroundStyle(StrataTheme.secondaryInk)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
