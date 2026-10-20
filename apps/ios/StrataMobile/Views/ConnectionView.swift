import SwiftUI

struct ConnectionView: View {
    @ObservedObject var store: CompanionStore
    let pair: () -> Void
    @State private var showingDisconnect = false
    @State private var showingForget = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                PageHeading(eyebrow: "Better together", title: "Connection", subtitle: "A direct link to your Strata workspace.")
                if store.isPaired {
                    connectedCard
                } else {
                    VStack(spacing: 14) {
                        PairingIllustration(compact: true)
                        VStack(spacing: 8) {
                            Text("Your Mac is the missing piece.")
                                .font(.system(.title3, design: .rounded, weight: .bold))
                                .foregroundStyle(StrataTheme.ink)
                                .multilineTextAlignment(.center)
                            Text("Scan a pairing code to bring your\nStrata workspace to your phone.")
                                .font(.subheadline)
                                .foregroundStyle(StrataTheme.secondaryInk)
                                .multilineTextAlignment(.center)
                                .lineSpacing(4)
                        }
                        Button(action: pair) {
                            Label("Pair your Mac", systemImage: "qrcode.viewfinder")
                        }
                        .buttonStyle(StrataPrimaryButtonStyle())
                        .padding(.top, 6)
                        .accessibilityIdentifier("connectionPairButton")
                    }
                    .padding(.horizontal, 18)
                    .padding(.bottom, 18)
                    .strataCard(padding: 0)
                }
                VStack(alignment: .leading, spacing: 22) {
                    DetailRow(icon: "wifi", title: "One local network", detail: "Keep your iPhone and Mac on the same network while using the companion.")
                    DetailRow(icon: "lock.shield", title: "A trusted connection", detail: "A one-time code pairs your phone. Your Mac’s certificate is checked on every connection.")
                    DetailRow(icon: "desktopcomputer", title: "Your workspace, connected", detail: "Use models and API connections configured on your Mac. Selected providers may run remotely.")
                }
                .padding(.horizontal, 2)
                HStack {
                    StrataMark(size: 22)
                    Text("Strata for iPhone").font(.caption)
                    Spacer()
                    Text(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")
                        .font(.caption.monospacedDigit())
                }
                .foregroundStyle(StrataTheme.secondaryInk)
                .padding(.top, 12)
            }
            .padding(.horizontal, 24)
            .padding(.top, 27)
            .padding(.bottom, 30)
        }
        .background(StrataTheme.background)
        .toolbar(.hidden, for: .navigationBar)
        .confirmationDialog("Disconnect this Mac?", isPresented: $showingDisconnect, titleVisibility: .visible) {
            Button("Disconnect Mac", role: .destructive) { Task { await store.disconnect() } }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("You can pair again at any time using a new code from your Mac.")
        }
        .confirmationDialog("Forget this Mac?", isPresented: $showingForget, titleVisibility: .visible) {
            Button("Forget this Mac", role: .destructive) { store.forgetMac() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This removes the pairing key from this iPhone. To revoke its access on the Mac too, choose Pair iPhone in Strata and remove this device.")
        }
        .refreshable { if store.isPaired { await store.refresh() } }
    }

    private var connectedCard: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 14) {
                Image(systemName: "desktopcomputer")
                    .font(.system(size: 27, weight: .light))
                    .foregroundStyle(StrataTheme.accentText)
                    .frame(width: 57, height: 57)
                    .background(StrataTheme.accentTint, in: RoundedRectangle(cornerRadius: 17))
                VStack(alignment: .leading, spacing: 6) {
                    Text(store.status?.macName ?? store.connection?.macName ?? "Your Mac")
                        .font(.system(.title3, design: .rounded, weight: .bold))
                        .foregroundStyle(StrataTheme.ink)
                    HStack(spacing: 6) {
                        Circle().fill(store.status?.hostConnected == true ? StrataTheme.accent : StrataTheme.secondaryInk).frame(width: 6, height: 6)
                        Text(store.status?.hostConnected == true ? "Connected" : "Paired · Mac unavailable")
                            .font(.caption)
                            .foregroundStyle(StrataTheme.secondaryInk)
                    }
                }
                Spacer(minLength: 0)
            }
            Divider().overlay(StrataTheme.line)
            if let address = store.connection?.baseURL.host {
                VStack(alignment: .leading, spacing: 6) {
                    Text("MAC ADDRESS").font(.system(.caption2, weight: .bold)).tracking(1.6)
                    Text(address).font(.subheadline.monospaced()).textSelection(.enabled)
                }
                .foregroundStyle(StrataTheme.secondaryInk)
            }
            Button { Task { await store.refresh() } } label: {
                HStack {
                    if store.isLoading { ProgressView().controlSize(.small) }
                    else { Image(systemName: "arrow.clockwise") }
                    Text(store.isLoading ? "Checking connection…" : "Check connection")
                }
            }
            .buttonStyle(StrataSecondaryButtonStyle())
            .disabled(store.isLoading)
            .accessibilityIdentifier("refreshConnectionButton")
            Button(action: pair) {
                Label("Pair again", systemImage: "qrcode.viewfinder")
            }
            .buttonStyle(StrataSecondaryButtonStyle())
            .disabled(store.isLoading || store.isSending)
            .accessibilityIdentifier("pairAgainButton")
            Button("Disconnect Mac", role: .destructive) { showingDisconnect = true }
                .font(.system(.subheadline, weight: .medium))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 5)
                .accessibilityIdentifier("disconnectMacButton")
            Button("Forget this Mac", role: .destructive) { showingForget = true }
                .font(.system(.caption, weight: .medium))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 5)
                .accessibilityIdentifier("forgetMacButton")
        }
        .strataCard()
    }
}

struct PairMacView: View {
    @ObservedObject var store: CompanionStore
    @Environment(\.dismiss) private var dismiss
    @State private var address = ""
    @State private var code = ""
    @State private var fingerprint = ""
    @State private var showingScanner = false
    @State private var scanError: String?
    @State private var showingManual = false
    @FocusState private var focusedField: PairingField?

    private enum PairingField: Hashable { case address, code, fingerprint }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 16) {
                        Image(systemName: "link")
                            .font(.system(size: 29, weight: .regular))
                            .foregroundStyle(StrataTheme.accentText)
                            .frame(width: 65, height: 65)
                            .background(StrataTheme.accentTint, in: RoundedRectangle(cornerRadius: 21))
                        PageHeading(eyebrow: "Hello, Mac", title: "Make it a pair.", subtitle: "Open Strata on your Mac, choose Pair iPhone, then enable pairing and scan its QR code. Keep both devices on the same network.")
                    }

                    Button { showingScanner = true } label: {
                        Label("Scan pairing code", systemImage: "qrcode.viewfinder")
                    }
                    .buttonStyle(StrataPrimaryButtonStyle())
                    .accessibilityIdentifier("scanPairingCodeButton")

                    HStack(spacing: 13) {
                        Rectangle().fill(StrataTheme.line).frame(height: 1)
                        Text("OR").font(.system(.caption2, weight: .bold)).foregroundStyle(StrataTheme.secondaryInk)
                        Rectangle().fill(StrataTheme.line).frame(height: 1)
                    }
                    .padding(.vertical, 2)

                    VStack(alignment: .leading, spacing: 18) {
                        Button {
                            withAnimation(.easeInOut(duration: 0.2)) { showingManual.toggle() }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 5) {
                                    Text("Enter details manually")
                                        .font(.system(.subheadline, weight: .semibold))
                                        .foregroundStyle(StrataTheme.ink)
                                    Text("Use the address and code on your Mac.")
                                        .font(.caption)
                                        .foregroundStyle(StrataTheme.secondaryInk)
                                }
                                Spacer()
                                Image(systemName: showingManual ? "chevron.up" : "chevron.down")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(StrataTheme.secondaryInk)
                            }
                        }
                        .accessibilityIdentifier("manualPairingButton")
                        if showingManual {
                            VStack(alignment: .leading, spacing: 17) {
                                field("MAC ADDRESS", hint: "https://192.168.1.10:9443", value: $address, field: .address)
                                    .keyboardType(.URL)
                                field("PAIRING CODE", hint: "6-digit code", value: $code, field: .code)
                                    .keyboardType(.numberPad)
                                field("CERTIFICATE FINGERPRINT", hint: "Paste the fingerprint from your Mac", value: $fingerprint, field: .fingerprint)
                                    .keyboardType(.asciiCapable)
                                Text("The fingerprint confirms you’re connecting to your Mac. Scanning the QR code fills all three fields for you.")
                                    .font(.caption)
                                    .foregroundStyle(StrataTheme.secondaryInk)
                                    .lineSpacing(3)
                                Button(action: connect) {
                                    HStack(spacing: 10) {
                                        if store.isLoading { ProgressView().tint(StrataTheme.accentInk) }
                                        Text(store.isLoading ? "Pairing with your Mac…" : "Pair with Mac")
                                    }
                                }
                                .buttonStyle(StrataPrimaryButtonStyle())
                                .disabled(!canPair || store.isLoading)
                                .opacity(canPair && !store.isLoading ? 1 : 0.5)
                                .accessibilityIdentifier("confirmPairingButton")
                            }
                        }
                    }
                    .strataCard()

                    if let error = scanError ?? store.error {
                        Label(error, systemImage: "exclamationmark.circle")
                            .font(.subheadline)
                            .foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("pairingError")
                    }
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "lock.shield").foregroundStyle(StrataTheme.accentText)
                        Text("Your pairing key is stored securely on this iPhone. You can disconnect at any time.")
                            .font(.caption)
                            .foregroundStyle(StrataTheme.secondaryInk)
                            .lineSpacing(3)
                    }
                    .padding(.horizontal, 2)
                }
                .padding(.horizontal, 24)
                .padding(.top, 20)
                .padding(.bottom, 30)
            }
            .background(StrataTheme.background)
            .navigationTitle("Pair your Mac")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        store.pendingInvitation = nil
                        dismiss()
                    }
                    .accessibilityIdentifier("dismissPairingButton")
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { focusedField = nil }
                        .accessibilityIdentifier("dismissPairingKeyboardButton")
                }
            }
            .sheet(isPresented: $showingScanner) {
                NavigationStack {
                    QRScannerView { payload in
                        showingScanner = false
                        receive(payload)
                    }
                    .navigationTitle("Scan pairing code")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button("Done") { showingScanner = false }
                        }
                    }
                }
            }
            .onChange(of: store.connection) { _, connection in
                if connection != nil {
                    store.pendingInvitation = nil
                    dismiss()
                }
            }
            .onAppear {
                if let invitation = store.pendingInvitation {
                    apply(invitation)
                }
            }
        }
        .tint(StrataTheme.accentText)
        .interactiveDismissDisabled(store.isLoading)
    }

    private var canPair: Bool {
        !address.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !fingerprint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func field(_ title: String, hint: String, value: Binding<String>, field: PairingField) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(.caption2, weight: .bold))
                .tracking(1.3)
                .foregroundStyle(StrataTheme.secondaryInk)
            TextField(hint, text: value)
                .font(.subheadline.monospaced())
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .focused($focusedField, equals: field)
                .padding(14)
                .foregroundStyle(StrataTheme.ink)
                .background(StrataTheme.surfaceSecondary, in: RoundedRectangle(cornerRadius: 12))
                .accessibilityIdentifier(field == .address ? "pairingAddressInput" : field == .code ? "pairingCodeInput" : "pairingFingerprintInput")
        }
    }

    private func receive(_ payload: String) {
        do {
            apply(try PairingInvitation.parse(payload))
        } catch {
            scanError = "That code isn’t a Strata pairing invitation. Choose Pair iPhone in Strata on your Mac and scan its code."
        }
    }

    private func apply(_ invitation: PairingInvitation) {
        address = invitation.url
        code = invitation.code
        fingerprint = invitation.fingerprint
        scanError = nil
        showingManual = true
    }

    private func connect() {
        focusedField = nil
        scanError = nil
        store.error = nil
        Task {
            await store.pair(url: address.trimmingCharacters(in: .whitespacesAndNewlines),
                             code: code.trimmingCharacters(in: .whitespacesAndNewlines),
                             fingerprint: fingerprint.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }
}
