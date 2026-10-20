import SwiftUI

struct ModelsView: View {
    @ObservedObject var store: CompanionStore
    let pair: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                PageHeading(eyebrow: "Your model library", title: "Models", subtitle: "Choose what powers your next idea.")
                if !store.isPaired {
                    ConnectionNeededView(icon: "square.stack.3d.up", title: "Meet your models", message: "Pair your Mac to see its model library and choose what runs next.", pair: pair)
                } else if store.models.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "square.stack.3d.up")
                            .font(.system(size: 34, weight: .light))
                            .foregroundStyle(StrataTheme.accentText)
                        Text(store.isLoading ? "Checking your library…" : "Your library is waiting")
                            .font(.system(.title3, design: .rounded, weight: .semibold))
                            .foregroundStyle(StrataTheme.ink)
                        Text("Add a model or API connection in Strata on your Mac. It will appear here when it’s ready.")
                            .font(.subheadline)
                            .foregroundStyle(StrataTheme.secondaryInk)
                            .multilineTextAlignment(.center)
                            .lineSpacing(4)
                        if store.isLoading { ProgressView().tint(StrataTheme.accentText) }
                    }
                    .frame(maxWidth: .infinity)
                    .strataCard(padding: 28)
                } else {
                    HStack {
                        Text("YOUR LIBRARY")
                            .font(.system(.caption2, weight: .bold))
                            .tracking(1.7)
                            .foregroundStyle(StrataTheme.secondaryInk)
                        Spacer()
                        if store.isLoading {
                            ProgressView().controlSize(.small).tint(StrataTheme.accentText)
                                .accessibilityLabel("Updating models")
                        } else {
                            Text("\(store.models.count) \(store.models.count == 1 ? "model" : "models")")
                                .font(.caption)
                                .foregroundStyle(StrataTheme.secondaryInk)
                        }
                    }
                    VStack(spacing: 14) {
                        ForEach(store.models) { model in modelCard(model) }
                    }
                }
                if store.isPaired {
                    Label("Requests go through your Mac. Your selected provider may run remotely.", systemImage: "desktopcomputer")
                        .font(.caption)
                        .foregroundStyle(StrataTheme.secondaryInk)
                        .lineSpacing(3)
                        .padding(.horizontal, 2)
                }
            }
            .padding(.horizontal, 24)
            .padding(.top, 27)
            .padding(.bottom, 30)
        }
        .background(StrataTheme.background)
        .toolbar(.hidden, for: .navigationBar)
        .refreshable { if store.isPaired { await store.refresh() } }
    }

    private func modelCard(_ model: CompanionModel) -> some View {
        let loaded = ["loaded", "selected", "running", "serving"].contains(model.state.lowercased())
        let active = model.selected ?? loaded
        let apiModel = model.id.hasPrefix("api-")
        let needsDownload = model.state.lowercased() == "available"
        let canChangeModel = !store.isLoading && !store.isSending && store.status?.hostConnected == true
        return VStack(alignment: .leading, spacing: 17) {
            HStack(alignment: .top, spacing: 13) {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 25, weight: .light))
                    .foregroundStyle(active ? StrataTheme.accentText : StrataTheme.ink)
                    .frame(width: 47, height: 47)
                    .background(active ? StrataTheme.accentTint : StrataTheme.surfaceSecondary, in: RoundedRectangle(cornerRadius: 14))
                VStack(alignment: .leading, spacing: 5) {
                    Text(model.name)
                        .font(.system(.headline, design: .rounded))
                        .foregroundStyle(StrataTheme.ink)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 7) {
                        if let size = model.size, !size.isEmpty { Text(size) }
                        if model.size?.isEmpty == false { Text("·") }
                        Text(active ? "In use" : loaded ? "Loaded on Mac" : model.state.capitalized)
                    }
                    .font(.caption)
                    .foregroundStyle(active ? StrataTheme.accentText : StrataTheme.secondaryInk)
                }
                Spacer(minLength: 0)
            }
            if let detail = model.detail, !detail.isEmpty {
                Text(detail).font(.caption).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(3)
            }
            if active {
                Button { Task { await store.unloadModel() } } label: {
                    HStack { Image(systemName: "stop.circle"); Text(apiModel ? "Stop using model" : "Unload model") }
                }
                .buttonStyle(StrataSecondaryButtonStyle())
                .disabled(!canChangeModel)
                .opacity(canChangeModel ? 1 : 0.55)
                .accessibilityIdentifier("unloadModelButton")
            } else {
                Button { Task { await store.loadModel(model.id) } } label: {
                    HStack {
                        Image(systemName: needsDownload ? "arrow.down.circle" : "play.fill")
                        Text(needsDownload ? "Download on your Mac" : apiModel || loaded ? "Use model" : "Load on Mac")
                    }
                }
                .buttonStyle(StrataPrimaryButtonStyle(compact: true))
                .disabled(!canChangeModel || needsDownload)
                .opacity(canChangeModel && !needsDownload ? 1 : 0.55)
                .accessibilityIdentifier("loadModel-\(model.id)")
            }
        }
        .strataCard()
    }
}
