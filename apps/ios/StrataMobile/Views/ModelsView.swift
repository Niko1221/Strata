import SwiftUI

struct ModelsView: View {
    @ObservedObject var store: CompanionStore
    let pair: () -> Void

    private var libraryModels: [CompanionModel] {
        store.models.filter { $0.state.lowercased() != "available" || $0.id.hasPrefix("api-") }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                PageHeading(eyebrow: "Your model library", title: "Models", subtitle: "Choose what powers your next idea.")
                if !store.isPaired {
                    ConnectionNeededView(icon: "square.stack.3d.up", title: "Meet your models", message: "Pair your Mac to see its model library and choose what runs next.", pair: pair)
                } else if libraryModels.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "square.stack.3d.up")
                            .font(.system(size: 34, weight: .light))
                            .foregroundStyle(StrataTheme.accentText)
                        Text(store.isLoading ? "Checking your library…" : "Your library is waiting")
                            .font(.system(.title3, design: .rounded, weight: .semibold))
                            .foregroundStyle(StrataTheme.ink)
                        Text("Download a model below, or add an API connection in Strata on your Mac. Ready models appear here.")
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
                            Text("\(libraryModels.count) \(libraryModels.count == 1 ? "model" : "models")")
                                .font(.caption)
                                .foregroundStyle(StrataTheme.secondaryInk)
                        }
                    }
                    VStack(spacing: 14) {
                        ForEach(libraryModels) { model in modelCard(model) }
                    }
                }
                if store.isPaired {
                    downloadSection
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
        .refreshable {
            if store.isPaired { await store.refresh(); await store.refreshModelManagement(refreshTargets: true) }
        }
        .task(id: store.isPaired) { if store.isPaired { await store.refreshModelManagement(refreshTargets: true) } }
        .sheet(isPresented: Binding(get: { store.reviewingModelDownload != nil }, set: { if !$0 { store.dismissModelDownloadReview() } })) {
            ModelDownloadReviewView(store: store)
        }
    }

    private func modelCard(_ model: CompanionModel) -> some View {
        let loaded = ["loaded", "selected", "running", "serving"].contains(model.state.lowercased())
        let active = model.selected ?? loaded
        let apiModel = model.id.hasPrefix("api-")
        let needsDownload = model.state.lowercased() == "available"
        let localDownloadRunning = store.selectedModelTargetId == "local" && store.modelCatalog?.jobs.contains(where: \.isRunning) == true
        let canChangeModel = !store.isLoading && !store.isSending && !store.isModelDownloadBusy && !localDownloadRunning && store.status?.hostConnected == true
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

    private var downloadSection: some View {
        VStack(alignment: .leading, spacing: 17) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Downloads").font(.system(.title2, design: .rounded, weight: .bold)).foregroundStyle(StrataTheme.ink)
                Text("Get model files on your computer. Nothing is downloaded to this iPhone.")
                    .font(.subheadline).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(3)
            }
            if !store.modelTargets.isEmpty {
                Picker("Download to", selection: Binding(
                    get: { store.modelTargetSelection },
                    set: { id in Task { await store.selectModelTarget(id) } }
                )) {
                    ForEach(store.modelTargets) { target in
                        Label(target.name, systemImage: target.isRemote ? "network" : "desktopcomputer").tag(target.id)
                    }
                }
                .pickerStyle(.menu).tint(StrataTheme.accentText)
                .disabled(store.isModelDownloadBusy)
                .accessibilityIdentifier("modelDownloadTargetPicker")
            }
            if store.isDiscoveringModelTargets || store.isLoadingModelCatalog {
                ProgressView("Checking model downloads…").tint(StrataTheme.accentText)
                    .font(.subheadline).foregroundStyle(StrataTheme.secondaryInk)
            }
            if let message = store.modelManagementMessage {
                VStack(alignment: .leading, spacing: 12) {
                    Label(message, systemImage: "info.circle").font(.subheadline).foregroundStyle(StrataTheme.secondaryInk)
                        .accessibilityIdentifier("modelDownloadsUnavailable")
                    Button("Check again") { Task { await store.refreshModelManagement(refreshTargets: true) } }
                        .font(.subheadline.weight(.semibold)).foregroundStyle(StrataTheme.accentText)
                        .disabled(store.isDiscoveringModelTargets || store.isLoadingModelCatalog)
                }
                .strataCard()
            }
            if let catalog = store.modelCatalog {
                if !catalog.jobs.isEmpty {
                    VStack(spacing: 13) { ForEach(catalog.jobs) { job in downloadJobCard(job, catalog: catalog) } }
                }
                if catalog.canDownload {
                    if catalog.models.isEmpty {
                        Text("No downloadable models are available on this computer.")
                            .font(.subheadline).foregroundStyle(StrataTheme.secondaryInk)
                    } else {
                        VStack(spacing: 14) { ForEach(catalog.models) { model in downloadModelCard(model, catalog: catalog) } }
                    }
                }
            }
        }
    }

    private func downloadModelCard(_ model: DownloadableModel, catalog: ModelManagementCatalog) -> some View {
        let working = catalog.jobs.contains { $0.modelId == model.id && $0.isRunning }
        let canReview = model.canDownload && !working && !store.isModelDownloadBusy && !store.isPlanningModelDownload && store.modelManagementMessage == nil
        return VStack(alignment: .leading, spacing: 13) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "cube.transparent").font(.title2).foregroundStyle(StrataTheme.accentText)
                    .frame(width: 38, height: 38).background(StrataTheme.accentTint, in: RoundedRectangle(cornerRadius: 11))
                VStack(alignment: .leading, spacing: 5) {
                    Text(model.name).font(.headline).foregroundStyle(StrataTheme.ink)
                    Text(model.sizeLabel + (model.readPictures ? " · Picture reading available" : ""))
                        .font(.caption).foregroundStyle(StrataTheme.secondaryInk)
                }
                Spacer(minLength: 0)
            }
            if let detail = model.detail, !detail.isEmpty { Text(detail).font(.caption).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(3) }
            if working {
                Label("Download in progress", systemImage: "arrow.down.circle").font(.subheadline).foregroundStyle(StrataTheme.accentText)
            } else if model.canDownload {
                Button { Task { await store.reviewModelDownload(model, readPictures: model.readPictures) } } label: {
                    Label("Review download", systemImage: "arrow.down.circle")
                }
                .buttonStyle(StrataPrimaryButtonStyle(compact: true)).disabled(!canReview).opacity(canReview ? 1 : 0.55)
                .accessibilityIdentifier("reviewModelDownload-" + model.id)
            } else {
                Label(["installed", "loaded", "ready"].contains(model.state) ? "Downloaded on computer" : model.state.capitalized,
                      systemImage: ["installed", "loaded", "ready"].contains(model.state) ? "checkmark.circle" : "cube")
                    .font(.subheadline).foregroundStyle(StrataTheme.secondaryInk)
            }
        }
        .strataCard()
    }

    private func downloadJobCard(_ job: ModelDownloadJob, catalog: ModelManagementCatalog) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(job.name).font(.headline).foregroundStyle(StrataTheme.ink)
                    Text(job.statusLabel).font(.caption).foregroundStyle(job.state == "error" ? StrataTheme.warning : StrataTheme.accentText)
                }
                Spacer(minLength: 4)
                Image(systemName: job.state == "done" ? "checkmark.circle.fill" : job.state == "error" ? "exclamationmark.circle" : "arrow.down.circle")
                    .foregroundStyle(job.state == "error" ? StrataTheme.warning : StrataTheme.accentText)
            }
            if !job.summary.isEmpty { Text(job.summary).font(.subheadline).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(3) }
            if job.isRunning {
                if let progress = job.progress { ProgressView(value: progress).tint(StrataTheme.accentText) }
                else { ProgressView().tint(StrataTheme.accentText) }
            }
            HStack {
                if let counter = job.counterLabel { Text(counter) }
                Spacer(minLength: 4)
                if let progress = job.progress { Text(progress.formatted(.percent.precision(.fractionLength(0)))) }
            }
            .font(.caption.monospacedDigit()).foregroundStyle(StrataTheme.secondaryInk)
            if let error = job.error, !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(StrataTheme.warning).textSelection(.enabled)
            }
            if job.kind == "download", job.isRunning, job.canCancel == true {
                Button { Task { await store.cancelModelDownload(job) } } label: {
                    Text(store.cancellingModelDownloadId == job.id ? "Stopping…" : "Stop download")
                }
                .buttonStyle(StrataSecondaryButtonStyle()).disabled(store.isModelDownloadBusy)
                .accessibilityIdentifier("cancelModelDownload-" + job.id)
            } else if job.isRunning {
                Text("Manage this download on the computer or device that started it.")
                    .font(.caption).foregroundStyle(StrataTheme.secondaryInk)
            } else if job.canRetry, let model = catalog.models.first(where: { $0.id == job.modelId }) {
                Button { Task { await store.reviewModelDownload(model, readPictures: model.readPictures) } } label: {
                    Label("Review and retry", systemImage: "arrow.clockwise")
                }
                .buttonStyle(StrataSecondaryButtonStyle()).disabled(store.isModelDownloadBusy || store.isPlanningModelDownload)
                .accessibilityIdentifier("retryModelDownload-" + job.id)
            }
        }
        .strataCard()
    }
}

private struct ModelDownloadReviewView: View {
    @ObservedObject var store: CompanionStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 21) {
                    Text(store.reviewingModelDownload?.name ?? "Model download")
                        .font(.system(.title2, design: .rounded, weight: .bold)).foregroundStyle(StrataTheme.ink)
                    if store.isPlanningModelDownload {
                        ProgressView("Checking size, storage, and required steps…").tint(StrataTheme.accentText)
                            .foregroundStyle(StrataTheme.secondaryInk).font(.subheadline)
                    }
                    if let plan = store.modelDownloadPlan {
                        planDetails(plan)
                    }
                    if let error = store.modelDownloadReviewError {
                        Label(error, systemImage: "exclamationmark.circle").font(.subheadline).foregroundStyle(StrataTheme.warning)
                            .accessibilityIdentifier("modelDownloadReviewError")
                    }
                    if store.modelDownloadPlan == nil, !store.isPlanningModelDownload { refreshPlanButton }
                }
                .padding(24)
            }
            .background(StrataTheme.background)
            .navigationTitle("Review download").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Close") { store.dismissModelDownloadReview() }.disabled(store.isStartingModelDownload)
                        .accessibilityIdentifier("dismissModelDownloadReview")
                }
            }
        }
        .interactiveDismissDisabled(store.isStartingModelDownload)
    }

    private func planDetails(_ plan: ModelDownloadPlan) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 15) {
                reviewRow("Download to", value: plan.targetName, icon: "desktopcomputer")
                reviewRow("Download size", value: ModelDownloadFormat.gigabytes(plan.downloadGb), icon: "arrow.down.circle")
                reviewRow("Storage needed", value: ModelDownloadFormat.gigabytes(plan.diskNeededGb), icon: "externaldrive")
                reviewRow("Free storage", value: ModelDownloadFormat.gigabytes(plan.diskFreeGb), icon: "internaldrive")
            }
            .strataCard()
            if store.reviewingModelDownload?.readPictures == true {
                Toggle("Include picture reading files", isOn: Binding(
                    get: { plan.readPictures },
                    set: { value in
                        if let model = store.reviewingModelDownload { Task { await store.reviewModelDownload(model, readPictures: value) } }
                    }
                ))
                .tint(StrataTheme.accent).font(.subheadline).foregroundStyle(StrataTheme.ink)
                .disabled(store.isPlanningModelDownload || store.isStartingModelDownload)
                .accessibilityIdentifier("modelDownloadVisionToggle")
            } else {
                Text("Picture reading files are not included for this model.").font(.caption).foregroundStyle(StrataTheme.secondaryInk)
            }
            ForEach(Array(plan.warnings.enumerated()), id: \.offset) { _, warning in
                Label(warning, systemImage: "info.circle").font(.subheadline).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(3)
            }
            if let shortage = plan.diskShort, !shortage.isEmpty {
                Label(shortage, systemImage: "externaldrive.badge.exclamationmark")
                    .font(.subheadline).foregroundStyle(StrataTheme.warning).accessibilityIdentifier("modelDownloadDiskWarning")
            }
            Text("Model files stay on the selected computer. The download continues if you leave this screen.")
                .font(.caption).foregroundStyle(StrataTheme.secondaryInk).lineSpacing(3)
            TimelineView(.periodic(from: .now, by: 1)) { context in
                VStack(alignment: .leading, spacing: 13) {
                    if plan.isExpired(at: context.date) {
                        Text("This review has expired. Refresh it to check storage and required steps again.")
                            .font(.caption).foregroundStyle(StrataTheme.warning)
                    }
                    Button { Task { await store.startModelDownload() } } label: {
                        HStack {
                            if store.isStartingModelDownload { ProgressView().controlSize(.small) }
                            else { Image(systemName: "arrow.down.circle") }
                            Text(store.isStartingModelDownload ? "Starting download…" : "Download on " + plan.targetName)
                        }
                    }
                    .buttonStyle(StrataPrimaryButtonStyle(compact: true))
                    .disabled(!plan.canStart(at: context.date) || store.isPlanningModelDownload || store.isStartingModelDownload)
                    .accessibilityIdentifier("confirmModelDownloadButton")
                    if plan.isExpired(at: context.date) || store.modelDownloadReviewError != nil { refreshPlanButton }
                }
            }
        }
    }

    private var refreshPlanButton: some View {
        Button("Refresh plan") {
            if let model = store.reviewingModelDownload { Task { await store.reviewModelDownload(model, readPictures: store.modelDownloadPlan?.readPictures ?? model.readPictures) } }
        }
        .buttonStyle(StrataSecondaryButtonStyle()).disabled(store.isPlanningModelDownload || store.isStartingModelDownload)
        .accessibilityIdentifier("refreshModelDownloadPlanButton")
    }

    private func reviewRow(_ title: String, value: String, icon: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon).foregroundStyle(StrataTheme.accentText).frame(width: 22)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.caption).foregroundStyle(StrataTheme.secondaryInk)
                Text(value).font(.subheadline.weight(.medium)).foregroundStyle(StrataTheme.ink)
            }
            Spacer(minLength: 0)
        }
    }
}
