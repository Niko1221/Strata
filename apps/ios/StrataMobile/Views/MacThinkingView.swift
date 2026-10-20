import SwiftUI

struct MacThinkingView: View {
    let message: MacChatMessage
    let isLive: Bool
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Button { expanded.toggle() } label: {
                HStack(spacing: 8) {
                    Image(systemName: "sparkle").font(.caption).foregroundStyle(StrataTheme.accentText)
                    Text(message.thinkingTitle(isLive: isLive)).font(.subheadline.weight(.medium))
                    if isLive { Circle().fill(StrataTheme.accent).frame(width: 6, height: 6).accessibilityLabel("Live thinking") }
                    Spacer(minLength: 4)
                    Image(systemName: expanded ? "chevron.up" : "chevron.down").font(.caption2)
                }
                .foregroundStyle(StrataTheme.secondaryInk).contentShape(Rectangle())
            }
            .buttonStyle(.plain).accessibilityIdentifier("thinkingDisclosure-" + message.id)
            .accessibilityValue(isLive ? "Thinking live" : "Recorded thinking")
            if expanded, let reasoning = message.thinkingText {
                Text(reasoning).font(.callout).foregroundStyle(StrataTheme.secondaryInk)
                    .lineSpacing(4).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("thinkingContent-" + message.id)
            } else if isLive, let preview = message.thinkingPreview {
                Text(preview).font(.caption).foregroundStyle(StrataTheme.secondaryInk).lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("thinkingPreview-" + message.id)
            }
        }
        .padding(13).background(StrataTheme.surfaceSecondary.opacity(0.7), in: RoundedRectangle(cornerRadius: 13))
    }
}

struct MacActivityGroupView<Content: View>: View {
    let group: MacActivityGroup
    let tools: [MacToolActivity]
    let isLive: Bool
    let pendingToolIds: Set<String>
    let activity: (MacToolActivity) -> Content
    @State private var expanded = false

    private var accessibilitySummary: String { group.title + (isLive && !group.closed ? ", Live activity" : "") }
    private var needsApproval: Bool { tools.contains { pendingToolIds.contains($0.id) } }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if tools.isEmpty {
                summary(showChevron: false)
                    .accessibilityElement(children: .combine).accessibilityLabel(accessibilitySummary)
                    .accessibilityIdentifier("activityGroup-" + group.id)
            } else {
                Button { expanded.toggle() } label: { summary(showChevron: true) }
                    .buttonStyle(.plain).accessibilityLabel(accessibilitySummary)
                    .accessibilityIdentifier("activityGroup-" + group.id)
                    .accessibilityValue(expanded ? "Expanded" : "Collapsed")
            }
            if expanded {
                VStack(alignment: .leading, spacing: 9) { ForEach(tools) { tool in activity(tool) } }
            }
        }
        .onChange(of: needsApproval, initial: true) { _, needsApproval in if needsApproval { expanded = true } }
    }
    private func summary(showChevron: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if showChevron {
                Image(systemName: "chevron.right").font(.caption2.weight(.semibold)).rotationEffect(.degrees(expanded ? 90 : 0))
            }
            Text(group.displayTitle).font(.subheadline).multilineTextAlignment(.leading)
            if isLive, !group.closed {
                Circle().fill(StrataTheme.accent).frame(width: 6, height: 6)
                    .accessibilityLabel("Live activity").accessibilityIdentifier("liveActivityGroup-" + group.id)
            }
            Spacer(minLength: 0)
            if let added = group.added, added > 0 { Text("+" + added.formatted(.number.precision(.fractionLength(0)))).foregroundStyle(StrataTheme.accentText) }
            if let removed = group.removed, removed > 0 { Text("−" + removed.formatted(.number.precision(.fractionLength(0)))).foregroundStyle(StrataTheme.warning) }
        }
        .font(.caption.monospaced()).foregroundStyle(StrataTheme.secondaryInk).contentShape(Rectangle())
    }

}
