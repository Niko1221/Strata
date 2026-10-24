import SwiftUI

enum StrataTheme {
    // These colors follow apps/host/src/styles/kit/tokens.css.
    static let background = adaptive(0xF9FAFB, 0x0E1113)
    static let surface = adaptive(0xFFFFFF, 0x161A1D)
    static let surfaceSecondary = adaptive(0xF7F7F7, 0x1D2226)
    static let line = adaptive(0xDEE5EE, 0x2A3036)
    static let ink = adaptive(0x221E1F, 0xEEF1F3)
    static let secondaryInk = adaptive(0x6F6F6F, 0x8E979F)
    static let accent = adaptive(0x10B981, 0x34D399)
    static let accentText = adaptive(0x0C8A60, 0x4ADE80)
    static let accentTint = adaptive(0xECFDF5, 0x15362D)
    static let accentInk = Color(hex: 0x052E20)
    static let warning = adaptive(0x8A5A00, 0xF8C96A)

    private static func adaptive(_ light: UInt, _ dark: UInt) -> Color {
        Color(uiColor: UIColor { traits in
            UIColor(rgb: traits.userInterfaceStyle == .dark ? dark : light)
        })
    }
}

private extension UIColor {
    convenience init(rgb: UInt) {
        self.init(red: Double((rgb >> 16) & 0xFF) / 255,
                  green: Double((rgb >> 8) & 0xFF) / 255,
                  blue: Double(rgb & 0xFF) / 255, alpha: 1)
    }
}

extension Color {
    init(hex: UInt) {
        self.init(uiColor: UIColor(rgb: hex))
    }
}

struct StrataPrimaryButtonStyle: ButtonStyle {
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.body, design: .rounded, weight: .semibold))
            .frame(maxWidth: .infinity)
            .padding(.vertical, compact ? 13 : 17)
            .foregroundStyle(StrataTheme.accentInk)
            .background(StrataTheme.accent, in: RoundedRectangle(cornerRadius: 17))
            .opacity(configuration.isPressed ? 0.78 : 1)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
    }
}

struct StrataSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.subheadline, weight: .semibold))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .foregroundStyle(StrataTheme.ink)
            .background(StrataTheme.surfaceSecondary, in: RoundedRectangle(cornerRadius: 15))
            .opacity(configuration.isPressed ? 0.6 : 1)
    }
}

extension View {
    func strataCard(padding: CGFloat = 20) -> some View {
        self.padding(padding)
            .background(StrataTheme.surface, in: RoundedRectangle(cornerRadius: 25))
            .overlay {
                RoundedRectangle(cornerRadius: 25)
                    .strokeBorder(StrataTheme.line.opacity(0.65), lineWidth: 0.7)
            }
    }
}
