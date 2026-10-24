import SwiftUI
import AVFoundation
import UIKit

struct QRScannerView: View {
    let onScan: (String) -> Void
    @State private var permission = AVCaptureDevice.authorizationStatus(for: .video)
    var body: some View {
        Group {
            if permission == .authorized {
                CameraScanner(onScan: onScan)
                    .overlay(alignment: .bottom) {
                        Text("Point your camera at the QR code on your Mac")
                            .font(.callout).padding(20).foregroundStyle(.white)
                            .frame(maxWidth: .infinity).background(.black.opacity(0.65))
                    }
                    .accessibilityLabel("Pairing QR code scanner")
            } else if permission == .denied || permission == .restricted {
                ContentUnavailableView {
                    Label("Camera access needed", systemImage: "camera")
                } description: {
                    Text("Allow camera access in Settings, or enter the pairing details manually.")
                } actions: {
                    Button("Open Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                    }
                }
            } else {
                ProgressView("Opening camera…")
            }
        }
        .task {
            if permission == .notDetermined {
                _ = await AVCaptureDevice.requestAccess(for: .video)
                permission = AVCaptureDevice.authorizationStatus(for: .video)
            }
        }
    }
}

private struct CameraScanner: UIViewControllerRepresentable {
    var onScan: (String) -> Void
    func makeUIViewController(context: Context) -> ScannerController { ScannerController(onScan: onScan) }
    func updateUIViewController(_ controller: ScannerController, context: Context) { }
    static func dismantleUIViewController(_ controller: ScannerController, coordinator: ()) { controller.stop() }
}

private final class ScannerController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    private let capture = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.strata.companion.camera")
    private var preview: AVCaptureVideoPreviewLayer?
    private var delivered = false
    private let onScan: (String) -> Void
    init(onScan: @escaping (String) -> Void) { self.onScan = onScan; super.init(nibName: nil, bundle: nil) }
    required init?(coder: NSCoder) { fatalError("init(coder:) is unavailable") }
    override func viewDidLoad() {
        super.viewDidLoad(); view.backgroundColor = .black
        guard let device = AVCaptureDevice.default(for: .video), let input = try? AVCaptureDeviceInput(device: device), capture.canAddInput(input) else {
            showUnavailable(); return
        }
        capture.addInput(input)
        let output = AVCaptureMetadataOutput()
        guard capture.canAddOutput(output) else { showUnavailable(); return }
        capture.addOutput(output); output.setMetadataObjectsDelegate(self, queue: .main); output.metadataObjectTypes = [.qr]
        let preview = AVCaptureVideoPreviewLayer(session: capture); preview.videoGravity = .resizeAspectFill
        self.preview = preview; view.layer.addSublayer(preview)
        queue.async { [capture] in capture.startRunning() }
    }
    override func viewDidLayoutSubviews() { super.viewDidLayoutSubviews(); preview?.frame = view.bounds }
    func stop() { queue.async { [capture] in if capture.isRunning { capture.stopRunning() } } }
    func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject], from connection: AVCaptureConnection) {
        guard !delivered, let code = metadataObjects.compactMap({ ($0 as? AVMetadataMachineReadableCodeObject)?.stringValue }).first,
              (try? PairingInvitation.parse(code)) != nil else { return }
        delivered = true; stop(); onScan(code)
    }
    private func showUnavailable() {
        let label = UILabel(); label.text = "Camera unavailable. Enter your Mac’s pairing details manually."; label.numberOfLines = 0
        label.textAlignment = .center; label.textColor = .white; label.translatesAutoresizingMaskIntoConstraints = false; view.addSubview(label)
        NSLayoutConstraint.activate([label.centerYAnchor.constraint(equalTo: view.centerYAnchor), label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24), label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24)])
    }
}
