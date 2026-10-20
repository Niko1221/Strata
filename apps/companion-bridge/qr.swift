import Foundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

// Local system frameworks only. No invitation is sent to an online QR service.
guard CommandLine.arguments.count == 3,
      let filter = CIFilter(name: "CIQRCodeGenerator") else { exit(1) }
filter.setValue(Data(CommandLine.arguments[1].utf8), forKey: "inputMessage")
filter.setValue("M", forKey: "inputCorrectionLevel")
guard let image = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 8, y: 8)),
      let rendered = CIContext().createCGImage(image, from: image.extent),
      let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: CommandLine.arguments[2]) as CFURL, UTType.png.identifier as CFString, 1, nil)
else { exit(1) }
CGImageDestinationAddImage(destination, rendered, nil)
guard CGImageDestinationFinalize(destination) else { exit(1) }
