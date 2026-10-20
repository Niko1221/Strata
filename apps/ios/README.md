# Strata for iPhone

A native SwiftUI companion for the Strata desktop app. It uses Strata's emerald, white and charcoal palette in light and dark mode. Requires iOS 17 or later.

This app requires the matching Mac iPhone-pairing and shared-conversation support in `apps/companion-bridge` and `apps/host`. Build the Mac companion changes first; the unmodified desktop release does not expose this API. Pairing is opt-in on the Mac.

## Connect

1. Keep Strata open on your Mac.
2. In the Mac Strata app, choose **Pair iPhone** in the sidebar, or **Settings → iPhone**, then choose **Enable iPhone pairing**.
3. Open **Strata** on iPhone, choose **Pair your Mac**, and scan the Mac's QR code. Check the address and choose **Pair with Mac**.
4. Allow local network access. Keep both devices on the same trusted network.

Each invitation expires after five minutes and works once. Use **New pairing code** on the Mac for another device. The camera is optional: enter the HTTPS address, six-digit code, and certificate fingerprint manually instead. Strata remembers that pairing is enabled and restores the listener on its next launch. **Disable pairing** closes the listener while preserving paired phones for later.

## What it controls

- Live connection and model status from the Mac.
- The Mac's configured API models, with credentials kept on the Mac.
- Loading and unloading models that are already installed locally. Download models in the desktop app first.
- The Mac's existing conversations, their saved message history, and current work.
- Continuing the same conversation and workspace from either device.
- The Mac's Thinking text in an expandable panel, with a live preview and recorded duration when available.
- Progress summaries between reply paragraphs, in the same order as the Mac.
- Live browser and tool activity, including the action, its target, status, and expandable results.
- Allowing or denying an action when the Mac asks for approval, and stopping the selected conversation.

Open **Chat → Conversation history** to choose a Mac conversation. Home also shows recent conversations. The active or most recent conversation opens automatically after connecting. New conversations started on the phone also appear on the Mac. Existing conversations keep their model and permission mode; new conversations start in the Mac's default workspace and ask before actions. Pairing already grants this access, so a paired phone does not need a new invitation after updating both apps.

Tap **Thinking** or **Thought for…** above a reply to read the Thinking text shared by the Mac. The live preview follows the Mac's current Thinking; it changes to a recorded panel once the answer begins. Progress such as **Ran 3 commands** stays between the relevant reply paragraphs. Expand a progress row when the Mac identifies its actions; other action cards remain visible below the reply. Older shared-conversation snapshots without Thinking or ordered progress still show their saved replies and activity cards.

While Chat is open in the foreground, its history and activity refresh about once per second. Other status and conversation lists refresh about every five seconds. On iOS 18 or later, scrolling back through history pauses automatic scrolling; choose **Latest** to follow new replies again. On iOS 17, use **Latest** to jump to current work; live updates leave the scroll position alone. Leaving Chat or putting the phone away does not stop the Mac's agent. Browser activity describes what the Mac is doing; it does not stream the Mac's screen. **Open page on iPhone** opens a reported web address locally.

Requests go through the Mac. A configured API provider may run remotely. Earlier phone-only chats remain available under **Earlier iPhone conversations**; continuing one uses its original text-only chat path.

**Connection → Disconnect Mac** revokes the phone's access and removes its local credential. **Forget this Mac** is available when the Mac is unreachable; it removes the credential only from the phone. Remove that device in the Mac pairing window to revoke its old session. **Pair again** also recovers a changed certificate or address.

## Build

Install Xcode and XcodeGen, then:

```sh
cd apps/ios
xcodegen generate
open StrataMobile.xcodeproj
```

Select the `StrataMobile` scheme and an installed iPhone simulator. No signing team is configured in the project. To install on your own device, select your development team under Signing & Capabilities, or set `STRATA_DEVELOPMENT_TEAM` when using the install script. Device installation requires a development profile containing your phone and the `com.strata.companion.ios` app identifier. If that identifier is unavailable to your team, set your own bundle identifier and use a matching profile. No paid dependencies or third-party Swift packages are used.

For a connected, unlocked iPhone, `STRATA_DEVELOPMENT_TEAM=<your-team> ./install-iphone.sh <device-identifier>` builds, installs, and launches the app. Set `STRATA_PROVISIONING_PROFILE` to an existing dedicated profile name when using manual signing. Set `STRATA_LAUNCH_AFTER_INSTALL=0` to install without opening the app.

## Verify

```sh
xcodebuild -project StrataMobile.xcodeproj -scheme StrataMobile \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:StrataMobileTests test
xcodebuild -project StrataMobile.xcodeproj -scheme StrataMobile \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -only-testing:StrataMobileUITests/CompanionSmokeTests/testHomePairingAndUnpairedTabs \
  -only-testing:StrataMobileUITests/CompanionSmokeTests/testManualPairingRejectsInsecureAddress \
  -only-testing:StrataMobileUITests/CompanionSmokeTests/testManualPairingRejectsMalformedFingerprint \
  -only-testing:StrataMobileUITests/CompanionSmokeTests/testManualPairingRejectsInvalidCode test
```

Choose an available simulator in the commands above. The [synthetic shared-conversation fixture](test-support/README.md) runs the full history, Thinking, ordered progress, browser, approval, continuation, Stop, and relaunch flow using the real TLS bridge and a synthetic host. It starts no model, browser, or desktop agent.

The optional real Mac read-only test takes `STRATA_READONLY_INVITATION`. It pairs, reads existing history, and verifies saved pairing after relaunch, without sending a model message or approving/stopping work.

The optional live TLS test requires `STRATA_TEST_INVITATION`; it checks the real certificate without sending a model message. The optional pairing/chat test uses a fresh invitation and the real selected model, creates a new QA conversation, and leaves the simulator paired. Run only that intended test with each one-use invitation. For command-line XCTest runs, prefix these environment variable names with `TEST_RUNNER_` so Xcode forwards them to the runner. Live UI tests use **Forget this Mac** locally before pairing; remove old simulator entries in the Mac pairing window when finished. Keep result bundles private because XCTest records entered pairing fields.

## Connection security

The QR invitation includes the SHA-256 fingerprint of the Mac's TLS certificate. The app checks that exact certificate before transmitting the pairing code or token, and refuses redirects. The phone saves its connection in the device-only Keychain. The bridge stores token hashes, checks every request, limits pairing guesses, and supports immediate revocation. The control routes are deliberately narrow; the desktop's whole API is never exposed to the phone.
