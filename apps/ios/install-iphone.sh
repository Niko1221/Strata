#!/bin/bash
set -euo pipefail

# Build, install, and launch on an explicitly chosen, connected iPhone.
# Uses project automatic signing, or an existing development profile if supplied.
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_device_id="${1:-${STRATA_DEVICE_ID:-}}"

if [[ -z "$task_device_id" ]]; then
    echo "Usage: ./apps/ios/install-iphone.sh <iPhone device identifier>"
    echo "Connect and unlock your iPhone, then use an identifier from this list:"
    xcrun devicectl list devices
    exit 2
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "iPhone installation requires a Mac with Xcode." >&2
    exit 1
fi

task_derived_data="${STRATA_DERIVED_DATA:-$task_script_dir/build-device}"
# devicectl displays a CoreDevice identifier; xcodebuild requires the hardware UDID.
task_device_info="$(mktemp -t strata-device-info)"
trap 'rm -f "$task_device_info"' EXIT
chmod 600 "$task_device_info"
xcrun devicectl device info details --device "$task_device_id" --json-output "$task_device_info" >/dev/null
task_xcode_device_id="$(/usr/bin/plutil -extract result.hardwareProperties.udid raw -o - "$task_device_info")"
task_build_args=(
    -project "$task_script_dir/StrataMobile.xcodeproj"
    -scheme StrataMobile
    -configuration Debug
    -destination "id=$task_xcode_device_id"
    -derivedDataPath "$task_derived_data"
    -allowProvisioningUpdates
    -allowProvisioningDeviceRegistration
)

if [[ -n "${STRATA_DEVELOPMENT_TEAM:-}" ]]; then
    task_build_args+=("DEVELOPMENT_TEAM=$STRATA_DEVELOPMENT_TEAM")
fi
if [[ -n "${STRATA_PROVISIONING_PROFILE:-}" ]]; then
    task_build_args+=("CODE_SIGN_STYLE=Manual" "PROVISIONING_PROFILE_SPECIFIER=$STRATA_PROVISIONING_PROFILE" "CODE_SIGN_IDENTITY=Apple Development")
fi

echo "Building Strata for your selected iPhone…"
xcodebuild "${task_build_args[@]}" build

task_app_bundle="$task_derived_data/Build/Products/Debug-iphoneos/StrataMobile.app"
if [[ ! -d "$task_app_bundle" ]]; then
    echo "The iPhone app was not produced at the expected build location." >&2
    exit 1
fi

echo "Installing Strata on your iPhone…"
xcrun devicectl device install app --device "$task_device_id" "$task_app_bundle"

if [[ "${STRATA_LAUNCH_AFTER_INSTALL:-1}" == "1" ]]; then
    task_bundle_id="$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$task_app_bundle/Info.plist")"
    xcrun devicectl device process launch --device "$task_device_id" "$task_bundle_id"
fi

echo "Strata is installed. Choose Pair iPhone in Strata on your Mac to pair it."
