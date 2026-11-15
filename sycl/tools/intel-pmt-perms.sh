#!/bin/sh
# Hand an Intel PMT telemetry blob to the render group, called from 99-intel-pmt-telemetry.rules.
#
# The kernel's pmt_class creates these attributes 0440 root:root and after device_add, so the uevent can arrive
# before the file exists: wait for it, then chgrp.  Crashlog blobs under the same subsystem stay root-only.
set -eu
name="${1:-}"
[ -n "$name" ] || exit 0
file="/sys/class/intel_pmt/$name/telem"
group="${PMT_GROUP:-render}"
i=0
while [ ! -e "$file" ] && [ "$i" -lt 60 ]; do
    sleep 0.05
    i=$((i + 1))
done
[ -e "$file" ] || exit 0                       # a blob the kernel never created is not an error
chgrp "$group" "$file" 2>/dev/null || exit 0   # no such group: leave the node as it was
chmod 0440 "$file" 2>/dev/null || true         # sysfs may refuse the mode; the group is what sysman needs
exit 0
