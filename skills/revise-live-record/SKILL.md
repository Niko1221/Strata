---
name: revise-live-record
description: "Apply an authorized revision to a canonical live record and verify it through independent readback."
---

# Revise Live Record

Resolve the project route, owner, allowed mutation, authorizing actor, current revision, and content hash. Use the owner's atomic expected-revision or hash guard and append a new revision. Keep the executing agent's attribution separate from human authorization. On a mismatch, reread and reconcile the new owner state before retrying. If the authorized guarded route is unavailable, record the concrete blocker.

After the write, read again through a fresh connection and verify revision, body hash, links and affected children. Retry a failed projection as a projection. Use the successful mutation receipt to avoid applying the same change twice.
