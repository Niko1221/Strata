---
name: coordinate-microjobs
description: "Coordinate bounded helper jobs across software audits or live database maintenance."
---

# Coordinate Microjobs

Make each job small enough to verify independently. Record its goal, project and revision, allowed write surface, evidence requirement, and completion condition. Parallelize independent reads or disjoint work. Keep a single owner for any overlapping write surface or serialized live writer. Inspect changed areas when accepting helper results.

On a live correction, identify affected jobs, update their briefs, and preserve unrelated work. Before restarting a slow writer, use its latest receipt or progress log. Finish only after reconciling helper evidence with the current workspace or canonical state.
