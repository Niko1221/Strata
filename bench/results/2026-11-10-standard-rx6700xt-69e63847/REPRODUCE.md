# Reproduce this custom-track draft run

Use the exact Terminal and signed RX6000 component revisions named in README. Obtain the two IQ2_XS shards from the pinned ISTA-DASLab repository and verify their hashes. Preserve the existing pack, draft vocabulary, expert profile and Q4-selector draft pack identities.

The following public JSON files retain the complete input structures with operational paths replaced:

- `provenance/run-config.json`: the source serving configuration, including drafting, tokenizer path and worker options.
- `provenance/profile.json`: the isolated profile and its source-config path.
- `provenance/command.json`: the exact official CLI argument array; quoting is unambiguous in this array.
- `provenance/chat_template.jinja`: the effective custom template.

Choose an empty local campaign directory. Replace `<campaign>`, `<terminal>` and `<engine-component>` in the JSON string values with its absolute path and the exact installed component paths. Put the source config at `<campaign>/standards-r2/config.json` and the profile at `<campaign>/standards-r2/profiles/benchmark-standard-r2.json`. Copy the original IQ2_XS tokenizer files into `<campaign>/standards-r2/tokenizer`, then replace only its `chat_template.jinja` with the attached file. The attached pre-run manifest records all original tokenizer file hashes; do not change model tensor or tokenizer ID data.

Set `STRATA_PROFILE_DIR` to the new isolated profiles directory, verify no server is listening on the profile port, and invoke the argument array from `provenance/command.json` using a subprocess without a shell. The output directory must not already exist. This starts a fresh engine, runs the unmodified 22-request official draft suite, and stops that engine. Do not reuse a previously exercised server or retry individual failed requests. Run `strata bench report <new-result-directory>` afterwards.

The raw CLI has identity-reporting omissions covered by the attached metadata corrections: its model-card constant is null even though the reference was pinned before this run, and the preview model list names the bf16-selector fallback while the actual resolved engine uses Q4-selector. Preserve newly generated raw output, capture the actual engine arguments and pack SHA256, and record any factual report metadata corrections separately. Verify the complete pack file hashes in `provenance/native-pack-identities.json`, including dense weights and pack indices; original runner discovery did not include these files. Do not alter measured requests, seeds, caps, scores, timing or cache limits.
