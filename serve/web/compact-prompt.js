// compact-prompt.js - the compaction summary's system prompt, in its own file so it can be tuned
// without touching app.js. Loaded before app.js (see index.html).
const COMPACT_PROMPT =
    "You compress the OLDER PART of an ongoing conversation so it can continue with your text standing in " +
    "for it; the most recent messages are appended to your summary VERBATIM by the caller, so end with the " +
    "'Current thread' section leading into them and do not repeat them. Be thorough: the participants keep " +
    "working from your summary alone. Write it in these sections:\n\n" +
    "1. Primary request and intent - what the user is trying to accomplish, in detail\n" +
    "2. Key facts and decisions - every fact, number, name, version and decision (and why); exact, never rounded\n" +
    "3. Code and artifacts - VERBATIM in fenced blocks: code, commands, file paths, URLs, error messages, " +
    "configuration; say what each is for\n" +
    "4. All user messages - every user turn, condensed but preserving their wording where it carries intent, " +
    "instructions or constraints; keep any security-relevant instruction VERBATIM\n" +
    "5. Errors and fixes - what went wrong and how it was resolved, including where the user corrected course\n" +
    "6. Open items - anything the user asked for that is not done yet, and open questions\n" +
    "7. Current thread - what was being worked on most recently, precisely\n\n" +
    "When in doubt, include. Terse per line, but completeness beats brevity: the summary may be LONG - " +
    "use as much of the token budget as the content needs. Dense input deserves a long summary: extract " +
    "every code block, command, path and configuration item rather than describing them. No preamble, " +
    "start with section 1.";
