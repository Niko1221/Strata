import type { ToolCall } from './types';

/** The actions of one chat that wait for an answer, oldest first (the approval dock above the message box shows them, #4725). */
export function pendingApprovals(session: { messages: { id: string; tools?: ToolCall[] }[] } | null | undefined): { messageId: string; tool: ToolCall }[] {
    const out: { messageId: string; tool: ToolCall }[] = [];
    for (const m of session?.messages ?? [])
        for (const tool of m.tools ?? [])
            if (tool.state === 'proposed')
                out.push({ messageId: m.id, tool });
    return out;
}
