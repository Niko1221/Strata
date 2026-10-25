import type { Session, Workspace } from './types';

/** The workspace a chat was made in: the sidebar groups chats by this same rule (the saved path, or an earlier folder of the workspace). */
export function workspaceIdForSession(workspaces: Workspace[], session: Pick<Session, 'workspace'> | null | undefined): string | null {
  const path = session?.workspace;
  if (!path) return null;
  return workspaces.find(w => w.path === path || !!w.aliases?.includes(path))?.id ?? null;
}
