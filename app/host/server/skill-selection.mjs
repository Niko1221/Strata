/** Match enabled instructions within their saved workspace scope. */
export function selectSkills(skills, message, workspaceId) {
  const text = String(message ?? '').toLowerCase();
  return skills.filter(skill => skill.enabled && (!skill.onlyIn || skill.onlyIn === workspaceId)
    && typeof skill.body === 'string' && skill.body.trim()
    && skill.triggers?.some(trigger => trigger && text.includes(String(trigger).toLowerCase())));
}

export function skillInstructions(skills) {
  return skills.map(skill => `Skill: ${skill.name}\n${skill.body.trim()}`).join('\n\n');
}

/**
 * Skills are chosen by the words of the message, so they must not sit in the system prompt: that is the head of every prompt, and a head that
 * changes with the message makes the engine read the whole conversation again (#6781, #6789). They go at the end of the user message they were
 * chosen for. The choice is a pure function of that message's text, so `historyWithSkills` gives every earlier user message of the history the
 * same block it had when it was the newest one: each request's prompt then starts with the previous request's, byte for byte.
 */
export function withSkills(skills, message, workspaceId) {
  if (typeof message !== 'string') return message;
  const block = skillInstructions(selectSkills(skills, message, workspaceId));
  return block ? `${message}\n\n[Instructions for this message]\n${block}` : message;
}

export function historyWithSkills(skills, history, workspaceId) {
  if (!Array.isArray(history) || !skills?.length) return history;
  return history.map(m => (m && m.role === 'user' && typeof m.content === 'string' ? { ...m, content: withSkills(skills, m.content, workspaceId) } : m));
}
