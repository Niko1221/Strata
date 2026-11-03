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

/* Strata Dev only (index.mjs calls these when STRATA_DEV=1; Live uses selectSkills above). */

const STOP = new Set(('the and for with that this from your you are was were have has had into when what which will would should could about there their them they then than also only just more most some such very over under each other into onto been being does did doing make makes made using used use '
  + 'der die das und oder mit für von auf aus bei ist sind war wird werden wenn dann als auch nur noch eine einer eines einem einen ein mein dein sein ihr ihre unser euer dies diese dieser dieses kann können soll sollen muss müssen nicht kein keine über unter nach vor').split(/\s+/));
const words = text => new Set(String(text ?? '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').split(/[^a-z0-9äöüß]+/).filter(w => w.length >= 4 && !STOP.has(w)));

/** How well a message fits a skill's description (and name): the significant words they share. */
export function descriptionMatch(skill, message) {
  const said = words(message);
  if (!said.size) return 0;
  let shared = 0;
  for (const w of words(`${skill.name ?? ''} ${skill.description ?? ''}`)) if (said.has(w) || [...said].some(s => s.length >= 6 && (s.startsWith(w) || w.startsWith(s)))) shared++;
  return shared;
}

/**
 * The trigger phrases as before, plus every enabled skill whose description shares at least `minShared` significant words with
 * the message (progressive disclosure for the direct chat: the description decides, the instructions follow). `allowed`, when
 * given, is the chat's own choice of skills: nothing else is used.
 */
export function selectSkillsDev(skills, message, workspaceId, { allowed = null, minShared = 2 } = {}) {
  const usable = skills.filter(s => (!allowed || allowed.includes(s.id)));
  const byTrigger = selectSkills(usable, message, workspaceId);
  const picked = new Set(byTrigger.map(s => s.id));
  const byDescription = usable.filter(s => !picked.has(s.id) && s.enabled && (!s.onlyIn || s.onlyIn === workspaceId)
    && typeof s.body === 'string' && s.body.trim() && descriptionMatch(s, message) >= minShared);
  return [...byTrigger, ...byDescription];
}
