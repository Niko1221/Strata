import { parseSkillFile, skillSlug } from './skill-files.mjs';

/** Read-only validation. Companion scripts, references and dependencies are not imported. */
export function previewSkillMarkdown(markdown, existing = []) {
  if(typeof markdown!=='string'||Buffer.byteLength(markdown,'utf8')>64*1024)throw new Error('Choose a SKILL.md file within 64 KiB.');
  const skill=parseSkillFile(markdown);
  if(!skill.name||skill.name.length>80||!skillSlug(skill.name))throw new Error('SKILL.md needs a name of up to 80 characters.');
  if(!skill.description?.trim()||skill.description.length>400||/^[|>][-+]?\d?$/.test(skill.description))throw new Error('SKILL.md needs a single-line description of up to 400 characters.');
  if(!skill.body.trim()||skill.body.length>40000)throw new Error('Skill instructions must contain 1–40000 characters.');
  if(existing.some(s=>skillSlug(s.name)===skillSlug(skill.name)))throw new Error('A skill with this name already exists. Rename it before importing.');
  return {name:skill.name,description:skill.description,body:skill.body,triggers:[skill.name.replace(/[-_]+/g,' ')],enabled:false,onlyIn:null};
}
