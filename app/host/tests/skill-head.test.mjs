/** #6781 / #6789: the system prompt is the head of every prompt the engine caches. Skills picked by the words of a message must not be in it. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { selectSkills, skillInstructions, withSkills, historyWithSkills } from '../server/skill-selection.mjs';

const SYSTEM = 'You are Strata, a helpful assistant.';
const skills = [
  { id: 'rev', name: 'Review', enabled: true, triggers: ['review'], body: 'Check correctness first.' },
  { id: 'doc', name: 'Docs', enabled: true, triggers: ['readme'], body: 'Write in plain English.' },
  { id: 'off', name: 'Off', enabled: false, triggers: ['review'], body: 'Never.' },
];

/** The messages a request carries, as the plain chat path builds them (inference.mjs proxyCompletion): system, history, the new user message. */
function request(history, message, workspaceId = 'w') {
  return [{ role: 'system', content: SYSTEM }, ...historyWithSkills(skills, history, workspaceId), { role: 'user', content: withSkills(skills, message, workspaceId) }];
}

/** What the host did before: the picked skills in the system prompt. */
function oldRequest(history, message, workspaceId = 'w') {
  const system = [SYSTEM, skillInstructions(selectSkills(skills, message, workspaceId))].filter(Boolean).join('\n\n');
  return [{ role: 'system', content: system }, ...history, { role: 'user', content: message }];
}

const startsWith = (next, prev) => prev.every((m, i) => JSON.stringify(m) === JSON.stringify(next[i]));

function session(build) {
  const turns = ['hello', 'please review this function', 'thanks, now the readme', 'and a plain question', 'review it again'];
  const history = [], prompts = [];
  for (const text of turns) {
    prompts.push(build(history, text));
    history.push({ role: 'user', content: text }, { role: 'assistant', content: `answer to: ${text}` });
  }
  return prompts;
}

test('with skills picked per message, every request starts with the previous one (the head never moves)', () => {
  const prompts = session(request);
  for (let i = 1; i < prompts.length; i++) {
    // the previous request ends with its own user message and then the answer is added: the new request must contain the previous one
    // up to and including that user message
    assert.ok(startsWith(prompts[i], prompts[i - 1]), `request ${i + 1} does not start with request ${i}`);
  }
  assert.equal(prompts[1][0].content, SYSTEM);                                  // the head is the plain system prompt on a skill turn too
  assert.match(prompts[1].at(-1).content, /Check correctness first\./);        // the skill reached the model, with the message
  assert.doesNotMatch(prompts[2].at(-1).content, /Check correctness first\./);  // ... only the skill the words of THIS message pick
});

test('the old placement fails the same check: the head changed on a skill turn', () => {
  const prompts = session(oldRequest);
  assert.notEqual(prompts[1][0].content, prompts[0][0].content);
  assert.equal(startsWith(prompts[1], prompts[0]), false);
});

test('a message no skill picks, and a disabled skill, add nothing', () => {
  assert.equal(withSkills(skills, 'hello there', 'w'), 'hello there');
  assert.equal(selectSkills(skills, 'review', 'w').some(s => s.id === 'off'), false);
  assert.deepEqual(historyWithSkills([], [{ role: 'user', content: 'review' }], 'w'), [{ role: 'user', content: 'review' }]);
  assert.deepEqual(historyWithSkills(skills, [{ role: 'assistant', content: 'review' }], 'w'), [{ role: 'assistant', content: 'review' }]);
});

test('no plain chat path of the host puts skill instructions in the system prompt', () => {
  const src = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.equal((src.match(/skillInstructions\(/g) ?? []).length, 0);
  assert.equal((src.match(/withSkills\(skillState/g) ?? []).length, 3);
  assert.equal((src.match(/historyWithSkills\(skillState/g) ?? []).length, 3);
});
