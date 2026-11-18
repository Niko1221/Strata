/**
 * Quoting a turn or a passage into the next message (#4785). The message that goes out carries the quote in one fixed, visible form, so the
 * model knows what is being answered and the transcript can show the quote apart from the reply:
 *
 *   Quoting:
 *   「the first passage」
 *   「the second passage」
 *   → Reply: what the person wrote
 */
export const MAX_QUOTES = 8;
export const MAX_QUOTE_CHARS = 4000;

export interface Quote { id: string; text: string; messageId: string; role: 'user' | 'assistant' }

/** The quote as it is carried: trimmed, bounded, and with the corner brackets of the form turned into other brackets so a quote cannot end itself early. */
export function cleanQuote(text: string): string {
  const flat = String(text ?? '').replace(/\r\n?/g, '\n').replace(/\n{3,}/g, '\n\n').trim().replace(/「/g, '『').replace(/」/g, '』');
  return flat.length > MAX_QUOTE_CHARS ? `${flat.slice(0, MAX_QUOTE_CHARS - 1).trimEnd()}…` : flat;
}

export function composeQuoted(quotes: readonly string[], reply: string): string {
  const list = quotes.map(cleanQuote).filter(Boolean).slice(0, MAX_QUOTES);
  if (!list.length) return reply;
  return `Quoting:\n${list.map(q => `「${q}」`).join('\n')}\n→ Reply: ${reply}`;
}

/** The quotes and the reply of a message made by composeQuoted, or null when the message is not one. */
export function parseQuoted(content: string): { quotes: string[]; reply: string } | null {
  const text = String(content ?? '');
  if (!text.startsWith('Quoting:\n「')) return null;
  const marker = text.indexOf('」\n→ Reply: ');
  if (marker < 0) return null;
  const head = text.slice('Quoting:\n'.length, marker + 1);
  const quotes = [...head.matchAll(/「([^」]*)」/g)].map(m => m[1]);
  if (!quotes.length) return null;
  return { quotes, reply: text.slice(marker + '」\n→ Reply: '.length) };
}

/** Adds a quote to a list: an empty one, a repeat and anything past the limit are left out. */
export function addToQuotes(list: readonly Quote[], quote: Omit<Quote, 'id'> & { id?: string }): Quote[] {
  const text = cleanQuote(quote.text);
  if (!text || list.length >= MAX_QUOTES || list.some(q => q.text === text && q.messageId === quote.messageId)) return [...list];
  return [...list, { id: quote.id ?? `q_${Math.random().toString(36).slice(2, 9)}`, text, messageId: quote.messageId, role: quote.role }];
}
