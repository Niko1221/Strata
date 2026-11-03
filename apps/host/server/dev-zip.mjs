/**
 * Strata Dev (docs/DEV_BUILD.md): a small .zip reader and writer for importing and exporting skills, with node:zlib only.
 * Writes deflated entries; reads stored and deflated ones (what Windows' "Send to compressed folder", 7-Zip and Info-ZIP make).
 * No zip64, no encryption. A path that would leave the target folder (`..`, an absolute path) is refused.
 */
import { deflateRawSync, inflateRawSync } from 'node:zlib';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
export function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** `entries`: [{ name: 'skill/SKILL.md', data: Buffer }] -> the .zip as one Buffer. */
export function writeZip(entries) {
  const locals = [], centrals = [];
  let offset = 0;
  for (const { name, data } of entries) {
    const nameBuf = Buffer.from(name.replace(/\\/g, '/'), 'utf8');
    const raw = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const packed = deflateRawSync(raw);
    const crc = crc32(raw);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x0800, 6); local.writeUInt16LE(8, 8);
    local.writeUInt32LE(0, 10); local.writeUInt32LE(crc, 14); local.writeUInt32LE(packed.length, 18); local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(nameBuf.length, 26); local.writeUInt16LE(0, 28);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(8, 10); central.writeUInt32LE(0, 12); central.writeUInt32LE(crc, 16); central.writeUInt32LE(packed.length, 20);
    central.writeUInt32LE(raw.length, 24); central.writeUInt16LE(nameBuf.length, 28); central.writeUInt32LE(offset, 42);
    locals.push(local, nameBuf, packed);
    centrals.push(central, nameBuf);
    offset += local.length + nameBuf.length + packed.length;
  }
  const centralBuf = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralBuf.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, centralBuf, end]);
}

/** A safe relative path inside the archive, or null for one that would leave the target folder. */
export function safeEntryName(name) {
  const clean = String(name).replace(/\\/g, '/');
  if (!clean || clean.startsWith('/') || /^[A-Za-z]:/.test(clean) || clean.split('/').some(p => p === '..')) return null;
  return clean;
}

/** The files of a .zip: [{ name, data }] (folders left out). Throws for a file that is not a zip or holds an unsafe path. */
export function readZip(buf, { maxBytes = 200 * 1024 * 1024 } = {}) {
  let end = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) if (buf.readUInt32LE(i) === 0x06054b50) { end = i; break; }
  if (end < 0) throw new Error('This file is not a .zip archive.');
  const count = buf.readUInt16LE(end + 10);
  let at = buf.readUInt32LE(end + 16);
  const out = [];
  let total = 0;
  for (let n = 0; n < count; n++) {
    if (buf.readUInt32LE(at) !== 0x02014b50) throw new Error('The .zip archive is damaged.');
    const method = buf.readUInt16LE(at + 10), size = buf.readUInt32LE(at + 20), rawSize = buf.readUInt32LE(at + 24);
    const nameLen = buf.readUInt16LE(at + 28), extraLen = buf.readUInt16LE(at + 30), commentLen = buf.readUInt16LE(at + 32), localAt = buf.readUInt32LE(at + 42);
    const name = buf.subarray(at + 46, at + 46 + nameLen).toString('utf8');
    at += 46 + nameLen + extraLen + commentLen;
    if (name.endsWith('/')) continue;
    const safe = safeEntryName(name);
    if (!safe) throw new Error(`The .zip archive holds a path outside its folder: ${name}`);
    total += rawSize;
    if (total > maxBytes) throw new Error('The .zip archive unpacks to more than 200 MB.');
    const dataAt = localAt + 30 + buf.readUInt16LE(localAt + 26) + buf.readUInt16LE(localAt + 28);
    const packed = buf.subarray(dataAt, dataAt + size);
    const data = method === 0 ? Buffer.from(packed) : method === 8 ? inflateRawSync(packed) : null;
    if (!data) throw new Error(`The .zip archive uses a packing method this reader does not know (${method}).`);
    out.push({ name: safe, data });
  }
  return out;
}
