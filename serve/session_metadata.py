"""Bounded metadata-only index of native session format v1.

Mirrors put_payload/put_checkpoint in src/core/conversation_file.cpp. It skips
state arrays and never loads KV. The native RESTORE reader remains responsible
for full payload hashes, execution identity and state-layout validation before
applying anything to the GPU. Unknown formats are not guessed.
"""
import struct


def session_prefixes(path, max_tokens):
    with open(path,"rb") as f:
        f.seek(0,2)
        size=f.tell()
        f.seek(0)
        header=f.read(64)
        if len(header)!=64 or header[:8]!=b"STRSESS\x01":
            raise ValueError("unknown native session magic")
        version,hsize=struct.unpack_from('<II',header,8)
        payload=struct.unpack_from('<Q',header,32)[0]
        if version!=1 or hsize!=64 or payload+80!=size or header[40:56]!=bytes(16):
            raise ValueError("unknown or invalid native session header")
        end=64+payload
        def read(n):
            if n<0 or n>end-f.tell():
                raise ValueError("native session metadata exceeds payload")
            data=f.read(n)
            if len(data)!=n:
                raise ValueError("truncated native session metadata")
            return data
        def count(limit):
            n=struct.unpack('<Q',read(8))[0]
            if n>limit:
                raise ValueError("native session metadata count exceeds limit")
            return n
        def skip(n):
            if n<0 or n>end-f.tell():
                raise ValueError("native session state exceeds payload")
            f.seek(n,1)
        skip((18+3)*8)  # geometry[18], layer_lo, layer_hi, cvec
        def checkpoint():
            n=count(max_tokens)
            ids=list(struct.unpack('<'+'i'*n,read(n*4)))
            if any(t<0 for t in ids):
                raise ValueError("negative native token ID")
            images=count(max_tokens)
            skip(images*16)
            for _ in range(5):  # gdn, ple, tails, dead, block_pos: byte vectors
                skip(count(end-f.tell()))
            skip(8)  # use counter
            return ids
        live=checkpoint()
        result=[live]
        for _ in range(count(256)):
            ids=checkpoint()
            if ids!=live[:len(ids)]:
                raise ValueError("native checkpoint is not an ancestor of live state")
            if ids and ids not in result:
                result.append(ids)
        return result
