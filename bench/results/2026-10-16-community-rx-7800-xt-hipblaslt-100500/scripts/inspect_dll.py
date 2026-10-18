import pathlib,struct,json
def exports(path):
    b=pathlib.Path(path).read_bytes(); pe=struct.unpack_from('<I',b,60)[0]
    sections=struct.unpack_from('<H',b,pe+6)[0];optlen=struct.unpack_from('<H',b,pe+20)[0]
    opt=pe+24;dd=opt+(112 if struct.unpack_from('<H',b,opt)[0]==0x20b else 96)
    exp=struct.unpack_from('<I',b,dd)[0]; sec=opt+optlen
    def off(rva):
        for i in range(sections):
            vs,va,sz,fp=struct.unpack_from('<IIII',b,sec+40*i+8)
            if va<=rva<va+max(vs,sz):return fp+rva-va
        raise ValueError(rva)
    p=off(exp);count=struct.unpack_from('<I',b,p+24)[0];names=off(struct.unpack_from('<I',b,p+32)[0])
    out=[]
    for i in range(count):
        p=off(struct.unpack_from('<I',b,names+4*i)[0]);out.append(b[p:b.index(0,p)].decode())
    return out
if __name__=='__main__':
    for name in ['libhipblaslt.dll','hipblas.dll']:
        p=pathlib.Path(__file__).resolve().parents[1]/'Strata'/'engine'/'rocm'/'bin'/name
        rows=exports(p);(pathlib.Path(__file__).parent/(name+'-exports.json')).write_text(json.dumps(rows,indent=2))
        print(name,[x for x in rows if any(s in x for s in ['getIndexFromAlgo','GetArchName','GemmEx','GetVersion'])])
