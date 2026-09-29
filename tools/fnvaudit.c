#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
static uint64_t fnv(uint64_t h, const unsigned char* p, size_t n){for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ULL;}return h;}
int main(int argc,char**argv){
    const char* dir = argc>1?argv[1]:"/local/strata/kvstore";
    DIR* d=opendir(dir); struct dirent* e; int nbad=0,ntot=0;
    char path[4096];
    while((e=readdir(d))){
        if(strncmp(e->d_name,"kv-",3))continue;
        snprintf(path,sizeof path,"%s/%s",dir,e->d_name);
        FILE* f=fopen(path,"rb"); if(!f)continue;
        fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
        unsigned char* buf=malloc(sz); fread(buf,1,sz,f); fclose(f);
        ntot++;
        if(sz<112){printf("BAD(short) %s (%ld)\n",path,sz);nbad++;free(buf);continue;}
        uint64_t want; memcpy(&want,buf+sz-8,8);
        uint64_t got=fnv(1469598103934665603ULL,buf+104,sz-112);
        if(got!=want){printf("BAD %s\n",path);nbad++;}
        free(buf);
    }
    closedir(d);
    printf("%d files, %d bad\n",ntot,nbad); return nbad?1:0;
}
