import{r as o}from"./index-CF55C8LQ.js";const s=new Map;function l(t,a){const[e,c]=o.useState(()=>s.has(t)?s.get(t):a);o.useEffect(()=>{s.set(t,e)},[t,e]);const n=o.useCallback(u=>c(p=>{const r=typeof u=="function"?u(p):u;return s.set(t,r),r}),[t]);return[e,n]}export{l as u};
//# sourceMappingURL=kept-state-cvoFhgci.js.map
