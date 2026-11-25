"use strict";
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
const tip = $("#tip");
const fmt = (n, d=1) => (n==null||isNaN(n)) ? "—" : Number(n).toLocaleString("en-US",{maximumFractionDigits:d});
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function fmtDur(s){
  if(s==null||isNaN(s)) return "—";
  s=Math.abs(s);
  if(s<1) return (s*1000).toFixed(0)+"ms";
  if(s<90) return s.toFixed(1)+"s";
  if(s<5400) return (s/60).toFixed(1)+"min";
  if(s<172800) return (s/3600).toFixed(1)+"h";
  return (s/86400).toFixed(1)+"d";
}
function showTip(e, html){ tip.innerHTML=html; tip.style.display="block";
  tip.style.left=Math.min(e.clientX+14, innerWidth-360)+"px";
  tip.style.top=Math.min(e.clientY+14, innerHeight-140)+"px"; }
function hideTip(){ tip.style.display="none"; }

// HiDPI canvas
function setupCanvas(cv, h){
  const dpr = window.devicePixelRatio||1;
  const w = cv.parentElement.clientWidth;
  cv.style.width = w+"px"; cv.style.height = h+"px";
  cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w, h};
}

const CAUSE_COLOR = {ok:"#3ecf8e", local:"#f5b83d", upstream:"#ff6b6b", network:"#4aa8ff", unknown:"#8b97ab"};
const CAUSE_CN = {ok:"正常", local:"本机", upstream:"上游", network:"网络", unknown:"未知", scheduled:"计划内"};
const SUB_CN = {local_jump:"本机跳变", dead_ntpd:"长期失锁", feed:"上游带偏", path:"路径劣化", "":""};

let STATE = {};

async function api(p){ const r = await fetch(p); return r.json(); }

// ---------------- header
async function initHeader(){
  const o = await api("/api/overview"); STATE.overview = o;
  const c = o.causes;
  $("#hdr-stats").innerHTML = `
    <div class="stat"><div class="v">${o.n_nodes}</div><div class="k">节点</div></div>
    <div class="stat"><div class="v">${o.n_servers}</div><div class="k">时间服务器</div></div>
    <div class="stat"><div class="v" style="color:var(--bad)">${(c.upstream||0)+(c.local||0)+(c.network||0)+(c.unknown||0)}</div><div class="k">异常节点</div></div>
    <div class="stat"><div class="v" style="color:var(--warn)">${o.n_upstream_incidents}</div><div class="k">上游故障</div></div>
    <div class="stat"><div class="v" style="color:var(--accent)">${o.n_network_incidents}</div><div class="k">网络故障</div></div>`;
}

// ---------------- overview
async function renderOverview(){
  const o = STATE.overview, e = o.eval;
  const fd = e.fault_detection;
  const el = $("#tab-overview");
  el.innerHTML = `
    <div class="banner">
      <b>场景</b> ${esc(o.scenario)} · ${o.n_nodes} 节点 (${o.n_machines} 业务机 + ${o.n_servers} 时间服务器) ·
      采集 ${o.days} 天, 每 ${o.base_dt/60} 分钟一点 · 闰秒 @ ${esc(o.leap_when||"—")}
    </div>
    <div class="kpi-row">
      <div class="kpi good"><div class="v">${fmt(fd.recall*100,1)}%</div><div class="k">故障机器召回</div></div>
      <div class="kpi ${fd.precision>0.8?'good':fd.precision>0.6?'warn':'bad'}"><div class="v">${fmt(fd.precision*100,1)}%</div><div class="k">故障机器精确率</div></div>
      <div class="kpi good"><div class="v">${fmt(e.cause_accuracy*100,1)}%</div><div class="k">根因归因准确率</div></div>
      <div class="kpi ${o.n_upstream_incidents?'warn':'good'}"><div class="v">${o.n_upstream_incidents}</div><div class="k">识别出的上游故障</div></div>
      <div class="kpi ${o.n_network_incidents?'warn':'good'}"><div class="v">${o.n_network_incidents}</div><div class="k">识别出的网络故障</div></div>
      <div class="kpi"><div class="v">${fmt(e.n_false_negative,0)}</div><div class="k">漏报节点</div></div>
    </div>
    <div class="grid cols2">
      <div class="card">
        <h3>偏差分布 (各业务等级 · 随时间的 p50/p95/p99) <span class="hint">滑动查看</span></h3>
        <canvas id="cv-dist"></canvas>
        <div class="legend" id="dist-legend"></div>
      </div>
      <div class="card">
        <h3>根因构成</h3>
        <div id="cause-bars"></div>
        <p class="muted" style="font-size:12px;margin-top:12px">
          本机 = 仅该机偏移 (ntpd 停摆 / 人工改时 / VM 暂停); 上游 = 与同源兄弟共因偏移;
          网络 = 偏差与路径时延/抖动/丢包同步恶化。闰秒摊开窗口按计划内处理, 不计入故障。
        </p>
      </div>
    </div>`;
  // cause bars
  const total = Object.values(o.causes).reduce((a,b)=>a+b,0);
  const order = ["upstream","local","network","unknown","ok"];
  $("#cause-bars").innerHTML = order.map(k=>{
    const v = o.causes[k]||0;
    if(!v) return "";
    return `<div style="margin-bottom:9px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
        <span><span class="sw" style="background:${CAUSE_COLOR[k]}"></span> ${CAUSE_CN[k]}</span>
        <span class="mono">${v} · ${(v/total*100).toFixed(1)}%</span></div>
      <div class="bar"><i style="width:${v/total*100}%;background:${CAUSE_COLOR[k]}"></i></div></div>`;
  }).join("");
  drawDistribution();
}

async function drawDistribution(){
  const d = await api("/api/distribution");
  const cv = $("#cv-dist"); if(!cv) return;
  const {ctx,w,h} = setupCanvas(cv, 260);
  const pad = {l:52,r:14,t:12,b:26};
  const pw = w-pad.l-pad.r, ph = h-pad.t-pad.b;
  const allv = [];
  Object.values(d.tiers).forEach(t=>{allv.push(...t.p99);});
  const ymax = Math.ceil(Math.log10(Math.max(...allv,10)));
  const ymap = v => pad.t + ph*(1 - Math.log10(Math.max(v,0.1))/ymax);
  ctx.clearRect(0,0,w,h);
  // axes
  ctx.strokeStyle="#263041"; ctx.fillStyle="#8b97ab"; ctx.font="10px ui-monospace";
  for(let k=0;k<=ymax;k++){
    const y = pad.t+ph*(1-k/ymax);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
    ctx.fillText("10^"+k+"ms", 8, y+3);
  }
  const T = d.tiers; const n = d.times.length;
  const colors = {critical:"#ff3d71",high:"#ff6b6b",standard:"#f5b83d",batch:"#4aa8ff"};
  // leap line
  if(d.leap_rel_s!=null){
    const rel = d.leap_rel_s; const tot = d.times[n-1]||1;
    const x = pad.l + pw*(rel/tot);
    if(x>pad.l && x<w-pad.r){ ctx.strokeStyle="#4aa8ff55"; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle="#4aa8ff"; ctx.fillText("闰秒", x+3, pad.t+10); }
  }
  Object.entries(T).forEach(([tier,t])=>{
    [["p99",0.35],["p95",0.6],["p50",1]].forEach(([k,alpha])=>{
      ctx.beginPath(); ctx.strokeStyle=colors[tier]; ctx.globalAlpha=alpha; ctx.lineWidth=k==="p50"?1.6:1;
      t[k].forEach((v,i)=>{ const x=pad.l+pw*(i/(n-1)); const y=ymap(v);
        i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
      ctx.stroke(); ctx.globalAlpha=1;
    });
    // threshold dashed
    const ty = ymap(t.threshold_ms);
    ctx.strokeStyle=colors[tier]; ctx.globalAlpha=0.25; ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,ty); ctx.lineTo(w-pad.r,ty); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha=1;
  });
  $("#dist-legend").innerHTML = Object.entries(T).map(([tier,t])=>
    `<span><span class="sw" style="background:${colors[tier]}"></span>${tier} (${t.n}台, 阈值${t.threshold_ms}ms)</span>`
  ).join("") + `<span class="muted">— 粗线=p50 细线=p99, 虚线=阈值</span>`;
}

// ---------------- heatmap
async function renderHeat(){
  const el = $("#tab-heat");
  if(!el.dataset.built){
    el.innerHTML = `
      <div class="banner">每行为一台机器 (按业务等级分组), 每列为时间 (取该格内最大偏差)。
        颜色越亮偏差越大。<b>点击任意行</b>查看该机详情。</div>
      <div class="card">
        <h3>机群偏差热力图 <span class="hint" id="hm-hint"></span></h3>
        <div style="position:relative"><canvas id="cv-heat"></canvas></div>
        <div class="legend">
          <span>偏差 (ms):</span>
          ${[0.1,1,10,100,1000,100000].map(v=>`<span><span class="sw" style="background:${heatColor(Math.log10(v)/Math.log10(3e5))}"></span>${v>=1000?(v/1000)+"s":v}</span>`).join("")}
          <span class="muted">· 色阶为对数</span>
        </div>
      </div>
      <div class="card" id="node-detail" style="display:none"></div>`;
    el.dataset.built = "1";
  }
  await drawHeat();
}

function heatColor(v){
  v = Math.max(0,Math.min(1,v));
  // dark -> blue -> yellow -> red
  const stops = [[0.08,0.10,0.15],[0.15,0.35,0.65],[0.25,0.75,0.75],[0.95,0.8,0.25],[1.0,0.3,0.3]];
  const x = v*(stops.length-1); const i = Math.min(stops.length-2, Math.floor(x)); const f = x-i;
  const a=stops[i], b=stops[i+1];
  const c = a.map((av,k)=> Math.round((av+(b[k]-av)*f)*255));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

let HEAT = null, heatGeom = null;
async function drawHeat(){
  const d = await api("/api/heatmap?rows=900&bins=420"); HEAT = d;
  const cv = $("#cv-heat"); const rows = d.data.length, cols = d.data[0].length;
  const rowH = Math.max(1, Math.min(4, Math.floor(620/rows)));
  const H = rows*rowH + 30;
  const {ctx,w,h} = setupCanvas(cv, H);
  const padT=18, padL=2, padR=2;
  const cw = (w-padL-padR)/cols;
  heatGeom = {padT, padL, cw, rowH, rows, cols};
  ctx.clearRect(0,0,w,h);
  // leap line
  const rel = d.leap_rel_s, tot = d.times[d.times.length-1]||1;
  const launchX = padL + (w-padL-padR)*(rel/tot);
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      ctx.fillStyle = heatColor(d.data[r][c]);
      ctx.fillRect(padL+c*cw, padT+r*rowH, Math.ceil(cw), rowH);
    }
  }
  if(launchX>padL&&launchX<w){ ctx.strokeStyle="#4aa8ff"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(launchX,padT); ctx.lineTo(launchX,padT+rows*rowH); ctx.stroke();
    ctx.fillStyle="#4aa8ff"; ctx.font="10px ui-monospace"; ctx.fillText("闰秒",launchX+3,padT-5); }
  // tier group markers
  ctx.fillStyle="#8b97ab"; ctx.font="10px -apple-system";
  let prev=null;
  d.tiers.forEach((t,r)=>{ if(t!==prev){ ctx.fillRect(0,padT+r*rowH,2,2);
    ctx.fillText(CAUSE_CN[t]||t, 4, padT+r*rowH+Math.min(rowH,10)); prev=t; }});
  $("#hm-hint").textContent = `显示 ${d.n_rows_shown}/${d.n_rows_total} 行 · ${cols} 时间格`;
  // events
  cv.onmousemove = (e)=>{
    const rect = cv.getBoundingClientRect();
    const x = e.clientX-rect.left, y = e.clientY-rect.top;
    if(y<padT||y>=padT+rows*rowH){ hideTip(); return; }
    const r = Math.floor((y-padT)/rowH), c = Math.floor((x-padL)/cw);
    if(r<0||r>=rows||c<0||c>=cols){ hideTip(); return; }
    const v = d.data[r][c]; const ms = Math.pow(10, v*Math.log10(3e5));
    showTip(e, `<b>${esc(d.labels[r])}</b><br>${d.tiers[r]} · t+${fmtDur(d.times[c])}<br>偏差 ≈ ${ms<1000?ms.toFixed(1)+"ms":(ms/1000).toFixed(1)+"s"}`);
  };
  cv.onmouseleave = hideTip;
  cv.onclick = async (e)=>{
    const rect = cv.getBoundingClientRect();
    const y = e.clientY-rect.top;
    const r = Math.floor((y-padT)/rowH);
    if(r>=0&&r<rows){ await renderNodeDetail(d.idx[r]); }
  };
}

async function showNodeByName(name){
  // find idx via node list or hierarchy; simplest: query all findings
  const nodes = await api("/api/nodes?limit=99999");
  let idx = (STATE.overview && null);
  // fallback: search by fetching detail isn't possible; use name->idx map built lazily
  if(!STATE.nameIdx){
    const ov = STATE.overview;
    // heatmap labels don't carry idx; use hierarchy+search. Build from /api/nodes not enough (ok nodes missing).
  }
  // Use a dedicated search: iterate heatmap not possible. We fetch node by matching name from a full map endpoint.
  const full = await api("/api/nodes?limit=99999&all=1");
  const hit = full.nodes.find(n=>n.name===name);
  if(hit){ await renderNodeDetail(hit.idx); }
  else { // ok node - need idx; derive from label position is unreliable, so just inform
    $("#node-detail").style.display="block";
    $("#node-detail").innerHTML = `<h3>${esc(name)}</h3><p class="muted">该节点当前偏差正常, 无异常窗口。</p>`;
  }
}

async function renderNodeDetail(idx){
  const d = await api("/api/node/"+idx);
  const box = $("#node-detail"); box.style.display="block";
  const causePill = `<span class="pill p-${d.cause}">${CAUSE_CN[d.cause]||d.cause}${d.subtype?" · "+(SUB_CN[d.subtype]||d.subtype):""}</span>`;
  box.innerHTML = `
    <h3>${esc(d.name)} ${causePill} <span class="hint">置信度 ${(d.confidence*100).toFixed(0)}%</span></h3>
    <div class="cols2">
      <div>
        <table>
          <tr><th>业务等级</th><td>${d.tier}</td></tr>
          <tr><th>区域</th><td>${d.region}</td></tr>
          <tr><th>告警阈值</th><td class="mono">${d.threshold_ms} ms</td></tr>
          <tr><th>基线/波动</th><td class="mono">${d.baseline_ms} ms</td></tr>
          <tr><th>峰值偏差</th><td class="mono">${fmt(d.peak_ms)} ms</td></tr>
          <tr><th>上游</th><td>${d.upstream.map(esc).join(", ")||"—"}</td></tr>
          <tr><th>肇事源</th><td>${d.culprit_name?`<a class="link" onclick="renderNodeDetail(${d.culprit})">${esc(d.culprit_name)}</a>`:"—"}</td></tr>
        </table>
        <h3 style="margin-top:14px">判据</h3>
        ${d.evidence.map(e=>`<div class="ev">• ${esc(e)}</div>`).join("")}
      </div>
      <div>
        <canvas id="cv-node"></canvas>
        <div class="legend"><span><span class="sw" style="background:#ff6b6b"></span>偏差 (ms)</span>
          <span><span class="sw" style="background:#4aa8ff"></span>时延 (ms)</span>
          <span><span class="sw" style="background:#3ecf8e"></span>抖动×100</span></div>
      </div>
    </div>`;
  drawNodeChart(d);
  box.scrollIntoView({behavior:"smooth",block:"nearest"});
}

function drawNodeChart(d){
  const cv = $("#cv-node"); if(!cv) return;
  const {ctx,w,h} = setupCanvas(cv, 220);
  const pad={l:44,r:10,t:10,b:22}; const pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const n=d.dev_ms.length;
  const devmax = Math.max(...d.dev_ms.map(Math.abs), d.threshold_ms)*1.1;
  const ymap = v => pad.t+ph*(1 - (v+devmax)/(2*devmax));
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#263041"; ctx.fillStyle="#8b97ab"; ctx.font="10px ui-monospace";
  [-devmax,-devmax/2,0,devmax/2,devmax].forEach(v=>{const y=ymap(v);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText(v.toFixed(0),2,y+3);});
  // threshold
  ctx.strokeStyle="#ff3b3b55"; ctx.setLineDash([3,3]);
  [d.threshold_ms,-d.threshold_ms].forEach(v=>{const y=ymap(v);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();});
  ctx.setLineDash([]);
  // leap
  if(d.leap_rel_s!=null){ const tot=d.times[n-1]||1; const x=pad.l+pw*(d.leap_rel_s/tot);
    if(x>pad.l&&x<w-pad.r){ctx.strokeStyle="#4aa8ff88";ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+ph);ctx.stroke();} }
  // dev line
  ctx.beginPath(); ctx.strokeStyle="#ff6b6b"; ctx.lineWidth=1.4;
  d.dev_ms.forEach((v,i)=>{const x=pad.l+pw*(i/(n-1));const y=ymap(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.stroke();
  // delay scaled onto secondary
  const dmax = Math.max(...d.delay_ms.filter(x=>x!=null),1);
  ctx.beginPath(); ctx.strokeStyle="#4aa8ff"; ctx.lineWidth=1; ctx.globalAlpha=.7;
  d.delay_ms.forEach((v,i)=>{if(v==null)return;const x=pad.l+pw*(i/(n-1));const y=pad.t+ph*(1-v/dmax);ctx.lineTo(x,y);});
  ctx.stroke(); ctx.globalAlpha=1;
}

// ---------------- hierarchy
async function renderHierarchy(){
  const h = await api("/api/hierarchy");
  const inc = await api("/api/incidents");
  const culpritSet = new Set(inc.upstream.map(u=>u.culprit));
  const s1 = h.nodes.filter(n=>n.stratum===1);
  const s2 = h.nodes.filter(n=>n.stratum===2);
  const el = $("#tab-hierarchy");
  el.innerHTML = `
    <div class="banner">时间服务器层级与健康。道红 = 被识别为故障源, 黄 = 自身异常, 绿 = 正常。
      括号内为直连下游台数。点节点看其详情。</div>
    <div class="card">
      <h3>Stratum-1 外部上游</h3>
      <div class="tier-row" id="s1-row"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>Stratum-2 内网中继 <span class="hint">${s2.length} 台</span></h3>
      <div class="tier-row" id="s2-row"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>上游异常事件</h3>
      ${inc.upstream.length? inc.upstream.map(u=>`
        <div style="border-left:3px solid var(--bad);padding:8px 12px;margin-bottom:10px;background:var(--panel2);border-radius:6px">
          <b>${esc(u.name)}</b> · 起始 ${esc(u.onset)} · 峰值 ${fmt(u.peak_ms)}ms ${u.direction>0?"(快)":"(慢)"}
          · 置信度 ${(u.confidence*100).toFixed(0)}%
          <div class="muted" style="font-size:12px;margin-top:4px">
            直连下游 ${u.direct_children} 台, 实际带偏 <b style="color:var(--bad)">${u.affected}</b> 台,
            因多数表决/冗余未受影响 ${u.protected} 台</div>
          ${u.evidence.map(e=>`<div class="ev">• ${esc(e)}</div>`).join("")}
        </div>`).join("") : `<p class="muted">无</p>`}
    </div>`;

  function nodeCard(n){
    const isCulprit = culpritSet.has(n.idx);
    const col = isCulprit?"var(--crit)":(n.cause==="ok"?"var(--good)":CAUSE_COLOR[n.cause]);
    return `<div onclick="renderNodeDetailInto(${n.idx})" style="cursor:pointer;background:var(--panel2);
      border:1px solid ${col};border-left:3px solid ${col};border-radius:7px;padding:8px 11px;min-width:150px">
      <div style="font-size:12px;font-weight:600">${esc(n.name)}</div>
      <div class="muted" style="font-size:10px">${esc(n.region)} · ${n.n_children} 下游</div>
      <div style="font-size:11px;margin-top:4px">
        <span style="color:${col}">${isCulprit?"故障源":CAUSE_CN[n.cause]}</span>
        <span class="mono muted"> ${n.baseline_ms}ms/σ${n.sigma_ms}</span></div>
    </div>`;
  }
  $("#s1-row").innerHTML = s1.map(nodeCard).join("");
  $("#s2-row").innerHTML = s2.map(nodeCard).join("");
  $("#tab-hierarchy").style.setProperty("--x","1");
  // attach row flex style
  $$(".tier-row").forEach(r=>{r.style.display="flex";r.style.gap="10px";r.style.flexWrap="wrap";});
}

async function renderNodeDetailInto(idx){
  // switch to heat tab and render detail in a floating card there
  showTab("heat");
  await new Promise(r=>setTimeout(r,120));
  await renderNodeDetail(idx);
}
window.renderNodeDetailInto = renderNodeDetailInto;

// ---------------- incidents
async function renderIncidents(){
  const inc = await api("/api/incidents");
  const nodes = await api("/api/nodes?limit=500");
  const el = $("#tab-incidents");
  const byCause = {upstream:[],local:[],network:[],unknown:[]};
  nodes.nodes.forEach(n=>{ (byCause[n.cause]||byCause.unknown).push(n); });
  el.innerHTML = `
    <div class="banner">按根因归类的异常机器。上游故障列出肇事服务器与受影响下游; 点击行看该机时间曲线与判据。</div>
    <div class="cols2">
      <div class="card">
        <h3 style="color:var(--bad)">上游故障 <span class="hint">${inc.upstream.length}</span></h3>
        ${inc.upstream.map(u=>`
          <div style="margin-bottom:12px;border-bottom:1px solid var(--line);padding-bottom:10px">
            <div><b>${esc(u.name)}</b> · ${esc(u.onset)} · 带偏 ${u.affected} 台 · 峰值 ${fmt(u.peak_ms)}ms</div>
            ${u.evidence.map(e=>`<div class="ev">• ${esc(e)}</div>`).join("")}
          </div>`).join("")||`<p class="muted">无</p>`}
        <h3 style="color:var(--accent);margin-top:16px">网络故障 <span class="hint">${inc.network.length}</span></h3>
        ${inc.network.map(n=>`
          <div style="margin-bottom:10px">
            <div><b>${esc(n.region)}</b> · ${esc(n.onset)} → ${esc(n.end)} · ${n.affected} 台受影响</div>
            <div class="ev">时延×${n.delay_ratio}, 抖动×${n.jitter_ratio}, 丢包 +${n.loss_add}%</div>
          </div>`).join("")||`<p class="muted">无</p>`}
      </div>
      <div class="card">
        <h3>异常机器 (按峰值排序, 共 ${nodes.total})</h3>
        <div class="scroll"><table>
          <thead><tr><th>机器</th><th>根因</th><th>峰值</th><th>起始</th></tr></thead>
          <tbody>${nodes.nodes.map(n=>`
            <tr onclick="renderNodeDetailInto(${n.idx})" style="cursor:pointer">
              <td>${esc(n.name)}</td>
              <td><span class="pill p-${n.cause}">${CAUSE_CN[n.cause]}${n.subtype?" "+(SUB_CN[n.subtype]||""):""}</span></td>
              <td class="mono">${fmt(n.peak_ms)}ms</td>
              <td class="muted">${n.onset_s!=null?fmtDur(n.onset_s):"—"}</td>
            </tr>`).join("")}</tbody>
        </table></div>
      </div>
    </div>`;
}

// ---------------- leap
async function renderLeap(){
  const d = await api("/api/leap");
  const cl = d.checklist;
  const el = $("#tab-leap");
  const sevOrder = ["critical","high","medium","low","ok"];
  const sevCN = {critical:"致命",high:"高",medium:"中",low:"低",ok:"无"};
  el.innerHTML = `
    <div class="banner">闰秒推演。左: 各策略下墙钟相对真实时间的偏差曲线 (中点为闰秒)。
      右: 各业务原语在每种策略下的严重度。下方为业务影响清单。</div>
    <div class="grid cols2">
      <div class="card">
        <h3>时钟偏差曲线 (ms)</h3>
        <canvas id="cv-leap"></canvas>
        <div class="legend" id="leap-legend"></div>
      </div>
      <div class="card">
        <h3>业务原语 × 策略 严重度矩阵</h3>
        <table>
          <thead><tr><th>原语</th><th>step</th><th>slew</th><th>smear</th><th>none</th></tr></thead>
          <tbody>${Object.entries(cl.severity_matrix).map(([p,row])=>`
            <tr><td>${p==="lease"?"分布式锁租约":p==="timeout"?"超时/重试":p==="log_order"?"跨机日志排序":"按时间生成编号"}</td>
            ${["step","slew","smear","none"].map(m=>`<td><span class="sev-${row[m]}">${sevCN[row[m]]||row[m]}</span></td>`).join("")}
            </tr>`).join("")}
          </tbody>
        </table>
        <h3 style="margin-top:16px">推荐</h3>
        <div style="font-size:12px;line-height:1.7">
          <b>全机群统一: <span class="sev-low">${cl.recommendation.primary}</span></b>
          <div class="muted" style="margin-top:6px">${esc(cl.recommendation.why)}</div>
          <div style="margin-top:8px;color:var(--warn)">⚠ ${esc(cl.recommendation.must_not_mix)}</div>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>混用策略的灾难性后果 (跨机事件时间戳排序)</h3>
      <table>
        <thead><tr><th>策略组合</th><th>相邻逆序比例</th><th>跨机 >100ms 严重逆序</th><th>租约严重度</th></tr></thead>
        <tbody>${cl.mixed_cases.map(c=>{
          const m=parseFloat(c.log_order.detail.match(/严重逆序 (\d+)/)?.[1]||0);
          return `<tr><td class="mono">${c.pair}</td>
            <td class="mono">${c.log_order.value.toFixed(2)}%</td>
            <td class="mono sev-${c.log_order.severity}">${m} 处</td>
            <td><span class="sev-${c.lease_severity}">${sevCN[c.lease_severity]}</span></td></tr>`;
        }).join("")}</tbody>
      </table>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>业务影响清单 (按策略)</h3>
      <div class="scroll"><table>
        <thead><tr><th>原语</th><th>工况</th><th>指标</th><th>数值</th><th>严重度</th><th>说明</th></tr></thead>
        <tbody>
        ${["step","slew","smear","none"].flatMap(m=>cl.per_mode[m].map(x=>`
          <tr><td>${x.primitive_cn}</td><td class="muted">${esc(x.scenario)}</td>
          <td>${esc(x.metric)}</td><td class="mono">${x.value} ${x.unit}</td>
          <td><span class="sev-${x.severity}">${sevCN[x.severity]}</span></td>
          <td class="muted" style="font-size:11px">${esc(x.detail)}</td></tr>`)).join("")}
        </tbody>
      </table></div>
    </div>`;
  drawLeapChart(d);
}

function drawLeapChart(d){
  const cv = $("#cv-leap"); if(!cv) return;
  const {ctx,w,h} = setupCanvas(cv, 300);
  const pad={l:46,r:12,t:12,b:26}; const pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const xs = d.times_h; const n=xs.length;
  let allv=[]; Object.values(d.curves).forEach(c=>allv.push(...c));
  const vmax = Math.max(1100, ...allv.map(Math.abs));
  const ymap = v => pad.t+ph*(1-(v+vmax)/(2*vmax));
  const xmap = i => pad.l+pw*(i/(n-1));
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#263041"; ctx.fillStyle="#8b97ab"; ctx.font="10px ui-monospace";
  [-1000,-500,0,500,1000].forEach(v=>{const y=ymap(v);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText(v+"",2,y+3);});
  // x axis labels (hours)
  const t0=xs[0], t1=xs[n-1];
  [t0,(t0+t1)/2,t1].forEach(v=>{const i=Math.round((v-t0)/(t1-t0)*(n-1));const x=xmap(i);
    ctx.fillText(v.toFixed(0)+"h",x-8,h-8);});
  const colors={step:"#ff3d71",slew:"#4aa8ff",smear:"#3ecf8e",none:"#f5b83d"};
  Object.entries(d.curves).forEach(([m,c])=>{
    ctx.beginPath(); ctx.strokeStyle=colors[m]; ctx.lineWidth=m==="step"?2:1.6;
    c.forEach((v,i)=>{i?ctx.lineTo(xmap(i),ymap(v)):ctx.moveTo(xmap(i),ymap(v));}); ctx.stroke();
  });
  $("#leap-legend").innerHTML = Object.entries(d.labels).map(([m,lab])=>
    `<span><span class="sw" style="background:${colors[m]}"></span>${esc(lab)}</span>`).join("");
}

// ---------------- scan
async function renderScan(){
  const s = await api("/api/scan");
  const ev = s.evaluation;
  const el = $("#tab-scan");
  const sevCN={critical:"致命",high:"高",medium:"中",low:"低"};
  el.innerHTML = `
    <div class="banner">静态扫描业务代码中用墙上时间算时长/超时/ID 的位置。这些在时间跳变或闰秒时最易出事。</div>
    <div class="kpi-row">
      <div class="kpi good"><div class="v">${fmt(ev.recall*100,0)}%</div><div class="k">召回率</div></div>
      <div class="kpi ${ev.precision>0.9?'good':'warn'}"><div class="v">${fmt(ev.precision*100,0)}%</div><div class="k">精确率</div></div>
      <div class="kpi"><div class="v">${s.total}</div><div class="k">发现问题</div></div>
      <div class="kpi ${ev.fn?'warn':'good'}"><div class="v">${ev.fn}</div><div class="k">漏报</div></div>
      <div class="kpi ${ev.fp?'warn':'good'}"><div class="v">${ev.fp}</div><div class="k">误报</div></div>
    </div>
    <div class="card">
      <h3>扫描结果 <span class="hint">按规则统计 ${JSON.stringify(s.by_rule)}</span></h3>
      <div class="scroll"><table>
        <thead><tr><th>位置</th><th>规则</th><th>函数</th><th>严重度</th><th>代码</th><th>建议</th></tr></thead>
        <tbody>${s.findings.sort((a,b)=>sevRank(b.severity)-sevRank(a.severity)).map(f=>`
          <tr><td class="mono muted">${esc(f.path.split("/").pop())}:${f.lineno}</td>
            <td><code>${f.rule}</code></td><td>${esc(f.func)}</td>
            <td><span class="sev-${f.severity}">${sevCN[f.severity]||f.severity}</span></td>
            <td class="mono" style="font-size:11px">${esc(f.snippet)}</td>
            <td class="muted" style="font-size:11px">${esc(f.suggestion)}</td></tr>`).join("")}</tbody>
      </table></div>
    </div>`;
  function sevRank(s){return {low:1,medium:2,high:3,critical:4}[s]||0;}
}

// ---------------- advice
async function renderAdvice(){
  const a = await api("/api/advice");
  const el = $("#tab-advice");
  const t=a.topology, l=a.leap, th=a.thresholds;
  el.innerHTML = `
    <div class="banner">机群对时配置建议: 上游编排 · 闰秒策略 · 分级阈值。</div>
    <div class="kpi-row">
      <div class="kpi ${t.n_single_source?'bad':'good'}"><div class="v">${t.n_single_source}</div><div class="k">单源无冗余机器</div></div>
      <div class="kpi ${t.n_common_mode?'warn':'good'}"><div class="v">${t.n_common_mode}</div><div class="k">共因风险机器</div></div>
      <div class="kpi ${t.n_insufficient_diversity?'warn':'good'}"><div class="v">${t.n_insufficient_diversity}</div><div class="k">tier-1 多样性不足</div></div>
      <div class="kpi good"><div class="v">${l.uniform_choice}</div><div class="k">推荐闰秒策略</div></div>
    </div>
    <div class="cols2">
      <div class="card">
        <h3>上游编排问题 (示例 ${t.examples.length}/${t.n_problem})</h3>
        <div class="scroll"><table>
          <thead><tr><th>机器</th><th>问题</th><th>当前→建议上游</th></tr></thead>
          <tbody>${t.examples.map(x=>`
            <tr><td>${esc(x.name)}</td>
              <td><span class="pill p-${x.severity==='critical'?'upstream':'local'}">${x.issues.join("; ")}</span></td>
              <td class="mono" style="font-size:11px">[${x.current.join(",")}] → [${x.recommended.join(",")}]</td></tr>`).join("")}</tbody>
        </table></div>
      </div>
      <div class="card">
        <h3>闰秒策略</h3>
        <p style="font-size:12px"><b>全机群统一: ${esc(l.uniform_choice)}</b></p>
        <p class="muted" style="font-size:12px">${esc(l.rationale)}</p>
        <h3 style="margin-top:14px;color:var(--bad)">禁止事项</h3>
        ${l.forbidden.map(f=>`<div class="ev" style="color:var(--bad)">✕ ${esc(f)}</div>`).join("")}
        <h3 style="margin-top:14px">迁移步骤</h3>
        ${l.migration.map((m,i)=>`<div class="ev">${i+1}. ${esc(m)}</div>`).join("")}
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>分级告警阈值</h3>
      <p class="muted" style="font-size:12px">${esc(th.rationale)}</p>
      <table>
        <thead><tr><th>业务等级</th><th>阈值</th><th>观测窗口</th><th>动作</th><th>说明</th></tr></thead>
        <tbody>${Object.entries(th.tiers).map(([k,v])=>`
          <tr><td><b>${k}</b></td><td class="mono">${v.threshold_ms} ms</td>
            <td class="mono">${v.window_s}s</td><td>${esc(v.action)}</td>
            <td class="muted" style="font-size:11px">${esc(v.why)}</td></tr>`).join("")}</tbody>
      </table>
    </div>`;
}

// ---------------- eval
async function renderEval(){
  const e = await api("/api/eval");
  const fd = e.fault_detection;
  const el = $("#tab-eval");
  const pc = e.per_class||{};
  el.innerHTML = `
    <div class="banner">分析结论 vs 仿真真值的逐项评测。</div>
    <div class="kpi-row">
      <div class="kpi good"><div class="v">${fmt(fd.recall*100,1)}%</div><div class="k">召回</div></div>
      <div class="kpi ${fd.precision>0.8?'good':fd.precision>0.6?'warn':'bad'}"><div class="v">${fmt(fd.precision*100,1)}%</div><div class="k">精确率</div></div>
      <div class="kpi"><div class="v">${fmt(fd.f1*100,1)}%</div><div class="k">F1</div></div>
      <div class="kpi good"><div class="v">${fmt(e.cause_accuracy*100,1)}%</div><div class="k">根因准确率</div></div>
      <div class="kpi bad"><div class="v">${fd.fn}</div><div class="k">漏报</div></div>
      <div class="kpi warn"><div class="v">${fd.fp}</div><div class="k">误报</div></div>
    </div>
    <div class="cols2">
      <div class="card"><h3>各根因分类准确率</h3>
        <table><thead><tr><th>真值根因</th><th>样本</th><th>判对</th><th>准确率</th></tr></thead>
        <tbody>${Object.entries(pc).map(([k,v])=>`
          <tr><td><span class="pill p-${k}">${CAUSE_CN[k]||k}</span></td>
            <td class="mono">${v.n}</td><td class="mono">${v.correct}</td>
            <td class="mono">${(v.accuracy*100).toFixed(1)}%</td></tr>`).join("")}</tbody></table>
        <h3 style="margin-top:14px">混淆矩阵</h3>
        <table><tbody>${Object.entries(e.cause_confusion).map(([k,v])=>
          `<tr><td class="mono">${esc(k.replace("->"," → "))}</td><td class="mono">${v}</td></tr>`).join("")}</tbody></table>
      </div>
      <div class="card"><h3>上游故障定位</h3>
        ${(e.upstream||[]).filter(u=>u.matched_ground_truth).map(u=>`
          <div style="margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--line)">
            <b>${esc(u.name)}</b>
            <div class="ev">识别为肇事源 ✓ · 起始误差 ${u.onset_error_s}s</div>
            <div class="ev">受影响: 预测 ${u.affected_pred} / 真值 ${u.affected_gt} ·
              精确率 ${(u.affected_precision*100).toFixed(0)}% · 召回 ${(u.affected_recall*100).toFixed(0)}%</div>
          </div>`).join("")}
        <h3 style="margin-top:14px">网络事件</h3>
        ${((e.detail&&e.detail.network)||[]).map(n=>`
          <div class="ev">${esc(n.region)}: ${n.detected?"检出 ✓":"未检出 ✗"}
            ${n.delay_ratio?`(时延×${n.delay_ratio}, 抖动×${n.jitter_ratio})`:""}</div>`).join("")}
      </div>
    </div>`;
}

// ---------------- tabs
const RENDER = {overview:renderOverview, heat:renderHeat, hierarchy:renderHierarchy,
  incidents:renderIncidents, leap:renderLeap, scan:renderScan, advice:renderAdvice, eval:renderEval};
function showTab(name){
  $$("#nav button").forEach(b=>b.classList.toggle("active", b.dataset.tab===name));
  $$("main section").forEach(s=>s.classList.toggle("active", s.id==="tab-"+name));
  RENDER[name] && RENDER[name]();
  history.replaceState(null,"","#"+name);
}
$("#nav").onclick = e=>{ const b=e.target.closest("button"); if(b) showTab(b.dataset.tab); };

(async function(){
  await initHeader();
  const start = location.hash.slice(1) || "overview";
  showTab(RENDER[start]?start:"overview");
})();
