"""Thin local progress UI for generation/gen_checkpoints.py.

Reads the modal.Dict "vea-gen-progress" that the generation containers update and
serves a self-refreshing page at http://localhost:8765 .

  /opt/anaconda3/bin/python generation/progress_server.py [--port 8765]
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import modal

progress = modal.Dict.from_name("vea-gen-progress", create_if_missing=True)
_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()


def snapshot():
    with _lock:
        if time.time() - _cache["t"] > 3 or _cache["data"] is None:
            try:
                items = dict(progress.items())
                items.pop("_ping", None)
                _cache["data"] = {"ok": True, "now": time.time(), "items": items}
            except Exception as e:
                _cache["data"] = {"ok": False, "now": time.time(), "error": str(e), "items": {}}
            _cache["t"] = time.time()
        return _cache["data"]


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VEA Rerun Progress</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff;--track:#f0efed;
--q:#a8a29e;--load:#d97706;--gen:#2563eb;--ok:#16a34a;--bad:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#e7e5e4;--mut:#a8a29e;--line:#292524;
--card:#1c1917;--track:#292524;--q:#57534e;--load:#f59e0b;--gen:#60a5fa;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:12.5px;margin-bottom:18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.tile span{color:var(--mut);font-size:12px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;min-width:760px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}
th{font-size:12px;color:var(--mut);font-weight:600}tr:last-child td{border-bottom:0}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;color:#fff}
.queued{background:var(--q)}.loading,.writing{background:var(--load)}.generating{background:var(--gen)}
.done{background:var(--ok)}.failed{background:var(--bad)}
.bar{width:160px;height:8px;background:var(--track);border-radius:4px;overflow:hidden;display:inline-block;vertical-align:middle;margin-right:8px}
.bar i{display:block;height:100%;background:var(--gen)}.bar i.ok{background:var(--ok)}
.mut{color:var(--mut)}.err{color:var(--bad);white-space:normal;font-size:12px;max-width:420px}
.overall{height:10px;margin-bottom:18px;width:100%}
</style></head><body><main>
<h1>VEA rerun · generation</h1>
<div class="sub" id="sub">loading…</div>
<div class="bar overall"><i id="ov" style="width:0"></i></div>
<div class="tiles" id="tiles"></div>
<div class="wrap"><table><thead><tr>
<th>Checkpoint</th><th>Status</th><th>Rollouts</th><th>Speed</th><th>Elapsed</th><th>ETA</th><th>Result</th>
</tr></thead><tbody id="rows"></tbody></table></div>
</main><script>
const fmt=s=>{if(s==null||!isFinite(s))return"–";s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return h?`${h}h ${m}m`:m?`${m}m ${x}s`:`${x}s`};
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function tick(){
 let d;try{d=await (await fetch("/api/progress")).json()}catch(e){document.getElementById("sub").textContent="server unreachable";return}
 const run=d.items._run||{};delete d.items._run;const now=d.now;
 const rows=Object.entries(d.items).sort((a,b)=>a[0].localeCompare(b[0]));
 let tot=0,done=0,c={queued:0,loading:0,generating:0,writing:0,done:0,failed:0};
 const html=rows.map(([k,r])=>{
  tot+=r.total||0;done+=r.done||0;c[r.status]=(c[r.status]||0)+1;
  const pct=r.total?100*(r.done||0)/r.total:0;
  const start=r.started||r.queued;const end=r.finished||now;const el=start?end-start:null;
  let spd="–",eta="–";
  if(r.status==="generating"&&r.loaded&&r.done>0){const rate=r.done/(now-r.loaded);spd=(rate*60).toFixed(0)+"/min";eta=fmt((r.total-r.done)/rate)}
  const m=/output:\s*([\d.]+)/.exec(r.postfix||"");if(m&&r.status==="generating")spd+=` · ${Number(m[1]).toFixed(0)} tok/s`;
  if(r.status==="loading")eta="loading weights";
  let res="";if(r.stats){const s=r.stats;res=`median ${s.median_tokens} tok · </think> ${s.has_close_think} · capped ${s.length_capped} · self-opened &lt;think&gt; ${s.self_opened_think}`}
  if(r.error)res=`<span class="err">${esc(r.error)}</span>`;
  return `<tr><td><b>${esc(k)}</b><div class="mut" style="font-size:12px">${esc(r.model||"")}@${esc(r.revision||"")}</div></td>
  <td><span class="pill ${esc(r.status)}">${esc(r.status)}</span></td>
  <td><span class="bar"><i class="${r.status==="done"?"ok":""}" style="width:${pct.toFixed(1)}%"></i></span>${r.done||0}/${r.total||0}</td>
  <td>${spd}</td><td>${fmt(el)}</td><td class="mut">${eta}</td><td class="mut">${res}</td></tr>`}).join("");
 document.getElementById("rows").innerHTML=html||`<tr><td colspan="7" class="mut">no jobs yet</td></tr>`;
 document.getElementById("ov").style.width=(tot?100*done/tot:0)+"%";
 const started=run.started?new Date(run.started*1000).toLocaleTimeString():"–";
 document.getElementById("sub").textContent=`template ${run.template||"?"} · ${run.n_prompts||"?"} prompts × ${run.g||"?"} · started ${started} · updated ${new Date(now*1000).toLocaleTimeString()}${d.ok?"":" · ERROR "+d.error}`;
 const t=(v,l)=>`<div class="tile"><b>${v}</b><span>${l}</span></div>`;
 document.getElementById("tiles").innerHTML=t(`${(tot?100*done/tot:0).toFixed(1)}%`,`${done.toLocaleString()} / ${tot.toLocaleString()} rollouts`)+
  t(c.done,"done")+t(c.generating,"generating")+t(c.loading+c.writing,"loading / writing")+t(c.queued,"queued")+t(c.failed,"failed");
}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/progress"):
            body, ctype = json.dumps(snapshot()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"progress UI: http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
