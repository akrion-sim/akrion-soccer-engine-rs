#!/usr/bin/env python3
"""Build a self-play viewer (out/index.html): a champion-vs-champion match animation
+ the champion-ladder climb chart. Unlike assemble.py (train-mode before/after vs a
scripted baseline), this fits the `selfplay` command's data (both teams learned,
promote-to-advance). Records a fresh match each build via the `fiveaside` binary.

  python3 viz/selfplay_viz.py            # record champ-vs-champ + write out/index.html
Serve it with: PORT=8081 python3 viz/serve_selfplay.py
"""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
BIN = os.path.join(ROOT, "target", "release", "fiveaside")
MAX_SEED = 2 ** 63 - 1


def env_int(name, default, lo=None, hi=None):
    try:
        value = int(os.environ.get(name, str(default)), 0)
    except ValueError:
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def policy_complete(path):
    return all(os.path.exists(os.path.join(path, name)) for name in ("actor.txt", "critic.txt", "speedor.txt"))


def latest_champion():
    champs = OUT if policy_complete(OUT) else None
    gens = []
    cdir = os.path.join(OUT, "champions")
    if os.path.isdir(cdir):
        for d in os.listdir(cdir):
            path = os.path.join(cdir, d)
            if d.startswith("gen") and d[3:].isdigit() and policy_complete(path):
                gens.append((int(d[3:]), path))
    gens.sort()
    if not champs and not gens:
        raise SystemExit(f"no complete champion policy found under {OUT}; run selfplay first")
    return champs or (gens[-1][1] if gens else OUT), (gens[-1][0] if gens else 0)


def record(seed, a_dir, opp_dir, out_json):
    timeout = env_int("NEWGAME_TIMEOUT", 120, lo=1, hi=3600)
    try:
        r = subprocess.run(
            [BIN, "play", str(seed), "--out-dir", a_dir, "--opponent", opp_dir, "--out", out_json],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"record timed out after {timeout}s") from exc
    except OSError as exc:
        raise SystemExit(f"record failed to start: {exc}") from exc
    if r.returncode != 0:
        raise SystemExit(f"record failed: {(r.stderr or r.stdout)[-800:]}")
    with open(out_json) as fh:
        match = json.load(fh)
    if not isinstance(match.get("frames"), list) or not match["frames"]:
        raise SystemExit(f"record wrote invalid match JSON at {out_json}")
    return match


def read_ladder():
    p = os.path.join(OUT, "selfplay_ladder.csv")
    rows = []
    if os.path.exists(p):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def main():
    champ, gen = latest_champion()
    # a fair, symmetric champion-vs-champion match (both teams the current champion)
    fd, match_path = tempfile.mkstemp(suffix=".json", prefix="selfplay_match_")
    os.close(fd)
    try:
        match = record(env_int("SEED", 20260712, lo=0, hi=MAX_SEED), champ, champ, match_path)
    finally:
        try:
            os.unlink(match_path)
        except OSError:
            pass
    ladder = read_ladder()
    # ladder curve: vs-champion + vs-scripted goal-diff per generation
    curve = []
    for r in ladder:
        try:
            row = {
                "gen": int(r["generation"]),
                "vs_champ": float(r["cand_vs_champ_gd"]),
                "vs_scripted": float(r["cand_vs_scripted_gd"]),
                "champ_gen": int(r["champion_gen"]),
                "promoted": r["promoted"].strip().lower() == "true",
            }
        except (KeyError, ValueError):
            continue
        if math.isfinite(row["vs_champ"]) and math.isfinite(row["vs_scripted"]):
            curve.append(row)
    promotions = sum(1 for r in ladder if r["promoted"].strip().lower() == "true")
    final_champ_gen = curve[-1]["champ_gen"] if curve else gen
    last_scripted = curve[-1]["vs_scripted"] if curve else 0.0
    meta = {
        "champ_gen": final_champ_gen,
        "promotions": promotions,
        "generations": len(curve),
        "vs_scripted_gd": round(last_scripted, 2),
        "match_score": f"{match['frames'][-1]['ga']}-{match['frames'][-1]['gb']}",
    }
    html = TEMPLATE.replace("/*__MATCH__*/null", json.dumps(match)) \
                   .replace("/*__CURVE__*/null", json.dumps(curve)) \
                   .replace("/*__META__*/null", json.dumps(meta))
    dst = os.path.join(OUT, "index.html")
    with open(dst, "w") as fh:
        fh.write(html)
    print("wrote", dst, "| champion gen", final_champ_gen, "| promotions", promotions,
          "| vs-scripted gd", round(last_scripted, 2), "| match", meta["match_score"])


TEMPLATE = r"""<!doctype html><meta charset=utf-8><title>5-a-side self-play champion</title><style>
:root{--bg:#0b140e;--panel:#101c14;--ink:#e8f0e9;--dim:#93a89a;--line:#22362a;--accent:#2fe0b6;
--teamA:#2fe0b6;--teamB:#ff6b5e;--ball:#f7f7f2;--pitch:#12351f;--pline:rgba(233,240,233,.5);--good:#57d979;--bad:#ff6b5e}
*{box-sizing:border-box}body{margin:0;font-family:ui-monospace,Menlo,monospace;background:var(--bg);color:var(--ink);
line-height:1.5;padding:clamp(16px,4vw,40px)}.wrap{max-width:1040px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
.eyebrow{font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim)}
h1{font-family:system-ui,sans-serif;font-weight:800;letter-spacing:-.02em;font-size:clamp(26px,5vw,46px);margin:.2em 0}
h1 .em{color:var(--accent)}.lede{color:var(--dim);max-width:70ch;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}@media(max-width:640px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi .v{font-family:system-ui,sans-serif;font-weight:800;font-size:26px;letter-spacing:-.02em}
.kpi .v.accent{color:var(--accent)}.kpi .k{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin-top:4px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
.pc-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.score{font-family:system-ui,sans-serif;font-weight:800;font-size:22px}.score .a{color:var(--teamA)}.score .b{color:var(--teamB)}.score .x{color:var(--dim)}
canvas{display:block;width:100%;height:auto;border-radius:10px}
.ctl{display:flex;gap:12px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;color:var(--bg);background:var(--accent);border:none;border-radius:9px;padding:9px 15px}
button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}.clock{color:var(--dim);font-variant-numeric:tabular-nums;font-size:13px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--dim);flex-wrap:wrap}.legend i{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px;vertical-align:middle}
.h2{font-family:system-ui,sans-serif;font-weight:700;font-size:18px;margin-bottom:2px}.note{font-size:12px;color:var(--dim)}
.foot{color:var(--dim);font-size:12px;text-align:center}</style>
<div class=wrap>
<header><div class=eyebrow><span style="color:var(--accent)">●</span> standalone 5-a-side · adversarial self-play · zero deps</div>
<h1>Watch the <span class=em>champion play itself.</span></h1>
<p class=lede>Both teams are learned policies. A frozen <b>champion</b> plays a challenger that keeps training; when the
challenger beats the champion by a margin it's <b>promoted</b> to the new champion — a self-play ladder. Below: the current
champion in a <b>champion-vs-champion</b> match, and the ladder's climb across generations.</p></header>
<div class=kpis id=kpis></div>
<section class=card>
<div class=pc-head><div><div class=h2>Champion vs champion</div><div class=note>current best policy, both teams · symmetric kickoff</div></div>
<span class=score><span class=a id=sa>0</span><span class=x>–</span><span class=b id=sb>0</span></span></div>
<canvas id=pitch></canvas>
<div class=ctl><button id=play>Pause</button><button class=ghost id=restart>Restart</button>
<button class=ghost id=newgame title="record a fresh champion-vs-champion match (needs the live server)">New Game ▸</button>
<span class=clock id=clock>0.0s</span>
<span class=legend><span><i style="background:var(--teamA)"></i>champion A</span><span><i style="background:var(--teamB)"></i>champion B</span><span><i style="background:var(--ball)"></i>ball</span></span></div>
</section>
<section class=card><div class=h2>The self-play ladder</div><div class=note>goal difference per generation — the challenger must beat the frozen champion (teal) to advance; grey = vs the scripted baseline (absolute skill)</div>
<canvas id=curve></canvas>
<div class=legend style="margin-top:8px"><span><i style="background:var(--accent)"></i>vs champion</span><span><i style="background:var(--dim)"></i>vs scripted</span><span style="color:var(--good)">▲ = promoted</span></div></section>
<p class=foot id=foot></p></div>
<script>
const MATCH=/*__MATCH__*/null, CURVE=/*__CURVE__*/null, META=/*__META__*/null;
document.getElementById('kpis').innerHTML=[
 ['v accent',META.champ_gen,'Champion gen'],['v',META.promotions,'Promotions'],
 ['v',META.generations,'Generations'],[(META.vs_scripted_gd>0?'v good':'v'),(META.vs_scripted_gd>0?'+':'')+META.vs_scripted_gd,'GD vs scripted']
].map(([c,v,k])=>`<div class=kpi><div class="${c}">${v}</div><div class=k>${k}</div></div>`).join('');
document.getElementById('foot').textContent='champion-vs-champion match '+META.match_score+' · hermetic zero-dep PPO self-play · New Game records a fresh match with the live server';
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
// pitch
let M=MATCH,NF=M.frames.length;const[L,W]=M.field,gh=M.goal_half,AR=W/L,cv=document.getElementById('pitch'),ctx=cv.getContext('2d');let pw,ph;const pad=10;
function rs(){const w=cv.clientWidth||880,dpr=Math.min(2,devicePixelRatio||1);cv.width=w*dpr;cv.height=w*AR*dpr;cv.style.height=(w*AR)+'px';ctx.setTransform(dpr,0,0,dpr,0,0);pw=w;ph=w*AR}
const X=x=>pad+x/L*(pw-2*pad),Y=y=>pad+y/W*(ph-2*pad);
function draw(fi){const f=M.frames[Math.max(0,Math.min(NF-1,fi|0))];ctx.clearRect(0,0,pw,ph);
 ctx.fillStyle=css('--pitch');ctx.fillRect(X(0),Y(0),X(L)-X(0),Y(W)-Y(0));
 ctx.strokeStyle=css('--pline');ctx.lineWidth=1.4;ctx.strokeRect(X(0),Y(0),X(L)-X(0),Y(W)-Y(0));
 ctx.beginPath();ctx.moveTo(X(L/2),Y(0));ctx.lineTo(X(L/2),Y(W));ctx.stroke();
 ctx.beginPath();ctx.arc(X(L/2),Y(W/2),(X(L)-X(0))*.09,0,7);ctx.stroke();
 ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(X(0),Y(W/2-gh));ctx.lineTo(X(0),Y(W/2+gh));ctx.stroke();
 ctx.beginPath();ctx.moveTo(X(L),Y(W/2-gh));ctx.lineTo(X(L),Y(W/2+gh));ctx.stroke();
 f.a.forEach((p,i)=>dot(p,i===0?css('--panel'):css('--teamA'),i===0,f.own===0&&f.oi===i));
 f.b.forEach((p,i)=>dot(p,i===0?css('--panel'):css('--teamB'),i===0,f.own===1&&f.oi===i));
 const bx=X(f.ball[0]),by=Y(f.ball[1]);ctx.beginPath();ctx.arc(bx,by,3.2,0,7);ctx.fillStyle=css('--ball');ctx.fill();
 document.getElementById('sa').textContent=f.ga;document.getElementById('sb').textContent=f.gb;
 document.getElementById('clock').textContent=(fi/(M.hz||20)).toFixed(1)+'s'}
function dot(p,col,gk,onball){const x=X(p[0]),y=Y(p[1]);if(onball){ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.strokeStyle=css('--ball');ctx.lineWidth=2;ctx.stroke()}
 ctx.beginPath();ctx.arc(x,y,gk?5:4.4,0,7);if(gk){ctx.fillStyle=css('--panel');ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=col;ctx.stroke()}else{ctx.fillStyle=col;ctx.fill()}}
let fi=0,playing=true,last=0;const hz=M.hz||20;
function loop(t){if(playing){if(!last)last=t;fi+=(t-last)/1000*hz*0.6;last=t;if(fi>=NF-1){fi=NF-1;playing=false;pb.textContent='Replay'}draw(fi)}else last=0;requestAnimationFrame(loop)}
const pb=document.getElementById('play');pb.onclick=()=>{if(fi>=NF-1)fi=0;playing=!playing;pb.textContent=playing?'Pause':'Play';last=0};
document.getElementById('restart').onclick=()=>{fi=0;playing=true;pb.textContent='Pause';last=0};
document.getElementById('newgame').onclick=async e=>{const b=e.target,o=b.textContent;b.textContent='…recording';b.disabled=true;
 try{const r=await fetch('/newgame?seed='+((Math.random()*1e9)|0));if(!r.ok)throw 0;const m=await r.json();if(!m.frames)throw 0;M=m;NF=m.frames.length;fi=0;playing=true;pb.textContent='Pause';last=0;b.textContent=o}
 catch(_){b.textContent='needs live server';setTimeout(()=>b.textContent=o,1600)}finally{b.disabled=false}};
rs();draw(0);requestAnimationFrame(loop);addEventListener('resize',()=>{rs();draw(fi)});
// ladder curve
(function(){const c=document.getElementById('curve'),x=c.getContext('2d');
 function rs2(){const w=c.clientWidth||880,dpr=Math.min(2,devicePixelRatio||1);c.width=w*dpr;c.height=Math.round(w*.32)*dpr;c.style.height=Math.round(w*.32)+'px';x.setTransform(dpr,0,0,dpr,0,0);dr(w,Math.round(w*.32))}
 function dr(w,h){x.clearRect(0,0,w,h);if(!CURVE.length){return}const pL=40,pR=14,pT=14,pB=22;
  const mi=Math.max(...CURVE.map(p=>p.gen),1);let lo=Math.min(-1,...CURVE.flatMap(p=>[p.vs_champ,p.vs_scripted])),hi=Math.max(1,...CURVE.flatMap(p=>[p.vs_champ,p.vs_scripted]));const sp=hi-lo||1;lo-=sp*.1;hi+=sp*.1;
  const PX=i=>pL+i/mi*(w-pL-pR),PY=v=>pT+(1-(v-lo)/(hi-lo))*(h-pT-pB);
  x.strokeStyle=css('--line');x.fillStyle=css('--dim');x.font='10px monospace';x.textAlign='right';x.textBaseline='middle';
  for(let g=Math.ceil(lo);g<=Math.floor(hi);g++){const y=PY(g);x.globalAlpha=g?.5:1;x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.globalAlpha=1;x.fillText((g>0?'+':'')+g,pL-6,y)}
  const line=(key,col,lw)=>{x.strokeStyle=col;x.lineWidth=lw;x.beginPath();CURVE.forEach((p,i)=>{const px=PX(p.gen),py=PY(p[key]);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()};
  line('vs_scripted',css('--dim'),1.4);line('vs_champ',css('--accent'),2.2);
  CURVE.filter(p=>p.promoted).forEach(p=>{x.fillStyle=css('--good');x.beginPath();x.moveTo(PX(p.gen),PY(p.vs_champ)-6);x.lineTo(PX(p.gen)-4,PY(p.vs_champ)-1);x.lineTo(PX(p.gen)+4,PY(p.vs_champ)-1);x.fill()});
  x.fillStyle=css('--dim');x.textAlign='center';x.textBaseline='top';[0,mi].forEach(i=>x.fillText('gen '+i,PX(i),h-pB+5))}
 rs2();addEventListener('resize',rs2)})();
</script>"""

if __name__ == "__main__":
    main()
