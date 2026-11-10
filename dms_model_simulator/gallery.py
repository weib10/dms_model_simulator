"""Overview page of a catalog run: browse the catalog one thing at a time
(`dms-sim catalog gallery RUN_DIR` -> RUN_DIR/index.html).

Three levels, one on screen at a time: the map (behaviour family x category, a count and the entry numbers per
cell), the list of one behaviour family (one line per entry), and one entry (one row of it: script, camera view,
timeline; the other rows behind a selector; previous / next to walk through the catalog). Every entry shows its
fixed number (`#045`, the entry's `number`), so "look at number 45" finds it: on the map, in the lists, in the search.
The bar on top stays on screen and marks where you are; on a wide screen an entry page also lists its family on
the left (one list per family in the file, shown and highlighted by the script).

One self-contained HTML file with relative links: it reads the run's manifest.json, each row's card.json and the
pictures `catalog run --timeline / --gif` wrote (repro.PICTURE_FILES). Rows without pictures say so; entries the
run did not select are listed without rows, so the page always shows the whole catalog. The page holds no data of
its own: rebuild it after a run. Without JavaScript every level is shown one after the other."""
from __future__ import annotations
import json
from collections import Counter
from html import escape
from pathlib import Path
from . import catalog as C
from .repro import PICTURE_FILES

MAX_SCRIPT_LINES = 12          # a row's script table shows at most this many segments (long random drives)
SOURCE_ZH = {                  # an entry's `source`, in words (catalog.SOURCES)
    "written": "手寫劇本",
    "generated": "隨機產生",
    "recorded": "真錄影整理",
}
SOURCE_HELP = {
    "written": "人照行為逐段寫的劇本",
    "generated": "隨機產生器排出的長段駕駛",
    "recorded": "從真實錄影整理出的劇本",
}

FLOW = (   # how a catalog entry becomes files: (step, what it is, where)
    ("1 條目", "一條寫「駕駛做什麼」的劇本大綱，帶可調的參數（講電話幾秒、頭轉幾度）。", "catalog/entries/"),
    ("2 展開", "排成一列一列具體的劇本：每個難度或參數值 × 相機 × 模型誤差設定各一列。", "dms-sim catalog expand"),
    ("3 模擬", "照劇本演一次，記下真值（駕駛實際做了什麼）和模型輸出（真值加上漏框、抖動等誤差）。", "dms-sim catalog run"),
    ("4 考卷", "挑一批列當題目。後處理讀模型輸出、交答案卷，評分只拿答案卷跟真值比。", "dms-sim exam build / grade"),
)
TIMELINE_LEGEND = (
    ("gaze zone", "視線落在哪一區。藍線是真值，灰點是模型估的。"),
    ("eye openness", "眼睛開合度，1 全開、0 全閉。往下的尖刺是眨眼，壓在底部一段是閉眼。"),
    ("truth behaviour", "真值的行為標籤：有色塊的時段，駕駛真的在做那件事。後處理該抓的就是這些。"),
    ("injected model errors", "模擬器故意加進模型輸出的錯（掉幀、漏框、誤報）。後處理在這些時段出錯，要另外看。"),
)
CSS = """
:root{--ink:#1d1d1f;--ink2:#55555c;--muted:#8a8a93;--line:#e2e1dc;--bg:#fdfcf9;--card:#fff;--accent:#2a6fdb;
--sel:#eef3fc;--barh:52px;
--POS:#c2410c;--NEG:#15803d;--BND:#6d28d9;--CLR:#0e7490;--ROB:#b45309;--NAT:#475569;--REG:#be123c}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 system-ui,"Segoe UI","Microsoft JhengHei","PingFang TC",sans-serif;color:var(--ink);background:var(--bg)}
code{font:13px ui-monospace,Consolas,monospace;background:#f1f0eb;padding:1px 5px;border-radius:4px;word-break:break-all}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.top{position:sticky;top:0;z-index:10;background:rgba(253,252,249,.97);border-bottom:1px solid var(--line)}
.bar{max-width:1280px;margin:0 auto;padding:9px 28px;display:flex;gap:4px 6px;align-items:center;flex-wrap:wrap;font-size:14px}
.bar a{color:var(--ink2);padding:2px 10px;border-radius:14px}.bar a.home{font-weight:700;color:var(--ink)}
.bar a:hover{background:#f1f0eb;text-decoration:none}
.bar a.cur,.bar a.cur:hover{background:var(--accent);color:#fff}.bar .sp{flex:1}
#q{width:230px;padding:5px 10px;border:1px solid var(--line);border-radius:6px;font:inherit;font-size:14px;background:#fff}
#hits{position:absolute;right:28px;top:100%;width:420px;max-width:calc(100vw - 20px);max-height:60vh;overflow:auto;
background:#fff;border:1px solid var(--line);border-radius:6px;box-shadow:0 6px 18px rgba(0,0,0,.08);padding:4px 0;font-size:14px}
#hits:empty{display:none}#hits a{display:block;padding:4px 12px;color:var(--ink)}#hits a:hover,#hits a:first-child{background:var(--sel);text-decoration:none}
#hits p{margin:0;padding:4px 12px;color:var(--muted)}
.wrap{max-width:1280px;margin:0 auto;display:flex;align-items:flex-start}
main{flex:1;min-width:0;max-width:980px;margin:0 auto;padding:22px 28px 90px}
.side{display:none}
@media (min-width:1100px){
 .js body.ent .side{display:block;position:sticky;top:var(--barh);width:280px;flex:none;
  height:calc(100vh - var(--barh));overflow:auto;border-right:1px solid var(--line);padding:14px 0 40px;font-size:13px}}
.side .fam{display:none}.side .fam.on{display:block}
.side .sh{margin:0 14px 4px;font-weight:700}.side .sh a{color:var(--ink)}
.side .sc{margin:12px 14px 2px;font-size:12px;color:var(--ink2)}
.side ul{list-style:none;margin:0;padding:0}
.side li a{display:block;padding:2px 14px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-left:3px solid transparent}
.side li a:hover{background:#f3f2ed;text-decoration:none}
.side li a.cur{background:var(--sel);border-left-color:var(--accent);font-weight:600}
.no{font:600 13px ui-monospace,Consolas,monospace;color:var(--ink2)}
.code{font:12px ui-monospace,Consolas,monospace;color:var(--muted)}
h1{font-size:26px;line-height:1.35;margin:0 0 6px}h1 .no{font-size:20px;color:var(--accent);margin-right:6px}
h2{font-size:19px;margin:34px 0 4px}
h2,li,section{scroll-margin-top:calc(var(--barh) + 10px)}
.lead{color:var(--ink2);margin:0 0 10px}.small{font-size:13px;color:var(--ink2)}.muted{color:var(--muted)}
table{border-collapse:collapse;font-size:14px}th,td{border:1px solid var(--line);padding:6px 11px;text-align:left;vertical-align:top}
th{background:#f3f2ed;font-weight:600}
.map td.n{text-align:center;padding:0}.map td.n a{display:block;padding:6px 10px;color:var(--ink)}
.map td.n b{display:block;font-size:16px}.map td.n .rg{display:block;font-size:11px;color:var(--muted);white-space:nowrap}
.map td.n a:hover{background:var(--sel);text-decoration:none}.map th a{color:var(--ink)}.map th .code{display:block}
.map tr>th:first-child .code{display:inline;margin-left:6px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}
ul.facts{margin:6px 0;padding-left:20px;font-size:14px;color:var(--ink2)}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:10px 0}
.step{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.step b{display:block}.step p{margin:2px 0 6px;font-size:13px;color:var(--ink2)}
ul.cases{list-style:none;margin:6px 0 0;padding:0;border-top:1px solid var(--line)}
ul.cases li{border-bottom:1px solid var(--line)}
ul.cases a{display:flex;gap:10px;align-items:baseline;padding:9px 6px;color:var(--ink);border-left:3px solid transparent}
ul.cases a:hover{background:#f3f2ed;text-decoration:none}ul.cases .t{flex:1}ul.cases .k{font-size:12px;color:var(--muted);white-space:nowrap}
ul.cases a.was{background:var(--sel);border-left-color:var(--accent)}ul.cases a.was .k::before{content:"剛才看的 · ";color:var(--accent)}
.chip{display:inline-block;font-size:12px;padding:0 8px;border-radius:10px;border:1px solid var(--line);color:var(--ink2);margin:0 5px 4px 0;background:#fff}
.pick{margin:14px 0 4px;font-size:14px}.pick select{font:inherit;font-size:14px;padding:5px 8px;max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.fig{margin:20px 0 0}.fig h3{font-size:15px;margin:0 0 2px}.fig p{margin:0 0 6px;font-size:13px;color:var(--ink2)}
.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;display:block}
details.more{margin:18px 0 0;font-size:14px}details.more summary{cursor:pointer;color:var(--accent)}
.none{color:var(--muted);font-size:14px;margin:10px 0}
.pn{display:flex;justify-content:space-between;gap:16px;margin-top:34px;padding-top:14px;border-top:1px solid var(--line);font-size:14px}
.pn a{max-width:48%}
.js .view,.js .row{display:none}.js .view.on,.js .row.on{display:block}
.view+.view{margin-top:60px}.js .view+.view{margin-top:0}
"""
JS = """
document.documentElement.classList.add('js');
const views=[...document.querySelectorAll('.view')],topbar=document.querySelector('.top'),side=document.querySelector('.side');
function barh(){document.documentElement.style.setProperty('--barh',topbar.offsetHeight+'px');}
addEventListener('resize',barh);barh();
const fam=id=>id.split('-')[1],cat=id=>id.split('-')[0];
const links=sel=>[...document.querySelectorAll(sel)];
let last=null;   // the entry seen last: marked in its family's list
function show(){const h=decodeURIComponent(location.hash.slice(1));const p=h.split('/');
 let id='home';if(p[0]==='g'&&document.getElementById('g-'+p[1]))id='g-'+p[1];
 if(p[0]==='e'&&document.getElementById('e-'+p[1]))id='e-'+p[1];
 views.forEach(v=>v.classList.toggle('on',v.id===id));
 const ent=id[0]==='e',f=id==='home'?'':ent?fam(p[1]):p[1];
 links('.bar a[data-f]').forEach(a=>a.classList.toggle('cur',a.dataset.f===f));
 document.body.classList.toggle('ent',ent);
 links('.side .fam').forEach(n=>n.classList.toggle('on',ent&&n.id==='s-'+f));
 links('[data-e].cur,[data-e].was').forEach(a=>a.classList.remove('cur','was'));
 if(ent){last=p[1];
  const a=links('.side [data-e]').find(x=>x.dataset.e===last);
  if(a){a.classList.add('cur');
   if(a.offsetTop<side.scrollTop||a.offsetTop+a.offsetHeight>side.scrollTop+side.clientHeight)side.scrollTop=a.offsetTop-side.clientHeight/3;}
  scrollTo(0,0);
 }else if(id!=='home'){
  const w=last&&fam(last)===f?links('#'+id+' [data-e]').find(x=>x.dataset.e===last):null;
  if(w)w.classList.add('was');
  const sub=p[2]?document.getElementById('g-'+f+'-'+p[2]):null;
  if(w&&(!p[2]||cat(last)===p[2]))w.scrollIntoView({block:'center'});else if(sub)sub.scrollIntoView();else scrollTo(0,0);
 }else scrollTo(0,0);
 document.title=(document.querySelector('#'+CSS.escape(id)+' h1')||{}).textContent||document.title;}
addEventListener('hashchange',show);show();
document.querySelectorAll('.pick select').forEach(s=>s.addEventListener('change',()=>{
 s.closest('.view').querySelectorAll('.row').forEach((r,i)=>r.classList.toggle('on',i===s.selectedIndex));}));
addEventListener('keydown',ev=>{if(ev.target.matches('input,select'))return;
 const v=document.querySelector('.view.on');if(!v)return;
 const a=v.querySelector(ev.key==='ArrowLeft'?'.pn .prev':ev.key==='ArrowRight'?'.pn .next':'x');if(a)location.hash=a.hash;});
const q=document.getElementById('q'),hits=document.getElementById('hits');
function clearQ(){q.value='';hits.textContent='';}
q.addEventListener('input',()=>{const t=q.value.trim().toLowerCase();hits.textContent='';if(!t)return;
 // a number (45, #45, #045) is an entry number: that entry first, then text matches
 const num=/^#?[0-9]+$/.test(t)?parseInt(t.replace('#',''),10):null;
 const m=[...INDEX.filter(e=>num!==null&&e[3]===num),...INDEX.filter(e=>(num===null||e[3]!==num)&&e[2].includes(t.replace(/^#/,'')))].slice(0,12);
 if(!m.length){const n=document.createElement('p');n.textContent='找不到';hits.appendChild(n);return;}
 m.forEach(e=>{const a=document.createElement('a');a.href='#e/'+e[0];a.textContent=e[1];
  a.addEventListener('click',clearQ);hits.appendChild(a);});});
q.addEventListener('keydown',ev=>{if(ev.key==='Escape')clearQ();
 if(ev.key==='Enter'){const a=hits.querySelector('a');if(a){location.hash=a.hash;clearQ();q.blur();}}});
"""


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _cat_name(c: str) -> str:
    """Short plain name of a category: the part of its description before the colon."""
    return C.CATEGORIES[c].split("：")[0]


def _dot(c: str) -> str:
    return f'<span class="dot" style="background:var(--{c})"></span>'


def _no(e: dict) -> str:
    """`<span class="no">#045</span>` (empty when the entry has no number)."""
    n = C.entry_no(e)
    return f'<span class="no">{n}</span>' if n else ""


def _ranges(nums) -> str:
    """Entry numbers as short ranges: [45, 46, 47, 48] -> "#45–48", [45, 46, 47, 90] -> "#45–47, 90" ("" if none)."""
    ns, out, i = sorted(set(nums)), [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}–{ns[j]}" if j > i else f"{ns[i]}")
        i = j + 1
    return "#" + ", ".join(out) if out else ""


def _nums(es: list[dict]) -> list[int]:
    return [e["number"] for e in es if isinstance(e.get("number"), int) and not isinstance(e.get("number"), bool)]


def _starred(e: dict, behaviors: dict) -> bool:
    """★: the entry's main behaviour (the first of `behaviors`) is a must-test behaviour (catalog.is_core)."""
    return any(C.is_core(behaviors.get(b) or {}) for b in (e.get("behaviors") or [])[:1])


def _row_label(r: dict, card: dict | None) -> str:
    """Plain text that tells a row from its siblings: level, parameters, camera, error profile, seeds, length."""
    parts = []
    if r.get("level"):
        parts.append(f"難度 {r['level']}")
    parts += [f"{k}={_fmt(v)}" for k, v in (r.get("params") or {}).items() if k != "level"]
    parts += [f"相機 {r['camera']}", f"誤差 {r['profile']}"]
    if r.get("variant") is not None:
        parts.append(f"體型 v{r['variant']}")
    if r.get("seed"):
        parts.append(f"行為 seed {r['seed']}")
    if r.get("emulator_seed"):
        parts.append(f"誤差 seed {r['emulator_seed']}")
    if card:
        parts.append(f"{_fmt(float(card['duration_s']))} 秒")
    return " · ".join(parts)


def _script(card: dict) -> str:
    """The row's script as a small table (card.json `timeline`): when the driver does what."""
    segs = card.get("timeline") or []
    L = ["<table><tr><th>時間（秒）</th><th>駕駛在做什麼</th></tr>"]
    L += [f"<tr><td>{_fmt(float(s['t0']))}–{_fmt(float(s['t1']))}</td><td>{escape(s['text'])}</td></tr>"
          for s in segs[:MAX_SCRIPT_LINES]]
    if len(segs) > MAX_SCRIPT_LINES:
        L.append(f"<tr><td>…</td><td>共 {len(segs)} 段，完整的在行為卡</td></tr>")
    return "".join(L) + "</table>"


def _row_html(r: dict, run: Path, card: dict | None, first: bool) -> str:
    """One row of an entry: script, camera view, timeline, files. Only one row of an entry is on screen."""
    rid = r["row_id"]
    rd, rel = run / "rows" / rid, f"rows/{rid}"
    L = [f'<div class="row{" on" if first else ""}">']
    if card is None:
        L.append('<p class="none">這一列還沒匯出（<code>dms-sim catalog run</code>）。</p></div>')
        return "".join(L)
    L.append(f'<div class="fig"><h3>劇本</h3><p>駕駛在這一列照這張表做。</p>{_script(card)}</div>')
    if (rd / PICTURE_FILES["gif"]).is_file():
        L.append(f'<div class="fig"><h3>相機看到的</h3><p>線框臉是駕駛。綠框是真值，紅框是模型輸出的框，'
                 f'右上角紅底字是當下成立的行為。</p>'
                 f'<img loading="lazy" alt="相機畫面 {escape(rid)}" src="{rel}/{PICTURE_FILES["gif"]}"></div>')
    if (rd / PICTURE_FILES["timeline"]).is_file():
        L.append(f'<div class="fig"><h3>整段的時間軸</h3><p>橫軸是時間。第三條的色塊是駕駛真的在做的行為，'
                 f'最下面一條是模擬器故意加的模型錯誤。</p>'
                 f'<img loading="lazy" alt="時間軸 {escape(rid)}" src="{rel}/{PICTURE_FILES["timeline"]}"></div>')
    if not any((rd / n).is_file() for n in PICTURE_FILES.values()):
        L.append('<p class="none">這一列沒有圖：重跑 <code>dms-sim catalog run … --timeline --gif</code>。</p>')
    L.append(f'<details class="more"><summary>這一列的檔案</summary><p class="small">行為卡 '
             f'<a href="{rel}/card.md">card.md</a> · 劇本 '
             f'<a href="scenarios/{escape(Path(r["scenario"].replace(chr(92), "/")).name)}">YAML</a> · '
             f'row id <code>{escape(rid)}</code></p></details></div>')
    return "".join(L)


def _pn_link(e: dict, cls: str) -> str:
    n = C.entry_no(e)
    label = escape(f"{n} {e.get('title', '')}".strip())
    if cls == "prev":
        return f'<a class="prev" href="#e/{escape(e["id"])}">← {label}</a>'
    return f'<a class="next" href="#e/{escape(e["id"])}" style="text-align:right">{label} →</a>'


def _entry_html(e: dict, rows: list[dict], run: Path, behaviors: dict, prev: dict | None, nxt: dict | None,
                pos: tuple[int, int]) -> str:
    """One entry. pos: (place of the entry in its family, entries in the family), from 1."""
    c, f = C.category(e), C.family(e)
    cards = [_load_json(run / "rows" / r["row_id"] / "card.json") for r in rows]
    beh = "".join(f'<span class="chip">{"★ " if C.is_core(behaviors.get(b) or {}) else ""}{escape(b)} '
                  f'{escape(str((behaviors.get(b) or {}).get("name", "")))}</span>' for b in e.get("behaviors") or [])
    n = C.entry_no(e)
    where = f"{n}，" if n else ""
    src = "" if e.get("_orphan") else f' · 來源：{escape(SOURCE_ZH.get(e.get("source", "written"), str(e.get("source"))))}'
    L = [f'<section class="view" id="e-{escape(e["id"])}">',
         f'<p class="small"><a href="#g/{f}">{escape(C.FAMILIES[f])}</a> › '
         f'<a href="#g/{f}/{c}">{_dot(c)}{escape(_cat_name(c))}</a> › {where}這一家族第 {pos[0]}／{pos[1]} 條</p>',
         f'<h1>{_no(e)} {escape(str(e.get("title", "")))}</h1>',
         f'<p class="small"><code>{escape(e["id"])}</code>{src}</p>', f"<p>{beh}</p>"]
    if e.get("status", "ready") != "ready":
        L.append(f'<p class="none">planned：模擬器還做不到，缺 {escape("、".join(map(str, e.get("needs") or [])))}。</p>')
    elif not rows:
        L.append('<p class="none">這次的 run 沒有選這一條。</p>')
    elif len(rows) > 1:
        opts = "".join(f"<option>{escape(_row_label(r, cd))}</option>" for r, cd in zip(rows, cards))
        L.append(f'<p class="pick">這條情境有 {len(rows)} 列（同一個劇本換難度、相機或模型誤差各演一次），現在看的是：<br>'
                 f'<select aria-label="選一列">{opts}</select></p>')
    else:
        L.append(f'<p class="pick">{escape(_row_label(rows[0], cards[0]))}</p>')
    L += [_row_html(r, run, cd, i == 0) for i, (r, cd) in enumerate(zip(rows, cards))]
    if e.get("summary"):
        L.append(f'<details class="more"><summary>這條情境的說明</summary><p class="small">{escape(str(e["summary"]))}</p>'
                 f'<p class="small">標籤：{escape("、".join(map(str, e.get("tags") or [])))}</p></details>')
    L.append('<p class="pn">' + (_pn_link(prev, "prev") if prev else "<span></span>")
             + (_pn_link(nxt, "next") if nxt else "") + "</p></section>")
    return "\n".join(L)


def _star_note(n_core: int) -> str:
    return (f"★ = 這條的主行為（列的第一個行為）是必測行為：<code>catalog/behaviors.yaml</code> 裡標 "
            f"<code>must_test: true</code> 的 {n_core} 個行為。這是本專案自己訂的清單，不是法規要求。")


def _group_html(f: str, cats: list[str], entries: list[dict], by_entry: dict, behaviors: dict, n_core: int) -> str:
    """The list of one behaviour family: per category, one line per entry (number first)."""
    es = [e for e in entries if C.family(e) == f]
    rg = _ranges(_nums(es))
    L = [f'<section class="view" id="g-{f}">', f'<h1>{escape(C.FAMILIES[f])} <span class="code">{f}</span></h1>',
         f'<p class="lead">{len(es)} 條情境{f"（{rg}）" if rg else ""}，照「在考什麼」分組。點一條看它的圖。</p>']
    if any(_starred(e, behaviors) for e in es):
        L.append(f'<p class="small">{_star_note(n_core)}</p>')
    for c in cats:
        ces = [e for e in es if C.category(e) == c]
        if not ces:
            continue
        L += [f'<h2 id="g-{f}-{c}">{_dot(c)}{escape(_cat_name(c))} <span class="code">{c}</span></h2>',
              f'<p class="small">{escape(C.CATEGORIES[c].split("：", 1)[-1])}</p>', '<ul class="cases">']
        for e in ces:
            n = len(by_entry.get(e["id"], []))
            star = "★ " if _starred(e, behaviors) else ""
            note = "planned" if e.get("status", "ready") != "ready" else f"{n} 列" if n else "沒跑"
            L.append(f'<li><a href="#e/{escape(e["id"])}" data-e="{escape(e["id"])}">{_no(e)}'
                     f'<span class="t">{star}{escape(str(e.get("title", "")))}</span><span class="k">{note}</span></a></li>')
        L.append("</ul>")
    L.append("</section>")
    return "\n".join(L)


def _side_html(f: str, cats: list[str], entries: list[dict], behaviors: dict) -> str:
    """The family column of the entry pages (wide screens only): one per family, the script shows the current one."""
    es = [e for e in entries if C.family(e) == f]
    L = [f'<nav class="fam" id="s-{f}" aria-label="{escape(C.FAMILIES[f], quote=True)}">',
         f'<p class="sh"><a href="#g/{f}">{escape(C.FAMILIES[f])}</a> <span class="muted">{len(es)} 條</span></p>']
    for c in cats:
        ces = [e for e in es if C.category(e) == c]
        if not ces:
            continue
        L.append(f'<p class="sc">{_dot(c)}{escape(_cat_name(c))}</p><ul>')
        for e in ces:
            t = f'{"★ " if _starred(e, behaviors) else ""}{e.get("title", "")}'
            L.append(f'<li><a href="#e/{escape(e["id"])}" data-e="{escape(e["id"])}" title="{escape(t, quote=True)}">'
                     f'{_no(e)} {escape(t)}</a></li>')
        L.append("</ul>")
    L.append("</nav>")
    return "\n".join(L)


def gallery_html(run_dir, entries: list[dict] | None = None, title: str | None = None) -> str:
    """The overview page of a run as one HTML document. entries: the catalog to list (default: the repo's)."""
    run = Path(run_dir)
    man = C.load_manifest(run)
    catalog = entries if entries is not None else C.load_entries()
    known = {e["id"] for e in catalog}
    # rows of an entry the catalog no longer has still get a page (built from the manifest)
    entries = catalog + [{"id": i, "title": next(r["title"] for r in man["rows"] if r["entry"] == i),
                          "summary": "這一條已不在目錄裡，內容來自 manifest。", "_orphan": True}
                         for i in man["entries"] if i not in known]
    behaviors = C.load_behaviors()
    n_core = sum(C.is_core(b) for b in behaviors.values())
    by_entry: dict[str, list[dict]] = {}
    for r in man["rows"]:
        by_entry.setdefault(r["entry"], []).append(r)
    n_pic = sum(any((run / "rows" / r["row_id"] / n).is_file() for n in PICTURE_FILES.values()) for r in man["rows"])
    cats = [c for c in C.CATEGORIES if any(C.category(e) == c for e in entries)]
    fams = [f for f in C.FAMILIES if any(C.family(e) == f for e in entries)]
    # reading order: family, then category, then number (entries without one last, by id): the order of the lists,
    # of previous / next, and of the first numbering, so a map cell starts as one run of numbers
    order = sorted(entries, key=lambda e: (fams.index(C.family(e)), cats.index(C.category(e)), not _nums([e]),
                                           _nums([e]) or [0], e["id"]))

    def cell(f, c=None):
        return [e for e in order if C.family(e) == f and (c is None or C.category(e) == c)]

    def cell_html(f, c=None):
        es = cell(f, c)
        if not es:
            return '<td class="n"></td>'
        rg = _ranges(_nums(es))
        rg = f'<span class="rg">{rg}</span>' if rg else ""
        return f'<td class="n"><a href="#g/{f}{"/" + c if c else ""}"><b>{len(es)}</b>{rg}</a></td>'

    head = "".join(f'<th title="{escape(C.CATEGORIES[c], quote=True)}">{_dot(c)}{escape(_cat_name(c))}'
                   f'<span class="code">{c}</span></th>' for c in cats)
    matrix = [f'<table class="map"><tr><th>行為＼在考什麼</th>{head}<th>共</th></tr>']
    for f in fams:
        matrix.append(f'<tr><th><a href="#g/{f}">{escape(C.FAMILIES[f])}</a><span class="code">{f}</span></th>'
                      f'{"".join(cell_html(f, c) for c in cats)}{cell_html(f)}</tr>')
    matrix.append("</table>")
    cat_help = "".join(f"<tr><td>{_dot(c)}{escape(_cat_name(c))} <span class=\"code\">{c}</span></td>"
                       f"<td>{escape(C.CATEGORIES[c].split('：', 1)[-1])}</td></tr>" for c in cats)
    flow = "".join(f'<div class="step"><b>{escape(a)}</b><p>{escape(b)}</p><code>{escape(cmd)}</code></div>'
                   for a, b, cmd in FLOW)
    legend = "".join(f"<tr><td><b>{escape(a)}</b></td><td>{escape(b)}</td></tr>" for a, b in TIMELINE_LEGEND)
    title = title or "DMS 模型輸出模擬器：情境總覽"
    in_run = sum(bool(by_entry.get(e["id"])) for e in entries)
    n_star = sum(_starred(e, behaviors) for e in catalog)
    src = Counter(e.get("source", "written") for e in catalog)
    src_txt = "、".join(f"{SOURCE_ZH[k]}（{SOURCE_HELP[k]}）{src.get(k, 0)} 條" for k in SOURCE_ZH)
    n_sum = sum(bool(e.get("summary")) for e in catalog)
    sum_txt = "每條" if n_sum == len(catalog) else f"有說明的 {n_sum} 條（共 {len(catalog)} 條），每條"
    index = [[e["id"], f"{C.entry_no(e)} {e.get('title', '')}".strip(),
              " ".join([e["id"], str(e.get("title", "")), *map(str, e.get("tags") or []),
                        *map(str, e.get("behaviors") or [])]).lower(), e.get("number") if _nums([e]) else None]
             for e in order]
    home = f"""<section class="view on" id="home">
<h1>{escape(title)}</h1>
<p class="lead">目錄有 {len(catalog)} 條情境，每條一個固定編號（#001 起）。先在下面的表選一種行為，再一條一條看：每條一段相機畫面、一張時間軸。</p>
{"".join(matrix)}
<p class="small">格子裡是條數，小字是那幾條的編號；點了看那一組的清單。要找某一號，右上角搜尋打號碼（例如 45）。
這次的 run 跑了 {in_run} 條、{len(man["rows"])} 列，{n_pic} 列有圖。</p>
<p class="small">{_star_note(n_core)}目錄裡 {n_star} 條標了 ★（標在清單的標題前）。<code>tests/test_behaviors.py</code>
檢查列了這些行為的條目，在每個 seed、每個難度的真值都符合預期。</p>
<h2>這些情境怎麼來的</h2>
<ul class="facts">
<li>每條標了來源（條目的 <code>source</code> 欄位）：{escape(src_txt)}。</li>
<li>{sum_txt}的門檻與時長為什麼這樣設，寫在該條的說明裡（單條頁下方「這條情境的說明」）。</li>
<li>目前沒有逐條對應到法規或標準條文。</li>
</ul>
<h2>欄位是什麼意思</h2>
<table>{cat_help}</table>
<h2>模擬器怎麼運作</h2>
<p class="lead">它只做一件事：照劇本產生「模型會輸出什麼」和「駕駛其實做了什麼」。裡面沒有告警邏輯。</p>
<div class="flow">{flow}</div>
<details class="more"><summary>時間軸圖怎麼讀</summary><table>{legend}</table></details>
<details class="more"><summary>這一頁怎麼來的</summary><p class="small">由 <code>dms-sim catalog gallery</code> 從這個 run 產生，不要手改；
版本指紋 <code>{escape(str(man.get("fingerprint_id", "")))}</code>。一條情境會變成很多列，row id 把差別寫在名字裡：
<code>條目__難度__劇本雜湊__相機__體型__誤差seed__誤差設定</code>。</p></details>
</section>"""
    groups = [_group_html(f, cats, order, by_entry, behaviors, n_core) for f in fams]
    sides = [_side_html(f, cats, order, behaviors) for f in fams]
    fam_of = {f: [e["id"] for e in order if C.family(e) == f] for f in fams}
    pages = [_entry_html(e, by_entry.get(e["id"], []), run, behaviors, order[i - 1] if i else None,
                         order[i + 1] if i + 1 < len(order) else None,
                         (fam_of[C.family(e)].index(e["id"]) + 1, len(fam_of[C.family(e)])))
             for i, e in enumerate(order)]
    nav = "".join(f'<a href="#g/{f}" data-f="{f}">{escape(C.FAMILIES[f])}</a>' for f in fams)
    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head><body>
<header class="top"><nav class="bar"><a class="home" href="#" data-f="">情境總覽</a>{nav}
<span class="sp"></span><input id="q" type="search" placeholder="搜尋：編號、標題、id、行為編號" aria-label="搜尋情境"></nav>
<div id="hits"></div></header>
<div class="wrap"><aside class="side">
{chr(10).join(sides)}
</aside><main>
{home}
{chr(10).join(groups)}
{chr(10).join(pages)}
</main></div><script>const INDEX={json.dumps(index, ensure_ascii=False).replace("</", "<\\/")};{JS}</script></body></html>
"""


def write_gallery(run_dir, entries: list[dict] | None = None, title: str | None = None) -> Path:
    """Write the overview page to <run>/index.html (its links are relative to the run directory)."""
    out = Path(run_dir) / "index.html"
    out.write_text(gallery_html(run_dir, entries, title), encoding="utf-8", newline="\n")
    return out
