"""Overview page of a catalog run: browse the catalog one thing at a time
(`dms-sim catalog gallery RUN_DIR` -> RUN_DIR/index.html).

Three levels, one on screen at a time: the map (behaviour family x category, a count per cell), the list of one
behaviour family (one line per entry), and one entry (one row of it: script, camera view, timeline; the other rows
behind a selector; previous / next to walk through the catalog).

One self-contained HTML file with relative links: it reads the run's manifest.json, each row's card.json and the
pictures `catalog run --timeline / --gif` wrote (repro.PICTURE_FILES). Rows without pictures say so; entries the
run did not select are listed without rows, so the page always shows the whole catalog. The page holds no data of
its own: rebuild it after a run. Without JavaScript every level is shown one after the other."""
from __future__ import annotations
import json
from html import escape
from pathlib import Path
from . import catalog as C
from .repro import PICTURE_FILES

MAX_SCRIPT_LINES = 12          # a row's script table shows at most this many segments (long random drives)

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
--POS:#c2410c;--NEG:#15803d;--BND:#6d28d9;--CLR:#0e7490;--ROB:#b45309;--NAT:#475569;--REG:#be123c}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 system-ui,"Segoe UI","Microsoft JhengHei","PingFang TC",sans-serif;color:var(--ink);background:var(--bg)}
code{font:13px ui-monospace,Consolas,monospace;background:#f1f0eb;padding:1px 5px;border-radius:4px;word-break:break-all}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
main{max-width:980px;margin:0 auto;padding:22px 28px 90px}
.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:14px;color:var(--ink2);margin-bottom:18px}
.bar a.home{font-weight:700;color:var(--ink)}.bar .sp{flex:1}
#q{width:230px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;font:inherit;font-size:14px;background:#fff}
#hits{max-width:980px;margin:-8px auto 12px}#hits a{display:block;padding:3px 0;font-size:14px}
h1{font-size:26px;line-height:1.35;margin:0 0 6px}h2{font-size:19px;margin:34px 0 4px}
.lead{color:var(--ink2);margin:0 0 10px}.small{font-size:13px;color:var(--ink2)}.muted{color:var(--muted)}
table{border-collapse:collapse;font-size:14px}th,td{border:1px solid var(--line);padding:6px 11px;text-align:left;vertical-align:top}
th{background:#f3f2ed;font-weight:600}
.map td.n{text-align:center;padding:0}.map td.n a{display:block;padding:8px 14px;font-weight:700;font-size:16px}
.map td.n a:hover{background:#eef3fc;text-decoration:none}.map th a{color:var(--ink)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:10px 0}
.step{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.step b{display:block}.step p{margin:2px 0 6px;font-size:13px;color:var(--ink2)}
ul.cases{list-style:none;margin:6px 0 0;padding:0;border-top:1px solid var(--line)}
ul.cases li{border-bottom:1px solid var(--line)}
ul.cases a{display:flex;gap:10px;align-items:baseline;padding:9px 6px;color:var(--ink)}
ul.cases a:hover{background:#f3f2ed;text-decoration:none}ul.cases .t{flex:1}ul.cases .k{font-size:12px;color:var(--muted);white-space:nowrap}
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
const views=[...document.querySelectorAll('.view')];
function show(){const h=decodeURIComponent(location.hash.slice(1));const p=h.split('/');
 let id='home';if(p[0]==='g'&&document.getElementById('g-'+p[1]))id='g-'+p[1];
 if(p[0]==='e'&&document.getElementById('e-'+p[1]))id='e-'+p[1];
 views.forEach(v=>v.classList.toggle('on',v.id===id));
 const sub=p[0]==='g'&&p[2]?document.getElementById('g-'+p[1]+'-'+p[2]):null;
 if(sub)sub.scrollIntoView();else scrollTo(0,0);
 document.title=(document.querySelector('#'+CSS.escape(id)+' h1')||{}).textContent||document.title;}
addEventListener('hashchange',show);show();
document.querySelectorAll('.pick select').forEach(s=>s.addEventListener('change',()=>{
 s.closest('.view').querySelectorAll('.row').forEach((r,i)=>r.classList.toggle('on',i===s.selectedIndex));}));
addEventListener('keydown',ev=>{if(ev.target.matches('input,select'))return;
 const v=document.querySelector('.view.on');if(!v)return;
 const a=v.querySelector(ev.key==='ArrowLeft'?'.pn .prev':ev.key==='ArrowRight'?'.pn .next':'x');if(a)location.hash=a.hash;});
const q=document.getElementById('q'),hits=document.getElementById('hits');
q.addEventListener('input',()=>{const t=q.value.trim().toLowerCase();hits.textContent='';if(!t)return;
 const m=INDEX.filter(e=>e[2].includes(t)).slice(0,12);
 if(!m.length){hits.textContent='找不到';return;}
 m.forEach(e=>{const a=document.createElement('a');a.href='#e/'+e[0];a.textContent=e[1];
  a.addEventListener('click',()=>{q.value='';hits.textContent='';});hits.appendChild(a);});});
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


def _entry_html(e: dict, rows: list[dict], run: Path, behaviors: dict, prev: dict | None, nxt: dict | None) -> str:
    c, f = C.category(e), C.family(e)
    cards = [_load_json(run / "rows" / r["row_id"] / "card.json") for r in rows]
    beh = "".join(f'<span class="chip">{"★ " if C.is_core(behaviors.get(b) or {}) else ""}{escape(b)} '
                  f'{escape(str((behaviors.get(b) or {}).get("name", "")))}</span>' for b in e.get("behaviors") or [])
    L = [f'<section class="view" id="e-{escape(e["id"])}">',
         f'<p class="small"><a href="#g/{f}">{escape(C.FAMILIES[f])}</a> › '
         f'<a href="#g/{f}/{c}">{_dot(c)}{escape(_cat_name(c))}</a></p>',
         f'<h1>{escape(str(e.get("title", "")))}</h1>',
         f'<p class="small"><code>{escape(e["id"])}</code></p>', f"<p>{beh}</p>"]
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
    L.append('<p class="pn">'
             + (f'<a class="prev" href="#e/{escape(prev["id"])}">← {escape(str(prev.get("title", "")))}</a>' if prev else "<span></span>")
             + (f'<a class="next" href="#e/{escape(nxt["id"])}" style="text-align:right">{escape(str(nxt.get("title", "")))} →</a>' if nxt else "")
             + "</p></section>")
    return "\n".join(L)


def _group_html(f: str, cats: list[str], entries: list[dict], by_entry: dict, behaviors: dict) -> str:
    """The list of one behaviour family: per category, one line per entry."""
    es = [e for e in entries if C.family(e) == f]
    L = [f'<section class="view" id="g-{f}">', f"<h1>{escape(C.FAMILIES[f])}</h1>",
         f'<p class="lead">{len(es)} 條情境，照「在考什麼」分組。點一條看它的圖。</p>']
    for c in cats:
        ces = [e for e in es if C.category(e) == c]
        if not ces:
            continue
        L += [f'<h2 id="g-{f}-{c}">{_dot(c)}{escape(_cat_name(c))}</h2>',
              f'<p class="small">{escape(C.CATEGORIES[c].split("：", 1)[-1])}</p>', '<ul class="cases">']
        for e in ces:
            n = len(by_entry.get(e["id"], []))
            star = "★ " if any(C.is_core(behaviors.get(b) or {}) for b in (e.get("behaviors") or [])[:1]) else ""
            note = "planned" if e.get("status", "ready") != "ready" else f"{n} 列" if n else "沒跑"
            L.append(f'<li><a href="#e/{escape(e["id"])}"><span class="t">{star}{escape(str(e.get("title", "")))}</span>'
                     f'<span class="k">{note}</span></a></li>')
        L.append("</ul>")
    L.append("</section>")
    return "\n".join(L)


def gallery_html(run_dir, entries: list[dict] | None = None, title: str | None = None) -> str:
    """The overview page of a run as one HTML document. entries: the catalog to list (default: the repo's)."""
    run = Path(run_dir)
    man = C.load_manifest(run)
    entries = entries if entries is not None else C.load_entries()
    known = {e["id"] for e in entries}
    # rows of an entry the catalog no longer has still get a page (built from the manifest)
    entries = entries + [{"id": i, "title": next(r["title"] for r in man["rows"] if r["entry"] == i),
                          "summary": "這一條已不在目錄裡，內容來自 manifest。"}
                         for i in man["entries"] if i not in known]
    behaviors = C.load_behaviors()
    by_entry: dict[str, list[dict]] = {}
    for r in man["rows"]:
        by_entry.setdefault(r["entry"], []).append(r)
    n_pic = sum(any((run / "rows" / r["row_id"] / n).is_file() for n in PICTURE_FILES.values()) for r in man["rows"])
    cats = [c for c in C.CATEGORIES if any(C.category(e) == c for e in entries)]
    fams = [f for f in C.FAMILIES if any(C.family(e) == f for e in entries)]
    # reading order: family, then category, then id (the order of the lists and of previous / next)
    order = sorted(entries, key=lambda e: (fams.index(C.family(e)), cats.index(C.category(e)), e["id"]))

    def count(f, c):
        return sum(C.family(e) == f and C.category(e) == c for e in entries)

    head = "".join(f'<th title="{escape(C.CATEGORIES[c], quote=True)}">{_dot(c)}{escape(_cat_name(c))}</th>' for c in cats)
    matrix = [f'<table class="map"><tr><th>行為＼在考什麼</th>{head}<th>共</th></tr>']
    for f in fams:
        cells = "".join(f'<td class="n"><a href="#g/{f}/{c}">{count(f, c)}</a></td>' if count(f, c) else '<td class="n"></td>'
                        for c in cats)
        matrix.append(f'<tr><th><a href="#g/{f}">{escape(C.FAMILIES[f])}</a></th>{cells}'
                      f'<td class="n"><a href="#g/{f}">{sum(C.family(e) == f for e in entries)}</a></td></tr>')
    matrix.append("</table>")
    cat_help = "".join(f"<tr><td>{_dot(c)}{escape(_cat_name(c))}</td><td>{escape(C.CATEGORIES[c].split('：', 1)[-1])}</td></tr>"
                       for c in cats)
    flow = "".join(f'<div class="step"><b>{escape(a)}</b><p>{escape(b)}</p><code>{escape(cmd)}</code></div>'
                   for a, b, cmd in FLOW)
    legend = "".join(f"<tr><td><b>{escape(a)}</b></td><td>{escape(b)}</td></tr>" for a, b in TIMELINE_LEGEND)
    title = title or "DMS 模型輸出模擬器：情境總覽"
    in_run = sum(bool(by_entry.get(e["id"])) for e in entries)
    index = [[e["id"], str(e.get("title", "")),
              " ".join([e["id"], str(e.get("title", "")), *map(str, e.get("tags") or []),
                        *map(str, e.get("behaviors") or [])]).lower()] for e in order]
    home = f"""<section class="view on" id="home">
<h1>{escape(title)}</h1>
<p class="lead">目錄有 {len(entries)} 條情境。先在下面的表選一種行為，再一條一條看：每條一段相機畫面、一張時間軸。</p>
{"".join(matrix)}
<p class="small">格子裡是條數，點了看那一組的清單。這次的 run 跑了 {in_run} 條、{len(man["rows"])} 列，{n_pic} 列有圖。</p>
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
    groups = [_group_html(f, cats, order, by_entry, behaviors) for f in fams]
    pages = [_entry_html(e, by_entry.get(e["id"], []), run, behaviors, order[i - 1] if i else None,
                         order[i + 1] if i + 1 < len(order) else None) for i, e in enumerate(order)]
    return f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head><body><main>
<div class="bar"><a class="home" href="#">情境總覽</a>{"".join(f'<a href="#g/{f}">{escape(C.FAMILIES[f])}</a>' for f in fams)}
<span class="sp"></span><input id="q" type="search" placeholder="搜尋：標題、id、行為編號" aria-label="搜尋情境"></div>
<div id="hits"></div>
{home}
{chr(10).join(groups)}
{chr(10).join(pages)}
</main><script>const INDEX={json.dumps(index, ensure_ascii=False).replace("</", "<\\/")};{JS}</script></body></html>
"""


def write_gallery(run_dir, entries: list[dict] | None = None, title: str | None = None) -> Path:
    """Write the overview page to <run>/index.html (its links are relative to the run directory)."""
    out = Path(run_dir) / "index.html"
    out.write_text(gallery_html(run_dir, entries, title), encoding="utf-8", newline="\n")
    return out
