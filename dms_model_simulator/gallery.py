"""Overview page of a catalog run: browse the catalog one thing at a time
(`dms-sim catalog gallery RUN_DIR` -> RUN_DIR/index.html in Chinese and RUN_DIR/index.en.html in English).

Three levels, one on screen at a time: the map (behaviour family x category, a count and the entry numbers per
cell), the list of one behaviour family (one line per entry), and one entry (one row of it: script, camera view,
timeline; the other rows behind a selector; previous / next to walk through the catalog). Every entry shows its
fixed number (`#045`, the entry's `number`), so "look at number 45" finds it: on the map, in the lists, in the search.
Lists, previous / next and search results show the number with the entry's short name (`short` / `short_en`); the
entry page puts the long title under it and the levels from `brief` / `brief_en` (small / medium / large, all).
The standards page sets our own `premise` (standard) and `note` (requirement) of the overlay apart from the texts.
The bar on top stays on screen and marks where you are; on a wide screen an entry page also lists its family on
the left (one list per family in the file, shown and highlighted by the script).

Standards (catalog/standards.yaml) get a page each (`#s/<id>`): the standard in a few sentences, then its
requirements grouped by coverage (entries related / no entry yet / out of the simulator's reach). Entries whose
`standards` field names a standard carry its code as a small mark (list line, family column, entry page, a count
per map cell). The mark means related, never compliant. The public standards.yaml has neutral markers only (an id,
requirement ids, coverage): the page then says so and lists ids. The names and texts come from the git-ignored local
overlay (catalog/standards.local.yaml, merged in by catalog.load_standards(local=True); `catalog gallery` does that
when the file exists, `--no-local` not); a standard is shown by its `name` when it has one, the route stays its id.

Two languages, one file each: every text of the page itself is in TEXT (one table per language, same keys);
entries, behaviours, categories and families bring their own English (`title_en`, `summary_en`, `name_en`,
catalog.CATEGORIES_EN / FAMILIES_EN), and a missing English text falls back to the Chinese. The script lines of a
row come from its card.json `timeline[].keys` (behavior_card.seg_text). The two pages link to each other at the
same place (`#...`).

One self-contained HTML file per language with relative links: it reads the run's manifest.json, each row's
card.json and the pictures `catalog run --timeline / --gif` wrote (repro.PICTURE_FILES). Rows without pictures say
so; entries the run did not select are listed without rows, so the page always shows the whole catalog. The page
holds no data of its own: rebuild it after a run. Without JavaScript every level is shown one after the other."""
from __future__ import annotations
import json
from collections import Counter
from html import escape
from pathlib import Path
from . import catalog as C
from .behavior_card import seg_text
from .repro import PICTURE_FILES

MAX_SCRIPT_LINES = 12          # a row's script table shows at most this many segments (long random drives)
LANGS = ("zh", "en")
FILES = {"zh": "index.html", "en": "index.en.html"}
COVERAGE_ORDER = ("partial", "gap", "out_of_scope")

# Every text of the page itself, per language (same keys in both; tests compare them). Values with HTML are ours.
TEXT = {
    "zh": {
        "html_lang": "zh-Hant",
        "title": "DMS 模型輸出模擬器：情境總覽",
        "home": "情境總覽",
        "search_ph": "搜尋：編號、標題、id、行為編號",
        "search_aria": "搜尋情境",
        "not_found": "找不到",
        "was": "剛才看的 · ",
        "other_lang": "English",
        "other_lang_hreflang": "en",
        "other_lang_title": "這一頁的英文版",
        "list_sep": "、",
        "source_names": {"written": "手寫劇本", "generated": "隨機產生", "recorded": "真錄影整理"},
        "source_help": {"written": "人照行為逐段寫的劇本", "generated": "隨機產生器排出的長段駕駛",
                        "recorded": "從真實錄影整理出的劇本"},
        "flow": (   # how a catalog entry becomes files: (step, what it is, where)
            ("1 條目", "一條寫「駕駛做什麼」的劇本大綱，帶可調的參數（講電話幾秒、頭轉幾度）。", "catalog/entries/"),
            ("2 展開", "排成一列一列具體的劇本：每個難度或參數值 × 相機 × 模型誤差設定各一列。", "dms-sim catalog expand"),
            ("3 模擬", "照劇本演一次，記下真值（駕駛實際做了什麼）和模型輸出（真值加上漏框、抖動等誤差）。", "dms-sim catalog run"),
            ("4 考卷", "挑一批列當題目。後處理讀模型輸出、交答案卷，評分只拿答案卷跟真值比。", "dms-sim exam build / grade"),
        ),
        "legend": (
            ("gaze zone", "視線落在哪一區。藍線是真值，灰點是模型估的。"),
            ("eye openness", "眼睛開合度，1 全開、0 全閉。往下的尖刺是眨眼，壓在底部一段是閉眼。"),
            ("truth behaviour", "真值的行為標籤：有色塊的時段，駕駛真的在做那件事。後處理該抓的就是這些。"),
            ("injected model errors", "模擬器故意加進模型輸出的錯（掉幀、漏框、誤報）。後處理在這些時段出錯，要另外看。"),
        ),
        # a row
        "row_level": "難度 {}", "row_camera": "相機 {}", "row_profile": "誤差 {}", "row_variant": "體型 v{}",
        "row_seed": "行為 seed {}", "row_eseed": "誤差 seed {}", "row_secs": "{} 秒",
        "script_time": "時間（秒）", "script_what": "駕駛在做什麼", "script_more": "共 {n} 段，完整的在行為卡",
        "not_exported": "這一列還沒匯出（<code>dms-sim catalog run</code>）。",
        "fig_script": "劇本", "fig_script_p": "駕駛在這一列照這張表做。",
        "fig_gif": "相機看到的", "fig_gif_p": "線框臉是駕駛。綠框是真值，紅框是模型輸出的框，右上角紅底字是當下成立的行為。",
        "gif_alt": "相機畫面 {}",
        "fig_tl": "整段的時間軸", "fig_tl_p": "橫軸是時間。第三條的色塊是駕駛真的在做的行為，最下面一條是模擬器故意加的模型錯誤。",
        "tl_alt": "時間軸 {}",
        "no_pics": "這一列沒有圖：重跑 <code>dms-sim catalog run … --timeline --gif</code>。",
        "row_files": "這一列的檔案", "row_files_card": "行為卡", "row_files_yaml": "劇本",
        # an entry
        "crumb_where": "{}，", "crumb_place": "這一家族第 {i}／{n} 條", "source_label": " · 來源：{}",
        "planned": "planned：模擬器還做不到，缺 {}。", "not_in_run": "這次的 run 沒有選這一條。",
        "pick": "這條情境有 {n} 列（同一個劇本換難度、相機或模型誤差各演一次），現在看的是：", "pick_aria": "選一列",
        "about_entry": "這條情境的說明", "tags": "標籤：{}",
        "orphan_summary": "這一條已不在目錄裡，內容來自 manifest。",
        "ent_std_lead": "跟標準的關係（標的是相關，不是符合）",
        "ent_std_topic": "跟這個標準管的主題有關，沒有對到特定一項。",
        "req_clause": "{rid}（{clause}）：", "req_id": "{rid}：",
        "note_lead": "我們的解讀：", "premise_h": "我們的前提（條文沒有確認）",
        "lv_lead": "同一個劇本的三個難度：small → medium → large，從最容易到最難分辨。", "lv_all": "每級相同",
        "lv_one": "這一條跑的：",
        "ent_std_ids": "{std}：{ids}",
        # a family list
        "group_lead": "{n} 條情境{rg}，照「在考什麼」分組。點一條看它的圖。", "group_rg": "（{}）",
        "rows_n": "{n} 列", "rows_1": "1 列", "not_run": "沒跑", "side_n": "{n} 條",
        "star_note": "★ = 這條的主行為（列的第一個行為）是必測行為：<code>catalog/behaviors.yaml</code> 裡標 "
                     "<code>must_test: true</code> 的 {n} 個行為。這是本專案自己訂的清單，不是法規要求。",
        # the home page
        "home_lead": "目錄有 {n} 條情境，每條一個固定編號（#001 起）。先在下面的表選一種行為，再一條一條看：每條一段相機畫面、一張時間軸。",
        "map_corner": "行為＼在考什麼", "map_total": "共",
        "map_note": "格子裡是條數，小字是那幾條的編號；點了看那一組的清單。要找某一號，右上角搜尋打號碼（例如 45）。\n"
                    "這次的 run 跑了 {e} 條、{r} 列，{p} 列有圖。",
        "map_std_title": "{n} 條跟 {name} 有關",
        "home_std": "{n} 條情境標了跟 <a href=\"#s/{sid}\">{name}</a> 哪幾項要求有關（地圖格子裡的 {name} 小字是條數；只標相關，不表示符合）。",
        "home_star": "目錄裡 {n} 條標了 ★（標在清單的標題前）。<code>tests/test_behaviors.py</code>\n"
                     "檢查列了這些行為的條目，在每個 seed、每個難度的真值都符合預期。",
        "h_origin": "這些情境怎麼來的",
        "origin_src": "每條標了來源（條目的 <code>source</code> 欄位）：{}。", "origin_src_item": "{name}（{help}）{n} 條",
        "origin_sum_all": "每條", "origin_sum_some": "有說明的 {n} 條（共 {total} 條），每條",
        "origin_sum": "{which}的門檻與時長為什麼這樣設，寫在該條的說明裡（單條頁下方「這條情境的說明」）。",
        "origin_no_std": "目前沒有逐條對應到法規或標準條文。",
        "h_fields": "欄位是什麼意思", "h_how": "模擬器怎麼運作",
        "how_lead": "它只做一件事：照劇本產生「模型會輸出什麼」和「駕駛其實做了什麼」。裡面沒有告警邏輯。",
        "legend_sum": "時間軸圖怎麼讀", "about_sum": "這一頁怎麼來的",
        "about": "由 <code>dms-sim catalog gallery</code> 從這個 run 產生，不要手改；\n版本指紋 <code>{fp}</code>。"
                 "一條情境會變成很多列，row id 把差別寫在名字裡：\n<code>條目__難度__劇本雜湊__相機__體型__誤差seed__誤差設定</code>。",
        # a standard
        "std_intro": "一個外部法規的標記。它的名稱和要求的內容不放在公開的檔案裡。",
        "std_related": "目錄有 {n} 條情境標了跟這個標準的關係。這裡只標情境跟哪一項要求有關，不表示符合。",
        "std_not_covered": "它沒管的：",
        "cov_h": {"partial": "目錄有相關情境", "gap": "還沒有情境", "out_of_scope": "模擬器管不到"},
        "cov_lead": {"partial": "{n} 項。目錄有條目跟這些要求有關，都只是部分涵蓋。",
                     "gap": "{n} 項。模擬器補得起來，目前沒有條目。",
                     "out_of_scope": "{n} 項。要靠真人、車輛訊號、HMI 或文件，模擬器管不到。"},
        "cov_last_col": "最後一欄是還差什麼。",
        "ids_cols": ("項目", "相關情境"),
        "cov_cols": {"partial": ("條號", "要求", "相關情境", "還差什麼"), "gap": ("條號", "要求", "缺什麼"),
                     "out_of_scope": ("條號", "要求", "為什麼管不到")},
        "std_topic_only": "另外 {n} 條跟這個標準管的主題有關，沒有對到特定一項：",
        "std_source": "來源：<a href=\"{url}\">標準全文</a>（{date} 查證）。",
    },
    "en": {
        "html_lang": "en",
        "title": "DMS model output simulator: scenario overview",
        "home": "Overview",
        "search_ph": "Search: number, title, id, behaviour code",
        "search_aria": "Search scenarios",
        "not_found": "No match",
        "was": "just viewed · ",
        "other_lang": "中文",
        "other_lang_hreflang": "zh-Hant",
        "other_lang_title": "This page in Chinese",
        "list_sep": ", ",
        "source_names": {"written": "hand-written script", "generated": "randomly generated",
                         "recorded": "taken from real video"},
        "source_help": {"written": "a script written segment by segment from the behaviour",
                        "generated": "a long drive laid out by the random generator",
                        "recorded": "a script worked out from real recordings"},
        "flow": (
            ("1 Entry", "An outline of what the driver does, with adjustable parameters (seconds on the phone, "
                        "degrees of head turn).", "catalog/entries/"),
            ("2 Expand", "Laid out as concrete scripts, one row each: per level or parameter value x camera x model "
                         "error profile.", "dms-sim catalog expand"),
            ("3 Simulate", "Each script is played once, recording the truth (what the driver really did) and the "
                           "model output (the truth plus errors such as missed boxes and jitter).",
             "dms-sim catalog run"),
            ("4 Exam", "A set of rows becomes the questions. A post-processing program reads the model output and "
                       "hands in an answer sheet; grading compares only the answer sheet with the truth.",
             "dms-sim exam build / grade"),
        ),
        "legend": (
            ("gaze zone", "Where the gaze falls. The blue line is the truth, the grey dots the model's estimate."),
            ("eye openness", "1 is fully open, 0 fully closed. Downward spikes are blinks; a stretch at the bottom "
                             "is eyes closed."),
            ("truth behaviour", "Truth behaviour labels: in the coloured spans the driver really does that. These "
                                "are what a post-processing program should catch."),
            ("injected model errors", "Errors the simulator adds to the model output on purpose (dropped frames, "
                                      "missed boxes, false positives). Mistakes of a post-processing program in "
                                      "these spans need a separate look."),
        ),
        "row_level": "level {}", "row_camera": "camera {}", "row_profile": "errors {}", "row_variant": "body v{}",
        "row_seed": "behaviour seed {}", "row_eseed": "error seed {}", "row_secs": "{} s",
        "script_time": "Time (s)", "script_what": "What the driver does",
        "script_more": "{n} segments in all; the behaviour card lists every one",
        "not_exported": "This row has not been exported yet (<code>dms-sim catalog run</code>).",
        "fig_script": "Script", "fig_script_p": "In this row the driver follows this table.",
        "fig_gif": "What the camera sees", "fig_gif_p": "The wire-frame face is the driver. Green boxes are the "
                   "truth, red boxes the model output; red text at the top right names the behaviour in progress.",
        "gif_alt": "Camera view {}",
        "fig_tl": "Timeline of the whole row", "fig_tl_p": "Time runs left to right. The coloured spans of the third "
                  "strip are what the driver really does; the bottom strip shows the model errors the simulator "
                  "added on purpose.",
        "tl_alt": "Timeline {}",
        "no_pics": "This row has no pictures: run <code>dms-sim catalog run … --timeline --gif</code> again.",
        "row_files": "Files of this row", "row_files_card": "behaviour card", "row_files_yaml": "script",
        "crumb_where": "{}, ", "crumb_place": "entry {i} of {n} in this family", "source_label": " · source: {}",
        "planned": "planned: the simulator cannot do this yet; it needs {}.",
        "not_in_run": "This run did not select this entry.",
        "pick": "This scenario has {n} rows (the same script played with another level, camera or model error "
                "profile); showing:", "pick_aria": "Choose a row",
        "about_entry": "About this scenario", "tags": "Tags: {}",
        "orphan_summary": "This entry is no longer in the catalog; its content comes from the manifest.",
        "ent_std_lead": "Relation to standards (related, not compliant)",
        "ent_std_topic": "Related to the topic this standard covers, not to one requirement in particular.",
        "req_clause": "{rid} ({clause}): ", "req_id": "{rid}: ",
        "note_lead": "Our reading: ", "premise_h": "Our premise (the text does not confirm it)",
        "lv_lead": "Three levels of the same script: small → medium → large, from easiest to hardest to tell "
                   "apart.", "lv_all": "Every level",
        "lv_one": "What it runs: ",
        "ent_std_ids": "{std}: {ids}",
        "group_lead": "{n} scenarios{rg}, grouped by what they test. Pick one to see its pictures.",
        "group_rg": " ({})",
        "rows_n": "{n} rows", "rows_1": "1 row", "not_run": "not run", "side_n": "{n} entries",
        "star_note": "★ = the entry's main behaviour (the first one listed) is a must-test behaviour: one of the {n} "
                     "behaviours marked <code>must_test: true</code> in <code>catalog/behaviors.yaml</code>. This "
                     "list is the project's own, not a regulatory requirement.",
        "home_lead": "The catalog has {n} scenarios, each with a fixed number (from #001). Pick a behaviour in the "
                     "table below, then go through it one scenario at a time: each has a camera clip and a timeline.",
        "map_corner": "Behaviour \\ what is tested", "map_total": "Total",
        "map_note": "A cell gives the number of scenarios, with their numbers in small print; click it for the list. "
                    "To find a number, type it in the search box at the top right (for example 45).\n"
                    "This run covers {e} scenarios, {r} rows; {p} rows have pictures.",
        "map_std_title": "{n} related to {name}",
        "home_std": "{n} scenarios are marked with the <a href=\"#s/{sid}\">{name}</a> requirements they relate to "
                    "(the small {name} count in a map cell; related, not compliant).",
        "home_star": "{n} catalog entries carry a ★ (before the title in the lists). <code>tests/test_behaviors.py</code>\n"
                     "checks that the entries listing these behaviours have the expected truth for every seed and "
                     "level.",
        "h_origin": "Where these scenarios come from",
        "origin_src": "Each entry names its origin (the entry's <code>source</code> field): {}.",
        "origin_src_item": "{name} ({help}): {n}",
        "origin_sum_all": "every entry", "origin_sum_some": "{n} of the {total} entries have one",
        "origin_sum": "Why its thresholds and durations are set as they are is written in the entry's summary "
                      "({which}; at the bottom of an entry page, “About this scenario”).",
        "origin_no_std": "No entry is mapped to a clause of a regulation or standard yet.",
        "h_fields": "What the columns mean", "h_how": "How the simulator works",
        "how_lead": "It does one thing: from a script, it produces what the model would output and what the driver "
                    "really did. There is no alert logic in it.",
        "legend_sum": "How to read the timeline", "about_sum": "How this page was made",
        "about": "Made by <code>dms-sim catalog gallery</code> from this run; do not edit by hand.\n"
                 "Fingerprint <code>{fp}</code>. One scenario becomes many rows; the row id spells out the "
                 "differences:\n<code>entry__level__script hash__camera__body__error seed__error profile</code>.",
        "std_intro": "A marker for an external regulation. Its name and requirement texts are not part of the "
                     "public files.",
        "std_related": "{n} catalog scenarios are marked with their relation to this standard. This page only says "
                       "which requirement a scenario relates to, never that it complies.",
        "std_not_covered": "Not covered: ",
        "cov_h": {"partial": "Scenarios in the catalog", "gap": "No scenario yet",
                  "out_of_scope": "Out of the simulator's reach"},
        "cov_lead": {"partial": "{n} requirements. Catalog entries relate to them, partly in each case.",
                     "gap": "{n} requirements. The simulator could cover them; no entry does yet.",
                     "out_of_scope": "{n} requirements. They need people, vehicle signals, the HMI or documents; "
                                     "no simulator reaches them."},
        "cov_last_col": " The last column says what is missing.",
        "ids_cols": ("Requirement", "Related scenarios"),
        "cov_cols": {"partial": ("Clause", "Requirement", "Related scenarios", "What is missing"),
                     "gap": ("Clause", "Requirement", "What is missing"),
                     "out_of_scope": ("Clause", "Requirement", "Why out of reach")},
        "std_topic_only": "{n} more relate to the topic this standard covers, not to one requirement in particular: ",
        "std_source": "Source: <a href=\"{url}\">full text of the standard</a> (checked {date}).",
    },
}

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
.bar a.std{font:600 13px ui-monospace,Consolas,monospace;color:#7c2d12}
.bar a.lang{border:1px solid var(--line)}
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
.sm{display:inline-block;font:600 11px/1.5 ui-monospace,Consolas,monospace;color:#7c2d12;background:#fff4e6;
border:1px solid #f1c48f;border-radius:4px;padding:0 5px;margin-left:6px;white-space:nowrap;vertical-align:1px}
a.sm:hover{text-decoration:none;background:#ffe8cc}
h1{font-size:26px;line-height:1.35;margin:0 0 6px}h1 .no{font-size:20px;color:var(--accent);margin-right:6px}
h1 .sm{font-size:13px;vertical-align:4px}
h2{font-size:19px;margin:34px 0 4px}
h2,li,section{scroll-margin-top:calc(var(--barh) + 10px)}
.lead{color:var(--ink2);margin:0 0 10px}.small{font-size:13px;color:var(--ink2)}.muted{color:var(--muted)}
table{border-collapse:collapse;font-size:14px}th,td{border:1px solid var(--line);padding:6px 11px;text-align:left;vertical-align:top}
th{background:#f3f2ed;font-weight:600}
.map td.n{text-align:center;padding:0}.map td.n a{display:block;padding:6px 10px;color:var(--ink)}
.map td.n b{display:block;font-size:16px}.map td.n .rg{display:block;font-size:11px;color:var(--muted);white-space:nowrap}
.map td.n .sm{margin:2px 0 0}
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
ul.cases a.was{background:var(--sel);border-left-color:var(--accent)}ul.cases a.was .k::before{content:__WAS__;color:var(--accent)}
.chip{display:inline-block;font-size:12px;padding:0 8px;border-radius:10px;border:1px solid var(--line);color:var(--ink2);margin:0 5px 4px 0;background:#fff}
.stdbox{background:#fffaf3;border:1px solid #f1dcc0;border-radius:6px;padding:8px 12px;margin:6px 0 4px;font-size:14px}
.stdbox p{margin:0}.stdbox ul{margin:4px 0 0;padding-left:20px}
.note{display:block;margin:3px 0 0;padding:2px 0 2px 8px;border-left:3px solid #c4b5fd;color:#4c1d95;font-size:13px;font-style:italic}
.note b{font-style:normal}
.premise{background:#f5f3ff;border:1px dashed #a78bfa;border-radius:6px;padding:8px 12px;margin:8px 0;font-size:14px}
.premise p{margin:0}.premise .ph{font-weight:700;color:#5b21b6}
p.sub{margin:-2px 0 8px;color:var(--ink2);font-size:15px}
.lvbox{margin:8px 0 4px}.lvbox p{margin:0 0 3px}table.lv th{font:600 13px ui-monospace,Consolas,monospace;width:90px}
p.lv1{margin:8px 0 4px;font-size:14px}
.std td.c{white-space:nowrap}.std td.c b{font-family:ui-monospace,Consolas,monospace}
.std td.c .code{display:block}
.pick{margin:14px 0 4px;font-size:14px}.pick select{font:inherit;font-size:14px;padding:5px 8px;max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.fig{margin:20px 0 0}.fig h3{font-size:15px;margin:0 0 2px}.fig p{margin:0 0 6px;font-size:13px;color:var(--ink2)}
.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;display:block}
details.more{margin:18px 0 0;font-size:14px}details.more summary{cursor:pointer;color:var(--accent)}
details.more>summary h2{display:inline;font-size:19px;color:var(--ink)}
.none{color:var(--muted);font-size:14px;margin:10px 0}
.pn{display:flex;justify-content:space-between;gap:16px;margin-top:34px;padding-top:14px;border-top:1px solid var(--line);font-size:14px}
.pn a{max-width:48%}
.js .view,.js .row{display:none}.js .view.on,.js .row.on{display:block}
.view+.view{margin-top:60px}.js .view+.view{margin-top:0}
"""
JS = """
document.documentElement.classList.add('js');
const views=[...document.querySelectorAll('.view')],topbar=document.querySelector('.top'),side=document.querySelector('.side');
const lang=document.getElementById('lang');
function barh(){document.documentElement.style.setProperty('--barh',topbar.offsetHeight+'px');}
addEventListener('resize',barh);barh();
const fam=id=>id.split('-')[1],cat=id=>id.split('-')[0];
const links=sel=>[...document.querySelectorAll(sel)];
let last=null;   // the entry seen last: marked in its family's list
function show(){const h=decodeURIComponent(location.hash.slice(1));const p=h.split('/');
 let id='home';if(p[0]==='g'&&document.getElementById('g-'+p[1]))id='g-'+p[1];
 if(p[0]==='e'&&document.getElementById('e-'+p[1]))id='e-'+p[1];
 if(p[0]==='s'&&document.getElementById('std-'+p[1]))id='std-'+p[1];
 views.forEach(v=>v.classList.toggle('on',v.id===id));
 const ent=id[0]==='e',std=id.startsWith('std-'),f=id==='home'?'':ent?fam(p[1]):std?'s/'+p[1]:p[1];
 links('.bar a[data-f]').forEach(a=>a.classList.toggle('cur',a.dataset.f===f));
 if(lang)lang.href=lang.dataset.file+location.hash;   // the other language, same place
 document.body.classList.toggle('ent',ent);
 links('.side .fam').forEach(n=>n.classList.toggle('on',ent&&n.id==='s-'+f));
 links('[data-e].cur,[data-e].was').forEach(a=>a.classList.remove('cur','was'));
 if(ent){last=p[1];
  const a=links('.side [data-e]').find(x=>x.dataset.e===last);
  if(a){a.classList.add('cur');
   if(a.offsetTop<side.scrollTop||a.offsetTop+a.offsetHeight>side.scrollTop+side.clientHeight)side.scrollTop=a.offsetTop-side.clientHeight/3;}
  scrollTo(0,0);
 }else if(id!=='home'&&!std){
  const w=last&&fam(last)===f?links('#'+id+' [data-e]').find(x=>x.dataset.e===last):null;
  if(w)w.classList.add('was');
  const sub=p[2]?document.getElementById('g-'+f+'-'+p[2]):null;
  if(w&&(!p[2]||cat(last)===p[2]))w.scrollIntoView({block:'center'});else if(sub)sub.scrollIntoView();else scrollTo(0,0);
 }else scrollTo(0,0);
 const h1=document.querySelector('#'+CSS.escape(id)+' h1');   // the tab title: the heading without its marks
 if(h1){const c=h1.cloneNode(true);c.querySelectorAll('.sm').forEach(x=>x.remove());document.title=c.textContent.trim();}}
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
 if(!m.length){const n=document.createElement('p');n.textContent=T.not_found;hits.appendChild(n);return;}
 m.forEach(e=>{const a=document.createElement('a');a.href='#e/'+e[0];a.textContent=e[1];
  a.addEventListener('click',clearQ);hits.appendChild(a);});});
q.addEventListener('keydown',ev=>{if(ev.key==='Escape')clearQ();
 if(ev.key==='Enter'){const a=hits.querySelector('a');if(a){location.hash=a.hash;clearQ();q.blur();}}});
"""


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


# ---- texts of the catalog in a language (English falls back to the Chinese when missing) -----------------------
def _cat(c: str, lg: str) -> tuple[str, str]:
    """(short name, what it covers) of a category."""
    if lg == "en" and c in C.CATEGORIES_EN:
        n, _, h = C.CATEGORIES_EN[c].partition(": ")
    else:
        n, _, h = C.CATEGORIES[c].partition("：")
    return n, h or n


def _cat_full(c: str, lg: str) -> str:
    return C.CATEGORIES_EN.get(c, C.CATEGORIES[c]) if lg == "en" else C.CATEGORIES[c]


def _fam(f: str, lg: str) -> str:
    return C.FAMILIES_EN.get(f, C.FAMILIES[f]) if lg == "en" else C.FAMILIES[f]


def _tr(d: dict, key: str, lg: str) -> str:
    """d[key] in the language: d[key + "_en"] on the English page when given, else d[key]."""
    v = d.get(f"{key}_en") if lg == "en" else None
    return str(v if v else d.get(key, "") or "")


def _name(e: dict, lg: str) -> str:
    """The entry's short name in the language (`short` / `short_en`), the long title when it has none: what the
    lists, previous / next and the search show next to the number."""
    return _tr(e, "short", lg) or _tr(e, "title", lg)


def _bi(d, lg: str) -> str:
    """A {zh, en} text of a standard in the language ("" when absent: markers only)."""
    return str((d or {}).get(lg) or (d or {}).get("zh") or "") if isinstance(d, dict) else str(d or "")


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


def _marks(e: dict, standards: dict) -> str:
    """Small marks of the standards an entry relates to (its `standards` field), in standards.yaml order."""
    return "".join(f'<span class="sm">{escape(C.standard_label(s, v))}</span>' for s, v in standards.items()
                   if s in (e.get("standards") or {}))


def _entry_link(e: dict, lg: str) -> str:
    """The entry's number (its id when it has none) linking to its page, the title on hover."""
    return (f'<a href="#e/{escape(e["id"])}" title="{escape(_tr(e, "title", lg), quote=True)}">'
            f'{escape(C.entry_no(e) or e["id"])}</a>')


# ---- one entry ----------------------------------------------------------------------------------------
def _row_label(r: dict, card: dict | None, lg: str = "zh") -> str:
    """Plain text that tells a row from its siblings: level, parameters, camera, error profile, seeds, length."""
    T, parts = TEXT[lg], []
    if r.get("level"):
        parts.append(T["row_level"].format(r["level"]))
    parts += [f"{k}={_fmt(v)}" for k, v in (r.get("params") or {}).items() if k != "level"]
    parts += [T["row_camera"].format(r["camera"]), T["row_profile"].format(r["profile"])]
    if r.get("variant") is not None:
        parts.append(T["row_variant"].format(r["variant"]))
    if r.get("seed"):
        parts.append(T["row_seed"].format(r["seed"]))
    if r.get("emulator_seed"):
        parts.append(T["row_eseed"].format(r["emulator_seed"]))
    if card:
        parts.append(T["row_secs"].format(_fmt(float(card["duration_s"]))))
    return " · ".join(parts)


def _script(card: dict, lg: str = "zh") -> str:
    """The row's script as a small table (card.json `timeline`): when the driver does what. The Chinese page shows
    the card's own text; other languages build it from the segment keys with the same rule."""
    T, segs = TEXT[lg], card.get("timeline") or []
    text = (lambda s: s["text"]) if lg == "zh" else (lambda s: seg_text(s.get("keys") or {}, lg))
    L = [f"<table><tr><th>{T['script_time']}</th><th>{T['script_what']}</th></tr>"]
    L += [f"<tr><td>{_fmt(float(s['t0']))}–{_fmt(float(s['t1']))}</td><td>{escape(text(s))}</td></tr>"
          for s in segs[:MAX_SCRIPT_LINES]]
    if len(segs) > MAX_SCRIPT_LINES:
        L.append(f"<tr><td>…</td><td>{T['script_more'].format(n=len(segs))}</td></tr>")
    return "".join(L) + "</table>"


def _row_html(r: dict, run: Path, card: dict | None, first: bool, lg: str = "zh") -> str:
    """One row of an entry: script, camera view, timeline, files. Only one row of an entry is on screen."""
    T, rid = TEXT[lg], r["row_id"]
    rd, rel = run / "rows" / rid, f"rows/{rid}"
    L = [f'<div class="row{" on" if first else ""}">']
    if card is None:
        L.append(f'<p class="none">{T["not_exported"]}</p></div>')
        return "".join(L)
    L.append(f'<div class="fig"><h3>{T["fig_script"]}</h3><p>{T["fig_script_p"]}</p>{_script(card, lg)}</div>')
    if (rd / PICTURE_FILES["gif"]).is_file():
        L.append(f'<div class="fig"><h3>{T["fig_gif"]}</h3><p>{T["fig_gif_p"]}</p>'
                 f'<img loading="lazy" alt="{escape(T["gif_alt"].format(rid), quote=True)}" '
                 f'src="{rel}/{PICTURE_FILES["gif"]}"></div>')
    if (rd / PICTURE_FILES["timeline"]).is_file():
        L.append(f'<div class="fig"><h3>{T["fig_tl"]}</h3><p>{T["fig_tl_p"]}</p>'
                 f'<img loading="lazy" alt="{escape(T["tl_alt"].format(rid), quote=True)}" '
                 f'src="{rel}/{PICTURE_FILES["timeline"]}"></div>')
    if not any((rd / n).is_file() for n in PICTURE_FILES.values()):
        L.append(f'<p class="none">{T["no_pics"]}</p>')
    L.append(f'<details class="more"><summary>{T["row_files"]}</summary><p class="small">{T["row_files_card"]} '
             f'<a href="{rel}/card.md">card.md</a> · {T["row_files_yaml"]} '
             f'<a href="scenarios/{escape(Path(r["scenario"].replace(chr(92), "/")).name)}">YAML</a> · '
             f'row id <code>{escape(rid)}</code></p></details></div>')
    return "".join(L)


def _pn_link(e: dict, cls: str, lg: str = "zh") -> str:
    n = C.entry_no(e)
    label = escape(f"{n} {_name(e, lg)}".strip())
    if cls == "prev":
        return f'<a class="prev" href="#e/{escape(e["id"])}">← {label}</a>'
    return f'<a class="next" href="#e/{escape(e["id"])}" style="text-align:right">{label} →</a>'


def _levels_html(e: dict, lg: str) -> str:
    """The entry's levels from `brief` / `brief_en`: small / medium / large (easiest to hardest to tell apart) and
    what is the same at every level; an entry without levels has the `all` line only ("" without a brief)."""
    T = TEXT[lg]
    b = e.get("brief_en") if lg == "en" and e.get("brief_en") else e.get("brief")
    if not isinstance(b, dict) or not b:
        return ""
    lv = [(k, b[k]) for k in C.LEVELS if b.get(k)]
    same = b.get(C.BRIEF_ALL)
    if not lv:
        return f'<p class="lv1"><b>{T["lv_one"]}</b>{escape(str(same))}</p>' if same else ""
    rows = "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>" for k, v in lv)
    if same:
        rows += f'<tr><th>{T["lv_all"]}</th><td>{escape(str(same))}</td></tr>'
    return f'<div class="lvbox"><p class="small">{T["lv_lead"]}</p><table class="lv">{rows}</table></div>'


def _note(q: dict, lg: str) -> str:
    """A requirement's `note` (our reading, not the text's), set apart from the requirement ("" without one)."""
    n = _bi(q.get("note"), lg)
    return f'<span class="note"><b>{TEXT[lg]["note_lead"]}</b>{escape(n)}</span>' if n else ""


def _entry_standards(e: dict, standards: dict, lg: str) -> str:
    """Which requirements of which standard the entry relates to, one sentence each ("" without `standards`). With
    markers only (no requirement texts) one line: `STD-A: R8, R20`."""
    T, own = TEXT[lg], e.get("standards") or {}
    parts = []
    for sid, s in standards.items():
        if sid not in own:
            continue
        reqs = s.get("requirements") or {}
        link = f'<a href="#s/{escape(sid)}">{escape(_bi(s.get("title"), lg) or C.standard_label(sid, s))}</a>'
        if own[sid] and not any((reqs.get(r) or {}).get("text") for r in own[sid]):
            parts.append(f'<p>{T["ent_std_ids"].format(std=link, ids=escape(", ".join(map(str, own[sid]))))}</p>')
        elif own[sid]:
            items = []
            for r in own[sid]:
                q = reqs.get(r) or {}
                head = (T["req_clause"].format(rid=r, clause=q["clause"]) if q.get("clause")
                        else T["req_id"].format(rid=r))
                items.append(f'<li>{escape(head)}{escape(_bi(q.get("text"), lg))}{_note(q, lg)}</li>')
            parts.append(f'<p>{link}</p><ul>{"".join(items)}</ul>')
        else:
            parts.append(f'<p>{link}</p><ul><li>{T["ent_std_topic"]}</li></ul>')
    if not parts:
        return ""
    return f'<div class="stdbox"><p class="small">{T["ent_std_lead"]}</p>{"".join(parts)}</div>'


def _entry_html(e: dict, rows: list[dict], run: Path, behaviors: dict, prev: dict | None, nxt: dict | None,
                pos: tuple[int, int], lg: str = "zh", standards: dict | None = None) -> str:
    """One entry. pos: (place of the entry in its family, entries in the family), from 1."""
    T, standards = TEXT[lg], standards or {}
    c, f = C.category(e), C.family(e)
    cards = [_load_json(run / "rows" / r["row_id"] / "card.json") for r in rows]
    beh = "".join(f'<span class="chip">{"★ " if C.is_core(behaviors.get(b) or {}) else ""}{escape(b)} '
                  f'{escape(_tr(behaviors.get(b) or {}, "name", lg))}</span>' for b in e.get("behaviors") or [])
    n = C.entry_no(e)
    where = T["crumb_where"].format(n) if n else ""
    src = "" if e.get("_orphan") else escape(T["source_label"].format(
        T["source_names"].get(e.get("source", "written"), str(e.get("source")))))
    marks = "".join(f' <a class="sm" href="#s/{escape(s)}">{escape(C.standard_label(s, v))}</a>'
                    for s, v in standards.items() if s in (e.get("standards") or {}))
    L = [f'<section class="view" id="e-{escape(e["id"])}">',
         f'<p class="small"><a href="#g/{f}">{escape(_fam(f, lg))}</a> › '
         f'<a href="#g/{f}/{c}">{_dot(c)}{escape(_cat(c, lg)[0])}</a> › {escape(where)}'
         f'{escape(T["crumb_place"].format(i=pos[0], n=pos[1]))}</p>',
         f'<h1>{_no(e)} {escape(_name(e, lg))}{marks}</h1>']
    if _tr(e, "short", lg) and _tr(e, "title", lg):
        L.append(f'<p class="sub">{escape(_tr(e, "title", lg))}</p>')
    L += [f'<p class="small"><code>{escape(e["id"])}</code>{src}</p>', f"<p>{beh}</p>", _levels_html(e, lg)]
    box = _entry_standards(e, standards, lg)
    if box:
        L.append(box)
    if e.get("status", "ready") != "ready":
        L.append(f'<p class="none">{escape(T["planned"].format(T["list_sep"].join(map(str, e.get("needs") or []))))}</p>')
    elif not rows:
        L.append(f'<p class="none">{T["not_in_run"]}</p>')
    elif len(rows) > 1:
        opts = "".join(f"<option>{escape(_row_label(r, cd, lg))}</option>" for r, cd in zip(rows, cards))
        L.append(f'<p class="pick">{escape(T["pick"].format(n=len(rows)))}<br>'
                 f'<select aria-label="{escape(T["pick_aria"], quote=True)}">{opts}</select></p>')
    else:
        L.append(f'<p class="pick">{escape(_row_label(rows[0], cards[0], lg))}</p>')
    L += [_row_html(r, run, cd, i == 0, lg) for i, (r, cd) in enumerate(zip(rows, cards))]
    summary = _tr(e, "summary", lg)
    if summary:
        L.append(f'<details class="more"><summary>{T["about_entry"]}</summary><p class="small">{escape(summary)}</p>'
                 f'<p class="small">{escape(T["tags"].format(T["list_sep"].join(map(str, e.get("tags") or []))))}</p>'
                 f'</details>')
    L.append('<p class="pn">' + (_pn_link(prev, "prev", lg) if prev else "<span></span>")
             + (_pn_link(nxt, "next", lg) if nxt else "") + "</p></section>")
    return "\n".join(L)


# ---- a family, a standard ---------------------------------------------------------------------------------
def _star_note(n_core: int, lg: str = "zh") -> str:
    return TEXT[lg]["star_note"].format(n=n_core)


def _group_html(f: str, cats: list[str], entries: list[dict], by_entry: dict, behaviors: dict, n_core: int,
                lg: str = "zh", standards: dict | None = None) -> str:
    """The list of one behaviour family: per category, one line per entry (number first)."""
    T, standards = TEXT[lg], standards or {}
    es = [e for e in entries if C.family(e) == f]
    rg = _ranges(_nums(es))
    L = [f'<section class="view" id="g-{f}">', f'<h1>{escape(_fam(f, lg))} <span class="code">{f}</span></h1>',
         f'<p class="lead">{escape(T["group_lead"].format(n=len(es), rg=T["group_rg"].format(rg) if rg else ""))}</p>']
    if any(_starred(e, behaviors) for e in es):
        L.append(f'<p class="small">{_star_note(n_core, lg)}</p>')
    for c in cats:
        ces = [e for e in es if C.category(e) == c]
        if not ces:
            continue
        name, help_ = _cat(c, lg)
        L += [f'<h2 id="g-{f}-{c}">{_dot(c)}{escape(name)} <span class="code">{c}</span></h2>',
              f'<p class="small">{escape(help_)}</p>', '<ul class="cases">']
        for e in ces:
            n = len(by_entry.get(e["id"], []))
            star = "★ " if _starred(e, behaviors) else ""
            note = ("planned" if e.get("status", "ready") != "ready" else T["rows_1"] if n == 1
                    else T["rows_n"].format(n=n) if n else T["not_run"])
            L.append(f'<li><a href="#e/{escape(e["id"])}" data-e="{escape(e["id"])}">{_no(e)}'
                     f'<span class="t">{star}{escape(_name(e, lg))}{_marks(e, standards)}</span>'
                     f'<span class="k">{note}</span></a></li>')
        L.append("</ul>")
    L.append("</section>")
    return "\n".join(L)


def _side_html(f: str, cats: list[str], entries: list[dict], behaviors: dict, lg: str = "zh",
               standards: dict | None = None) -> str:
    """The family column of the entry pages (wide screens only): one per family, the script shows the current one."""
    T, standards = TEXT[lg], standards or {}
    es = [e for e in entries if C.family(e) == f]
    L = [f'<nav class="fam" id="s-{f}" aria-label="{escape(_fam(f, lg), quote=True)}">',
         f'<p class="sh"><a href="#g/{f}">{escape(_fam(f, lg))}</a> '
         f'<span class="muted">{escape(T["side_n"].format(n=len(es)))}</span></p>']
    for c in cats:
        ces = [e for e in es if C.category(e) == c]
        if not ces:
            continue
        L.append(f'<p class="sc">{_dot(c)}{escape(_cat(c, lg)[0])}</p><ul>')
        for e in ces:
            t = f'{"★ " if _starred(e, behaviors) else ""}{_name(e, lg)}'
            L.append(f'<li><a href="#e/{escape(e["id"])}" data-e="{escape(e["id"])}" '
                     f'title="{escape(_tr(e, "title", lg), quote=True)}">'
                     f'{_no(e)}{_marks(e, standards)} {escape(t)}</a></li>')   # mark before the title: long
                                                                               # titles are cut off at the end
        L.append("</ul>")
    L.append("</nav>")
    return "\n".join(L)


def _has_texts(s: dict) -> bool:
    """True when the standard's requirements carry their texts (the local overlay was merged in)."""
    return any(isinstance(r, dict) and r.get("text") for r in (s.get("requirements") or {}).values())


def _standard_html(sid: str, s: dict, entries: list[dict], lg: str = "zh") -> str:
    """The page of one standard: what it covers and not, then its requirements grouped by coverage (entries
    related / no entry yet / out of the simulator's reach, the last group folded), source and date checked.
    Without the local overlay (markers only) the page has a generic intro and lists requirement ids: the related
    entries for `partial`, the ids alone for the other groups."""
    T, reqs = TEXT[lg], s.get("requirements") or {}
    related, full = C.standard_entries(entries, sid), _has_texts(s)
    title, summary, nc = (_bi(s.get(k), lg) for k in ("title", "summary", "not_covered"))
    L = [f'<section class="view std" id="std-{escape(sid)}">', f'<h1>{escape(title or C.standard_label(sid, s))}</h1>',
         f'<p class="lead">{escape(summary or T["std_intro"])}</p>']
    premise = _bi(s.get("premise"), lg)          # our own premise, which the text does not confirm: set apart
    if premise:
        L.append(f'<div class="premise"><p class="ph">{T["premise_h"]}</p><p>{escape(premise)}</p></div>')
    if nc:
        L.append(f'<p class="small"><b>{T["std_not_covered"]}</b>{escape(nc)}</p>')
    L.append(f'<p class="small">{escape(T["std_related"].format(n=len(related)))}</p>')
    for cov in COVERAGE_ORDER:
        rs = [(rid, r) for rid, r in reqs.items() if (r or {}).get("coverage") == cov]
        if not rs:
            continue
        links = {rid: ", ".join(_entry_link(e, lg) for e in C.standard_entries(entries, sid, rid)) for rid, _ in rs}
        if full:
            cols = "".join(f"<th>{escape(x)}</th>" for x in T["cov_cols"][cov])
            rows = []
            for rid, r in rs:
                clause = f'<span class="code">{escape(str(r["clause"]))}</span>' if r.get("clause") else ""
                cells = [f'<td class="c"><b>{escape(rid)}</b>{clause}</td>',
                         f'<td>{escape(_bi(r.get("text"), lg))}{_note(r, lg)}</td>']
                if cov == "partial":
                    cells.append(f"<td>{links[rid]}</td>")
                cells.append(f'<td>{escape(_bi(r.get("gap"), lg))}</td>')
                rows.append(f'<tr id="std-{escape(sid)}-{escape(rid)}">{"".join(cells)}</tr>')
            body = f"<table><tr>{cols}</tr>{''.join(rows)}</table>"
        elif cov == "partial":                      # markers only: the ids and their related entries
            cols = "".join(f"<th>{escape(x)}</th>" for x in T["ids_cols"])
            body = (f"<table><tr>{cols}</tr>" + "".join(f'<tr id="std-{escape(sid)}-{escape(rid)}"><td class="c">'
                                                        f'<b>{escape(rid)}</b></td><td>{links[rid]}</td></tr>'
                                                        for rid, _ in rs) + "</table>")
        else:
            body = '<p class="small">' + ", ".join(f'<b id="std-{escape(sid)}-{escape(rid)}">{escape(rid)}</b>'
                                                   for rid, _ in rs) + "</p>"
        lead = T["cov_lead"][cov].format(n=len(rs)) + (T["cov_last_col"] if full and cov == "partial" else "")
        lead = f'<p class="small">{escape(lead)}</p>'
        if cov == "out_of_scope":
            L.append(f'<details class="more"><summary><h2>{escape(T["cov_h"][cov])}</h2></summary>{lead}{body}</details>')
        else:
            L += [f'<h2>{escape(T["cov_h"][cov])}</h2>', lead, body]
    topic = [e for e in related if not (e.get("standards") or {}).get(sid)]
    if topic:
        L.append(f'<p class="small">{escape(T["std_topic_only"].format(n=len(topic)))}'
                 f'{", ".join(_entry_link(e, lg) for e in topic)}</p>')
    if s.get("source"):
        src = T["std_source"].format(url=escape(str(s["source"]), quote=True), date=escape(str(s.get("checked", ""))))
        L.append(f'<p class="small">{src}</p>')
    L.append("</section>")
    return "\n".join(L)


# ---- the page -------------------------------------------------------------------------------------------
def gallery_html(run_dir, entries: list[dict] | None = None, title: str | None = None, lang: str = "zh",
                 standards: dict | None = None) -> str:
    """The overview page of a run as one HTML document in `lang` (zh | en). entries: the catalog to list (default:
    the repo's); standards: catalog/standards.yaml's content (default: the public markers, without the local
    overlay; the CLI passes the merged one)."""
    if lang not in TEXT:
        raise ValueError(f"lang must be one of {list(TEXT)}, got {lang!r}")
    lg, T = lang, TEXT[lang]
    run = Path(run_dir)
    man = C.load_manifest(run)
    catalog = entries if entries is not None else C.load_entries()
    standards = C.load_standards() if standards is None else standards
    known = {e["id"] for e in catalog}
    # rows of an entry the catalog no longer has still get a page (built from the manifest)
    entries = catalog + [{"id": i, "title": next(r["title"] for r in man["rows"] if r["entry"] == i),
                          "summary": T["orphan_summary"], "_orphan": True}
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
        sm = "".join(f'<span class="sm" title="{escape(T["map_std_title"].format(n=k, name=lb), quote=True)}">'
                     f'{escape(lb)} · {k}</span>' for sid in standards
                     for k, lb in [(len(C.standard_entries(es, sid)), C.standard_label(sid, standards[sid]))] if k)
        return f'<td class="n"><a href="#g/{f}{"/" + c if c else ""}"><b>{len(es)}</b>{rg}{sm}</a></td>'

    head = "".join(f'<th title="{escape(_cat_full(c, lg), quote=True)}">{_dot(c)}{escape(_cat(c, lg)[0])}'
                   f'<span class="code">{c}</span></th>' for c in cats)
    matrix = [f'<table class="map"><tr><th>{escape(T["map_corner"])}</th>{head}<th>{T["map_total"]}</th></tr>']
    for f in fams:
        matrix.append(f'<tr><th><a href="#g/{f}">{escape(_fam(f, lg))}</a><span class="code">{f}</span></th>'
                      f'{"".join(cell_html(f, c) for c in cats)}{cell_html(f)}</tr>')
    matrix.append("</table>")
    cat_help = "".join(f"<tr><td>{_dot(c)}{escape(_cat(c, lg)[0])} <span class=\"code\">{c}</span></td>"
                       f"<td>{escape(_cat(c, lg)[1])}</td></tr>" for c in cats)
    flow = "".join(f'<div class="step"><b>{escape(a)}</b><p>{escape(b)}</p><code>{escape(cmd)}</code></div>'
                   for a, b, cmd in T["flow"])
    legend = "".join(f"<tr><td><b>{escape(a)}</b></td><td>{escape(b)}</td></tr>" for a, b in T["legend"])
    title = title or T["title"]
    in_run = sum(bool(by_entry.get(e["id"])) for e in entries)
    n_star = sum(_starred(e, behaviors) for e in catalog)
    src = Counter(e.get("source", "written") for e in catalog)
    src_txt = T["list_sep"].join(T["origin_src_item"].format(name=T["source_names"][k], help=T["source_help"][k],
                                                             n=src.get(k, 0)) for k in T["source_names"])
    n_sum = sum(bool(e.get("summary")) for e in catalog)
    which = T["origin_sum_all"] if n_sum == len(catalog) else T["origin_sum_some"].format(n=n_sum, total=len(catalog))
    index = [[e["id"], f"{C.entry_no(e)} {_name(e, lg)}".strip(),
              " ".join([e["id"], *(str(e[k]) for k in ("short", "short_en", "title", "title_en") if e.get(k)),
                        *map(str, e.get("tags") or []), *map(str, e.get("behaviors") or [])]).lower(),
              e.get("number") if _nums([e]) else None]
             for e in order]
    std_lines = "".join('<p class="small">' + T["home_std"].format(n=len(C.standard_entries(catalog, sid)), sid=escape(sid),
                                                                   name=escape(C.standard_label(sid, s))) + "</p>\n"
                        for sid, s in standards.items())
    any_std = any(C.standard_entries(catalog, sid) for sid in standards)
    no_std = "" if any_std else f"\n<li>{T['origin_no_std']}</li>"
    home = f"""<section class="view on" id="home">
<h1>{escape(title)}</h1>
<p class="lead">{escape(T["home_lead"].format(n=len(catalog)))}</p>
{"".join(matrix)}
<p class="small">{escape(T["map_note"].format(e=in_run, r=len(man["rows"]), p=n_pic))}</p>
{std_lines}<p class="small">{_star_note(n_core, lg)}{T["home_star"].format(n=n_star)}</p>
<h2>{T["h_origin"]}</h2>
<ul class="facts">
<li>{T["origin_src"].format(escape(src_txt))}</li>
<li>{escape(T["origin_sum"].format(which=which))}</li>{no_std}
</ul>
<h2>{T["h_fields"]}</h2>
<table>{cat_help}</table>
<h2>{T["h_how"]}</h2>
<p class="lead">{T["how_lead"]}</p>
<div class="flow">{flow}</div>
<details class="more"><summary>{T["legend_sum"]}</summary><table>{legend}</table></details>
<details class="more"><summary>{T["about_sum"]}</summary><p class="small">{T["about"].format(fp=escape(str(man.get("fingerprint_id", ""))))}</p></details>
</section>"""
    groups = [_group_html(f, cats, order, by_entry, behaviors, n_core, lg, standards) for f in fams]
    sides = [_side_html(f, cats, order, behaviors, lg, standards) for f in fams]
    stds = [_standard_html(sid, s, order, lg) for sid, s in standards.items()]
    fam_of = {f: [e["id"] for e in order if C.family(e) == f] for f in fams}
    pages = [_entry_html(e, by_entry.get(e["id"], []), run, behaviors, order[i - 1] if i else None,
                         order[i + 1] if i + 1 < len(order) else None,
                         (fam_of[C.family(e)].index(e["id"]) + 1, len(fam_of[C.family(e)])), lg, standards)
             for i, e in enumerate(order)]
    nav = "".join(f'<a href="#g/{f}" data-f="{f}">{escape(_fam(f, lg))}</a>' for f in fams)
    nav += "".join(f'<a class="std" href="#s/{escape(sid)}" data-f="s/{escape(sid)}">{escape(C.standard_label(sid, s))}</a>'
                   for sid, s in standards.items())
    other = next(x for x in LANGS if x != lg)
    lang_link = (f'<a class="lang" id="lang" href="{FILES[other]}" data-file="{FILES[other]}" '
                 f'hreflang="{T["other_lang_hreflang"]}" lang="{T["other_lang_hreflang"]}" '
                 f'title="{escape(T["other_lang_title"], quote=True)}">{T["other_lang"]}</a>')
    css = CSS.replace("__WAS__", '"' + T["was"] + '"')
    script_text = json.dumps({"not_found": T["not_found"]}, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="{T["html_lang"]}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{css}</style></head><body>
<header class="top"><nav class="bar"><a class="home" href="#" data-f="">{T["home"]}</a>{nav}
<span class="sp"></span><input id="q" type="search" placeholder="{escape(T["search_ph"], quote=True)}" aria-label="{escape(T["search_aria"], quote=True)}">{lang_link}</nav>
<div id="hits"></div></header>
<div class="wrap"><aside class="side">
{chr(10).join(sides)}
</aside><main>
{home}
{chr(10).join(stds)}
{chr(10).join(groups)}
{chr(10).join(pages)}
</main></div><script>const T={script_text};const INDEX={json.dumps(index, ensure_ascii=False).replace("</", "<\\/")};{JS}</script></body></html>
"""


def write_gallery(run_dir, entries: list[dict] | None = None, title: str | None = None,
                  lang: str | None = None, standards: dict | None = None) -> list[Path]:
    """Write the overview page to <run>/index.html (Chinese) and <run>/index.en.html (English), or only the one of
    `lang`; returns the paths written. Their links are relative to the run directory."""
    langs = LANGS if lang is None else (lang,)
    out = []
    for lg in langs:
        p = Path(run_dir) / FILES[lg]
        p.write_text(gallery_html(run_dir, entries, title, lg, standards), encoding="utf-8", newline="\n")
        out.append(p)
    return out
