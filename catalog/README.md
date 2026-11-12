# 情境目錄（scenario catalog）

把「要模擬什麼駕駛行為」寫成有定義的條目，挑出要的條目自動展開成大量具體劇本，每一列匯出模型輸出、真值和一張行為卡。
拿去測後處理的人，用行為卡對照後處理的告警；目錄本身不判對錯。要有成績單，用考卷（`dms-sim exam`，見 [docs/EXAM.md](../docs/EXAM.md)）。

- 全部條目一覽：[INDEX.md](INDEX.md)（`dms-sim catalog index` 產生）
- 審閱表：[REVIEW.md](REVIEW.md)、[REVIEW.en.md](REVIEW.en.md)、`REVIEW.csv`、`REVIEW.en.csv`（`dms-sim catalog review` 產生，不手改）。
  所有條目一張表、照編號排，給不寫程式的人逐條確認：編號、短名（planned 的加「（未完成）」）、類別、行為、small／medium／large、
  每級相同／不分級、標記、id，最後兩欄「確認」「備註」留給審閱的人。CSV 是帶 BOM 的 UTF-8，Excel 直接開。
  標記欄只用公開的代號（`STD-A: R8, R20`），這個指令永遠不讀本機的 `standards.local.yaml`。`--out-dir` 換輸出資料夾。
  改了條目後跟 `INDEX.md` 一起重產（`tests/test_catalog.py` 檢查兩者是最新的）。
- 條目：`entries/<類別>/<id>.yaml`，一條一個檔
- 行為清單：[behaviors.yaml](behaviors.yaml)。每個行為類別一個編號（V 違規、N 正常、C 外觀條件），標常見度、好不好分、會跟哪幾號搞混；
  條目用 `behaviors:` 指過去（第一個是主行為）。★ = 必測（`must_test: true`，獨立欄位，跟常見度、好分度無關），一定不能出問題：
  `tests/test_behaviors.py` 檢查每個必測行為都有條目，而且條目在每個 seed（隨機長段 10 個）、每個難度的真值都符合 `expect`
  （違規每級都要觸發、POS/CLR 要達到最短時長、手機和菸要有相機看得到；正常行為不能越界）。常見度、好分度是估計值，有真實案例再改。
- 哈欠用劇本事件：段落寫 `yawn: {start: true, duration_s: 5, peak_openness: 0.9}`，該段開頭開始一次哈欠（sin 輪廓），真值 `yawning` 才會成立；
  只寫 `mouth_openness: 0.9` 是張嘴，不是哈欠。可加 `eye_openness_at_peak` 讓頂點時眼睛瞇起。
- 難度：參數寫成 `{levels: {small: …, medium: …, large: …}}`。難度 = 離它像的那個違規門檻多近（small 一看就知道，large 貼著門檻但不跨過），
  每條在 summary 寫明對照的門檻。一列只取一個難度，條目裡所有 levels 參數同時取那一級（不做笛卡兒積）。
- 標準：[standards.yaml](standards.yaml)。外部法規的中性代號（目前只有 `STD-A`）與它的要求項目編號，條目用 `standards:` 指過去，標「跟哪幾項要求有關」，不是「符合」。見下面「標準」一節。
- 模型誤差設定：[profiles.yaml](profiles.yaml)。分維度分等級（`jit` 抖動、`jmp` 跳動、`mis` 漏框、`drp` 掉幀、`lat` 延遲、`cnf` 分數偏低、`flp` 開閉眼嘴判錯、`fp` 誤報、`pg` 頭姿／臉朝向／視線誤差（每人固定偏差＋快速抖動＋偶發離群），各 1–3 級），用 `+` 疊加，例如 `jit2+jmp2+mis1`；`bad`、`worst` 是現成組合（分別含 `pg2`、`pg3`）。等級是刻意粗略的，用來看後處理撐得住多少誤差。

## 怎麼讀 id

`<類別>-<行為家族>-<主題>`，例如 `BND-DRW-SINGLE-CLOSURE` = 閉眼家族的時長掃描：單次閉眼。

| 類別 | 內容 |
|---|---|
| POS | 行為真的發生：分心、閉眼、打哈欠、講手機、抽菸、沒繫安全帶 |
| NEG | 長得像但不是：看後照鏡、喝東西、手放臉旁、正常眨眼 |
| BND | 時長掃描：同一個行為從短到長成對排開 |
| CLR | 停止與再發生：行為停下、隔一段時間再來一次 |
| ROB | 偵測變差：漏框、掉幀、延遲、抖動加重 |
| NAT | 長時間日常：隨機產生的長段駕駛 |
| REG | 回歸：之前出過問題的片段 |

| 行為家族 | 內容 |
|---|---|
| DIS | 視線或頭離開前方 |
| DRW | 閉眼、瞌睡 |
| YWN | 打哈欠 |
| PHN | 手機 |
| SMK | 抽菸 |
| SBT | 安全帶 |
| ALL | 混合或日常 |

相機、駕駛體型（`--variant`）、模型雜訊 seed、誤差設定不另外開條目，是每條都會套的另一個軸（條目的 `variation`，或 `expand` 的參數覆蓋）。

## 用法

```bash
dms-sim catalog list                         # 全部；也可以 list BND、list PHN、list BND-PHN、--tag basic
dms-sim catalog expand --tag basic --out runs/b1   # 基本款：每個家族一條最單純的行為＋全乾淨基準
dms-sim catalog expand --tag ncap --out runs/n1    # Euro NCAP 風格：owl / lizard、VATS、sleep、無反應、墨鏡 / 口罩
dms-sim catalog expand --tag basic --profile nominal jit2 jmp2+mis2 worst --out runs/s1   # 同一批條目在不同模型誤差下重跑
dms-sim catalog validate                     # 檢查所有條目
dms-sim catalog expand BND PHN --out runs/r1 # 展開成具體劇本 + runs/r1/manifest.json
dms-sim catalog run runs/r1 --jobs 4         # 每列匯出到 runs/r1/rows/<row_id>/，並寫 rows.tsv
dms-sim catalog run runs/r1 --layout my_layout.yaml   # frame_yaml 改用自己的類別 id 與門檻；layout 路徑與 sha256 記進每列 row.json
dms-sim catalog run runs/r1 --include-face            # pipeline JSONL 帶臉框（--labels mapped：改用 layout 的類別）
dms-sim catalog run runs/r1 --timeline --gif          # 每列多出 timeline.png、preview.gif（預設都不出）
dms-sim catalog gallery runs/r1                       # 寫 runs/r1/index.html（中文）和 index.en.html（英文）：整個目錄，每條是這次 run 的圖、劇本、行為卡
dms-sim catalog gallery runs/r1 --lang en             # 只寫其中一種語言
dms-sim catalog expand NEG --level large --out runs/h1        # 只要最難的一級（預設三級都出，row id 帶難度）
dms-sim catalog expand NEG-DRW-SNEEZE --param close_s=0.55 --out runs/p1   # 直接覆蓋參數（值用 YAML 解析；只套到有這個參數的條目）
```

每一列（`rows/<row_id>/`）有：

- `<stem>.frames/labels/<frame>.yaml`：每幀的模型輸出（frame_yaml）
- `<stem>.pipeline.jsonl`、`<stem>.truth.jsonl`、`<stem>.jsonl`（真值與模型輸出並排，含誤差標記）
- `card.md`、`card.json`：行為卡（行為時間軸＋注入的模型誤差）
- `row.json`：路徑、幀數、`road_box`（這一列「看路」的視線範圍，車身座標的 yaw／pitch，度）
- `timeline.png`（`--timeline`）：真值視線區域、眼睛開合度、真值行為標籤、注入的模型誤差，對時間畫成一張圖
- `preview.gif`（`--gif`）：相機畫面，真值框綠色、模型框紅色。`--gif-every N` 每 N 幀畫一張（預設 3），`--gif-scale` 是相對相機影像的大小（預設 0.4）。
  13 s 的一列約 1 MB、幾秒；300 s 的一列約 20 MB、一分多鐘

圖用這一列自己的模擬結果畫，不改其他檔案；`catalog verify` 不比圖。重跑時沒指定的圖會刪掉（免得留著舊輸出的圖）。

**總覽頁**（`catalog gallery`）：一個 HTML 檔，連結都是相對路徑，用瀏覽器直接開。一次只顯示一層：

1. 地圖：行為家族 × 類別的條數表（一個畫面），列和欄都附代碼（`PHN`、`POS`，對得上 id），每格小字是那幾條的編號（`#45–48`；不連續的寫 `#45–47, 90`）。
   底下是 ★ 的意思、這些情境怎麼來的（`source` 的分布，數字從目錄算）、模擬器怎麼運作。點格子進清單。
2. 清單：一個行為家族的所有條目，照類別分組，一條一行：編號＋短名（`short`）。這次沒跑的、planned 的也列。
3. 單條：標題是編號＋短名，下面一行是長標題，再下面是難度表（`brief`：small → medium → large 從最容易到最難分辨，加上每級相同的一行；
   不分級的只有一行）。上方標出位置（「手機 › 行為真的發生 › #052，這一家族第 1／11 條」）和來源（`source`）。
   只看這一條的一列：劇本表、相機畫面 GIF、時間軸圖。其他列（難度、相機、誤差設定）用下拉選；左右鍵或頁尾連結換上一條／下一條（連結帶編號）。

最上面那排固定在頂端，目前所在的家族（首頁時是「情境總覽」）會標成選中。寬螢幕（約 1100px 以上）的單條頁左邊多一欄，列出同一家族的所有條，
目前這條高亮；點了直接換條。從單條回到清單時，剛才看的那條會標出來並捲到位。
★ = 這條的主行為（`behaviors` 的第一個）是必測行為。右上角可以搜尋編號（`45`、`#45`、`#045`）、標題、id、行為編號，Enter 開第一筆。
`standards.yaml` 的每個標準在頂端那排有一個入口（`#s/STD-A`），照 `coverage` 分三組列要求
（目錄有相關情境：列相關條目的編號／還沒有情境／模擬器管不到，最後一組預設收合）。只有公開檔時，頁面只寫要求編號；
有本機的 `standards.local.yaml` 時（見「標準」一節），多出標準的名稱、摘要、它沒管什麼、每項要求的說明與還差什麼、來源與查證日期。
有 `standards` 的條目在家族清單、單條頁左欄、單條頁標題旁有標準代號的小標記，單條頁寫出是哪幾項（有本機檔時每項一句要求）；
首頁只多一行（幾條跟該標準有關），地圖每格用小標記寫有幾條相關。標準從 `standards.yaml` 讀，加第二個標準不用改程式。
英文版 `index.en.html` 跟中文版同樣的內容；右上角互相連結，連到對方的同一個位置。英文頁的劇本表由 `card.json` 的 `timeline[].keys`
產生（跟中文句子同一套規則，`behavior_card.seg_text`），條目或行為缺英文時顯示中文。
它不存資料，只讀 manifest、每列的 `card.json` 和圖；重跑之後重產一次。沒有 JavaScript 時三層依序全部顯示。
考卷的答案側也是一個 run：`dms-sim catalog gallery runs/A/key`。

全目錄各出一張的做法（一個難度、一個模型雜訊 seed；長時間的 NAT 只出時間軸圖）。444 列、`--jobs 4` 約 17 分鐘、3 GB
（JSONL 約 2 GB、GIF 約 340 MB，其餘是每幀一個的小檔）：

```bash
dms-sim catalog expand all --level medium --emulator-seeds 1 --out runs/gallery
dms-sim catalog run runs/gallery POS NEG BND CLR ROB --jobs 4 --timeline --gif --gif-every 4
dms-sim catalog run runs/gallery NAT --jobs 4 --timeline
dms-sim catalog gallery runs/gallery
```

`rows.tsv` 每列一行：row_id、labels 目錄、truth、card、fps。run 目錄放短路徑（如 `runs/`），Windows 的 260 字元上限會讓長路徑失敗。

## 條目格式

```yaml
id: BND-PHN-CALL-DURATION
number: 59                                     # 固定編號（總覽頁、INDEX.md、list 顯示成 #059）。給了就不變、不重用
title: "講電話的時間從 0.25 s 掃到 6 s"         # 一句話：駕駛做什麼
title_en: "Phone call duration swept from 0.25 s to 6 s"   # title 的英文（必填，總覽頁英文版用）
short: "講電話時長掃描"                          # 短名（必填）：這條是什麼，幾個字；總覽頁清單與審閱表用
short_en: "Phone call duration sweep"          # 短名的英文（必填）
brief: {all: "講 0.25–6 s，共 7 種"}            # 難度表（必填）：沒有 levels 參數只寫 all
brief_en: {all: "0.25–6 s calls, 7 lengths"}   # 有 levels 參數：{small: …, medium: …, large: …}，可再加 all
summary: "..."                                 # 補充（選填）：細節、為什麼這樣寫
summary_en: "..."                              # summary 的英文（有 summary 就必填）
status: ready                                  # ready | planned（模擬器還做不到，要寫 needs）
needs: []                                      # planned 時：缺的能力
source: written                                # written | recorded（真錄影整理而來）| generated
tags: [phone, call, duration]
behaviors: [V08]                               # catalog/behaviors.yaml 的編號，主行為放第一個
standards: {STD-A: [R8, R20]}                  # 選填（這裡只示範寫法）：跟 standards.yaml 哪個標準的哪幾項要求有關；[] = 只跟它管的主題有關
params:                                        # 參數，模板裡用 "${名稱}"
  call_s: {grid: [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]}
  # 其他寫法：{fixed: v} {choice: [...]} {uniform: [lo, hi]} {levels: {small: v, medium: v, large: v}}
template:                                      # 就是 scenarios/*.yaml 的格式（behavior / cabin / camera / segments）
  segments:
    - {target: road_ahead, duration: 3.0}
    - {target: road_ahead, duration: "${call_s}", hand: call, phone: call}
    - {target: road_ahead, duration: 3.0}
    # 重複一段：{repeat: N, segments: [...]}（N 可以是參數，可巢狀），例如「看乘客 ${glance_s} 秒、回前方 3 秒」重複 ${n} 次
# 或 from_scenario: scenarios/xxx.yaml（原封包現有劇本），或 generator: {duration_s, config}（gen-scenario）
variation:                                     # 選填，預設兩台相機 × nominal × 3 個模型雜訊 seed
  cameras: [dash_right, steering_column]
  variants: 0                                  # nominal + variant 0..N-1（configs/variation.yaml）
  seeds: [0]                                   # 行為 seed
  emulator_seeds: [0, 1, 2]
  profiles: [nominal]                          # profiles.yaml
fps: 20                                        # 選填，預設 20
```

規則：時間一律是一幀（預設 1/20 s）的倍數；模板不能做運算，只能代入參數；覆寫 `behavior:` / `cabin:` 的鍵必須真的存在於 configs（`validate` 會抓打錯字）。

編號（`number`）：每條必填、正整數、整個目錄不重複（`validate` 與載入時檢查）。給了就不變；刪掉的條目，它的號碼也不再給別人。
新條目拿目前最大值 +1，所以後來加的條目號碼會接在最後，不跟著地圖的格子連續。最初的 88 條照總覽頁的閱讀順序
（行為家族 → 類別 → id）編成 1–88。

英文欄位：`title_en`、`summary_en` 跟中文寫在同一個條目檔；行為的英文名是 `behaviors.yaml` 每個行為的 `name_en`。
`validate` 把它們當必填：每條要有 `title_en`，有 `summary` 就要有 `summary_en`（沒有就不能有），每個行為要有 `name_en`。改中文時英文一起改。
類別與行為家族的英文在 `catalog.py` 的 `CATEGORIES_EN`、`FAMILIES_EN`。只有總覽頁英文版用；`INDEX.md`、行為卡、考卷不出英文版。

短名與難度表（`validate` 檢查，上限在 `catalog.py` 的 `SHORT_MAX`、`BRIEF_MAX`，單位是字元）：

- `short`／`short_en`：這條情境是什麼，幾個字（中文最多 14、英文最多 45），整個目錄不能重複。總覽頁的清單、上一條／下一條、搜尋結果、
  審閱表都用「編號＋短名」，長標題放在單條頁的標題下面。
- `brief`／`brief_en`：同一組鍵的對照表，每格是非空字串。條目有 levels 參數時，鍵是 `small`、`medium`、`large`（必有），可以再加 `all`；
  沒有 levels 參數時只有 `all`。`small`／`medium`／`large` 只寫各級**不同**的地方（中文最多 36、英文最多 80）；
  `all` 寫每一級都一樣的部分，或不分級的條目跑的是什麼（中文最多 50、英文最多 110）。

## 標準

[standards.yaml](standards.yaml) 是一個外部法規的**標記**，不是它的內容：每個標準一個中性代號（`STD-A`），
底下是要求項目的編號（`R1`…`R21`）和各自的 `coverage`。法規的名稱、摘要、條號與要求的說明都不在公開的檔案裡。

```yaml
standards:
  STD-A:
    requirements:
      R1: {coverage: partial}
      R2: {coverage: out_of_scope}
```

| coverage | 意思 |
|---|---|
| `partial` | 目錄有條目跟它有關（部分涵蓋） |
| `gap` | 模擬器補得起來，但目前沒有條目 |
| `out_of_scope` | 要靠真人、車輛訊號、HMI 或文件，模擬器管不到 |

條目的 `standards: {STD-A: [R8, R20]}` 標的是「跟這幾項要求有關」；`{STD-A: []}` 是跟它管的主題有關，但沒有對到特定一項。
標的永遠是「相關」，**沒有條目能說「符合」**。`coverage` 只寫在 `standards.yaml`。

**本機的名稱與說明**（選用）：同一個資料夾可以放一份 `standards.local.yaml`，它在 `.gitignore` 裡，不會進 git。
結構跟 `standards.yaml` 一樣，替列出的標準補上 `name`（顯示用的名稱）、中英文的 `title`、`summary`、`not_covered`、
`source`、`checked`，每項要求補 `clause`、中英文的 `text`、`gap`（partial 寫還差什麼、gap 寫缺什麼情境、out_of_scope 寫為什麼管不到）。
它不能有 `coverage`，只能寫 `standards.yaml` 裡有的標準與要求，而且列出的標準要寫齊（每一項要求都要有）。
另外兩個欄位是選填、中英文都要寫：標準層的 `premise`（我們自己的前提，條文沒有確認），總覽頁放在摘要下面另一個框，
標題「我們的前提（條文沒有確認）」；要求層的 `note`（我們對這一項的解讀），放在要求說明下面、樣式不同，前面加「我們的解讀：」
（標準頁和單條頁都會出現）。條文本身寫得出來的放 `text`，是我們推論的才放這兩個欄位。

- `dms-sim catalog gallery` 有這個檔就用它：頂端那排與小標記顯示 `name`（網址仍是 `#s/STD-A`），標準頁與單條頁出現說明。
  `--no-local` 不用它，只出代號；指令會印一行說用了哪一種。總覽頁本來就不進 git。
- `index`、`review`、`list` 與所有會 commit 的產出只讀公開檔（`catalog.load_standards()` 預設 `local=False`），
  所以本機檔的內容不會進到公開的檔案。`validate` 會讀本機檔，但只檢查它、不寫任何檔。

`validate` 檢查：條目引用的標準與要求存在、不重複；整個目錄的一致性：`partial` 的要求至少一條條目引用，`gap`、`out_of_scope` 的沒有任何條目引用
（改了 `coverage` 或條目的 `standards`，兩邊要一起改）。有 `standards.local.yaml` 時也檢查它（上面的規則），多印一行 `standards.local.yaml: ok`。

## 新增條目

1. 一句話寫出駕駛在做什麼，決定類別、家族、`tags`；`number` 取目前最大的編號 +1（`dms-sim catalog list` 第一欄）。
   標題與說明附英文（`title_en`、`summary_en`）；跟某個標準的要求有關就寫 `standards`。
2. 現有積木（視線目標、`head_pose`、`eye_openness`、`mouth_openness`、`closure`、`yawn`、`hand`、`phone`、`cigarette`、`seatbelt`、`held`、劇本裡的 `cabin:` / `behavior:` 覆寫）做得到 → `ready`；做不到 → `planned` 並寫 `needs`。
3. 數字寫成參數；時長掃描用 `grid`。
4. `dms-sim catalog validate && dms-sim catalog index`。

## 相機看得到什麼（寫條目前先想這個）

- 只用框（不用視線）的後處理，分心只能從臉框和五官框的變化看出來；只動眼睛從框上幾乎看不出來，低頭的部分在框上變化也小（看中控時頭照預設還會右轉約 21°，這部分看得出來）。分心類條目多用 `head_pose` 的 yaw 來寫。
- 手機放大腿上，模擬的相機都拍不到，只有貼耳或舉起來看得到（真值的 `phone_use_visible`）。
- 手在嘴前會遮住嘴（dash_right 看不到嘴框）；手在耳邊只碰到臉框的邊緣，眼睛和嘴都還看得到。
