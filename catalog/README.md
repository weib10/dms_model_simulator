# 情境目錄（scenario catalog）

把「要模擬什麼駕駛行為」寫成有定義的條目，挑出要的條目自動展開成大量具體劇本，每一列匯出模型輸出、真值和一張行為卡。
拿去測後處理的人，用行為卡對照後處理的告警；目錄本身不判對錯。

- 全部條目一覽：[INDEX.md](INDEX.md)（`dms-sim catalog index` 產生）
- 條目：`entries/<類別>/<id>.yaml`，一條一個檔
- 行為清單：[behaviors.yaml](behaviors.yaml)。每個行為類別一個編號（V 違規、N 正常、C 外觀條件），標常見度、好不好分、會跟哪幾號搞混；
  條目用 `behaviors:` 指過去（第一個是主行為）。★ = 必測（`must_test: true`，獨立欄位，跟常見度、好分度無關），一定不能出問題：
  `tests/test_behaviors.py` 檢查每個必測行為都有條目，而且條目在每個 seed（隨機長段 10 個）、每個難度的真值都符合 `expect`
  （違規每級都要觸發、POS/CLR 要達到最短時長、手機和菸要有相機看得到；正常行為不能越界）。常見度、好分度是估計值，有真實案例再改。
- 哈欠用劇本事件：段落寫 `yawn: {start: true, duration_s: 5, peak_openness: 0.9}`，該段開頭開始一次哈欠（sin 輪廓），真值 `yawning` 才會成立；
  只寫 `mouth_openness: 0.9` 是張嘴，不是哈欠。可加 `eye_openness_at_peak` 讓頂點時眼睛瞇起。
- 難度：參數寫成 `{levels: {small: …, medium: …, large: …}}`。難度 = 離它像的那個違規門檻多近（small 一看就知道，large 貼著門檻但不跨過），
  每條在 summary 寫明對照的門檻。一列只取一個難度，條目裡所有 levels 參數同時取那一級（不做笛卡兒積）。
- 模型誤差設定：[profiles.yaml](profiles.yaml)。分維度分等級（`jit` 抖動、`jmp` 跳動、`mis` 漏框、`drp` 掉幀、`lat` 延遲、`cnf` 分數偏低、`flp` 開閉眼嘴判錯、`fp` 誤報，各 1–3 級），用 `+` 疊加，例如 `jit2+jmp2+mis1`；`bad`、`worst` 是現成組合。等級是刻意粗略的，用來看後處理撐得住多少誤差。

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
dms-sim catalog expand NEG --level large --out runs/h1        # 只要最難的一級（預設三級都出，row id 帶難度）
dms-sim catalog expand NEG-DRW-SNEEZE --param close_s=0.55 --out runs/p1   # 直接覆蓋參數（值用 YAML 解析；只套到有這個參數的條目）
```

每一列（`rows/<row_id>/`）有：

- `<stem>.frames/labels/<frame>.yaml`：每幀的模型輸出（frame_yaml）
- `<stem>.pipeline.jsonl`、`<stem>.truth.jsonl`、`<stem>.jsonl`（真值與模型輸出並排，含誤差標記）
- `card.md`、`card.json`：行為卡（行為時間軸＋注入的模型誤差）
- `row.json`

`rows.tsv` 每列一行：row_id、labels 目錄、truth、card、fps。run 目錄放短路徑（如 `runs/`），Windows 的 260 字元上限會讓長路徑失敗。

## 條目格式

```yaml
id: BND-PHN-CALL-DURATION
title: "講電話的時間從 0.25 s 掃到 6 s"         # 一句話：駕駛做什麼
summary: "..."                                 # 補充（選填）：細節、為什麼這樣寫
status: ready                                  # ready | planned（模擬器還做不到，要寫 needs）
needs: []                                      # planned 時：缺的能力
source: written                                # written | recorded（真錄影整理而來）| generated
tags: [phone, call, duration]
behaviors: [V08]                               # catalog/behaviors.yaml 的編號，主行為放第一個
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

## 新增條目

1. 一句話寫出駕駛在做什麼，決定類別、家族、`tags`。
2. 現有積木（視線目標、`head_pose`、`eye_openness`、`mouth_openness`、`closure`、`yawn`、`hand`、`phone`、`cigarette`、`seatbelt`、`held`、劇本裡的 `cabin:` / `behavior:` 覆寫）做得到 → `ready`；做不到 → `planned` 並寫 `needs`。
3. 數字寫成參數；時長掃描用 `grid`。
4. `dms-sim catalog validate && dms-sim catalog index`。

## 相機看得到什麼（寫條目前先想這個）

- 只用框（不用視線）的後處理，分心只能從臉框和五官框的變化看出來；低頭看大腿、看中控、只動眼睛，從框上幾乎看不出來。分心類條目多用 `head_pose` 的 yaw 來寫。
- 手機放大腿上，模擬的相機都拍不到，只有貼耳或舉起來看得到（真值的 `phone_use_visible`）。
- 手在臉旁會遮住臉和嘴。
