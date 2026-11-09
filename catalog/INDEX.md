# 情境目錄索引

由 `dms-sim catalog index` 產生，不要手改。id = `<類別>-<行為家族>-<主題>`，說明見 [README.md](README.md)。

共 71 條：ready 70、planned 1。

| 類別 | 內容 | 條數 |
|---|---|---|
| POS | 行為真的發生：分心、閉眼、打哈欠、講手機、抽菸、沒繫安全帶 | 26 |
| NEG | 長得像但不是：看後照鏡、喝東西、手放臉旁、正常眨眼 | 17 |
| BND | 時長掃描：同一個行為從短到長成對排開 | 11 |
| CLR | 停止與再發生：行為停下、隔一段時間再來一次 | 7 |
| ROB | 偵測變差：漏框、掉幀、延遲、抖動加重 | 7 |
| NAT | 長時間日常：隨機產生的長段駕駛 | 3 |
| REG | 回歸：之前出過問題的片段 | 0 |

## POS：行為真的發生：分心、閉眼、打哈欠、講手機、抽菸、沒繫安全帶

| id | 情境 | 標籤 | 狀態 |
|---|---|---|---|
| `POS-DIS-BASIC-TURN` | 基本款：頭向右轉 60° 看乘客 7 s，再回正 | basic, head_turn | ready |
| `POS-DIS-NCAP-LIZARD` | Euro NCAP 長分心（lizard，只動眼睛）：頭不動，眼睛看中控 4 s | ncap, long_distraction, gaze_only, blind_spot | ready |
| `POS-DIS-NCAP-OWL-DOWN` | Euro NCAP 長分心（owl，往下看）：看中控或乘客腳踏區 4 s | ncap, head_turn, long_distraction, look_down, blind_spot | ready |
| `POS-DIS-NCAP-OWL-PASSENGER` | Euro NCAP 長分心（owl，頭跟著轉）：轉頭看乘客 4 s | ncap, head_turn, long_distraction | ready |
| `POS-DIS-NCAP-OWL-WINDOW` | Euro NCAP 長分心（owl，頭跟著轉）：轉頭看駕駛側車窗 4 s，角度 30°–75° | ncap, head_turn, long_distraction, blind_spot | ready |
| `POS-DIS-NCAP-VATS` | Euro NCAP 短分心（VATS）：30 s 內反覆看乘客側 1.5 s、回路面 1 s，累計 18 s | ncap, short_distraction, glance, accumulation | ready |
| `POS-DIS-PASSENGER-6S` | 頭向右轉 60° 看乘客 6 s | head_turn, passenger | ready |
| `POS-DRW-BASIC-CLOSURE` | 基本款：專心開車 8 s 後閉眼 3 s，再睜眼 | basic, closure | ready |
| `POS-DRW-CLOSURE-HEAD-TURNED` | 頭向右轉 40° 時閉眼 3 s | closure, head_turn, interaction | ready |
| `POS-DRW-MICROSLEEP` | 正常開車 8 s 後微睡 2 s | closure, microsleep | ready |
| `POS-DRW-NCAP-SLEEP` | Euro NCAP 睡著（sleep）：閉眼 6 s、頭慢慢往前點 | ncap, closure, sleep | ready |
| `POS-DRW-NCAP-UNRESPONSIVE` | Euro NCAP 無反應駕駛：閉眼、頭垂到胸前 10 s 不動 | ncap, closure, sleep, unresponsive, head_down | ready |
| `POS-DRW-PROGRESSIVE` | 一路變睏：開合度下降、長閉眼、哈欠、低頭，最後 2 s 微睡 | drowsy, progressive, existing | ready |
| `POS-DRW-YAWN-THEN-CLOSURE` | 先打哈欠，3 s 後閉眼 2.5 s | sleepiness, yawn, closure, interaction | ready |
| `POS-YWN-BASIC` | 基本款：打一次哈欠（嘴張大 3 s） | basic, yawn | ready |
| `POS-YWN-ONLY` | 眼睛正常，只打一次大哈欠（嘴張大 4 s） | yawn | ready |
| `POS-PHN-BASIC-CALL` | 基本款：看著路講電話 5 s，再放下 | basic, phone, call | ready |
| `POS-PHN-HELD-UP` | 手機舉在臉前 30 cm（打字、看訊息），眼睛看手機或看路 | phone, held_up | ready |
| `POS-PHN-LAP-THEN-CALL` | 低頭看大腿上的手機，再接電話（通話時看乘客 1.5 s） | phone, call, existing | ready |
| `POS-PHN-WINDSHIELD` | 手機舉在擋風玻璃前（Euro NCAP advanced） | phone, ncap | planned：camera_ahead_of_the_phone |
| `POS-SMK-BASIC` | 基本款：菸含在嘴上、雙手在方向盤 5 s，再拿掉 | basic, smoking | ready |
| `POS-SMK-MOUTH-HAND` | 抽菸：手到嘴抽、菸含嘴手回方向盤、菸拿在手上、再抽一次 | smoking, existing | ready |
| `POS-SMK-PUFF` | 手拿菸到嘴邊抽 3 s | smoking, puff, occlusion | ready |
| `POS-SBT-BASIC` | 基本款：沒繫安全帶 8 s，再繫回 | basic, seatbelt | ready |
| `POS-SBT-UNBUCKLE-MID` | 開車中解開安全帶 10 s 再繫回 | seatbelt | ready |
| `POS-ALL-SMOKE-AND-CALL` | 菸含在嘴上，同時講電話 6 s | smoking, phone, interaction | ready |

## NEG：長得像但不是：看後照鏡、喝東西、手放臉旁、正常眨眼

| id | 情境 | 標籤 | 狀態 |
|---|---|---|---|
| `NEG-DIS-INTERSECTION-SCAN` | 路口左右張望：看左前、右前路面各 2.5 s | head_turn, intersection | ready |
| `NEG-DIS-LOOKDOWN-CENTER` | 看中控或儀表 1.0–1.5 s，連續 6 次 | glance, center_display, blind_spot | ready |
| `NEG-DIS-MIRROR-CHECK` | 正常開車，短暫看左右鏡、後照鏡、儀表（每次 0.6–0.9 s） | mirror, glance, existing | ready |
| `NEG-DIS-SHOULDER-CHECK` | 變換車道前快速轉頭看盲點 0.8 s，重複 3 次 | head_turn, shoulder_check | ready |
| `NEG-DRW-ALERT-30` | 30 s 專心開車（正常眨眼） | sleepiness, baseline | ready |
| `NEG-DRW-BLINK-HEAVY` | 很常眨眼（每分鐘 30–40 次），每次都正常長度 | blink | ready |
| `NEG-DRW-LOOKDOWN-EYES-OPEN` | 低頭看大腿 3 s，眼睛是睜開的 | look_down, class_flip | ready |
| `NEG-DRW-SQUINT` | 瞇眼（陽光或習慣）：開合度 0.3–0.4，持續 15 s | squint | ready |
| `NEG-YWN-TALKING` | 講話：嘴巴微開（開合度 0.3–0.45）持續 10 s | talking, mouth | ready |
| `NEG-PHN-HAND-AT-EAR` | 手摸耳朵或撐頭，沒拿手機 | phone, hand_near_face, occlusion | ready |
| `NEG-PHN-HAND-AT-EAR-FP` | 手摸耳朵，模擬模型在手的位置成段報出假手機框 | phone, hand_near_face, false_positive | ready |
| `NEG-SMK-CIG-IN-LAP` | 菸夾在手上、手放在大腿（沒在抽） | smoking | ready |
| `NEG-SMK-HAND-AT-MOUTH` | 手摸嘴或托下巴，沒有菸 | smoking, hand_near_face, occlusion | ready |
| `NEG-SMK-HAND-AT-MOUTH-FP` | 手摸嘴或托下巴，模擬模型在嘴前成段報出假香菸框 | smoking, hand_near_face, false_positive | ready |
| `NEG-SBT-HAND-ACROSS-CHEST` | 安全帶有繫，手摸耳朵、摸嘴、講電話時擋到一段安全帶 | seatbelt, occlusion, hand_near_face | ready |
| `NEG-ALL-BASIC-CLEAN-30` | 基本款：30 s 只看前方、正常眨眼 | basic, baseline | ready |
| `NEG-ALL-DRINK-EAT` | 喝水、吃東西（手拿東西到嘴邊） | hand_near_face, drinking, eating, occlusion | ready |

## BND：時長掃描：同一個行為從短到長成對排開

| id | 情境 | 標籤 | 狀態 |
|---|---|---|---|
| `BND-DIS-GLITCH-RETURN` | 頭轉 2 s，短暫回正 0.1–1.0 s，再轉 2 s | head_turn, glitch | ready |
| `BND-DIS-HEADTURN-ANGLE` | 頭轉 4 s，角度從 ±15° 掃到 ±60° | head_turn, angle | ready |
| `BND-DIS-HEADTURN-DURATION` | 頭向右轉 60°，持續時間從 0.5 s 掃到 8 s | head_turn, duration | ready |
| `BND-DRW-LONG-CLOSURE-ENDS` | 單次閉眼 1.2 s 或 8 s | sleepiness, closure, duration | ready |
| `BND-DRW-PERCLOS` | 反覆短閉眼（每次 0.5 s），閉眼比例從 50% 掃到 14% | closure, perclos | ready |
| `BND-DRW-SINGLE-CLOSURE` | 正常開車中單次閉眼，閉眼時間從 0.3 s 掃到 6 s | closure, duration | ready |
| `BND-YWN-DURATION` | 嘴張大的時間從 0.25 s 掃到 5 s | yawn, duration | ready |
| `BND-PHN-CALL-DURATION` | 講電話的時間從 0.25 s 掃到 6 s | phone, call, duration | ready |
| `BND-PHN-CALL-GAP` | 講電話 3 s、放下一下、再講 3 s | phone, call, gap | ready |
| `BND-SMK-DURATION` | 菸含在嘴上（手在方向盤）的時間從 0.25 s 掃到 5 s | smoking, duration | ready |
| `BND-SBT-OFF-DURATION` | 沒繫安全帶的時間從 0.5 s 掃到 10 s | seatbelt, duration | ready |

## CLR：停止與再發生：行為停下、隔一段時間再來一次

| id | 情境 | 標籤 | 狀態 |
|---|---|---|---|
| `CLR-DIS-HEADTURN-REARM` | 頭向右轉 6 s、回正 4 s、再轉 6 s | head_turn, clear, rearm | ready |
| `CLR-DRW-AFTER-YAWN` | 打一次哈欠，之後 15 s 正常開車 | sleepiness, yawn, decay | ready |
| `CLR-DRW-CLOSURE-REARM` | 閉眼 2.5 s、睜眼 12 s、再閉眼 2.5 s | closure, clear, rearm | ready |
| `CLR-YWN-TWO-YAWNS` | 兩次哈欠，間隔 10 s | yawn, clear, rearm | ready |
| `CLR-PHN-CALL-REARM` | 講電話 4 s、放下 6 s、再講 4 s | phone, call, clear, rearm | ready |
| `CLR-SMK-REARM` | 叼菸 4 s、拿掉 6 s、再叼 4 s | smoking, clear, rearm | ready |
| `CLR-SBT-REBUCKLE` | 解開安全帶 8 s、繫回 8 s、再解開 8 s | seatbelt, clear, rearm | ready |

## ROB：偵測變差：漏框、掉幀、延遲、抖動加重

| id | 情境 | 標籤 | 狀態 |
|---|---|---|---|
| `ROB-DIS-LOW-FPS` | 15 fps 時頭向右轉，持續時間從 1 s 掃到 6 s | robustness, frame_rate, head_turn | ready |
| `ROB-DRW-EYE-FLIP` | 睜眼被認成閉眼，每次 0.5–2 s | robustness, class_flip | ready |
| `ROB-DRW-NCAP-SUNGLASSES` | Euro NCAP 遮擋（墨鏡）：眼框完全偵測不到 | ncap, occlusion, sunglasses, robustness | ready |
| `ROB-PHN-LOW-CONF` | 手機框分數偏低，講電話 6 s | robustness, phone, conf | ready |
| `ROB-ALL-BURST-MISS` | 連續漏框 1 / 5 / 15 / 30 幀 | robustness, burst_miss | ready |
| `ROB-ALL-ERROR-PROFILES` | 通話、抽菸、微睡、未繫安全帶各一段，在五種模型誤差設定下重跑 | robustness, profiles | ready |
| `ROB-ALL-NCAP-FACE-MASK` | Euro NCAP 遮擋（口罩）：嘴框完全偵測不到 | ncap, occlusion, mask, robustness | ready |

## NAT：長時間日常：隨機產生的長段駕駛

| id | 情境 | 標籤 | 狀態 |
|---|---|---|---|
| `NAT-DRW-SLEEPY-600` | 600 s 睏倦駕駛：只開疲勞片段（每分鐘約 0.6 段） | natural, drowsy | ready |
| `NAT-ALL-CLEAN-300` | 300 s 日常駕駛：看路、看鏡子、看儀表，沒有手機、疲勞、抽菸、未繫 | natural, false_alarm | ready |
| `NAT-ALL-MIXED-120` | 120 s 隨機日常駕駛，夾雜手機、疲勞、抽菸、未繫安全帶片段 | natural, existing | ready |

## REG：回歸：之前出過問題的片段

（還沒有條目）
