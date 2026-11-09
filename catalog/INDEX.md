# 情境目錄索引

由 `dms-sim catalog index` 產生，不要手改。id = `<類別>-<行為家族>-<主題>`，說明見 [README.md](README.md)。

共 89 條：ready 88、planned 1。

| 類別 | 內容 | 條數 |
|---|---|---|
| POS | 行為真的發生：分心、閉眼、打哈欠、講手機、抽菸、沒繫安全帶 | 27 |
| NEG | 長得像但不是：看後照鏡、喝東西、手放臉旁、正常眨眼 | 33 |
| BND | 時長掃描：同一個行為從短到長成對排開 | 11 |
| CLR | 停止與再發生：行為停下、隔一段時間再來一次 | 7 |
| ROB | 偵測變差：漏框、掉幀、延遲、抖動加重 | 7 |
| NAT | 長時間日常：隨機產生的長段駕駛 | 4 |
| REG | 回歸：之前出過問題的片段 | 0 |

## 行為清單

編號定義在 [behaviors.yaml](behaviors.yaml)。常見度、好分度是估計值；★ = 常見且好分，一定不能出問題（有測試把關）。

| 編號 | 行為 | 類型 | 常見 | 好分 | 會跟誰搞混 | 條目數 |
|---|---|---|---|---|---|---|
| ★ V01 | 轉頭長分心（看乘客、窗外） | 違規 | 常見 | 好分 | N02, N04, N05, N06, N07, N08, N25 | 15 |
| V02 | 只動眼睛的長分心（lizard） | 違規 | 常見 | 難分 | N09, N03 | 1 |
| V03 | 低頭長分心（看大腿、腳踏區） | 違規 | 常見 | 普通 | N03, N12, N14, N24, V06 | 4 |
| V04 | 短分心累計（VATS） | 違規 | 常見 | 難分 | N06, N07, N03, N02, N09 | 1 |
| V05 | 閉眼、微睡 | 違規 | 偶爾 | 普通 | N10, N11, N12, N13, N14, N17, N27, N30, C01 | 15 |
| V06 | 睡著、無反應（閉眼＋垂頭） | 違規 | 少見 | 好分 | V03, N27 | 2 |
| V07 | 打哈欠 | 違規 | 常見 | 普通 | N15, N16, N17, N18, N19, C02 | 11 |
| ★ V08 | 講電話（手機貼耳） | 違規 | 常見 | 好分 | N20, N19 | 11 |
| V09 | 手拿手機看或打字 | 違規 | 常見 | 普通 | N28, N19, N24 | 4 |
| V10 | 抽菸 | 違規 | 偶爾 | 普通 | N21, N22, N19, N23, C02 | 9 |
| ★ V11 | 沒繫安全帶 | 違規 | 常見 | 好分 | N29 | 6 |
| ★ N01 | 專心看前方、正常眨眼 | 正常 | 常見 | 好分 |  | 7 |
| ★ N02 | 看後照鏡、側鏡 | 正常 | 常見 | 好分 | V01, V04 | 8 |
| ★ N03 | 看儀表、中控（短瞥） | 正常 | 常見 | 好分 | V03, V04, V02 | 8 |
| N04 | 盲點檢查（轉頭看後方側邊） | 正常 | 常見 | 普通 | V01 | 1 |
| ★ N05 | 路口左右張望（視線仍在路上） | 正常 | 常見 | 好分 | V01 | 3 |
| N06 | 東張西望（視線快速在多處跳） | 正常 | 常見 | 普通 | V04, V01 | 1 |
| N07 | 聊天時看乘客一眼 | 正常 | 常見 | 普通 | V01, V04 | 3 |
| ★ N08 | 抬頭看紅綠燈 | 正常 | 常見 | 好分 | V01 | 1 |
| N09 | 只動眼睛的短瞥 | 正常 | 常見 | 難分 | V02, V04 | 1 |
| ★ N10 | 頻繁眨眼 | 正常 | 常見 | 好分 | V05 | 1 |
| N11 | 瞇眼（陽光、習慣） | 正常 | 偶爾 | 普通 | V05 | 1 |
| N12 | 低頭但眼睛睜開 | 正常 | 常見 | 普通 | V05, V03 | 1 |
| N13 | 揉眼睛、推眼鏡 | 正常 | 常見 | 普通 | V05 | 3 |
| N14 | 打噴嚏、咳嗽 | 正常 | 偶爾 | 普通 | V05, V03 | 1 |
| ★ N15 | 講話 | 正常 | 常見 | 好分 | V07 | 3 |
| N16 | 唱歌 | 正常 | 常見 | 普通 | V07 | 1 |
| N17 | 大笑 | 正常 | 常見 | 難分 | V07, V05 | 3 |
| N18 | 嚼口香糖 | 正常 | 偶爾 | 好分 | V07 | 2 |
| N19 | 喝水、吃東西 | 正常 | 常見 | 普通 | V07, V10, V09, V08 | 2 |
| N20 | 手摸耳朵、撐頭（沒拿手機） | 正常 | 常見 | 普通 | V08 | 4 |
| N21 | 手摸嘴、托下巴、抓鼻子（沒拿菸） | 正常 | 常見 | 普通 | V10 | 5 |
| N22 | 叼吸管、牙籤、筆、棒棒糖 | 正常 | 偶爾 | 難分 | V10 | 2 |
| N23 | 菸拿在手上沒在抽 | 正常 | 少見 | 難分 | V10 | 1 |
| N24 | 調冷氣、音響（手伸到中控） | 正常 | 常見 | 普通 | V03, V09 | 1 |
| N25 | 調後照鏡 | 正常 | 偶爾 | 普通 | V01 | 1 |
| N26 | 轉彎交叉換手（手擋住方向盤柱相機） | 正常 | 常見 | 普通 | C03 | 1 |
| N27 | 跟音樂點頭 | 正常 | 偶爾 | 普通 | V05, V06 | 1 |
| N28 | 手機放車架（沒用手拿） | 正常 | 常見 | 普通 | V09 | 1 |
| N29 | 有繫安全帶但手擋住一段 | 正常 | 常見 | 普通 | V11 | 1 |
| N30 | 手擋陽光、翻遮陽板 | 正常 | 偶爾 | 難分 | V05 | — |
| C01 | 戴墨鏡（眼框看不到） | 外觀條件 | 常見 | 難分 | V05 | 1 |
| C02 | 戴口罩（嘴框看不到） | 外觀條件 | 偶爾 | 難分 | V07, V10 | 1 |
| C03 | 模型輸出變差（漏框、掉幀、延遲、抖動） | 外觀條件 | 常見 | 普通 | N26 | 7 |

## POS：行為真的發生：分心、閉眼、打哈欠、講手機、抽菸、沒繫安全帶

| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |
|---|---|---|---|---|---|
| `POS-DIS-BASIC-TURN` | 基本款：頭向右轉 60° 看乘客 7 s，再回正（60 / 60 / 45°，8 / 7 / 3.5 s） | V01 | basic, head_turn | 三段 | ready |
| `POS-DIS-NCAP-LIZARD` | Euro NCAP 長分心（lizard，只動眼睛）：頭不動，眼睛看中控 5 / 4 / 3.5 s | V02 | ncap, long_distraction, gaze_only, blind_spot | 三段 | ready |
| `POS-DIS-NCAP-OWL-DOWN` | Euro NCAP 長分心（owl，往下看）：看中控或乘客腳踏區 5 / 4 / 3.5 s | V03 | ncap, head_turn, long_distraction, look_down, blind_spot | 三段 | ready |
| `POS-DIS-NCAP-OWL-PASSENGER` | Euro NCAP 長分心（owl，頭跟著轉）：轉頭看乘客 5 / 4 / 3.5 s | V01 | ncap, head_turn, long_distraction | 三段 | ready |
| `POS-DIS-NCAP-OWL-WINDOW` | Euro NCAP 長分心（owl，頭跟著轉）：轉頭看駕駛側車窗 5 / 4 / 3.5 s，角度 30°–75° | V01 | ncap, head_turn, long_distraction, blind_spot | 三段 | ready |
| `POS-DIS-NCAP-VATS` | Euro NCAP 短分心（VATS）：30 s 內反覆看乘客側 2 / 1.5 / 1 s、回路面 0.5 / 1 / 1.5 s，累計 24 / 18 / 12 s | V04 | ncap, short_distraction, glance, accumulation | 三段 | ready |
| `POS-DIS-PASSENGER-6S` | 頭向右轉 60° 看乘客 6 s（60 / 60 / 45°，8 / 6 / 3.5 s） | V01 | head_turn, passenger | 三段 | ready |
| `POS-DRW-BASIC-CLOSURE` | 基本款：專心開車 8 s 後閉眼 4 / 3 / 1.5 s，再睜眼 | V05 | basic, closure | 三段 | ready |
| `POS-DRW-CLOSURE-HEAD-TURNED` | 頭向右轉 40° 時閉眼 3 s（20 / 40 / 50°，4 / 3 / 1.5 s） | V05 | closure, head_turn, interaction | 三段 | ready |
| `POS-DRW-MICROSLEEP` | 正常開車 8 s 後微睡 2.5 / 2 / 1.2 s | V05 | closure, microsleep | 三段 | ready |
| `POS-DRW-NCAP-SLEEP` | Euro NCAP 睡著（sleep）：閉眼 8 / 6 / 3.5 s、頭慢慢往前點 | V06 | ncap, closure, sleep | 三段 | ready |
| `POS-DRW-NCAP-UNRESPONSIVE` | Euro NCAP 無反應駕駛：閉眼、頭垂到胸前 12 / 10 / 7 s 不動 | V06 | ncap, closure, sleep, unresponsive, head_down | 三段 | ready |
| `POS-DRW-PROGRESSIVE` | 一路變睏：開合度下降、長閉眼、哈欠、低頭，最後 2 s 微睡 | V05, V07, V03 | drowsy, progressive, existing |  | ready |
| `POS-DRW-YAWN-THEN-CLOSURE` | 先打哈欠，3 s 後閉眼 3 / 2.5 / 1.5 s | V05, V07 | sleepiness, yawn, closure, interaction | 三段 | ready |
| `POS-YWN-BASIC` | 基本款：打一次哈欠（嘴張大 5 / 3 / 2.5 s） | V07 | basic, yawn | 三段 | ready |
| `POS-YWN-COVERED` | 用手遮住嘴打哈欠 | V07, N21 | yawn, hand_near_face, occlusion | 三段 | ready |
| `POS-YWN-ONLY` | 眼睛正常，只打一次大哈欠（嘴張大 6 / 4 / 2.5 s） | V07 | yawn | 三段 | ready |
| `POS-PHN-BASIC-CALL` | 基本款：看著路講電話 6 / 5 / 2 s，再放下 | V08 | basic, phone, call | 三段 | ready |
| `POS-PHN-HELD-UP` | 手機舉在臉前 30 cm（打字、看訊息），眼睛看手機或看路 | V09 | phone, held_up | 三段 | ready |
| `POS-PHN-LAP-THEN-CALL` | 低頭看大腿上的手機，再接電話（通話時看乘客 1.5 s） | V09, V08, V01 | phone, call, existing |  | ready |
| `POS-PHN-WINDSHIELD` | 手機舉在擋風玻璃前（Euro NCAP advanced） | V09 | phone, ncap |  | planned：camera_ahead_of_the_phone |
| `POS-SMK-BASIC` | 基本款：菸含在嘴上、雙手在方向盤 6 / 5 / 2 s，再拿掉 | V10 | basic, smoking | 三段 | ready |
| `POS-SMK-MOUTH-HAND` | 抽菸：手到嘴抽、菸含嘴手回方向盤、菸拿在手上、再抽一次 | V10, N02 | smoking, existing | 三段 | ready |
| `POS-SMK-PUFF` | 手拿菸到嘴邊抽 4 / 3 / 1.5 s | V10 | smoking, puff, occlusion | 三段 | ready |
| `POS-SBT-BASIC` | 基本款：沒繫安全帶 10 / 8 / 3 s，再繫回 | V11 | basic, seatbelt | 三段 | ready |
| `POS-SBT-UNBUCKLE-MID` | 開車中解開安全帶 12 / 10 / 3 s 再繫回 | V11 | seatbelt | 三段 | ready |
| `POS-ALL-SMOKE-AND-CALL` | 菸含在嘴上，同時講電話 8 / 6 / 3 s | V10, V08 | smoking, phone, interaction | 三段 | ready |

## NEG：長得像但不是：看後照鏡、喝東西、手放臉旁、正常眨眼

| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |
|---|---|---|---|---|---|
| `NEG-DIS-ADJUST-HVAC` | 調冷氣、音響：右手伸到中控台，視線在中控與路面之間來回 | N24 | hand_off_wheel, glance, hvac | 三段 | ready |
| `NEG-DIS-ADJUST-MIRROR` | 調後照鏡：手握住後照鏡、眼睛看著鏡子（1 / 2 / 2.8 s） | N25 | mirror, hand_off_wheel, occlusion | 三段 | ready |
| `NEG-DIS-EYES-ONLY-GLANCE` | 只動眼睛的短瞥：頭不動，眼睛看儀表或中控一下（POS-DIS-NCAP-LIZARD 的正常對照） | N09, N03 | glance, gaze_only | 三段 | ready |
| `NEG-DIS-INTERSECTION-SCAN` | 路口左右張望：看左前、右前路面各 1.0 / 2.0 / 2.8 s | N05 | head_turn, intersection | 三段 | ready |
| `NEG-DIS-LOOK-AROUND` | 東張西望：視線在左鏡、右前路面、右側窗、左前路面之間快速跳，每次很短就回前方 | N06, N02, N05 | glance, head_turn, look_around | 三段 | ready |
| `NEG-DIS-LOOKDOWN-CENTER` | 看中控（或排檔）與儀表，每次 0.8 / 1.2 / 1.8 s | N03 | glance, center_display, blind_spot | 三段 | ready |
| `NEG-DIS-MIRROR-CHECK` | 正常開車，短暫看左右鏡、後照鏡、儀表（看鏡子每次 0.5 / 0.7–0.9 / 1.4 s） | N02, N03 | mirror, glance, existing | 三段 | ready |
| `NEG-DIS-PASSENGER-CHAT` | 跟乘客聊天：反覆看乘客一眼（每次 0.5 / 0.8 / 1.2 s） | N07 | glance, head_turn, passenger | 三段 | ready |
| `NEG-DIS-SHOULDER-CHECK` | 變換車道前快速轉頭看盲點 0.5 / 0.8 / 2.0 s，重複 3 次 | N04 | head_turn, shoulder_check | 三段 | ready |
| `NEG-DIS-TRAFFIC-LIGHT` | 抬頭看紅綠燈：視線仰角 10° / 20° / 30°（頭約轉一半），1 / 2 / 3 s | N08 | look_up, traffic_light | 三段 | ready |
| `NEG-DRW-ALERT-30` | 30 s 專心開車（正常眨眼） | N01 | sleepiness, baseline |  | ready |
| `NEG-DRW-BLINK-HEAVY` | 很常眨眼（每分鐘 20 / 35 / 50 次），每次都正常長度 | N10 | blink | 三段 | ready |
| `NEG-DRW-HAND-AT-EYE` | 揉眼睛、推眼鏡：右手指節擋住右眼（0.4 / 0.7 / 0.95 s，1 / 2 / 3 次） | N13 | hand_near_face, occlusion, rub_eyes | 三段 | ready |
| `NEG-DRW-LOOKDOWN-EYES-OPEN` | 低頭看大腿 1.0 / 1.8 / 2.6 s，眼睛是睜開的 | N12 | look_down, class_flip | 三段 | ready |
| `NEG-DRW-MUSIC-NOD` | 跟著音樂點頭：頭規律地上下點，眼睛一直看路 | N27 | look_down, head_nod | 三段 | ready |
| `NEG-DRW-SNEEZE` | 打噴嚏、咳嗽：短閉眼＋頭往前點＋嘴張開 | N14 | look_down, mouth, sneeze | 三段 | ready |
| `NEG-DRW-SQUINT` | 瞇眼（陽光或習慣）：開合度 0.5 / 0.35 / 0.25，持續 15 s | N11 | squint | 三段 | ready |
| `NEG-YWN-CHEWING` | 嚼口香糖：嘴小幅度規律開合 | N18 | mouth, chewing | 三段 | ready |
| `NEG-YWN-SINGING` | 唱歌：嘴巴一開一合 | N16 | mouth, singing | 三段 | ready |
| `NEG-YWN-TALKING` | 講話：嘴巴微開（開合度 0.3 / 0.4 / 0.5）持續 5 / 10 / 15 s | N15 | talking, mouth | 三段 | ready |
| `NEG-PHN-HAND-AT-EAR` | 手摸耳朵或撐頭，沒拿手機（1.5 / 4 / 8 s） | N20 | phone, hand_near_face, occlusion | 三段 | ready |
| `NEG-PHN-HAND-AT-EAR-FP` | 手摸耳朵，模擬模型在手的位置成段報出假手機框（每次 1 / 3 / 6 s，1 / 2 / 3 次） | N20, C03 | phone, hand_near_face, false_positive | 三段 | ready |
| `NEG-PHN-PHONE-MOUNT` | 手機放在儀表板的手機架上（導航），手在方向盤上，偶爾看一眼 | N28 | phone, glance | 三段 | ready |
| `NEG-SMK-CIG-IN-LAP` | 菸夾在手上、手放在大腿（沒在抽）：2 / 5 / 10 s | N23 | smoking | 三段 | ready |
| `NEG-SMK-HAND-AT-MOUTH` | 手摸嘴或托下巴，沒有菸（每次 1.5 / 3 / 4 s，1 / 2 / 3 次） | N21 | smoking, hand_near_face, occlusion | 三段 | ready |
| `NEG-SMK-HAND-AT-MOUTH-FP` | 手摸嘴或托下巴，模擬模型在嘴前成段報出假香菸框（每次 1 / 3 / 6 s，1 / 2 / 3 次） | N21, C03 | smoking, hand_near_face, false_positive | 三段 | ready |
| `NEG-SMK-HELD-STICK` | 嘴裡叼東西（沒用手扶）：牙籤 / 棒棒糖 / 吸管（3 / 6 / 10 s） | N22 | held_object, mouth, false_positive | 三段 | ready |
| `NEG-SBT-HAND-ACROSS-CHEST` | 安全帶有繫，手摸耳朵、摸嘴、講電話時擋到一段安全帶（每次 1.5 / 4 / 8 s） | N29, V08, N20, N21 | seatbelt, occlusion, hand_near_face | 三段 | ready |
| `NEG-ALL-BASIC-CLEAN-30` | 基本款：30 s 只看前方、正常眨眼 | N01 | basic, baseline |  | ready |
| `NEG-ALL-DRINK-EAT` | 喝水、吃東西（手拿東西到嘴邊）：每口喝 1.5 / 3 / 5 s，吃東西 1 / 2 / 4 口 | N19 | hand_near_face, drinking, eating, occlusion | 三段 | ready |
| `NEG-ALL-HABITS-COMBO` | 兩種正常行為疊在一起：邊笑邊看乘客、揉眼時低頭看儀表、邊講話邊看左鏡 | N17, N07, N13, N03, N15, N02 | interaction, laugh, rub_eyes, glance | 三段 | ready |
| `NEG-ALL-HAND-OVER-HAND` | 轉彎交叉換手：手臂掃過方向盤上緣，擋住方向盤柱相機（每次約 0.2 / 0.4 / 0.7 s，2 / 4 / 6 次） | N26, N05 | occlusion, hand_over_hand | 三段 | ready |
| `NEG-ALL-LAUGH` | 大笑：嘴張開、眼睛瞇起 | N17 | yawn, closure, mouth, squint, laugh | 三段 | ready |

## BND：時長掃描：同一個行為從短到長成對排開

| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |
|---|---|---|---|---|---|
| `BND-DIS-GLITCH-RETURN` | 頭轉 2 s，短暫回正 0.1–1.0 s，再轉 2 s | V01 | head_turn, glitch |  | ready |
| `BND-DIS-HEADTURN-ANGLE` | 頭轉 4 s，角度從 ±15° 掃到 ±60° | V01 | head_turn, angle |  | ready |
| `BND-DIS-HEADTURN-DURATION` | 頭向右轉 60°，持續時間從 0.5 s 掃到 8 s | V01 | head_turn, duration |  | ready |
| `BND-DRW-LONG-CLOSURE-ENDS` | 單次閉眼 1.2 s 或 8 s | V05 | sleepiness, closure, duration |  | ready |
| `BND-DRW-PERCLOS` | 反覆短閉眼（每次 0.5 s），閉眼比例從 50% 掃到 14% | V05 | closure, perclos |  | ready |
| `BND-DRW-SINGLE-CLOSURE` | 正常開車中單次閉眼，閉眼時間從 0.3 s 掃到 6 s | V05 | closure, duration |  | ready |
| `BND-YWN-DURATION` | 嘴張大的時間從 0.25 s 掃到 5 s | V07 | yawn, duration |  | ready |
| `BND-PHN-CALL-DURATION` | 講電話的時間從 0.25 s 掃到 6 s | V08 | phone, call, duration |  | ready |
| `BND-PHN-CALL-GAP` | 講電話 3 s、放下一下、再講 3 s | V08 | phone, call, gap |  | ready |
| `BND-SMK-DURATION` | 菸含在嘴上（手在方向盤）的時間從 0.25 s 掃到 5 s | V10 | smoking, duration |  | ready |
| `BND-SBT-OFF-DURATION` | 沒繫安全帶的時間從 0.5 s 掃到 10 s | V11 | seatbelt, duration |  | ready |

## CLR：停止與再發生：行為停下、隔一段時間再來一次

| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |
|---|---|---|---|---|---|
| `CLR-DIS-HEADTURN-REARM` | 頭向右轉 8 / 6 / 3.5 s、回正 8 / 4 / 2 s、再轉一次 | V01 | head_turn, clear, rearm | 三段 | ready |
| `CLR-DRW-AFTER-YAWN` | 打一次哈欠，之後 20 / 14 / 8 s 正常開車 | V07 | sleepiness, yawn, decay | 三段 | ready |
| `CLR-DRW-CLOSURE-REARM` | 閉眼 3 / 2.5 / 1.5 s、睜眼 15 / 12 / 5 s、再閉眼一次 | V05 | closure, clear, rearm | 三段 | ready |
| `CLR-YWN-TWO-YAWNS` | 兩次哈欠，間隔 12 / 10 / 5 s | V07 | yawn, clear, rearm | 三段 | ready |
| `CLR-PHN-CALL-REARM` | 講電話 5 / 4 / 2 s、放下 8 / 6 / 3 s、再講一次 | V08 | phone, call, clear, rearm | 三段 | ready |
| `CLR-SMK-REARM` | 叼菸 5 / 4 / 2 s、拿掉 8 / 6 / 3 s、再叼一次 | V10 | smoking, clear, rearm | 三段 | ready |
| `CLR-SBT-REBUCKLE` | 解開安全帶 10 / 8 / 3 s、繫回 10 / 8 / 3 s、再解開一次 | V11 | seatbelt, clear, rearm | 三段 | ready |

## ROB：偵測變差：漏框、掉幀、延遲、抖動加重

| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |
|---|---|---|---|---|---|
| `ROB-DIS-LOW-FPS` | 15 fps 時頭向右轉，持續時間從 1 s 掃到 6 s | V01, C03 | robustness, frame_rate, head_turn |  | ready |
| `ROB-DRW-EYE-FLIP` | 睜眼被認成閉眼，每次 0.5–2 s | C03, N01, V05 | robustness, class_flip |  | ready |
| `ROB-DRW-NCAP-SUNGLASSES` | Euro NCAP 遮擋（墨鏡）：眼框完全偵測不到 | C01, V01 | ncap, occlusion, sunglasses, robustness |  | ready |
| `ROB-PHN-LOW-CONF` | 手機框分數偏低，講電話 6 s | V08, C03 | robustness, phone, conf |  | ready |
| `ROB-ALL-BURST-MISS` | 連續漏框 1 / 5 / 15 / 30 幀 | C03, V08, V10, V05, V11, V01 | robustness, burst_miss |  | ready |
| `ROB-ALL-ERROR-PROFILES` | 通話、抽菸、微睡、未繫安全帶各一段，在五種模型誤差設定下重跑 | C03, V08, V10, V05, V11 | robustness, profiles |  | ready |
| `ROB-ALL-NCAP-FACE-MASK` | Euro NCAP 遮擋（口罩）：嘴框完全偵測不到 | C02, V05, V01, V07 | ncap, occlusion, mask, robustness |  | ready |

## NAT：長時間日常：隨機產生的長段駕駛

| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |
|---|---|---|---|---|---|
| `NAT-DRW-SLEEPY-600` | 600 s 睏倦駕駛：只開疲勞片段（每分鐘約 0.6 段） | V05, V07, V03, V01, N01, N02, N03 | natural, drowsy |  | ready |
| `NAT-ALL-CLEAN-300` | 300 s 日常駕駛：看路、看鏡子、看儀表，沒有手機、疲勞、抽菸、未繫 | N01, N02, N03 | natural, false_alarm |  | ready |
| `NAT-ALL-MIXED-120` | 120 s 隨機日常駕駛，夾雜手機、疲勞、抽菸、未繫安全帶片段 | N01, N02, N03, V08, V09, V03, V01, V05, V07, V10 | natural, existing |  | ready |
| `NAT-ALL-NORMAL-HABITS-300` | 300 s 日常駕駛，夾雜正常習慣：講話、嚼口香糖、大笑、揉眼、喝水、托下巴、撐頭、叼牙籤 | N01, N02, N03, N07, N13, N15, N17, N18, N19, N20, N21, N22 | natural, false_alarm, habits | 三段 | ready |

## REG：回歸：之前出過問題的片段

（還沒有條目）
