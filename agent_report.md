# Agent report

- LLM mode: off; model: gpt-4.1-mini
- Elapsed: 1.35 s; events: 33

## Pilots

| # | index | arm | channel | sub | n_req | n | cost | y | score | reason | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | tariff_8/HIGH/tariff_14 | call | tariff_8/HIGH/LITE/HIGH | 150 | 150 | 24 000 | -0.0573 | 2 544 120 | kg: kg=2671192.4 imm=-84332.5 pen=42739.5 bonus=0.0 total=2544120.5 | [-0.1859, 0.0714] |
| 2 | 1 | tariff_8/HIGH/tariff_18 | sms | tariff_8/HIGH/LITE/HIGH | 200 | 200 | 800 | 0.0439 | 2 209 348 | kg: kg=2245766.6 imm=-34813.0 pen=1605.2 bonus=0.0 total=2209348.4 | [-0.0675, 0.1554] |
| 3 | 2 | tariff_8/HIGH/tariff_19 | push | tariff_8/HIGH/LITE/LOW | 200 | 200 | 0 | -0.1079 | 2 084 550 | kg: kg=2120298.2 imm=-35748.3 pen=0.0 bonus=0.0 total=2084549.9 | [-0.2193, 0.0035] |
| 4 | 3 | tariff_8/HIGH/tariff_20 | push | tariff_8/HIGH/LITE/LOW | 200 | 200 | 0 | -0.0721 | 2 108 026 | kg: kg=2129245.1 imm=-21219.3 pen=0.0 bonus=0.0 total=2108025.8 | [-0.1835, 0.0393] |
| 5 | 4 | tariff_8/HIGH/tariff_10 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | -0.0148 | 2 010 706 | kg: kg=2051160.0 imm=-40454.2 pen=0.0 bonus=0.0 total=2010705.9 | [-0.1262, 0.0967] |
| 6 | 5 | tariff_8/HIGH/tariff_11 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.001 | 1 998 708 | kg: kg=2027818.0 imm=-29110.1 pen=0.0 bonus=0.0 total=1998707.9 | [-0.1104, 0.1124] |
| 7 | 6 | tariff_8/HIGH/tariff_12 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.0518 | 2 008 131 | kg: kg=2024657.5 imm=-16526.0 pen=0.0 bonus=0.0 total=2008131.5 | [-0.0596, 0.1632] |
| 8 | 7 | tariff_8/HIGH/tariff_21 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | 0.0441 | 1 965 134 | kg: kg=2011125.7 imm=-45991.2 pen=0.0 bonus=0.0 total=1965134.5 | [-0.0673, 0.1555] |
| 9 | 8 | tariff_8/HIGH/tariff_5 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | 0.0641 | 1 910 185 | kg: kg=1948891.1 imm=-38705.9 pen=0.0 bonus=0.0 total=1910185.2 | [-0.0473, 0.1755] |
| 10 | 9 | tariff_8/HIGH/tariff_5 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | 0.0266 | 2 205 783 | kg: kg=2205610.8 imm=172.0 pen=0.0 bonus=0.0 total=2205782.8 | [-0.0849, 0.138] |
| 11 | 10 | tariff_8/HIGH/tariff_6 | push | tariff_8/HIGH/HEAVY/HIGH | 200 | 200 | 0 | 0.021 | 1 622 210 | kg: kg=1622209.9 imm=0.0 pen=0.0 bonus=0.0 total=1622209.9 | [-0.0905, 0.1324] |
| 12 | 11 | tariff_8/HIGH/tariff_7 | push | tariff_8/HIGH/HEAVY/HIGH | 200 | 200 | 0 | -0.0545 | 1 624 138 | kg: kg=1624138.4 imm=0.0 pen=0.0 bonus=0.0 total=1624138.4 | [-0.1659, 0.0569] |
| 13 | 12 | tariff_8/HIGH/tariff_5 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | -0.0028 | 108 802 | verify: expected_loss=108802.5 deploy_channel=push v_contact=0.0 | [-0.1143, 0.1086] |
| 14 | 13 | tariff_8/HIGH/tariff_12 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | -0.0087 | 222 031 | verify: expected_loss=222030.8 deploy_channel=push v_contact=0.0 | [-0.1202, 0.1027] |
| 15 | 14 | tariff_4/MID/tariff_8 | digital_ads | tariff_4/MID/LITE/LOW | 200 | 200 | 4 400 | 0.2737 | 63 283 | verify: expected_loss=63282.6 deploy_channel=digital_ads v_contact=0.0 | [0.1623, 0.3851] |
| 16 | 15 | tariff_13/MID/tariff_8 | push | tariff_13/MID/LITE/LOW | 200 | 200 | 0 | 0.1194 | 38 068 | verify: expected_loss=38068.0 deploy_channel=digital_ads v_contact=0.0 | [0.008, 0.2308] |
| 17 | 16 | tariff_10/LOW/tariff_9 | push | tariff_10/LOW/LITE/LOW | 200 | 200 | 0 | 0.1094 | 16 463 | verify: expected_loss=16463.0 deploy_channel=push v_contact=0.0 | [-0.0021, 0.2208] |
| 18 | 17 | tariff_12/MID/tariff_8 | push | tariff_12/MID/HEAVY/LOW | 145 | 145 | 0 | 0.1743 | 11 964 | verify: expected_loss=11963.8 deploy_channel=digital_ads v_contact=0.0 | [0.0435, 0.3052] |

Pilots: 18, contacts: 3 495, cost: 29 200. CI = observed lift ratio ± 1.96·noise_sd/√n.

## Final plan

| campaign_name | filter_arpu_segment | filter_data_segment | filter_call_segment | filter_current_tariff | target_tariff | channel |
|---|---|---|---|---|---|---|
| c01_t8_digital_ads_MID_ALL_ALL_t1+t2+t3+t4+t12+t13+6more | MID | — | — | tariff_1;tariff_2;tariff_3;tariff_4;tariff_12;tariff_13;tariff_15;tariff_16;tariff_17;tariff_18;tariff_19;tariff_20 | tariff_8 | digital_ads |
| c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more | LOW | — | — | tariff_1;tariff_2;tariff_3;tariff_4;tariff_7;tariff_8;tariff_13;tariff_14;tariff_15;tariff_16;tariff_17;tariff_18;tariff_19 | tariff_9 | sms |
| c03_t13_push_LOW_ALL_ALL_t2+t5+t6+t20 | LOW | — | — | tariff_2;tariff_5;tariff_6;tariff_20 | tariff_13 | push |
| c04_t9_push_LOW_ALL_ALL_t6+t10 | LOW | — | — | tariff_6;tariff_10 | tariff_9 | push |
| c05_t8_sms_LOW_ALL_ALL_t2+t13+t17+t18+t20 | LOW | — | — | tariff_2;tariff_13;tariff_17;tariff_18;tariff_20 | tariff_8 | sms |
| c06_t8_push_LOW_ALL_ALL_t9+t11+t12+t13+t17+t20 | LOW | — | — | tariff_9;tariff_11;tariff_12;tariff_13;tariff_17;tariff_20 | tariff_8 | push |
| c07_t13_sms_LOW_ALL_ALL_t6 | LOW | — | — | tariff_6 | tariff_13 | sms |
| c08_t5_push_HIGH_ALL_ALL_t8 | HIGH | — | — | tariff_8 | tariff_5 | push |
| c09_t8_push_MID_ALL_ALL_t9+t11 | MID | — | — | tariff_9;tariff_11 | tariff_8 | push |

- Expected net: 5 757 330 (sd —); P(net>0): —

## LLM decisions

| agent | source | mode | n_arms | ok |
|---|---|---|---|---|
| HypothesisAnalyst | off | off | 40 | no |
| RiskReviewer | off | off | — | no |

## Timings

| t | kind | fields |
|---|---|---|
| 0 | start | {"llm_mode":"off","time_budget_s":420.0} |
| 0.026 | research | {"cells":63,"history":14823,"subs":359} |
| 0.05 | llm | {"agent":"HypothesisAnalyst","mode":"off","ok":false,"source":"off"} |
| 0.05 | hypotheses | {"arms":40,"down":0,"up":0} |
| 0.145 | pilot | {"arm":"tariff_8/HIGH/tariff_14","channel":"call","cost":24000.0,"n":150,"y":-0.0573} |
| 0.21 | pilot | {"arm":"tariff_8/HIGH/tariff_18","channel":"sms","cost":800.0,"n":200,"y":0.0439} |
| 0.264 | pilot | {"arm":"tariff_8/HIGH/tariff_19","channel":"push","cost":0.0,"n":200,"y":-0.1079} |
| 0.322 | pilot | {"arm":"tariff_8/HIGH/tariff_20","channel":"push","cost":0.0,"n":200,"y":-0.0721} |
| 0.379 | pilot | {"arm":"tariff_8/HIGH/tariff_10","channel":"push","cost":0.0,"n":200,"y":-0.0148} |
| 0.437 | pilot | {"arm":"tariff_8/HIGH/tariff_11","channel":"push","cost":0.0,"n":200,"y":0.001} |
| 0.495 | pilot | {"arm":"tariff_8/HIGH/tariff_12","channel":"push","cost":0.0,"n":200,"y":0.0518} |
| 0.554 | pilot | {"arm":"tariff_8/HIGH/tariff_21","channel":"push","cost":0.0,"n":200,"y":0.0441} |
| 0.619 | pilot | {"arm":"tariff_8/HIGH/tariff_5","channel":"push","cost":0.0,"n":200,"y":0.0641} |
| 0.686 | pilot | {"arm":"tariff_8/HIGH/tariff_5","channel":"push","cost":0.0,"n":200,"y":0.0266} |
| 0.74 | pilot | {"arm":"tariff_8/HIGH/tariff_6","channel":"push","cost":0.0,"n":200,"y":0.021} |
| 0.804 | pilot | {"arm":"tariff_8/HIGH/tariff_7","channel":"push","cost":0.0,"n":200,"y":-0.0545} |
| 0.836 | pilot | {"arm":"tariff_8/HIGH/tariff_5","channel":"push","cost":0.0,"n":200,"y":-0.0028} |
| 0.863 | pilot | {"arm":"tariff_8/HIGH/tariff_12","channel":"push","cost":0.0,"n":200,"y":-0.0087} |
| 0.887 | pilot | {"arm":"tariff_4/MID/tariff_8","channel":"digital_ads","cost":4400.0,"n":200,"y":0.2737} |
| 0.925 | pilot | {"arm":"tariff_13/MID/tariff_8","channel":"push","cost":0.0,"n":200,"y":0.1194} |
| 0.959 | pilot | {"arm":"tariff_10/LOW/tariff_9","channel":"push","cost":0.0,"n":200,"y":0.1094} |
| 0.992 | pilot | {"arm":"tariff_12/MID/tariff_8","channel":"push","cost":0.0,"n":145,"y":0.1743} |
| 1.031 | verify_skip | {"arm":"tariff_11/MID/tariff_8","channel":"push","expected_loss":40424.7,"pilot_cost":42433.0} |
| 1.031 | verify_stop | {"pilots":18} |
| 1.335 | allocate | {"campaigns":9,"chosen":"coarse","options":181} |
| 1.338 | llm | {"agent":"RiskReviewer","mode":"off","ok":false,"source":"off"} |
| 1.338 | review | {"applied":false,"decision":"not_applied","source":"off","veto":[]} |
| 1.341 | guardrail | {"issue":"overlap","name":"c03_t13_push_LOW_ALL_ALL_t2+t5+t6+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more"} |
| 1.341 | guardrail | {"issue":"overlap","name":"c04_t9_push_LOW_ALL_ALL_t6+t10","others":"c03_t13_push_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.342 | guardrail | {"issue":"overlap","name":"c05_t8_sms_LOW_ALL_ALL_t2+t13+t17+t18+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more;c03_t13_push_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.342 | guardrail | {"issue":"overlap","name":"c06_t8_push_LOW_ALL_ALL_t9+t11+t12+t13+t17+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more;c03_t13_push_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.342 | guardrail | {"issue":"overlap","name":"c07_t13_sms_LOW_ALL_ALL_t6","others":"c03_t13_push_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.345 | final | {"campaigns":9} |

## Run details

- **allocate**
  - chosen: coarse
  - coarse: {"campaigns":9,"net":5066919.5553,"overlap_gain":690410.6224,"within_limits":true}
  - exclude: {"campaigns":10,"net":3210040.3614,"overlap_gain":0.0,"within_limits":true}
  - overlap_cost: {"campaigns":10,"net":3360974.4721,"overlap_gain":226393.9959,"within_limits":true}
- **calibration**
  - beta: 1.1198
  - n_arms: 15
  - tau: 0
- **explore**
  - money: 29 200
  - pilots: 18
  - reach: 3 495
- **final_sim**
  - contacts: 11 465
  - cost: 70 784
  - gross: 5 828 114
  - net: 5 757 330
  - within_limits: yes
- **review**
  - applied: no
  - decision: not_applied
  - rationale: —
  - source: off
  - summary: —
  - veto: —
- **stages**
  - allocate: ok
  - explore: ok
  - finalize: ok
  - hypotheses: ok
  - research: ok
  - review: ok
