# Agent report

- LLM mode: off; model: gpt-4.1-mini
- Elapsed: 1.37 s; events: 31

## Pilots

| # | index | arm | channel | sub | n_req | n | cost | y | score | reason | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | tariff_8/HIGH/tariff_14 | call | tariff_8/HIGH/LITE/HIGH | 150 | 150 | 24 000 | 0.0649 | 2 544 120 | kg: kg=2671192.4 imm=-84332.5 pen=42739.5 bonus=0.0 total=2544120.5 | [-0.0637, 0.1936] |
| 2 | 1 | tariff_8/HIGH/tariff_18 | sms | tariff_8/HIGH/LITE/HIGH | 200 | 200 | 800 | 0.0201 | 2 170 072 | kg: kg=2206489.8 imm=-34813.0 pen=1605.2 bonus=0.0 total=2170071.5 | [-0.0914, 0.1315] |
| 3 | 2 | tariff_8/HIGH/tariff_19 | push | tariff_8/HIGH/LITE/LOW | 200 | 200 | 0 | 0.0545 | 2 047 168 | kg: kg=2082916.5 imm=-35748.3 pen=0.0 bonus=0.0 total=2047168.2 | [-0.0569, 0.1659] |
| 4 | 3 | tariff_8/HIGH/tariff_20 | push | tariff_8/HIGH/LITE/LOW | 200 | 200 | 0 | 0.0273 | 2 042 773 | kg: kg=2063935.1 imm=-21162.1 pen=0.0 bonus=0.0 total=2042773.1 | [-0.0841, 0.1388] |
| 5 | 4 | tariff_8/HIGH/tariff_19 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.0361 | 1 983 200 | kg: kg=1984537.3 imm=-1337.0 pen=0.0 bonus=0.0 total=1983200.3 | [-0.0754, 0.1475] |
| 6 | 5 | tariff_8/HIGH/tariff_10 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.0239 | 1 606 310 | kg: kg=1606310.1 imm=0.0 pen=0.0 bonus=0.0 total=1606310.1 | [-0.0875, 0.1354] |
| 7 | 6 | tariff_8/HIGH/tariff_11 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.0036 | 1 543 332 | kg: kg=1543332.3 imm=0.0 pen=0.0 bonus=0.0 total=1543332.3 | [-0.1078, 0.115] |
| 8 | 7 | tariff_8/HIGH/tariff_12 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | 0.0329 | 1 540 166 | kg: kg=1540165.5 imm=0.0 pen=0.0 bonus=0.0 total=1540165.5 | [-0.0786, 0.1443] |
| 9 | 8 | tariff_8/HIGH/tariff_21 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | -0.0094 | 1 526 745 | kg: kg=1526745.2 imm=0.0 pen=0.0 bonus=0.0 total=1526745.2 | [-0.1208, 0.1021] |
| 10 | 9 | tariff_8/HIGH/tariff_5 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | -0.0274 | 1 333 416 | kg: kg=1333415.9 imm=0.0 pen=0.0 bonus=0.0 total=1333415.9 | [-0.1388, 0.084] |
| 11 | 10 | tariff_8/HIGH/tariff_6 | push | tariff_8/HIGH/HEAVY/HIGH | 200 | 200 | 0 | 0.0023 | 1 331 068 | kg: kg=1331068.2 imm=0.0 pen=0.0 bonus=0.0 total=1331068.2 | [-0.1092, 0.1137] |
| 12 | 11 | tariff_8/HIGH/tariff_7 | push | tariff_8/HIGH/HEAVY/HIGH | 200 | 200 | 0 | -0.0166 | 1 331 855 | kg: kg=1331855.0 imm=0.0 pen=0.0 bonus=0.0 total=1331855.0 | [-0.1281, 0.0948] |
| 13 | 12 | tariff_8/HIGH/tariff_19 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | -0.0116 | 101 625 | verify: expected_loss=101625.1 deploy_channel=push v_contact=0.0 | [-0.123, 0.0998] |
| 14 | 13 | tariff_8/HIGH/tariff_14 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | 0.005 | 86 493 | verify: expected_loss=86493.3 deploy_channel=push v_contact=0.0 | [-0.1064, 0.1165] |
| 15 | 14 | tariff_8/HIGH/tariff_14 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | 0.0782 | 86 522 | verify: expected_loss=86522.4 deploy_channel=push v_contact=0.0 | [-0.0333, 0.1896] |
| 16 | 15 | tariff_4/MID/tariff_8 | digital_ads | tariff_4/MID/LITE/LOW | 200 | 200 | 4 400 | 0.2614 | 58 167 | verify: expected_loss=58166.8 deploy_channel=digital_ads v_contact=0.0 | [0.15, 0.3729] |
| 17 | 16 | tariff_13/MID/tariff_8 | push | tariff_13/MID/LITE/LOW | 200 | 200 | 0 | 0.0208 | 36 788 | verify: expected_loss=36787.7 deploy_channel=digital_ads v_contact=0.0 | [-0.0906, 0.1322] |
| 18 | 17 | tariff_13/MID/tariff_8 | push | tariff_13/MID/LITE/MEDIUM | 200 | 200 | 0 | 0.1183 | 26 304 | verify: expected_loss=26304.3 deploy_channel=push v_contact=0.0 | [0.0068, 0.2297] |
| 19 | 18 | tariff_10/LOW/tariff_9 | push | tariff_10/LOW/LITE/LOW | 200 | 200 | 0 | 0.1415 | 18 651 | verify: expected_loss=18651.3 deploy_channel=push v_contact=0.0 | [0.0301, 0.2529] |
| 20 | 19 | tariff_12/MID/tariff_8 | push | tariff_12/MID/HEAVY/LOW | 145 | 145 | 0 | 0.1533 | 15 594 | verify: expected_loss=15594.3 deploy_channel=digital_ads v_contact=0.0 | [0.0224, 0.2842] |

Pilots: 20, contacts: 3 895, cost: 29 200. CI = observed lift ratio ± 1.96·noise_sd/√n.

## Final plan

| campaign_name | filter_arpu_segment | filter_data_segment | filter_call_segment | filter_current_tariff | target_tariff | channel |
|---|---|---|---|---|---|---|
| c01_t8_digital_ads_MID_ALL_ALL_t1+t2+t3+t4+t12+t13+6more | MID | — | — | tariff_1;tariff_2;tariff_3;tariff_4;tariff_12;tariff_13;tariff_15;tariff_16;tariff_17;tariff_18;tariff_19;tariff_20 | tariff_8 | digital_ads |
| c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more | LOW | — | — | tariff_1;tariff_2;tariff_3;tariff_4;tariff_7;tariff_8;tariff_10;tariff_14;tariff_15;tariff_16;tariff_17;tariff_18;tariff_19 | tariff_9 | sms |
| c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20 | LOW | — | — | tariff_2;tariff_5;tariff_6;tariff_20 | tariff_13 | sms |
| c04_t8_push_LOW_ALL_ALL_t9+t11+t13+t17+t20 | LOW | — | — | tariff_9;tariff_11;tariff_13;tariff_17;tariff_20 | tariff_8 | push |
| c05_t14_push_HIGH_ALL_ALL_t8 | HIGH | — | — | tariff_8 | tariff_14 | push |
| c06_t8_sms_LOW_ALL_ALL_t9+t15+t17+t18 | LOW | — | — | tariff_9;tariff_15;tariff_17;tariff_18 | tariff_8 | sms |
| c07_t8_push_MID_ALL_ALL_t9 | MID | — | — | tariff_9 | tariff_8 | push |

- Expected net: 5 097 871 (sd —); P(net>0): —

## LLM decisions

| agent | source | mode | n_arms | ok |
|---|---|---|---|---|
| HypothesisAnalyst | off | off | 40 | no |
| RiskReviewer | off | off | — | no |

## Timings

| t | kind | fields |
|---|---|---|
| 0 | start | {"llm_mode":"off","time_budget_s":420.0} |
| 0.029 | research | {"cells":63,"history":14823,"subs":359} |
| 0.052 | llm | {"agent":"HypothesisAnalyst","mode":"off","ok":false,"source":"off"} |
| 0.052 | hypotheses | {"arms":40,"down":0,"up":0} |
| 0.208 | pilot | {"arm":"tariff_8/HIGH/tariff_14","channel":"call","cost":24000.0,"n":150,"y":0.0649} |
| 0.28 | pilot | {"arm":"tariff_8/HIGH/tariff_18","channel":"sms","cost":800.0,"n":200,"y":0.0201} |
| 0.337 | pilot | {"arm":"tariff_8/HIGH/tariff_19","channel":"push","cost":0.0,"n":200,"y":0.0545} |
| 0.398 | pilot | {"arm":"tariff_8/HIGH/tariff_20","channel":"push","cost":0.0,"n":200,"y":0.0273} |
| 0.457 | pilot | {"arm":"tariff_8/HIGH/tariff_19","channel":"push","cost":0.0,"n":200,"y":0.0361} |
| 0.513 | pilot | {"arm":"tariff_8/HIGH/tariff_10","channel":"push","cost":0.0,"n":200,"y":0.0239} |
| 0.574 | pilot | {"arm":"tariff_8/HIGH/tariff_11","channel":"push","cost":0.0,"n":200,"y":0.0036} |
| 0.635 | pilot | {"arm":"tariff_8/HIGH/tariff_12","channel":"push","cost":0.0,"n":200,"y":0.0329} |
| 0.696 | pilot | {"arm":"tariff_8/HIGH/tariff_21","channel":"push","cost":0.0,"n":200,"y":-0.0094} |
| 0.757 | pilot | {"arm":"tariff_8/HIGH/tariff_5","channel":"push","cost":0.0,"n":200,"y":-0.0274} |
| 0.817 | pilot | {"arm":"tariff_8/HIGH/tariff_6","channel":"push","cost":0.0,"n":200,"y":0.0023} |
| 0.878 | pilot | {"arm":"tariff_8/HIGH/tariff_7","channel":"push","cost":0.0,"n":200,"y":-0.0166} |
| 0.912 | pilot | {"arm":"tariff_8/HIGH/tariff_19","channel":"push","cost":0.0,"n":200,"y":-0.0116} |
| 0.942 | pilot | {"arm":"tariff_8/HIGH/tariff_14","channel":"push","cost":0.0,"n":200,"y":0.005} |
| 0.97 | pilot | {"arm":"tariff_8/HIGH/tariff_14","channel":"push","cost":0.0,"n":200,"y":0.0782} |
| 0.997 | pilot | {"arm":"tariff_4/MID/tariff_8","channel":"digital_ads","cost":4400.0,"n":200,"y":0.2614} |
| 1.037 | pilot | {"arm":"tariff_13/MID/tariff_8","channel":"push","cost":0.0,"n":200,"y":0.0208} |
| 1.069 | pilot | {"arm":"tariff_13/MID/tariff_8","channel":"push","cost":0.0,"n":200,"y":0.1183} |
| 1.096 | pilot | {"arm":"tariff_10/LOW/tariff_9","channel":"push","cost":0.0,"n":200,"y":0.1415} |
| 1.126 | pilot | {"arm":"tariff_12/MID/tariff_8","channel":"push","cost":0.0,"n":145,"y":0.1533} |
| 1.364 | allocate | {"campaigns":7,"chosen":"coarse","options":169} |
| 1.366 | llm | {"agent":"RiskReviewer","mode":"off","ok":false,"source":"off"} |
| 1.366 | review | {"applied":false,"decision":"not_applied","source":"off","veto":[]} |
| 1.369 | guardrail | {"issue":"overlap","name":"c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more"} |
| 1.369 | guardrail | {"issue":"overlap","name":"c04_t8_push_LOW_ALL_ALL_t9+t11+t13+t17+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more;c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.37 | guardrail | {"issue":"overlap","name":"c06_t8_sms_LOW_ALL_ALL_t9+t15+t17+t18","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more;c04_t8_push_LOW_ALL_ALL_t9+t11+t13+t17+t20"} |
| 1.372 | final | {"campaigns":7} |

## Run details

- **allocate**
  - chosen: coarse
  - coarse: {"campaigns":7,"net":4169372.551,"overlap_gain":928498.4346,"within_limits":true}
  - exclude: {"campaigns":10,"net":2409989.8918,"overlap_gain":0.0,"within_limits":true}
  - overlap_cost: {"campaigns":10,"net":3129982.4677,"overlap_gain":216615.3653,"within_limits":true}
- **calibration**
  - beta: 1.0062
  - n_arms: 15
  - tau: 0
- **explore**
  - money: 29 200
  - pilots: 20
  - reach: 3 895
- **final_sim**
  - contacts: 10 127
  - cost: 70 692
  - gross: 5 168 563
  - net: 5 097 871
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
