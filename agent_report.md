# Agent report

- LLM mode: off; model: gpt-4.1-mini
- Elapsed: 1.38 s; events: 32

## Pilots

| # | index | arm | channel | sub | n_req | n | cost | y | score | reason | ci95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | tariff_8/HIGH/tariff_14 | call | tariff_8/HIGH/LITE/HIGH | 150 | 150 | 24 000 | -0.0457 | 2 544 120 | kg: kg=2671192.4 imm=-84332.5 pen=42739.5 bonus=0.0 total=2544120.5 | [-0.1744, 0.0829] |
| 2 | 1 | tariff_8/HIGH/tariff_18 | sms | tariff_8/HIGH/LITE/HIGH | 200 | 200 | 800 | -0.0297 | 2 209 348 | kg: kg=2245766.6 imm=-34813.0 pen=1605.2 bonus=0.0 total=2209348.4 | [-0.1411, 0.0818] |
| 3 | 2 | tariff_8/HIGH/tariff_19 | push | tariff_8/HIGH/LITE/LOW | 200 | 200 | 0 | 0.0976 | 2 084 550 | kg: kg=2120298.2 imm=-35748.3 pen=0.0 bonus=0.0 total=2084549.9 | [-0.0139, 0.209] |
| 4 | 3 | tariff_8/HIGH/tariff_20 | push | tariff_8/HIGH/LITE/LOW | 200 | 200 | 0 | 0.0603 | 1 462 017 | kg: kg=1462016.7 imm=0.0 pen=0.0 bonus=0.0 total=1462016.7 | [-0.0511, 0.1717] |
| 5 | 4 | tariff_8/HIGH/tariff_10 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.0387 | 1 368 076 | kg: kg=1368076.0 imm=0.0 pen=0.0 bonus=0.0 total=1368076.0 | [-0.0728, 0.1501] |
| 6 | 5 | tariff_8/HIGH/tariff_11 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.009 | 1 315 037 | kg: kg=1315036.6 imm=0.0 pen=0.0 bonus=0.0 total=1315036.6 | [-0.1024, 0.1204] |
| 7 | 6 | tariff_8/HIGH/tariff_12 | push | tariff_8/HIGH/HEAVY/LOW | 200 | 200 | 0 | 0.0427 | 1 311 652 | kg: kg=1311651.8 imm=0.0 pen=0.0 bonus=0.0 total=1311651.8 | [-0.0687, 0.1541] |
| 8 | 7 | tariff_8/HIGH/tariff_21 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | -0.0178 | 1 297 630 | kg: kg=1297630.0 imm=0.0 pen=0.0 bonus=0.0 total=1297630.0 | [-0.1292, 0.0936] |
| 9 | 8 | tariff_8/HIGH/tariff_19 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | -0.0498 | 1 271 936 | kg: kg=1264103.8 imm=7832.5 pen=0.0 bonus=0.0 total=1271936.3 | [-0.1612, 0.0616] |
| 10 | 9 | tariff_8/HIGH/tariff_20 | push | tariff_8/HIGH/LITE/MEDIUM | 200 | 200 | 0 | 0.0211 | 2 134 904 | kg: kg=2134818.8 imm=84.8 pen=0.0 bonus=0.0 total=2134903.7 | [-0.0903, 0.1325] |
| 11 | 10 | tariff_8/HIGH/tariff_5 | push | tariff_8/HIGH/HEAVY/HIGH | 200 | 200 | 0 | 0.0262 | 1 792 755 | kg: kg=1848483.5 imm=-55728.8 pen=0.0 bonus=0.0 total=1792754.8 | [-0.0852, 0.1377] |
| 12 | 11 | tariff_8/HIGH/tariff_6 | push | tariff_8/HIGH/HEAVY/HIGH | 200 | 200 | 0 | -0.0393 | 1 810 225 | kg: kg=1854622.3 imm=-44397.6 pen=0.0 bonus=0.0 total=1810224.7 | [-0.1507, 0.0721] |
| 13 | 12 | tariff_8/HIGH/tariff_20 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | -0.0555 | 124 470 | verify: expected_loss=124470.3 deploy_channel=push v_contact=0.0 | [-0.1669, 0.0559] |
| 14 | 13 | tariff_8/HIGH/tariff_12 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | 0.035 | 281 423 | verify: expected_loss=281423.5 deploy_channel=push v_contact=0.0 | [-0.0764, 0.1464] |
| 15 | 14 | tariff_8/HIGH/tariff_12 | push | tariff_8/HIGH/HEAVY/MEDIUM | 200 | 200 | 0 | 0.0059 | 138 953 | verify: expected_loss=138952.8 deploy_channel=push v_contact=0.0 | [-0.1055, 0.1174] |
| 16 | 15 | tariff_4/MID/tariff_8 | digital_ads | tariff_4/MID/LITE/LOW | 200 | 200 | 4 400 | 0.2086 | 58 672 | verify: expected_loss=58671.6 deploy_channel=digital_ads v_contact=0.0 | [0.0971, 0.32] |
| 17 | 16 | tariff_13/MID/tariff_8 | push | tariff_13/MID/LITE/LOW | 200 | 200 | 0 | 0.1472 | 43 540 | verify: expected_loss=43539.5 deploy_channel=digital_ads v_contact=0.0 | [0.0358, 0.2587] |
| 18 | 17 | tariff_10/LOW/tariff_9 | push | tariff_10/LOW/LITE/LOW | 200 | 200 | 0 | 0.256 | 16 498 | verify: expected_loss=16498.1 deploy_channel=push v_contact=0.0 | [0.1445, 0.3674] |
| 19 | 18 | tariff_11/MID/tariff_8 | push | tariff_11/MID/LITE/LOW | 169 | 169 | 0 | -0.0329 | 38 916 | verify: expected_loss=38916.1 deploy_channel=push v_contact=0.0 | [-0.1541, 0.0883] |
| 20 | 19 | tariff_12/LOW/tariff_8 | push | tariff_12/LOW/LITE/LOW | 200 | 200 | 0 | 0.0837 | 20 231 | verify: expected_loss=20231.3 deploy_channel=push v_contact=0.0 | [-0.0278, 0.1951] |

Pilots: 20, contacts: 3 919, cost: 29 200. CI = observed lift ratio ± 1.96·noise_sd/√n.

## Final plan

| campaign_name | filter_arpu_segment | filter_data_segment | filter_call_segment | filter_current_tariff | target_tariff | channel |
|---|---|---|---|---|---|---|
| c01_t8_digital_ads_MID_ALL_ALL_t1+t2+t3+t4+t12+t13+6more | MID | — | — | tariff_1;tariff_2;tariff_3;tariff_4;tariff_12;tariff_13;tariff_15;tariff_16;tariff_17;tariff_18;tariff_19;tariff_20 | tariff_8 | digital_ads |
| c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more | LOW | — | — | tariff_1;tariff_2;tariff_3;tariff_4;tariff_7;tariff_8;tariff_10;tariff_14;tariff_15;tariff_16;tariff_17;tariff_18;tariff_19 | tariff_9 | sms |
| c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20 | LOW | — | — | tariff_2;tariff_5;tariff_6;tariff_20 | tariff_13 | sms |
| c04_t8_push_LOW_ALL_ALL_t9+t11+t12+t13+t17+t20 | LOW | — | — | tariff_9;tariff_11;tariff_12;tariff_13;tariff_17;tariff_20 | tariff_8 | push |
| c05_t8_sms_LOW_ALL_ALL_t9+t17+t18 | LOW | — | — | tariff_9;tariff_17;tariff_18 | tariff_8 | sms |
| c06_t13_digital_ads_LOW_ALL_ALL_t20 | LOW | — | — | tariff_20 | tariff_13 | digital_ads |
| c07_t12_push_HIGH_ALL_ALL_t8 | HIGH | — | — | tariff_8 | tariff_12 | push |
| c08_t8_push_MID_ALL_ALL_t9 | MID | — | — | tariff_9 | tariff_8 | push |

- Expected net: 5 593 777 (sd —); P(net>0): —

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
| 0.047 | llm | {"agent":"HypothesisAnalyst","mode":"off","ok":false,"source":"off"} |
| 0.047 | hypotheses | {"arms":40,"down":0,"up":0} |
| 0.139 | pilot | {"arm":"tariff_8/HIGH/tariff_14","channel":"call","cost":24000.0,"n":150,"y":-0.0457} |
| 0.203 | pilot | {"arm":"tariff_8/HIGH/tariff_18","channel":"sms","cost":800.0,"n":200,"y":-0.0297} |
| 0.256 | pilot | {"arm":"tariff_8/HIGH/tariff_19","channel":"push","cost":0.0,"n":200,"y":0.0976} |
| 0.318 | pilot | {"arm":"tariff_8/HIGH/tariff_20","channel":"push","cost":0.0,"n":200,"y":0.0603} |
| 0.381 | pilot | {"arm":"tariff_8/HIGH/tariff_10","channel":"push","cost":0.0,"n":200,"y":0.0387} |
| 0.444 | pilot | {"arm":"tariff_8/HIGH/tariff_11","channel":"push","cost":0.0,"n":200,"y":0.009} |
| 0.54 | pilot | {"arm":"tariff_8/HIGH/tariff_12","channel":"push","cost":0.0,"n":200,"y":0.0427} |
| 0.602 | pilot | {"arm":"tariff_8/HIGH/tariff_21","channel":"push","cost":0.0,"n":200,"y":-0.0178} |
| 0.663 | pilot | {"arm":"tariff_8/HIGH/tariff_19","channel":"push","cost":0.0,"n":200,"y":-0.0498} |
| 0.715 | pilot | {"arm":"tariff_8/HIGH/tariff_20","channel":"push","cost":0.0,"n":200,"y":0.0211} |
| 0.769 | pilot | {"arm":"tariff_8/HIGH/tariff_5","channel":"push","cost":0.0,"n":200,"y":0.0262} |
| 0.826 | pilot | {"arm":"tariff_8/HIGH/tariff_6","channel":"push","cost":0.0,"n":200,"y":-0.0393} |
| 0.858 | pilot | {"arm":"tariff_8/HIGH/tariff_20","channel":"push","cost":0.0,"n":200,"y":-0.0555} |
| 0.888 | pilot | {"arm":"tariff_8/HIGH/tariff_12","channel":"push","cost":0.0,"n":200,"y":0.035} |
| 0.917 | pilot | {"arm":"tariff_8/HIGH/tariff_12","channel":"push","cost":0.0,"n":200,"y":0.0059} |
| 0.943 | pilot | {"arm":"tariff_4/MID/tariff_8","channel":"digital_ads","cost":4400.0,"n":200,"y":0.2086} |
| 0.978 | pilot | {"arm":"tariff_13/MID/tariff_8","channel":"push","cost":0.0,"n":200,"y":0.1472} |
| 1.015 | pilot | {"arm":"tariff_10/LOW/tariff_9","channel":"push","cost":0.0,"n":200,"y":0.256} |
| 1.055 | pilot | {"arm":"tariff_11/MID/tariff_8","channel":"push","cost":0.0,"n":169,"y":-0.0329} |
| 1.092 | pilot | {"arm":"tariff_12/LOW/tariff_8","channel":"push","cost":0.0,"n":200,"y":0.0837} |
| 1.368 | allocate | {"campaigns":8,"chosen":"coarse","options":174} |
| 1.37 | llm | {"agent":"RiskReviewer","mode":"off","ok":false,"source":"off"} |
| 1.37 | review | {"applied":false,"source":"off","veto":[]} |
| 1.373 | guardrail | {"issue":"overlap","name":"c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more"} |
| 1.373 | guardrail | {"issue":"overlap","name":"c04_t8_push_LOW_ALL_ALL_t9+t11+t12+t13+t17+t20","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more;c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.373 | guardrail | {"issue":"overlap","name":"c05_t8_sms_LOW_ALL_ALL_t9+t17+t18","others":"c02_t9_sms_LOW_ALL_ALL_t1+t2+t3+t4+t7+t8+7more;c04_t8_push_LOW_ALL_ALL_t9+t11+t12+t13+t17+t20"} |
| 1.374 | guardrail | {"issue":"overlap","name":"c06_t13_digital_ads_LOW_ALL_ALL_t20","others":"c03_t13_sms_LOW_ALL_ALL_t2+t5+t6+t20"} |
| 1.376 | final | {"campaigns":8} |

## Run details

- **allocate**
  - chosen: coarse
  - coarse: {"campaigns":8,"net":4825106.5501,"overlap_gain":768670.0738,"within_limits":true}
  - exclude: {"campaigns":10,"net":3041343.3135,"overlap_gain":0.0,"within_limits":true}
  - overlap_cost: {"campaigns":10,"net":2890856.1158,"overlap_gain":288945.2783,"within_limits":true}
- **calibration**
  - beta: 1.1445
  - n_arms: 15
  - tau: 0
- **explore**
  - money: 29 200
  - pilots: 20
  - reach: 3 919
- **final_sim**
  - contacts: 10 481
  - cost: 70 746
  - gross: 5 664 523
  - net: 5 593 777
  - within_limits: yes
- **review**
  - applied: no
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
