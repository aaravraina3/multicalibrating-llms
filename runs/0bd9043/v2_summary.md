# v2 summary (test)

Run folder `runs/0bd9043`. Calibrators fit on calib; IGLB and boosted multicalibration early stop on val_tune; conformal thresholds chosen on val_conformal. SHAP diagnostics are on val_tune by design (safeguard 4) and explain B4 before calibration (safeguard 8). Intervals: 95% task clustered bootstrap.

## Split sizes

```
                       rows  problems  pass_rate
model   role                                    
gpt-oss base_train     3160       316      0.533
        calib          2110       211      0.501
        val_conformal  1320       132      0.503
        val_tune       1320       132      0.487
qwen3   base_train     3160       316      0.450
        calib          2110       211      0.438
        val_conformal  1320       132      0.428
        val_tune       1320       132      0.467
```

## Base predictor x calibrator, Qwen3 Coder

| start | calibrator | Brier | BSS [95%] | ECE | max gASCE (hand) | AUROC |
|---|---|---|---|---|---|---|
| avg_prob | uncalibrated | 0.383 | -0.543 [-0.738, -0.378] | 0.428 | 0.671 | 0.865 |
| avg_prob | platt | 0.151 | 0.390 [0.318, 0.451] | 0.060 | 0.047 | 0.865 |
| avg_prob | hb | 0.153 | 0.382 [0.313, 0.443] | 0.052 | 0.048 | 0.853 |
| avg_prob | linr | 0.136 | 0.452 [0.389, 0.508] | 0.052 | 0.050 | 0.889 |
| avg_prob | logr | 0.137 | 0.447 [0.384, 0.507] | 0.061 | 0.056 | 0.886 |
| avg_prob | ighb | 0.149 | 0.401 [0.323, 0.470] | 0.062 | 0.044 | 0.863 |
| avg_prob | iglb | 0.147 | 0.409 [0.340, 0.474] | 0.053 | 0.043 | 0.866 |
| avg_prob | bmc | 0.135 | 0.454 [0.378, 0.519] | 0.054 | 0.073 | 0.887 |
| B2 | uncalibrated | 0.109 | 0.560 [0.487, 0.625] | 0.041 | 0.050 | 0.928 |
| B2 | platt | 0.109 | 0.559 [0.477, 0.630] | 0.037 | 0.055 | 0.928 |
| B2 | hb | 0.114 | 0.540 [0.458, 0.611] | 0.037 | 0.062 | 0.924 |
| B2 | linr | 0.109 | 0.560 [0.481, 0.630] | 0.041 | 0.054 | 0.928 |
| B2 | logr | 0.110 | 0.556 [0.474, 0.628] | 0.038 | 0.058 | 0.928 |
| B2 | ighb | 0.118 | 0.527 [0.449, 0.596] | 0.046 | 0.080 | 0.916 |
| B2 | iglb | 0.109 | 0.560 [0.487, 0.625] | 0.041 | 0.050 | 0.928 |
| B2 | bmc | 0.109 | 0.559 [0.483, 0.625] | 0.039 | 0.058 | 0.928 |
| B4 | uncalibrated | 0.115 | 0.538 [0.455, 0.606] | 0.040 | 0.062 | 0.921 |
| B4 | platt | 0.117 | 0.527 [0.435, 0.602] | 0.041 | 0.074 | 0.921 |
| B4 | hb | 0.118 | 0.526 [0.438, 0.598] | 0.030 | 0.070 | 0.914 |
| B4 | linr | 0.111 | 0.552 [0.469, 0.622] | 0.039 | 0.054 | 0.925 |
| B4 | logr | 0.112 | 0.549 [0.463, 0.622] | 0.045 | 0.055 | 0.926 |
| B4 | ighb | 0.121 | 0.514 [0.432, 0.582] | 0.047 | 0.072 | 0.911 |
| B4 | iglb | 0.115 | 0.535 [0.450, 0.610] | 0.042 | 0.066 | 0.919 |
| B4 | bmc | 0.116 | 0.534 [0.448, 0.605] | 0.030 | 0.070 | 0.920 |
| B6 | uncalibrated | 0.124 | 0.501 [0.424, 0.568] | 0.042 | 0.046 | 0.909 |
| B6 | platt | 0.122 | 0.507 [0.421, 0.579] | 0.034 | 0.060 | 0.909 |
| B6 | hb | 0.125 | 0.498 [0.410, 0.569] | 0.038 | 0.065 | 0.906 |
| B6 | linr | 0.119 | 0.522 [0.441, 0.590] | 0.035 | 0.051 | 0.913 |
| B6 | logr | 0.117 | 0.528 [0.446, 0.597] | 0.029 | 0.053 | 0.916 |
| B6 | ighb | 0.123 | 0.503 [0.427, 0.568] | 0.041 | 0.046 | 0.909 |
| B6 | iglb | 0.122 | 0.508 [0.422, 0.580] | 0.034 | 0.064 | 0.909 |
| B6 | bmc | 0.124 | 0.501 [0.414, 0.573] | 0.034 | 0.063 | 0.908 |
| B7 | uncalibrated | 0.119 | 0.520 [0.433, 0.594] | 0.055 | 0.048 | 0.916 |
| B7 | platt | 0.117 | 0.528 [0.439, 0.604] | 0.041 | 0.064 | 0.916 |
| B7 | hb | 0.118 | 0.526 [0.438, 0.600] | 0.036 | 0.066 | 0.914 |
| B7 | linr | 0.117 | 0.529 [0.442, 0.605] | 0.043 | 0.060 | 0.916 |
| B7 | logr | 0.117 | 0.530 [0.443, 0.606] | 0.035 | 0.061 | 0.916 |
| B7 | ighb | 0.120 | 0.516 [0.428, 0.592] | 0.050 | 0.060 | 0.912 |
| B7 | iglb | 0.117 | 0.529 [0.442, 0.603] | 0.032 | 0.067 | 0.916 |
| B7 | bmc | 0.116 | 0.531 [0.444, 0.604] | 0.037 | 0.066 | 0.916 |

## Base predictor x calibrator, GPT OSS

| start | calibrator | Brier | BSS [95%] | ECE | max gASCE (hand) | AUROC |
|---|---|---|---|---|---|---|
| avg_prob | uncalibrated | 0.269 | -0.075 [-0.165, -0.008] | 0.199 | 0.458 | 0.760 |
| avg_prob | platt | 0.201 | 0.197 [0.139, 0.251] | 0.103 | 0.178 | 0.760 |
| avg_prob | hb | 0.197 | 0.213 [0.155, 0.266] | 0.068 | 0.172 | 0.754 |
| avg_prob | linr | 0.057 | 0.773 [0.713, 0.826] | 0.045 | 0.017 | 0.949 |
| avg_prob | logr | 0.055 | 0.782 [0.723, 0.832] | 0.031 | 0.014 | 0.954 |
| avg_prob | ighb | 0.059 | 0.762 [0.703, 0.817] | 0.015 | 0.015 | 0.945 |
| avg_prob | iglb | 0.053 | 0.789 [0.730, 0.840] | 0.021 | 0.011 | 0.964 |
| avg_prob | bmc | 0.126 | 0.497 [0.430, 0.553] | 0.049 | 0.077 | 0.890 |
| B2 | uncalibrated | 0.045 | 0.820 [0.759, 0.870] | 0.022 | 0.010 | 0.968 |
| B2 | platt | 0.044 | 0.823 [0.763, 0.874] | 0.014 | 0.009 | 0.968 |
| B2 | hb | 0.046 | 0.816 [0.756, 0.868] | 0.018 | 0.014 | 0.965 |
| B2 | linr | 0.045 | 0.819 [0.758, 0.870] | 0.018 | 0.012 | 0.964 |
| B2 | logr | 0.045 | 0.819 [0.759, 0.871] | 0.014 | 0.014 | 0.964 |
| B2 | ighb | 0.046 | 0.816 [0.754, 0.868] | 0.022 | 0.012 | 0.965 |
| B2 | iglb | 0.045 | 0.821 [0.760, 0.874] | 0.016 | 0.010 | 0.968 |
| B2 | bmc | 0.045 | 0.820 [0.759, 0.871] | 0.020 | 0.013 | 0.966 |
| B4 | uncalibrated | 0.044 | 0.825 [0.765, 0.874] | 0.023 | 0.010 | 0.970 |
| B4 | platt | 0.044 | 0.825 [0.764, 0.875] | 0.022 | 0.015 | 0.970 |
| B4 | hb | 0.045 | 0.820 [0.758, 0.872] | 0.015 | 0.013 | 0.968 |
| B4 | linr | 0.044 | 0.823 [0.762, 0.874] | 0.016 | 0.011 | 0.970 |
| B4 | logr | 0.045 | 0.822 [0.762, 0.873] | 0.014 | 0.012 | 0.970 |
| B4 | ighb | 0.044 | 0.825 [0.765, 0.874] | 0.023 | 0.010 | 0.970 |
| B4 | iglb | 0.044 | 0.825 [0.765, 0.874] | 0.023 | 0.010 | 0.970 |
| B4 | bmc | 0.044 | 0.824 [0.764, 0.874] | 0.021 | 0.010 | 0.970 |
| B6 | uncalibrated | 0.050 | 0.800 [0.742, 0.849] | 0.023 | 0.012 | 0.966 |
| B6 | platt | 0.050 | 0.799 [0.743, 0.848] | 0.023 | 0.010 | 0.966 |
| B6 | hb | 0.052 | 0.790 [0.733, 0.841] | 0.015 | 0.010 | 0.959 |
| B6 | linr | 0.050 | 0.801 [0.742, 0.850] | 0.031 | 0.014 | 0.960 |
| B6 | logr | 0.050 | 0.801 [0.742, 0.852] | 0.019 | 0.011 | 0.960 |
| B6 | ighb | 0.048 | 0.807 [0.749, 0.855] | 0.025 | 0.010 | 0.965 |
| B6 | iglb | 0.050 | 0.800 [0.742, 0.849] | 0.023 | 0.012 | 0.966 |
| B6 | bmc | 0.050 | 0.800 [0.742, 0.849] | 0.023 | 0.012 | 0.966 |
| B7 | uncalibrated | 0.044 | 0.825 [0.764, 0.876] | 0.017 | 0.010 | 0.967 |
| B7 | platt | 0.044 | 0.825 [0.765, 0.875] | 0.018 | 0.011 | 0.967 |
| B7 | hb | 0.045 | 0.821 [0.762, 0.872] | 0.015 | 0.010 | 0.964 |
| B7 | linr | 0.044 | 0.823 [0.763, 0.873] | 0.016 | 0.013 | 0.964 |
| B7 | logr | 0.045 | 0.820 [0.760, 0.871] | 0.011 | 0.011 | 0.965 |
| B7 | ighb | 0.044 | 0.825 [0.764, 0.876] | 0.017 | 0.010 | 0.967 |
| B7 | iglb | 0.044 | 0.825 [0.764, 0.876] | 0.017 | 0.010 | 0.967 |
| B7 | bmc | 0.044 | 0.823 [0.762, 0.874] | 0.020 | 0.012 | 0.965 |

## Boosted multicalibration: discovered groups and worst group error

* qwen3/avg_prob: sc_mean_sim <= 0.681 (leaf -0.60); sc_mean_sim > 0.681 (leaf -0.11)
* qwen3/B2: none
* qwen3/B4: none
* qwen3/B6: none
* qwen3/B7: none
* gpt-oss/avg_prob: all_std <= 0.454 (leaf -0.58); all_std > 0.454 (leaf -0.24); sc_mean_sim <= 0.354 (leaf -0.52); sc_mean_sim > 0.354 (leaf +0.18)
* gpt-oss/B2: avg_prob <= 0.689 (leaf -0.00); avg_prob > 0.689 (leaf -0.00); loops <= 1.5 (leaf +0.02); loops > 1.5 (leaf -0.04)
* gpt-oss/B4: avg_prob <= 0.689 (leaf -0.00); avg_prob > 0.689 (leaf -0.01)
* gpt-oss/B6: none
* gpt-oss/B7: sc_others_empty <= 0.944 (leaf -0.01); sc_others_empty > 0.944 (leaf -0.00); loops <= 1.5 (leaf +0.02); loops > 1.5 (leaf -0.05)

| model | start | method | max gASCE hand | max gASCE discovered |
|---|---|---|---|---|
| qwen3 | avg_prob | uncalibrated | 0.671 | 0.409 |
| qwen3 | avg_prob | bmc | 0.073 | 0.010 |
| qwen3 | B2 | uncalibrated | 0.050 |  |
| qwen3 | B2 | bmc | 0.058 |  |
| qwen3 | B4 | uncalibrated | 0.062 |  |
| qwen3 | B4 | bmc | 0.070 |  |
| qwen3 | B6 | uncalibrated | 0.046 |  |
| qwen3 | B6 | bmc | 0.063 |  |
| qwen3 | B7 | uncalibrated | 0.048 |  |
| qwen3 | B7 | bmc | 0.066 |  |
| gpt-oss | avg_prob | uncalibrated | 0.458 | 0.366 |
| gpt-oss | avg_prob | bmc | 0.077 | 0.069 |
| gpt-oss | B2 | uncalibrated | 0.010 | 0.005 |
| gpt-oss | B2 | bmc | 0.013 | 0.006 |
| gpt-oss | B4 | uncalibrated | 0.010 | 0.005 |
| gpt-oss | B4 | bmc | 0.010 | 0.004 |
| gpt-oss | B6 | uncalibrated | 0.012 |  |
| gpt-oss | B6 | bmc | 0.012 |  |
| gpt-oss | B7 | uncalibrated | 0.010 | 0.004 |
| gpt-oss | B7 | bmc | 0.012 | 0.004 |

## Base predictor differences (uncalibrated)

| model | comparison | metric | diff [95%] |
|---|---|---|---|
| qwen3 | B4 - B2 | brier | +0.0056 [-0.0015, +0.0130] |
| qwen3 | B4 - B2 | log_loss | +0.0113 [-0.0091, +0.0330] |
| qwen3 | B6 - B4 | brier | +0.0092 [-0.0016, +0.0196] |
| qwen3 | B6 - B4 | log_loss | +0.0329 [+0.0026, +0.0630] |
| qwen3 | B7 - B4 | brier | +0.0045 [-0.0027, +0.0121] |
| qwen3 | B7 - B4 | log_loss | +0.0200 [-0.0051, +0.0506] |
| qwen3 | B7 - B6 | brier | -0.0046 [-0.0120, +0.0031] |
| qwen3 | B7 - B6 | log_loss | -0.0129 [-0.0340, +0.0117] |
| gpt-oss | B4 - B2 | brier | -0.0012 [-0.0035, +0.0010] |
| gpt-oss | B4 - B2 | log_loss | -0.0082 [-0.0210, +0.0026] |
| gpt-oss | B6 - B4 | brier | +0.0062 [+0.0025, +0.0101] |
| gpt-oss | B6 - B4 | log_loss | +0.0264 [+0.0122, +0.0418] |
| gpt-oss | B7 - B4 | brier | -0.0000 [-0.0021, +0.0019] |
| gpt-oss | B7 - B4 | log_loss | +0.0054 [-0.0036, +0.0148] |
| gpt-oss | B7 - B6 | brier | -0.0063 [-0.0096, -0.0033] |
| gpt-oss | B7 - B6 | log_loss | -0.0210 [-0.0358, -0.0062] |

## SHAP (B4, val_tune, probability scale unless noted)

### Qwen3 Coder

Top features (mean |SHAP|, seed 0): sc_mean_sim 0.119, syntax_valid 0.062, log_output_tokens 0.050, log_code_lines 0.048, log_prompt_chars 0.036, sc_identical 0.023, structure_missing 0.019, sc_others_empty 0.018

Block importance: self consistency 0.149 [0.136, 0.163]; size 0.087 [0.079, 0.094]; AST structure 0.082 [0.075, 0.088]; logprob statistics 0.051 [0.047, 0.055]

Seed stability (Spearman): 0-1 0.987, 0-2 0.989, 1-2 0.986. In the top 10 for all 3 seeds: sc_mean_sim, syntax_valid, log_output_tokens, log_code_lines, log_prompt_chars, sc_identical, sc_others_empty, loops.

Noise check: noise ranks 22 of 28; real features below it: imports, code_std, empty_code, code_low, sc_missing, code_lp_missing.

Stable interaction pairs (top 5 in all 3 seeds, log odds scale): val_tune [['log_code_lines', 'sc_mean_sim'], ['log_code_lines', 'syntax_valid'], ['log_prompt_chars', 'log_output_tokens']]; out of fold base_train [['log_code_lines', 'sc_mean_sim'], ['log_code_lines', 'syntax_valid'], ['log_code_chars', 'sc_mean_sim'], ['log_prompt_chars', 'log_output_tokens']].

SHAP groups (frozen): sc_mean_sim > 0.546 (194 problems); syntax_valid <= 0.5 (155 problems); log_output_tokens > 6.72 (209 problems); log_code_lines > 2.94 (215 problems); log_code_lines > 2.94 & sc_mean_sim > 0.546 (132 problems)

| block dropped | val_tune log loss | change | AUROC | log loss, retuned |
|---|---|---|---|---|
| (none) | 0.3544 | 0.0000 | 0.9190 |  |
| logprob statistics | 0.3643 | 0.0099 | 0.9129 | 0.3647 |
| size | 0.3303 | -0.0241 | 0.9319 | 0.3392 |
| AST structure | 0.3672 | 0.0128 | 0.9122 | 0.3763 |
| self consistency | 0.3499 | -0.0045 | 0.9204 | 0.3583 |

Worst slices of uncalibrated B4 (block SHAP profile vs overall, probability scale):

* loc_high, bin 5: predicted 0.55, passed 0.83, 109 rows. Blocks: logprob statistics +0.025, size -0.028, AST structure +0.061, self consistency +0.086. Top features: sc_mean_sim +0.068 (slice mean 0.68 vs 0.46), log_code_lines -0.058 (slice mean 3.56 vs 2.29), syntax_valid +0.049 (slice mean 1.00 vs 0.67).
* nested, bin 5: predicted 0.56, passed 0.91, 47 rows. Blocks: logprob statistics +0.026, size -0.017, AST structure +0.062, self consistency +0.080. Top features: sc_mean_sim +0.066 (slice mean 0.63 vs 0.46), log_code_lines -0.057 (slice mean 3.56 vs 2.29), syntax_valid +0.050 (slice mean 1.00 vs 0.67).
* prompt_len_high, bin 5: predicted 0.55, passed 0.85, 54 rows. Blocks: logprob statistics +0.036, size -0.060, AST structure +0.063, self consistency +0.106. Top features: sc_mean_sim +0.078 (slice mean 0.76 vs 0.46), syntax_valid +0.049 (slice mean 1.00 vs 0.67), log_prompt_chars -0.049 (slice mean 7.52 vs 7.21).
* shap: log_code_lines > 2.94 & sc_mean_sim > 0.546, bin 6: predicted 0.64, passed 0.92, 61 rows. Blocks: logprob statistics +0.039, size -0.018, AST structure +0.082, self consistency +0.125. Top features: sc_mean_sim +0.089 (slice mean 0.77 vs 0.46), syntax_valid +0.052 (slice mean 1.00 vs 0.67), log_output_tokens +0.047 (slice mean 6.35 vs 6.70).
* nested, bin 8: predicted 0.85, passed 0.53, 40 rows. Blocks: logprob statistics +0.080, size +0.070, AST structure +0.075, self consistency +0.217. Top features: sc_mean_sim +0.171 (slice mean 0.86 vs 0.46), syntax_valid +0.056 (slice mean 1.00 vs 0.67), log_output_tokens +0.054 (slice mean 5.96 vs 6.70).

### GPT OSS

Top features (mean |SHAP|, seed 0): log_code_lines 0.129, sc_mean_sim 0.123, syntax_valid 0.085, sc_others_empty 0.041, log_output_tokens 0.022, nesting 0.016, log_prompt_chars 0.014, log_code_chars 0.013

Block importance: size 0.165 [0.159, 0.171]; self consistency 0.163 [0.156, 0.169]; AST structure 0.114 [0.109, 0.118]; logprob statistics 0.019 [0.018, 0.020]

Seed stability (Spearman): 0-1 0.946, 0-2 0.901, 1-2 0.933. In the top 10 for all 3 seeds: log_code_lines, sc_mean_sim, syntax_valid, sc_others_empty, log_output_tokens, nesting, log_prompt_chars, structure_missing, log_code_chars, all_p10.

Noise check: noise ranks 22 of 28; real features below it: all_mean, all_low, sc_identical, code_low, truncated, code_lp_missing.

Stable interaction pairs (top 5 in all 3 seeds, log odds scale): val_tune [['log_code_chars', 'sc_mean_sim'], ['log_code_lines', 'sc_mean_sim'], ['nesting', 'sc_mean_sim']]; out of fold base_train [['log_code_chars', 'sc_mean_sim'], ['log_code_lines', 'sc_mean_sim'], ['log_output_tokens', 'sc_mean_sim'], ['nesting', 'sc_mean_sim'], ['sc_others_empty', 'sc_mean_sim']].

SHAP groups (frozen): log_code_lines <= 2.77 (246 problems); sc_mean_sim > 0.373 (203 problems); syntax_valid <= 0.5 (197 problems); sc_others_empty <= 0.222 (175 problems); log_code_chars > 5.85 & sc_mean_sim > 0.373 (191 problems); log_code_lines <= 2.77 & sc_mean_sim > 0.373 (65 problems)

| block dropped | val_tune log loss | change | AUROC | log loss, retuned |
|---|---|---|---|---|
| (none) | 0.0886 | 0.0000 | 0.9954 |  |
| logprob statistics | 0.0924 | 0.0038 | 0.9940 | 0.0938 |
| size | 0.0874 | -0.0013 | 0.9959 | 0.0878 |
| AST structure | 0.0899 | 0.0013 | 0.9954 | 0.0882 |
| self consistency | 0.0919 | 0.0033 | 0.9950 | 0.0948 |

Worst slices of uncalibrated B4 (block SHAP profile vs overall, probability scale):

* disc: avg_prob > 0.689, bin 8: predicted 0.86, passed 0.98, 100 rows. Blocks: logprob statistics +0.019, size +0.139, AST structure +0.121, self consistency +0.117. Top features: log_code_lines +0.127 (slice mean 3.59 vs 1.68), syntax_valid +0.086 (slice mean 1.00 vs 0.52), sc_mean_sim +0.079 (slice mean 0.44 vs 0.30).
* prompt_len_high, bin 7: predicted 0.75, passed 0.94, 31 rows. Blocks: logprob statistics +0.026, size +0.111, AST structure +0.111, self consistency +0.042. Top features: log_code_lines +0.121 (slice mean 3.57 vs 1.68), syntax_valid +0.081 (slice mean 1.00 vs 0.52), sc_mean_sim +0.044 (slice mean 0.37 vs 0.30).
* all, bin 6: predicted 0.65, passed 0.82, 33 rows. Blocks: logprob statistics -0.003, size +0.096, AST structure +0.111, self consistency -0.014. Top features: log_code_lines +0.102 (slice mean 3.73 vs 1.68), syntax_valid +0.077 (slice mean 1.00 vs 0.52), nesting +0.021 (slice mean 4.70 vs 1.96).
* nested, bin 8: predicted 0.86, passed 0.98, 52 rows. Blocks: logprob statistics +0.018, size +0.136, AST structure +0.117, self consistency +0.127. Top features: log_code_lines +0.125 (slice mean 3.57 vs 1.68), syntax_valid +0.086 (slice mean 1.00 vs 0.52), sc_mean_sim +0.080 (slice mean 0.45 vs 0.30).
* all, bin 9: predicted 0.96, passed 1.00, 453 rows. Blocks: logprob statistics +0.012, size +0.182, AST structure +0.110, self consistency +0.199. Top features: sc_mean_sim +0.151 (slice mean 0.70 vs 0.30), log_code_lines +0.130 (slice mean 3.06 vs 1.68), syntax_valid +0.085 (slice mean 1.00 vs 0.52).

### Three group sets, IGLB on B4 (test)

| model | group set | groups | Brier | max gASCE union | slices | slices > 2 SE | IGLB patches |
|---|---|---|---|---|---|---|---|
| qwen3 | none (uncalibrated B4) | 0 | 0.1147 | 0.0288 | 100.0000 | 17.0000 | 0.0000 |
| qwen3 | hand | 8 | 0.1154 | 0.0411 | 99.0000 | 22.0000 | 1.0000 |
| qwen3 | discovered | 0 |  |  |  |  |  |
| qwen3 | shap | 5 | 0.1151 | 0.0457 | 99.0000 | 22.0000 | 1.0000 |
| gpt-oss | none (uncalibrated B4) | 0 | 0.0438 | 0.0072 | 76.0000 | 12.0000 | 0.0000 |
| gpt-oss | hand | 7 | 0.0438 | 0.0072 | 76.0000 | 12.0000 | 0.0000 |
| gpt-oss | discovered | 2 | 0.0437 | 0.0081 | 69.0000 | 9.0000 | 1.0000 |
| gpt-oss | shap | 6 | 0.0438 | 0.0072 | 76.0000 | 12.0000 | 0.0000 |

## Murphy decomposition

| model | start | method | Brier | reliability | resolution | uncertainty | gap | change in reliability vs Platt | change in resolution vs Platt |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | avg_prob | uncalibrated | 0.3831 | 0.2340 | 0.0980 | 0.2482 | -0.0011 | 0.2284 [0.1876, 0.2664] | -0.0041 [-0.0091, -0.0024] |
| qwen3 | avg_prob | platt | 0.1514 | 0.0057 | 0.1021 | 0.2482 | -0.0003 |  |  |
| qwen3 | avg_prob | hb | 0.1535 | 0.0038 | 0.0986 | 0.2482 | 0.0000 | -0.0019 [-0.0052, -0.0012] | -0.0035 [-0.0079, -0.0018] |
| qwen3 | avg_prob | linr | 0.1361 | 0.0046 | 0.1168 | 0.2482 | 0.0000 | -0.0011 [-0.0055, 0.0052] | 0.0147 [0.0050, 0.0268] |
| qwen3 | avg_prob | logr | 0.1372 | 0.0057 | 0.1164 | 0.2482 | -0.0004 | 0.0001 [-0.0045, 0.0077] | 0.0143 [0.0046, 0.0272] |
| qwen3 | avg_prob | ighb | 0.1486 | 0.0080 | 0.1072 | 0.2482 | -0.0004 | 0.0023 [-0.0028, 0.0105] | 0.0050 [-0.0009, 0.0130] |
| qwen3 | avg_prob | iglb | 0.1466 | 0.0037 | 0.1048 | 0.2482 | -0.0004 | -0.0020 [-0.0051, 0.0015] | 0.0027 [-0.0038, 0.0104] |
| qwen3 | avg_prob | bmc | 0.1354 | 0.0045 | 0.1171 | 0.2482 | -0.0002 | -0.0012 [-0.0087, 0.0045] | 0.0150 [0.0024, 0.0276] |
| qwen3 | B2 | uncalibrated | 0.1091 | 0.0029 | 0.1418 | 0.2482 | -0.0002 | 0.0000 [-0.0026, 0.0023] | 0.0004 [-0.0011, 0.0020] |
| qwen3 | B2 | platt | 0.1094 | 0.0029 | 0.1414 | 0.2482 | -0.0002 |  |  |
| qwen3 | B2 | hb | 0.1141 | 0.0062 | 0.1402 | 0.2482 | -0.0001 | 0.0033 [-0.0008, 0.0061] | -0.0012 [-0.0044, 0.0004] |
| qwen3 | B2 | linr | 0.1093 | 0.0040 | 0.1428 | 0.2482 | -0.0002 | 0.0011 [-0.0029, 0.0044] | 0.0013 [-0.0053, 0.0075] |
| qwen3 | B2 | logr | 0.1102 | 0.0049 | 0.1426 | 0.2482 | -0.0003 | 0.0020 [-0.0027, 0.0064] | 0.0012 [-0.0053, 0.0079] |
| qwen3 | B2 | ighb | 0.1175 | 0.0042 | 0.1347 | 0.2482 | -0.0002 | 0.0014 [-0.0018, 0.0049] | -0.0068 [-0.0141, -0.0005] |
| qwen3 | B2 | iglb | 0.1091 | 0.0029 | 0.1418 | 0.2482 | -0.0002 | 0.0000 [-0.0026, 0.0023] | 0.0004 [-0.0011, 0.0020] |
| qwen3 | B2 | bmc | 0.1095 | 0.0041 | 0.1427 | 0.2482 | -0.0001 | 0.0012 [-0.0016, 0.0041] | 0.0013 [-0.0010, 0.0048] |
| qwen3 | B4 | uncalibrated | 0.1147 | 0.0028 | 0.1359 | 0.2482 | -0.0004 | -0.0009 [-0.0067, 0.0060] | 0.0022 [-0.0001, 0.0060] |
| qwen3 | B4 | platt | 0.1174 | 0.0037 | 0.1337 | 0.2482 | -0.0008 |  |  |
| qwen3 | B4 | hb | 0.1178 | 0.0028 | 0.1330 | 0.2482 | -0.0003 | -0.0009 [-0.0053, 0.0027] | -0.0007 [-0.0054, 0.0024] |
| qwen3 | B4 | linr | 0.1113 | 0.0029 | 0.1395 | 0.2482 | -0.0004 | -0.0008 [-0.0054, 0.0033] | 0.0058 [-0.0008, 0.0123] |
| qwen3 | B4 | logr | 0.1121 | 0.0040 | 0.1398 | 0.2482 | -0.0004 | 0.0003 [-0.0043, 0.0051] | 0.0061 [-0.0005, 0.0130] |
| qwen3 | B4 | ighb | 0.1207 | 0.0056 | 0.1328 | 0.2482 | -0.0003 | 0.0019 [-0.0037, 0.0087] | -0.0009 [-0.0056, 0.0046] |
| qwen3 | B4 | iglb | 0.1154 | 0.0029 | 0.1353 | 0.2482 | -0.0005 | -0.0008 [-0.0054, 0.0051] | 0.0016 [-0.0027, 0.0071] |
| qwen3 | B4 | bmc | 0.1157 | 0.0025 | 0.1347 | 0.2482 | -0.0003 | -0.0012 [-0.0054, 0.0044] | 0.0010 [-0.0015, 0.0051] |
| qwen3 | B6 | uncalibrated | 0.1239 | 0.0032 | 0.1273 | 0.2482 | -0.0003 | 0.0009 [-0.0027, 0.0047] | -0.0008 [-0.0023, 0.0007] |
| qwen3 | B6 | platt | 0.1223 | 0.0023 | 0.1281 | 0.2482 | -0.0001 |  |  |
| qwen3 | B6 | hb | 0.1247 | 0.0028 | 0.1257 | 0.2482 | -0.0006 | 0.0005 [-0.0019, 0.0016] | -0.0024 [-0.0053, -0.0012] |
| qwen3 | B6 | linr | 0.1187 | 0.0018 | 0.1311 | 0.2482 | -0.0002 | -0.0005 [-0.0034, 0.0023] | 0.0030 [-0.0017, 0.0085] |
| qwen3 | B6 | logr | 0.1172 | 0.0017 | 0.1328 | 0.2482 | 0.0001 | -0.0006 [-0.0034, 0.0026] | 0.0047 [-0.0001, 0.0103] |
| qwen3 | B6 | ighb | 0.1234 | 0.0030 | 0.1276 | 0.2482 | -0.0003 | 0.0007 [-0.0030, 0.0045] | -0.0005 [-0.0024, 0.0014] |
| qwen3 | B6 | iglb | 0.1222 | 0.0019 | 0.1278 | 0.2482 | -0.0001 | -0.0004 [-0.0024, 0.0012] | -0.0003 [-0.0020, 0.0011] |
| qwen3 | B6 | bmc | 0.1238 | 0.0019 | 0.1258 | 0.2482 | -0.0006 | -0.0004 [-0.0025, 0.0016] | -0.0023 [-0.0048, -0.0001] |
| qwen3 | B7 | uncalibrated | 0.1193 | 0.0053 | 0.1341 | 0.2482 | -0.0002 | 0.0024 [-0.0019, 0.0066] | 0.0003 [-0.0011, 0.0016] |
| qwen3 | B7 | platt | 0.1172 | 0.0030 | 0.1338 | 0.2482 | -0.0002 |  |  |
| qwen3 | B7 | hb | 0.1177 | 0.0026 | 0.1330 | 0.2482 | -0.0001 | -0.0004 [-0.0049, 0.0027] | -0.0009 [-0.0034, 0.0000] |
| qwen3 | B7 | linr | 0.1170 | 0.0029 | 0.1340 | 0.2482 | -0.0001 | -0.0001 [-0.0030, 0.0026] | 0.0002 [-0.0049, 0.0051] |
| qwen3 | B7 | logr | 0.1167 | 0.0023 | 0.1334 | 0.2482 | -0.0005 | -0.0007 [-0.0040, 0.0026] | -0.0005 [-0.0057, 0.0045] |
| qwen3 | B7 | ighb | 0.1201 | 0.0051 | 0.1331 | 0.2482 | -0.0001 | 0.0021 [-0.0009, 0.0048] | -0.0008 [-0.0046, 0.0020] |
| qwen3 | B7 | iglb | 0.1168 | 0.0017 | 0.1329 | 0.2482 | -0.0002 | -0.0013 [-0.0043, 0.0010] | -0.0009 [-0.0031, 0.0008] |
| qwen3 | B7 | bmc | 0.1165 | 0.0023 | 0.1337 | 0.2482 | -0.0004 | -0.0007 [-0.0043, 0.0018] | -0.0002 [-0.0024, 0.0010] |
| gpt-oss | avg_prob | uncalibrated | 0.2685 | 0.0810 | 0.0608 | 0.2499 | -0.0016 | 0.0632 [0.0373, 0.0866] | -0.0060 [-0.0096, -0.0041] |
| gpt-oss | avg_prob | platt | 0.2006 | 0.0178 | 0.0668 | 0.2499 | -0.0003 |  |  |
| gpt-oss | avg_prob | hb | 0.1967 | 0.0081 | 0.0613 | 0.2499 | 0.0000 | -0.0097 [-0.0151, -0.0061] | -0.0055 [-0.0089, -0.0037] |
| gpt-oss | avg_prob | linr | 0.0568 | 0.0066 | 0.1996 | 0.2499 | -0.0002 | -0.0112 [-0.0214, -0.0042] | 0.1328 [0.1152, 0.1471] |
| gpt-oss | avg_prob | logr | 0.0545 | 0.0048 | 0.2002 | 0.2499 | 0.0001 | -0.0130 [-0.0222, -0.0064] | 0.1335 [0.1160, 0.1488] |
| gpt-oss | avg_prob | ighb | 0.0594 | 0.0014 | 0.1919 | 0.2499 | 0.0001 | -0.0164 [-0.0258, -0.0111] | 0.1252 [0.1071, 0.1405] |
| gpt-oss | avg_prob | iglb | 0.0528 | 0.0018 | 0.1989 | 0.2499 | 0.0001 | -0.0160 [-0.0258, -0.0090] | 0.1322 [0.1150, 0.1471] |
| gpt-oss | avg_prob | bmc | 0.1258 | 0.0041 | 0.1280 | 0.2499 | -0.0002 | -0.0137 [-0.0220, -0.0078] | 0.0613 [0.0465, 0.0749] |
| gpt-oss | B2 | uncalibrated | 0.0450 | 0.0023 | 0.2070 | 0.2499 | -0.0001 | 0.0015 [0.0002, 0.0033] | 0.0006 [-0.0001, 0.0019] |
| gpt-oss | B2 | platt | 0.0442 | 0.0008 | 0.2064 | 0.2499 | -0.0001 |  |  |
| gpt-oss | B2 | hb | 0.0460 | 0.0011 | 0.2048 | 0.2499 | -0.0001 | 0.0003 [-0.0014, 0.0007] | -0.0016 [-0.0042, -0.0006] |
| gpt-oss | B2 | linr | 0.0451 | 0.0018 | 0.2064 | 0.2499 | -0.0002 | 0.0011 [-0.0003, 0.0021] | 0.0000 [-0.0019, 0.0013] |
| gpt-oss | B2 | logr | 0.0451 | 0.0015 | 0.2062 | 0.2499 | -0.0001 | 0.0007 [-0.0008, 0.0025] | -0.0002 [-0.0024, 0.0018] |
| gpt-oss | B2 | ighb | 0.0461 | 0.0020 | 0.2056 | 0.2499 | -0.0002 | 0.0012 [-0.0004, 0.0028] | -0.0007 [-0.0033, 0.0007] |
| gpt-oss | B2 | iglb | 0.0446 | 0.0010 | 0.2062 | 0.2499 | -0.0000 | 0.0002 [-0.0018, 0.0010] | -0.0002 [-0.0022, 0.0003] |
| gpt-oss | B2 | bmc | 0.0449 | 0.0019 | 0.2070 | 0.2499 | 0.0001 | 0.0011 [-0.0000, 0.0028] | 0.0006 [-0.0004, 0.0020] |
| gpt-oss | B4 | uncalibrated | 0.0438 | 0.0015 | 0.2074 | 0.2499 | -0.0002 | -0.0003 [-0.0019, 0.0010] | -0.0004 [-0.0019, 0.0009] |
| gpt-oss | B4 | platt | 0.0438 | 0.0019 | 0.2078 | 0.2499 | -0.0001 |  |  |
| gpt-oss | B4 | hb | 0.0450 | 0.0016 | 0.2068 | 0.2499 | 0.0002 | -0.0002 [-0.0020, 0.0007] | -0.0011 [-0.0034, -0.0000] |
| gpt-oss | B4 | linr | 0.0443 | 0.0009 | 0.2065 | 0.2499 | -0.0000 | -0.0009 [-0.0026, 0.0003] | -0.0013 [-0.0037, 0.0004] |
| gpt-oss | B4 | logr | 0.0446 | 0.0014 | 0.2067 | 0.2499 | -0.0001 | -0.0004 [-0.0022, 0.0012] | -0.0011 [-0.0033, 0.0005] |
| gpt-oss | B4 | ighb | 0.0438 | 0.0015 | 0.2074 | 0.2499 | -0.0002 | -0.0003 [-0.0019, 0.0010] | -0.0004 [-0.0019, 0.0009] |
| gpt-oss | B4 | iglb | 0.0438 | 0.0015 | 0.2074 | 0.2499 | -0.0002 | -0.0003 [-0.0019, 0.0010] | -0.0004 [-0.0019, 0.0009] |
| gpt-oss | B4 | bmc | 0.0439 | 0.0014 | 0.2074 | 0.2499 | -0.0000 | -0.0004 [-0.0020, 0.0004] | -0.0004 [-0.0023, 0.0006] |
| gpt-oss | B6 | uncalibrated | 0.0501 | 0.0021 | 0.2018 | 0.2499 | -0.0001 | 0.0000 [-0.0008, 0.0013] | 0.0000 [-0.0009, 0.0012] |
| gpt-oss | B6 | platt | 0.0501 | 0.0020 | 0.2018 | 0.2499 | -0.0000 |  |  |
| gpt-oss | B6 | hb | 0.0524 | 0.0009 | 0.1987 | 0.2499 | 0.0002 | -0.0011 [-0.0033, 0.0001] | -0.0031 [-0.0063, -0.0013] |
| gpt-oss | B6 | linr | 0.0498 | 0.0032 | 0.2032 | 0.2499 | -0.0001 | 0.0012 [-0.0003, 0.0032] | 0.0014 [-0.0010, 0.0043] |
| gpt-oss | B6 | logr | 0.0496 | 0.0025 | 0.2027 | 0.2499 | -0.0001 | 0.0005 [-0.0015, 0.0024] | 0.0009 [-0.0021, 0.0041] |
| gpt-oss | B6 | ighb | 0.0483 | 0.0021 | 0.2038 | 0.2499 | 0.0000 | 0.0001 [-0.0008, 0.0013] | 0.0020 [-0.0003, 0.0049] |
| gpt-oss | B6 | iglb | 0.0501 | 0.0021 | 0.2018 | 0.2499 | -0.0001 | 0.0000 [-0.0008, 0.0013] | 0.0000 [-0.0009, 0.0012] |
| gpt-oss | B6 | bmc | 0.0501 | 0.0021 | 0.2018 | 0.2499 | -0.0001 | 0.0000 [-0.0008, 0.0013] | 0.0000 [-0.0009, 0.0012] |
| gpt-oss | B7 | uncalibrated | 0.0438 | 0.0010 | 0.2070 | 0.2499 | -0.0001 | -0.0002 [-0.0010, 0.0008] | -0.0002 [-0.0010, 0.0008] |
| gpt-oss | B7 | platt | 0.0438 | 0.0012 | 0.2072 | 0.2499 | -0.0001 |  |  |
| gpt-oss | B7 | hb | 0.0446 | 0.0012 | 0.2064 | 0.2499 | -0.0000 | -0.0000 [-0.0017, 0.0012] | -0.0008 [-0.0028, 0.0005] |
| gpt-oss | B7 | linr | 0.0443 | 0.0011 | 0.2067 | 0.2499 | 0.0001 | -0.0002 [-0.0015, 0.0013] | -0.0005 [-0.0018, 0.0009] |
| gpt-oss | B7 | logr | 0.0449 | 0.0016 | 0.2065 | 0.2499 | -0.0000 | 0.0004 [-0.0019, 0.0025] | -0.0007 [-0.0027, 0.0009] |
| gpt-oss | B7 | ighb | 0.0438 | 0.0010 | 0.2070 | 0.2499 | -0.0001 | -0.0002 [-0.0010, 0.0008] | -0.0002 [-0.0010, 0.0008] |
| gpt-oss | B7 | iglb | 0.0438 | 0.0010 | 0.2070 | 0.2499 | -0.0001 | -0.0002 [-0.0010, 0.0008] | -0.0002 [-0.0010, 0.0008] |
| gpt-oss | B7 | bmc | 0.0442 | 0.0015 | 0.2074 | 0.2499 | 0.0001 | 0.0003 [-0.0007, 0.0011] | 0.0001 [-0.0009, 0.0008] |

## Holm family (28 comparisons, Brier differences)

| model | comparison | Brier diff | lo | hi | p | Holm p | significant |
|---|---|---|---|---|---|---|---|
| qwen3 | avg_prob: linr - platt | -0.0154 | -0.0253 | -0.0061 | 0.0000 | 0.0000 | True |
| qwen3 | avg_prob: iglb - platt | -0.0048 | -0.0113 | 0.0024 | 0.1680 | 1.0000 | False |
| qwen3 | avg_prob: bmc - platt | -0.0160 | -0.0273 | -0.0044 | 0.0100 | 0.2400 | False |
| qwen3 | B2: linr - platt | -0.0001 | -0.0049 | 0.0044 | 0.9670 | 1.0000 | False |
| qwen3 | B2: iglb - platt | -0.0003 | -0.0029 | 0.0021 | 0.8090 | 1.0000 | False |
| qwen3 | B2: bmc - platt | 0.0001 | -0.0022 | 0.0020 | 0.8980 | 1.0000 | False |
| qwen3 | B4: linr - platt | -0.0062 | -0.0134 | 0.0009 | 0.0890 | 1.0000 | False |
| qwen3 | B4: iglb - platt | -0.0021 | -0.0064 | 0.0025 | 0.3580 | 1.0000 | False |
| qwen3 | B4: bmc - platt | -0.0017 | -0.0055 | 0.0021 | 0.3660 | 1.0000 | False |
| qwen3 | B6: linr - platt | -0.0036 | -0.0084 | 0.0012 | 0.1320 | 1.0000 | False |
| qwen3 | B6: iglb - platt | -0.0001 | -0.0012 | 0.0009 | 0.8020 | 1.0000 | False |
| qwen3 | B6: bmc - platt | 0.0015 | -0.0002 | 0.0032 | 0.0820 | 1.0000 | False |
| qwen3 | B4 - B2 (uncalibrated) | 0.0056 | -0.0015 | 0.0130 | 0.1490 | 1.0000 | False |
| qwen3 | B6 - B4 (uncalibrated) | 0.0092 | -0.0016 | 0.0196 | 0.0950 | 1.0000 | False |
| gpt-oss | avg_prob: linr - platt | -0.1439 | -0.1618 | -0.1251 | 0.0000 | 0.0000 | True |
| gpt-oss | avg_prob: iglb - platt | -0.1478 | -0.1654 | -0.1292 | 0.0000 | 0.0000 | True |
| gpt-oss | avg_prob: bmc - platt | -0.0749 | -0.0881 | -0.0612 | 0.0000 | 0.0000 | True |
| gpt-oss | B2: linr - platt | 0.0010 | -0.0005 | 0.0024 | 0.2070 | 1.0000 | False |
| gpt-oss | B2: iglb - platt | 0.0005 | -0.0008 | 0.0018 | 0.4890 | 1.0000 | False |
| gpt-oss | B2: bmc - platt | 0.0007 | -0.0001 | 0.0019 | 0.0720 | 1.0000 | False |
| gpt-oss | B4: linr - platt | 0.0004 | -0.0009 | 0.0018 | 0.5760 | 1.0000 | False |
| gpt-oss | B4: iglb - platt | 0.0000 | -0.0004 | 0.0004 | 0.8860 | 1.0000 | False |
| gpt-oss | B4: bmc - platt | 0.0001 | -0.0006 | 0.0007 | 0.8970 | 1.0000 | False |
| gpt-oss | B7: linr - platt | 0.0005 | -0.0009 | 0.0020 | 0.4970 | 1.0000 | False |
| gpt-oss | B7: iglb - platt | 0.0000 | -0.0001 | 0.0002 | 0.7150 | 1.0000 | False |
| gpt-oss | B7: bmc - platt | 0.0004 | -0.0002 | 0.0011 | 0.2560 | 1.0000 | False |
| gpt-oss | B4 - B2 (uncalibrated) | -0.0012 | -0.0035 | 0.0010 | 0.2740 | 1.0000 | False |
| gpt-oss | B7 - B4 (uncalibrated) | -0.0000 | -0.0021 | 0.0019 | 0.9420 | 1.0000 | False |

## Conformal risk control (Qwen3 Coder primary, GPT OSS fallback)

| calibrator | target | threshold | realized risk | escalation | system pass | oracle at same budget | regret | regret, rank based raw P4 |
|---|---|---|---|---|---|---|---|---|
| uncalibrated | 0.050 | 0.602 | 0.063 | 0.577 | 0.549 | 0.614 | 0.066 | 0.066 |
| uncalibrated | 0.100 | 0.407 | 0.141 | 0.438 | 0.542 | 0.614 | 0.073 | 0.073 |
| uncalibrated | 0.150 | 0.259 | 0.195 | 0.357 | 0.536 | 0.614 | 0.078 | 0.078 |
| platt | 0.050 | 0.645 | 0.063 | 0.577 | 0.549 | 0.614 | 0.066 | 0.066 |
| platt | 0.100 | 0.415 | 0.141 | 0.438 | 0.542 | 0.614 | 0.073 | 0.073 |
| platt | 0.150 | 0.240 | 0.195 | 0.357 | 0.536 | 0.614 | 0.078 | 0.078 |
| iglb | 0.050 | 0.602 | 0.063 | 0.577 | 0.549 | 0.614 | 0.066 | 0.066 |
| iglb | 0.100 | 0.407 | 0.141 | 0.438 | 0.542 | 0.614 | 0.073 | 0.073 |
| iglb | 0.150 | 0.259 | 0.195 | 0.357 | 0.536 | 0.614 | 0.078 | 0.078 |
| bmc | 0.050 | 0.636 | 0.064 | 0.574 | 0.549 | 0.614 | 0.065 | 0.065 |
| bmc | 0.100 | 0.458 | 0.141 | 0.438 | 0.542 | 0.614 | 0.073 | 0.073 |
| bmc | 0.150 | 0.227 | 0.195 | 0.357 | 0.536 | 0.614 | 0.078 | 0.078 |

Simulation check (500 fresh calibration draws, known pass probabilities):

| target | mean realized risk | share of draws above target |
|---|---|---|
| 0.050 | 0.043 | 0.202 |
| 0.100 | 0.094 | 0.290 |
| 0.150 | 0.145 | 0.368 |

Routing explanation (platt scores, target 0.10, threshold 0.415; 696 of 1320 val_conformal answers escalated). Mean B4 block SHAP, escalated vs accepted (explains B4 before calibration):

| block | escalated_mean | accepted_mean | difference |
|---|---|---|---|
| self consistency | -0.1616 | 0.1427 | -0.3043 |
| AST structure | -0.0792 | 0.0650 | -0.1442 |
| size | -0.0722 | 0.0462 | -0.1185 |
| logprob statistics | -0.0547 | 0.0288 | -0.0835 |

Top features for escalated answers: sc_mean_sim -0.130, syntax_valid -0.054, log_output_tokens -0.047, log_code_lines -0.007, log_prompt_chars -0.013

## Post-run reporting fixes (no test number changed)

**Exact routing explanation for B2**, the logistic model that makes the cascade's decisions: contribution of feature j = coefficient x (standardized value - base_train mean), log odds scale, val_conformal, Platt at target 0.10. This replaces the B4 SHAP version above as the main routing explanation.

| block | escalated_mean_logodds | accepted_mean_logodds | difference |
|---|---|---|---|
| self consistency | -0.891 | 0.893 | -1.785 |
| logprob statistics | -0.771 | 0.532 | -1.304 |
| AST structure | -0.676 | 0.613 | -1.289 |
| size | -0.510 | 0.425 | -0.935 |

Top features by escalated minus accepted: sc_mean_sim -0.619, sc_identical -0.551, structure_missing -0.519, syntax_valid -0.519, truncated -0.369, sc_missing -0.339, empty_code -0.328, code_lp_missing -0.307

**Worst slices with at least 20 problems** (replaces the 30 row floor; uncalibrated B4, val_tune, probability scale SHAP profiles). Slices not significant at 2 standard errors are tentative.

* qwen3, loc_high, bin 5: 30 problems, predicted 0.55, passed 0.83, gap +0.273 (SE 0.053). Blocks: logprob statistics +0.025, size -0.028, AST structure +0.061, self consistency +0.086.
* qwen3, shap: log_code_lines > 2.94 & sc_mean_sim > 0.546, bin 6: 22 problems, predicted 0.64, passed 0.92, gap +0.281 (SE 0.056). Blocks: logprob statistics +0.039, size -0.018, AST structure +0.082, self consistency +0.125.
* qwen3, shap: log_code_lines > 2.94 & sc_mean_sim > 0.546, bin 4: 23 problems, predicted 0.46, passed 0.71, gap +0.249 (SE 0.133) (tentative: within 2 SE). Blocks: logprob statistics +0.011, size -0.067, AST structure +0.045, self consistency +0.063.
* qwen3, shap: log_output_tokens > 6.72, bin 2: 32 problems, predicted 0.25, passed 0.43, gap +0.179 (SE 0.089). Blocks: logprob statistics -0.042, size -0.115, AST structure +0.023, self consistency -0.026.
* qwen3, prompt_len_high, bin 3: 21 problems, predicted 0.35, passed 0.56, gap +0.207 (SE 0.117) (tentative: within 2 SE). Blocks: logprob statistics -0.015, size -0.092, AST structure +0.028, self consistency +0.025.
* gpt-oss, disc: [['avg_prob', '>', 0.6894271671772003]], bin 8: 37 problems, predicted 0.86, passed 0.98, gap +0.121 (SE 0.019). Blocks: logprob statistics +0.019, size +0.139, AST structure +0.121, self consistency +0.117.
* gpt-oss, all, bin 9: 59 problems, predicted 0.96, passed 1.00, gap +0.031 (SE 0.003). Blocks: logprob statistics +0.012, size +0.182, AST structure +0.110, self consistency +0.199.
* gpt-oss, shap: log_code_lines <= 2.77, bin 9: 20 problems, predicted 0.98, passed 1.00, gap +0.021 (SE 0.004). Blocks: logprob statistics +0.005, size +0.198, AST structure +0.104, self consistency +0.209.
* gpt-oss, disc: [['avg_prob', '>', 0.6894271671772003]], bin 7: 22 problems, predicted 0.75, passed 0.72, gap -0.030 (SE 0.118) (tentative: within 2 SE). Blocks: logprob statistics +0.020, size +0.123, AST structure +0.114, self consistency +0.033.
* gpt-oss, disc: [['avg_prob', '<=', 0.6894271671772003]], bin 9: 21 problems, predicted 0.97, passed 0.99, gap +0.020 (SE 0.015) (tentative: within 2 SE). Blocks: logprob statistics -0.002, size +0.187, AST structure +0.106, self consistency +0.212.

Interaction values (the stable pairs above) come from path dependent TreeSHAP on the log odds scale, the only mode shap supports for interactions, so they don't follow safeguard 2 like the main effects do.

Conformal diagnosis: `conformal_diagnosis.md`.

## Distribution shift (optional V7)

| base | fit on | evaluated on | calibrator | BSS | ECE | risk at target 0.10 | coverage at 0.10 |
|---|---|---|---|---|---|---|---|
| B2 | qwen3 | qwen3 | uncalibrated | 0.560 | 0.041 | 0.141 | 0.562 |
| B2 | qwen3 | qwen3 | platt | 0.559 | 0.037 | 0.141 | 0.562 |
| B2 | qwen3 | qwen3 | iglb | 0.560 | 0.041 | 0.141 | 0.562 |
| B2 | qwen3 | qwen3 | bmc | 0.559 | 0.039 | 0.141 | 0.562 |
| B2 | gpt-oss | gpt-oss | uncalibrated | 0.820 | 0.022 | 0.118 | 0.628 |
| B2 | gpt-oss | gpt-oss | platt | 0.823 | 0.014 | 0.118 | 0.628 |
| B2 | gpt-oss | gpt-oss | iglb | 0.821 | 0.016 | 0.118 | 0.628 |
| B2 | gpt-oss | gpt-oss | bmc | 0.820 | 0.020 | 0.126 | 0.637 |
| B2 | qwen3 | gpt-oss | uncalibrated | -0.124 | 0.357 | 0.005 | 0.094 |
| B2 | qwen3 | gpt-oss | platt | -0.165 | 0.364 | 0.005 | 0.094 |
| B2 | qwen3 | gpt-oss | iglb | -0.124 | 0.357 | 0.005 | 0.094 |
| B2 | qwen3 | gpt-oss | bmc | -0.133 | 0.359 | 0.005 | 0.094 |
| B2 | gpt-oss | qwen3 | uncalibrated | 0.059 | 0.243 | 0.402 | 0.860 |
| B2 | gpt-oss | qwen3 | platt | 0.051 | 0.248 | 0.402 | 0.860 |
| B2 | gpt-oss | qwen3 | iglb | 0.039 | 0.249 | 0.402 | 0.860 |
| B2 | gpt-oss | qwen3 | bmc | 0.074 | 0.237 | 0.404 | 0.862 |
| B4 | qwen3 | qwen3 | uncalibrated | 0.538 | 0.040 | 0.123 | 0.532 |
| B4 | qwen3 | qwen3 | platt | 0.535 | 0.036 | 0.123 | 0.532 |
| B4 | qwen3 | qwen3 | iglb | 0.535 | 0.042 | 0.119 | 0.527 |
| B4 | qwen3 | qwen3 | bmc | 0.534 | 0.030 | 0.123 | 0.532 |
| B4 | gpt-oss | gpt-oss | uncalibrated | 0.825 | 0.023 | 0.106 | 0.616 |
| B4 | gpt-oss | gpt-oss | platt | 0.825 | 0.022 | 0.106 | 0.616 |
| B4 | gpt-oss | gpt-oss | iglb | 0.825 | 0.023 | 0.106 | 0.616 |
| B4 | gpt-oss | gpt-oss | bmc | 0.824 | 0.021 | 0.108 | 0.619 |
| B4 | qwen3 | gpt-oss | uncalibrated | 0.153 | 0.285 | 0.014 | 0.243 |
| B4 | qwen3 | gpt-oss | platt | 0.179 | 0.273 | 0.014 | 0.243 |
| B4 | qwen3 | gpt-oss | iglb | 0.167 | 0.280 | 0.013 | 0.225 |
| B4 | qwen3 | gpt-oss | bmc | 0.190 | 0.274 | 0.014 | 0.242 |
| B4 | gpt-oss | qwen3 | uncalibrated | 0.163 | 0.219 | 0.387 | 0.845 |
| B4 | gpt-oss | qwen3 | platt | 0.142 | 0.225 | 0.387 | 0.845 |
| B4 | gpt-oss | qwen3 | iglb | 0.163 | 0.219 | 0.387 | 0.845 |
| B4 | gpt-oss | qwen3 | bmc | 0.136 | 0.225 | 0.387 | 0.845 |
