# v2 summary (val_tune)

Run folder `runs/v2/dev_val_tune`. Calibrators fit on calib; IGLB and boosted multicalibration early stop on val_tune; conformal thresholds chosen on val_conformal. SHAP diagnostics are on val_tune by design (safeguard 4) and explain B4 before calibration (safeguard 8). Intervals: 95% task clustered bootstrap.

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
| avg_prob | uncalibrated | 0.376 | -0.511 [-0.813, -0.303] | 0.417 | 0.665 | 0.823 |
| avg_prob | platt | 0.165 | 0.335 [0.220, 0.440] | 0.053 | 0.054 | 0.823 |
| avg_prob | hb | 0.165 | 0.338 [0.228, 0.439] | 0.049 | 0.048 | 0.824 |
| avg_prob | linr | 0.159 | 0.361 [0.249, 0.462] | 0.094 | 0.064 | 0.835 |
| avg_prob | logr | 0.159 | 0.360 [0.244, 0.466] | 0.079 | 0.068 | 0.833 |
| avg_prob | ighb | 0.168 | 0.326 [0.198, 0.438] | 0.081 | 0.052 | 0.825 |
| avg_prob | iglb | 0.168 | 0.324 [0.198, 0.435] | 0.090 | 0.066 | 0.816 |
| avg_prob | bmc | 0.143 | 0.424 [0.318, 0.511] | 0.060 | 0.071 | 0.868 |
| B2 | uncalibrated | 0.113 | 0.545 [0.436, 0.643] | 0.063 | 0.051 | 0.930 |
| B2 | platt | 0.112 | 0.549 [0.433, 0.656] | 0.053 | 0.055 | 0.930 |
| B2 | hb | 0.116 | 0.535 [0.410, 0.645] | 0.067 | 0.053 | 0.923 |
| B2 | linr | 0.114 | 0.541 [0.426, 0.643] | 0.068 | 0.085 | 0.918 |
| B2 | logr | 0.116 | 0.535 [0.415, 0.642] | 0.073 | 0.081 | 0.926 |
| B2 | ighb | 0.114 | 0.543 [0.437, 0.640] | 0.067 | 0.069 | 0.931 |
| B2 | iglb | 0.113 | 0.545 [0.436, 0.643] | 0.063 | 0.051 | 0.930 |
| B2 | bmc | 0.112 | 0.549 [0.434, 0.650] | 0.056 | 0.052 | 0.930 |
| B4 | uncalibrated | 0.121 | 0.514 [0.405, 0.611] | 0.089 | 0.072 | 0.919 |
| B4 | platt | 0.115 | 0.537 [0.417, 0.641] | 0.060 | 0.060 | 0.919 |
| B4 | hb | 0.119 | 0.522 [0.408, 0.625] | 0.068 | 0.060 | 0.912 |
| B4 | linr | 0.119 | 0.523 [0.410, 0.621] | 0.059 | 0.076 | 0.917 |
| B4 | logr | 0.121 | 0.514 [0.389, 0.622] | 0.075 | 0.092 | 0.920 |
| B4 | ighb | 0.127 | 0.491 [0.384, 0.589] | 0.083 | 0.094 | 0.907 |
| B4 | iglb | 0.114 | 0.542 [0.429, 0.638] | 0.074 | 0.062 | 0.919 |
| B4 | bmc | 0.117 | 0.529 [0.419, 0.626] | 0.075 | 0.061 | 0.918 |
| B6 | uncalibrated | 0.122 | 0.510 [0.410, 0.596] | 0.082 | 0.050 | 0.921 |
| B6 | platt | 0.117 | 0.530 [0.422, 0.619] | 0.059 | 0.039 | 0.921 |
| B6 | hb | 0.118 | 0.524 [0.418, 0.616] | 0.049 | 0.035 | 0.918 |
| B6 | linr | 0.121 | 0.515 [0.414, 0.607] | 0.065 | 0.057 | 0.918 |
| B6 | logr | 0.123 | 0.508 [0.396, 0.610] | 0.077 | 0.059 | 0.917 |
| B6 | ighb | 0.122 | 0.511 [0.414, 0.598] | 0.081 | 0.050 | 0.921 |
| B6 | iglb | 0.116 | 0.536 [0.431, 0.623] | 0.053 | 0.036 | 0.921 |
| B6 | bmc | 0.117 | 0.530 [0.425, 0.620] | 0.055 | 0.042 | 0.921 |
| B7 | uncalibrated | 0.126 | 0.493 [0.378, 0.595] | 0.090 | 0.079 | 0.922 |
| B7 | platt | 0.118 | 0.527 [0.417, 0.626] | 0.064 | 0.061 | 0.922 |
| B7 | hb | 0.119 | 0.521 [0.408, 0.621] | 0.065 | 0.048 | 0.917 |
| B7 | linr | 0.120 | 0.517 [0.410, 0.613] | 0.063 | 0.069 | 0.919 |
| B7 | logr | 0.122 | 0.508 [0.393, 0.617] | 0.081 | 0.086 | 0.919 |
| B7 | ighb | 0.122 | 0.508 [0.394, 0.612] | 0.079 | 0.054 | 0.916 |
| B7 | iglb | 0.116 | 0.532 [0.419, 0.634] | 0.059 | 0.053 | 0.922 |
| B7 | bmc | 0.118 | 0.527 [0.414, 0.627] | 0.052 | 0.050 | 0.920 |

## Base predictor x calibrator, GPT OSS

| start | calibrator | Brier | BSS [95%] | ECE | max gASCE (hand) | AUROC |
|---|---|---|---|---|---|---|
| avg_prob | uncalibrated | 0.276 | -0.107 [-0.252, -0.002] | 0.240 | 0.452 | 0.781 |
| avg_prob | platt | 0.194 | 0.224 [0.139, 0.302] | 0.068 | 0.173 | 0.781 |
| avg_prob | hb | 0.189 | 0.245 [0.161, 0.323] | 0.026 | 0.161 | 0.777 |
| avg_prob | linr | 0.037 | 0.852 [0.800, 0.894] | 0.054 | 0.022 | 0.975 |
| avg_prob | logr | 0.037 | 0.852 [0.799, 0.895] | 0.034 | 0.019 | 0.979 |
| avg_prob | ighb | 0.039 | 0.844 [0.795, 0.888] | 0.028 | 0.012 | 0.971 |
| avg_prob | iglb | 0.033 | 0.869 [0.822, 0.909] | 0.040 | 0.015 | 0.988 |
| avg_prob | bmc | 0.117 | 0.534 [0.452, 0.608] | 0.037 | 0.099 | 0.908 |
| B2 | uncalibrated | 0.025 | 0.901 [0.863, 0.932] | 0.030 | 0.010 | 0.995 |
| B2 | platt | 0.023 | 0.907 [0.869, 0.938] | 0.027 | 0.006 | 0.995 |
| B2 | hb | 0.025 | 0.900 [0.855, 0.937] | 0.025 | 0.010 | 0.992 |
| B2 | linr | 0.024 | 0.903 [0.859, 0.938] | 0.028 | 0.011 | 0.991 |
| B2 | logr | 0.025 | 0.902 [0.856, 0.938] | 0.030 | 0.013 | 0.991 |
| B2 | ighb | 0.027 | 0.894 [0.850, 0.928] | 0.026 | 0.010 | 0.987 |
| B2 | iglb | 0.024 | 0.904 [0.864, 0.937] | 0.022 | 0.005 | 0.994 |
| B2 | bmc | 0.024 | 0.905 [0.866, 0.936] | 0.028 | 0.010 | 0.995 |
| B4 | uncalibrated | 0.023 | 0.906 [0.872, 0.936] | 0.036 | 0.012 | 0.995 |
| B4 | platt | 0.023 | 0.909 [0.874, 0.939] | 0.030 | 0.010 | 0.995 |
| B4 | hb | 0.024 | 0.905 [0.861, 0.941] | 0.020 | 0.008 | 0.990 |
| B4 | linr | 0.023 | 0.908 [0.868, 0.940] | 0.025 | 0.007 | 0.994 |
| B4 | logr | 0.024 | 0.903 [0.859, 0.939] | 0.034 | 0.016 | 0.994 |
| B4 | ighb | 0.023 | 0.906 [0.872, 0.936] | 0.036 | 0.012 | 0.995 |
| B4 | iglb | 0.023 | 0.906 [0.872, 0.936] | 0.036 | 0.012 | 0.995 |
| B4 | bmc | 0.023 | 0.909 [0.871, 0.940] | 0.027 | 0.008 | 0.995 |
| B6 | uncalibrated | 0.031 | 0.874 [0.828, 0.911] | 0.027 | 0.016 | 0.992 |
| B6 | platt | 0.031 | 0.874 [0.828, 0.911] | 0.023 | 0.015 | 0.992 |
| B6 | hb | 0.033 | 0.867 [0.820, 0.904] | 0.026 | 0.015 | 0.983 |
| B6 | linr | 0.031 | 0.875 [0.825, 0.915] | 0.031 | 0.014 | 0.988 |
| B6 | logr | 0.032 | 0.871 [0.819, 0.913] | 0.024 | 0.014 | 0.987 |
| B6 | ighb | 0.026 | 0.897 [0.855, 0.932] | 0.026 | 0.005 | 0.992 |
| B6 | iglb | 0.031 | 0.874 [0.828, 0.911] | 0.027 | 0.016 | 0.992 |
| B6 | bmc | 0.031 | 0.874 [0.828, 0.911] | 0.027 | 0.016 | 0.992 |
| B7 | uncalibrated | 0.023 | 0.906 [0.862, 0.940] | 0.024 | 0.011 | 0.994 |
| B7 | platt | 0.024 | 0.905 [0.862, 0.939] | 0.028 | 0.012 | 0.994 |
| B7 | hb | 0.023 | 0.907 [0.862, 0.943] | 0.021 | 0.005 | 0.993 |
| B7 | linr | 0.024 | 0.902 [0.854, 0.940] | 0.028 | 0.012 | 0.990 |
| B7 | logr | 0.027 | 0.893 [0.842, 0.932] | 0.037 | 0.020 | 0.992 |
| B7 | ighb | 0.023 | 0.906 [0.862, 0.940] | 0.024 | 0.011 | 0.994 |
| B7 | iglb | 0.023 | 0.906 [0.862, 0.940] | 0.024 | 0.011 | 0.994 |
| B7 | bmc | 0.023 | 0.907 [0.864, 0.941] | 0.021 | 0.008 | 0.994 |

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
| qwen3 | avg_prob | uncalibrated | 0.665 | 0.359 |
| qwen3 | avg_prob | bmc | 0.071 | 0.016 |
| qwen3 | B2 | uncalibrated | 0.051 |  |
| qwen3 | B2 | bmc | 0.052 |  |
| qwen3 | B4 | uncalibrated | 0.072 |  |
| qwen3 | B4 | bmc | 0.061 |  |
| qwen3 | B6 | uncalibrated | 0.050 |  |
| qwen3 | B6 | bmc | 0.042 |  |
| qwen3 | B7 | uncalibrated | 0.079 |  |
| qwen3 | B7 | bmc | 0.050 |  |
| gpt-oss | avg_prob | uncalibrated | 0.452 | 0.343 |
| gpt-oss | avg_prob | bmc | 0.099 | 0.082 |
| gpt-oss | B2 | uncalibrated | 0.010 | 0.008 |
| gpt-oss | B2 | bmc | 0.010 | 0.007 |
| gpt-oss | B4 | uncalibrated | 0.012 | 0.006 |
| gpt-oss | B4 | bmc | 0.008 | 0.004 |
| gpt-oss | B6 | uncalibrated | 0.016 |  |
| gpt-oss | B6 | bmc | 0.016 |  |
| gpt-oss | B7 | uncalibrated | 0.011 | 0.006 |
| gpt-oss | B7 | bmc | 0.008 | 0.006 |

## Base predictor differences (uncalibrated)

| model | comparison | metric | diff [95%] |
|---|---|---|---|
| qwen3 | B4 - B2 | brier | +0.0077 [-0.0033, +0.0187] |
| qwen3 | B4 - B2 | log_loss | +0.0064 [-0.0252, +0.0369] |
| qwen3 | B6 - B4 | brier | +0.0011 [-0.0113, +0.0137] |
| qwen3 | B6 - B4 | log_loss | +0.0113 [-0.0204, +0.0421] |
| qwen3 | B7 - B4 | brier | +0.0052 [-0.0061, +0.0161] |
| qwen3 | B7 - B4 | log_loss | +0.0147 [-0.0168, +0.0466] |
| qwen3 | B7 - B6 | brier | +0.0041 [-0.0081, +0.0164] |
| qwen3 | B7 - B6 | log_loss | +0.0034 [-0.0301, +0.0395] |
| gpt-oss | B4 - B2 | brier | -0.0014 [-0.0043, +0.0017] |
| gpt-oss | B4 - B2 | log_loss | -0.0014 [-0.0086, +0.0061] |
| gpt-oss | B6 - B4 | brier | +0.0080 [+0.0029, +0.0137] |
| gpt-oss | B6 - B4 | log_loss | +0.0360 [+0.0158, +0.0603] |
| gpt-oss | B7 - B4 | brier | +0.0001 [-0.0030, +0.0035] |
| gpt-oss | B7 - B4 | log_loss | -0.0005 [-0.0100, +0.0107] |
| gpt-oss | B7 - B6 | brier | -0.0079 [-0.0124, -0.0040] |
| gpt-oss | B7 - B6 | log_loss | -0.0365 [-0.0589, -0.0190] |

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

### Three group sets, IGLB on B4 (val_tune)

| model | group set | groups | Brier | max gASCE union | slices | slices > 2 SE | IGLB patches |
|---|---|---|---|---|---|---|---|
| qwen3 | none (uncalibrated B4) | 0 | 0.1209 | 0.0722 | 91.0000 | 28.0000 | 0.0000 |
| qwen3 | hand | 8 | 0.1140 | 0.0619 | 90.0000 | 23.0000 | 1.0000 |
| qwen3 | discovered | 0 |  |  |  |  |  |
| qwen3 | shap | 5 | 0.1137 | 0.0486 | 90.0000 | 22.0000 | 1.0000 |
| gpt-oss | none (uncalibrated B4) | 0 | 0.0234 | 0.0118 | 69.0000 | 38.0000 | 0.0000 |
| gpt-oss | hand | 7 | 0.0234 | 0.0118 | 69.0000 | 38.0000 | 0.0000 |
| gpt-oss | discovered | 2 | 0.0229 | 0.0070 | 59.0000 | 29.0000 | 1.0000 |
| gpt-oss | shap | 6 | 0.0234 | 0.0118 | 69.0000 | 38.0000 | 0.0000 |

## Murphy decomposition

| model | start | method | Brier | reliability | resolution | uncertainty | gap | change in reliability vs Platt | change in resolution vs Platt |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | avg_prob | uncalibrated | 0.3761 | 0.2131 | 0.0852 | 0.2489 | -0.0007 | 0.2051 [0.1461, 0.2603] | -0.0063 [-0.0166, -0.0037] |
| qwen3 | avg_prob | platt | 0.1655 | 0.0080 | 0.0915 | 0.2489 | 0.0001 |  |  |
| qwen3 | avg_prob | hb | 0.1648 | 0.0031 | 0.0873 | 0.2489 | 0.0000 | -0.0049 [-0.0138, -0.0023] | -0.0042 [-0.0122, -0.0019] |
| qwen3 | avg_prob | linr | 0.1592 | 0.0167 | 0.1059 | 0.2489 | -0.0006 | 0.0087 [-0.0006, 0.0239] | 0.0143 [-0.0032, 0.0340] |
| qwen3 | avg_prob | logr | 0.1593 | 0.0175 | 0.1066 | 0.2489 | -0.0005 | 0.0095 [0.0012, 0.0277] | 0.0151 [0.0009, 0.0366] |
| qwen3 | avg_prob | ighb | 0.1677 | 0.0112 | 0.0917 | 0.2489 | -0.0007 | 0.0032 [-0.0054, 0.0156] | 0.0002 [-0.0101, 0.0125] |
| qwen3 | avg_prob | iglb | 0.1684 | 0.0164 | 0.0960 | 0.2489 | -0.0009 | 0.0084 [-0.0026, 0.0205] | 0.0045 [-0.0062, 0.0150] |
| qwen3 | avg_prob | bmc | 0.1434 | 0.0059 | 0.1115 | 0.2489 | 0.0000 | -0.0020 [-0.0128, 0.0045] | 0.0200 [0.0010, 0.0376] |
| qwen3 | B2 | uncalibrated | 0.1132 | 0.0088 | 0.1445 | 0.2489 | -0.0001 | 0.0005 [-0.0051, 0.0048] | -0.0005 [-0.0049, 0.0029] |
| qwen3 | B2 | platt | 0.1121 | 0.0083 | 0.1450 | 0.2489 | -0.0001 |  |  |
| qwen3 | B2 | hb | 0.1159 | 0.0072 | 0.1391 | 0.2489 | -0.0012 | -0.0011 [-0.0108, 0.0039] | -0.0059 [-0.0163, -0.0014] |
| qwen3 | B2 | linr | 0.1143 | 0.0080 | 0.1420 | 0.2489 | -0.0007 | -0.0002 [-0.0094, 0.0066] | -0.0030 [-0.0157, 0.0073] |
| qwen3 | B2 | logr | 0.1158 | 0.0144 | 0.1471 | 0.2489 | -0.0005 | 0.0062 [-0.0008, 0.0137] | 0.0021 [-0.0076, 0.0123] |
| qwen3 | B2 | ighb | 0.1136 | 0.0091 | 0.1445 | 0.2489 | 0.0001 | 0.0009 [-0.0070, 0.0066] | -0.0005 [-0.0086, 0.0073] |
| qwen3 | B2 | iglb | 0.1132 | 0.0088 | 0.1445 | 0.2489 | -0.0001 | 0.0005 [-0.0051, 0.0048] | -0.0005 [-0.0049, 0.0029] |
| qwen3 | B2 | bmc | 0.1124 | 0.0079 | 0.1442 | 0.2489 | -0.0003 | -0.0004 [-0.0050, 0.0026] | -0.0008 [-0.0050, 0.0018] |
| qwen3 | B4 | uncalibrated | 0.1209 | 0.0166 | 0.1450 | 0.2489 | 0.0004 | 0.0077 [-0.0023, 0.0188] | 0.0024 [-0.0022, 0.0120] |
| qwen3 | B4 | platt | 0.1152 | 0.0088 | 0.1426 | 0.2489 | -0.0000 |  |  |
| qwen3 | B4 | hb | 0.1189 | 0.0110 | 0.1411 | 0.2489 | 0.0001 | 0.0022 [-0.0066, 0.0090] | -0.0015 [-0.0094, 0.0054] |
| qwen3 | B4 | linr | 0.1188 | 0.0094 | 0.1388 | 0.2489 | -0.0008 | 0.0006 [-0.0078, 0.0095] | -0.0038 [-0.0121, 0.0061] |
| qwen3 | B4 | logr | 0.1211 | 0.0151 | 0.1426 | 0.2489 | -0.0004 | 0.0063 [-0.0034, 0.0170] | 0.0000 [-0.0093, 0.0098] |
| qwen3 | B4 | ighb | 0.1268 | 0.0154 | 0.1380 | 0.2489 | 0.0005 | 0.0065 [-0.0024, 0.0145] | -0.0046 [-0.0164, 0.0059] |
| qwen3 | B4 | iglb | 0.1140 | 0.0109 | 0.1459 | 0.2489 | 0.0001 | 0.0021 [-0.0056, 0.0129] | 0.0034 [-0.0039, 0.0142] |
| qwen3 | B4 | bmc | 0.1173 | 0.0110 | 0.1418 | 0.2489 | -0.0008 | 0.0022 [-0.0063, 0.0129] | -0.0008 [-0.0077, 0.0101] |
| qwen3 | B6 | uncalibrated | 0.1221 | 0.0110 | 0.1377 | 0.2489 | -0.0002 | 0.0033 [-0.0033, 0.0084] | -0.0022 [-0.0053, -0.0001] |
| qwen3 | B6 | platt | 0.1171 | 0.0077 | 0.1399 | 0.2489 | 0.0003 |  |  |
| qwen3 | B6 | hb | 0.1184 | 0.0044 | 0.1345 | 0.2489 | -0.0005 | -0.0033 [-0.0093, -0.0010] | -0.0054 [-0.0116, -0.0034] |
| qwen3 | B6 | linr | 0.1207 | 0.0106 | 0.1383 | 0.2489 | -0.0005 | 0.0029 [-0.0037, 0.0086] | -0.0015 [-0.0100, 0.0070] |
| qwen3 | B6 | logr | 0.1225 | 0.0107 | 0.1367 | 0.2489 | -0.0004 | 0.0030 [-0.0038, 0.0126] | -0.0031 [-0.0122, 0.0091] |
| qwen3 | B6 | ighb | 0.1217 | 0.0108 | 0.1378 | 0.2489 | -0.0002 | 0.0031 [-0.0034, 0.0080] | -0.0021 [-0.0068, 0.0012] |
| qwen3 | B6 | iglb | 0.1156 | 0.0055 | 0.1388 | 0.2489 | -0.0001 | -0.0022 [-0.0055, 0.0012] | -0.0010 [-0.0039, 0.0019] |
| qwen3 | B6 | bmc | 0.1171 | 0.0068 | 0.1380 | 0.2489 | -0.0006 | -0.0009 [-0.0061, 0.0025] | -0.0019 [-0.0055, 0.0006] |
| qwen3 | B7 | uncalibrated | 0.1262 | 0.0183 | 0.1415 | 0.2489 | 0.0004 | 0.0082 [0.0010, 0.0146] | 0.0008 [-0.0024, 0.0037] |
| qwen3 | B7 | platt | 0.1177 | 0.0102 | 0.1408 | 0.2489 | -0.0007 |  |  |
| qwen3 | B7 | hb | 0.1193 | 0.0092 | 0.1389 | 0.2489 | 0.0001 | -0.0009 [-0.0076, 0.0029] | -0.0018 [-0.0077, 0.0002] |
| qwen3 | B7 | linr | 0.1203 | 0.0104 | 0.1394 | 0.2489 | 0.0004 | 0.0003 [-0.0064, 0.0068] | -0.0014 [-0.0099, 0.0069] |
| qwen3 | B7 | logr | 0.1224 | 0.0146 | 0.1407 | 0.2489 | -0.0004 | 0.0045 [-0.0035, 0.0134] | -0.0001 [-0.0089, 0.0087] |
| qwen3 | B7 | ighb | 0.1225 | 0.0134 | 0.1401 | 0.2489 | 0.0002 | 0.0032 [-0.0026, 0.0075] | -0.0007 [-0.0100, 0.0053] |
| qwen3 | B7 | iglb | 0.1164 | 0.0101 | 0.1423 | 0.2489 | -0.0003 | -0.0001 [-0.0038, 0.0034] | 0.0015 [-0.0029, 0.0051] |
| qwen3 | B7 | bmc | 0.1177 | 0.0088 | 0.1398 | 0.2489 | -0.0002 | -0.0014 [-0.0061, 0.0022] | -0.0010 [-0.0054, 0.0014] |
| gpt-oss | avg_prob | uncalibrated | 0.2765 | 0.0922 | 0.0643 | 0.2498 | -0.0013 | 0.0802 [0.0449, 0.1130] | -0.0034 [-0.0098, -0.0013] |
| gpt-oss | avg_prob | platt | 0.1939 | 0.0121 | 0.0677 | 0.2498 | -0.0004 |  |  |
| gpt-oss | avg_prob | hb | 0.1886 | 0.0017 | 0.0630 | 0.2498 | -0.0000 | -0.0104 [-0.0219, -0.0033] | -0.0047 [-0.0109, -0.0021] |
| gpt-oss | avg_prob | linr | 0.0371 | 0.0102 | 0.2231 | 0.2498 | 0.0002 | -0.0019 [-0.0175, 0.0038] | 0.1555 [0.1273, 0.1740] |
| gpt-oss | avg_prob | logr | 0.0371 | 0.0063 | 0.2194 | 0.2498 | 0.0004 | -0.0058 [-0.0207, -0.0001] | 0.1518 [0.1235, 0.1715] |
| gpt-oss | avg_prob | ighb | 0.0389 | 0.0023 | 0.2132 | 0.2498 | -0.0001 | -0.0097 [-0.0247, -0.0053] | 0.1455 [0.1167, 0.1650] |
| gpt-oss | avg_prob | iglb | 0.0326 | 0.0041 | 0.2212 | 0.2498 | -0.0001 | -0.0080 [-0.0227, -0.0030] | 0.1535 [0.1255, 0.1729] |
| gpt-oss | avg_prob | bmc | 0.1165 | 0.0025 | 0.1355 | 0.2498 | -0.0003 | -0.0096 [-0.0226, -0.0036] | 0.0679 [0.0463, 0.0853] |
| gpt-oss | B2 | uncalibrated | 0.0248 | 0.0035 | 0.2285 | 0.2498 | -0.0000 | 0.0015 [-0.0004, 0.0042] | 0.0001 [-0.0007, 0.0017] |
| gpt-oss | B2 | platt | 0.0232 | 0.0019 | 0.2284 | 0.2498 | -0.0001 |  |  |
| gpt-oss | B2 | hb | 0.0249 | 0.0016 | 0.2264 | 0.2498 | -0.0001 | -0.0003 [-0.0028, 0.0012] | -0.0020 [-0.0060, -0.0000] |
| gpt-oss | B2 | linr | 0.0242 | 0.0022 | 0.2279 | 0.2498 | 0.0000 | 0.0003 [-0.0019, 0.0021] | -0.0006 [-0.0034, 0.0020] |
| gpt-oss | B2 | logr | 0.0245 | 0.0031 | 0.2284 | 0.2498 | 0.0001 | 0.0011 [-0.0013, 0.0036] | -0.0000 [-0.0031, 0.0025] |
| gpt-oss | B2 | ighb | 0.0266 | 0.0033 | 0.2268 | 0.2498 | 0.0002 | 0.0014 [-0.0008, 0.0037] | -0.0017 [-0.0054, 0.0012] |
| gpt-oss | B2 | iglb | 0.0240 | 0.0020 | 0.2278 | 0.2498 | 0.0000 | 0.0000 [-0.0027, 0.0019] | -0.0006 [-0.0038, 0.0012] |
| gpt-oss | B2 | bmc | 0.0238 | 0.0029 | 0.2289 | 0.2498 | -0.0001 | 0.0010 [-0.0010, 0.0029] | 0.0005 [-0.0012, 0.0021] |
| gpt-oss | B4 | uncalibrated | 0.0234 | 0.0049 | 0.2316 | 0.2498 | 0.0003 | 0.0011 [-0.0009, 0.0029] | 0.0008 [-0.0012, 0.0027] |
| gpt-oss | B4 | platt | 0.0227 | 0.0038 | 0.2308 | 0.2498 | -0.0001 |  |  |
| gpt-oss | B4 | hb | 0.0237 | 0.0018 | 0.2282 | 0.2498 | 0.0003 | -0.0020 [-0.0047, -0.0008] | -0.0027 [-0.0071, -0.0004] |
| gpt-oss | B4 | linr | 0.0230 | 0.0025 | 0.2295 | 0.2498 | 0.0001 | -0.0013 [-0.0033, -0.0000] | -0.0014 [-0.0040, 0.0003] |
| gpt-oss | B4 | logr | 0.0242 | 0.0041 | 0.2296 | 0.2498 | -0.0002 | 0.0003 [-0.0023, 0.0031] | -0.0012 [-0.0041, 0.0010] |
| gpt-oss | B4 | ighb | 0.0234 | 0.0049 | 0.2316 | 0.2498 | 0.0003 | 0.0011 [-0.0009, 0.0029] | 0.0008 [-0.0012, 0.0027] |
| gpt-oss | B4 | iglb | 0.0234 | 0.0049 | 0.2316 | 0.2498 | 0.0003 | 0.0011 [-0.0009, 0.0029] | 0.0008 [-0.0012, 0.0027] |
| gpt-oss | B4 | bmc | 0.0227 | 0.0025 | 0.2295 | 0.2498 | -0.0000 | -0.0013 [-0.0038, 0.0007] | -0.0013 [-0.0043, 0.0008] |
| gpt-oss | B6 | uncalibrated | 0.0314 | 0.0022 | 0.2204 | 0.2498 | -0.0002 | 0.0008 [-0.0011, 0.0019] | 0.0007 [-0.0012, 0.0019] |
| gpt-oss | B6 | platt | 0.0314 | 0.0014 | 0.2197 | 0.2498 | -0.0001 |  |  |
| gpt-oss | B6 | hb | 0.0333 | 0.0020 | 0.2187 | 0.2498 | 0.0002 | 0.0006 [-0.0019, 0.0015] | -0.0010 [-0.0041, -0.0000] |
| gpt-oss | B6 | linr | 0.0312 | 0.0036 | 0.2222 | 0.2498 | -0.0000 | 0.0022 [-0.0012, 0.0052] | 0.0025 [-0.0025, 0.0073] |
| gpt-oss | B6 | logr | 0.0321 | 0.0014 | 0.2187 | 0.2498 | -0.0004 | 0.0000 [-0.0024, 0.0011] | -0.0010 [-0.0051, 0.0012] |
| gpt-oss | B6 | ighb | 0.0258 | 0.0021 | 0.2260 | 0.2498 | -0.0001 | 0.0007 [-0.0016, 0.0017] | 0.0063 [0.0012, 0.0111] |
| gpt-oss | B6 | iglb | 0.0314 | 0.0022 | 0.2204 | 0.2498 | -0.0002 | 0.0008 [-0.0011, 0.0019] | 0.0007 [-0.0012, 0.0019] |
| gpt-oss | B6 | bmc | 0.0314 | 0.0022 | 0.2204 | 0.2498 | -0.0002 | 0.0008 [-0.0011, 0.0019] | 0.0007 [-0.0012, 0.0019] |
| gpt-oss | B7 | uncalibrated | 0.0235 | 0.0028 | 0.2290 | 0.2498 | -0.0001 | -0.0004 [-0.0015, 0.0003] | -0.0005 [-0.0018, 0.0004] |
| gpt-oss | B7 | platt | 0.0237 | 0.0032 | 0.2295 | 0.2498 | 0.0002 |  |  |
| gpt-oss | B7 | hb | 0.0233 | 0.0013 | 0.2278 | 0.2498 | -0.0000 | -0.0018 [-0.0048, -0.0004] | -0.0017 [-0.0052, -0.0003] |
| gpt-oss | B7 | linr | 0.0244 | 0.0035 | 0.2289 | 0.2498 | -0.0000 | 0.0003 [-0.0023, 0.0023] | -0.0006 [-0.0039, 0.0015] |
| gpt-oss | B7 | logr | 0.0268 | 0.0063 | 0.2295 | 0.2498 | 0.0002 | 0.0032 [-0.0005, 0.0066] | 0.0001 [-0.0031, 0.0022] |
| gpt-oss | B7 | ighb | 0.0235 | 0.0028 | 0.2290 | 0.2498 | -0.0001 | -0.0004 [-0.0015, 0.0003] | -0.0005 [-0.0018, 0.0004] |
| gpt-oss | B7 | iglb | 0.0235 | 0.0028 | 0.2290 | 0.2498 | -0.0001 | -0.0004 [-0.0015, 0.0003] | -0.0005 [-0.0018, 0.0004] |
| gpt-oss | B7 | bmc | 0.0232 | 0.0016 | 0.2282 | 0.2498 | -0.0000 | -0.0015 [-0.0036, 0.0001] | -0.0013 [-0.0035, 0.0004] |

## Holm family (28 comparisons, Brier differences)

| model | comparison | Brier diff | lo | hi | p | Holm p | significant |
|---|---|---|---|---|---|---|---|
| qwen3 | avg_prob: linr - platt | -0.0063 | -0.0193 | 0.0077 | 0.3600 | 1.0000 | False |
| qwen3 | avg_prob: iglb - platt | 0.0029 | -0.0051 | 0.0122 | 0.5170 | 1.0000 | False |
| qwen3 | avg_prob: bmc - platt | -0.0221 | -0.0412 | -0.0044 | 0.0150 | 0.3750 | False |
| qwen3 | B2: linr - platt | 0.0022 | -0.0048 | 0.0090 | 0.5460 | 1.0000 | False |
| qwen3 | B2: iglb - platt | 0.0010 | -0.0024 | 0.0042 | 0.5240 | 1.0000 | False |
| qwen3 | B2: bmc - platt | 0.0002 | -0.0019 | 0.0020 | 0.7840 | 1.0000 | False |
| qwen3 | B4: linr - platt | 0.0036 | -0.0068 | 0.0144 | 0.5180 | 1.0000 | False |
| qwen3 | B4: iglb - platt | -0.0011 | -0.0069 | 0.0044 | 0.7070 | 1.0000 | False |
| qwen3 | B4: bmc - platt | 0.0022 | -0.0031 | 0.0072 | 0.4380 | 1.0000 | False |
| qwen3 | B6: linr - platt | 0.0036 | -0.0050 | 0.0115 | 0.4100 | 1.0000 | False |
| qwen3 | B6: iglb - platt | -0.0015 | -0.0031 | 0.0001 | 0.0710 | 1.0000 | False |
| qwen3 | B6: bmc - platt | -0.0000 | -0.0033 | 0.0031 | 0.9770 | 1.0000 | False |
| qwen3 | B4 - B2 (uncalibrated) | 0.0077 | -0.0033 | 0.0187 | 0.1790 | 1.0000 | False |
| qwen3 | B6 - B4 (uncalibrated) | 0.0011 | -0.0113 | 0.0137 | 0.8320 | 1.0000 | False |
| gpt-oss | avg_prob: linr - platt | -0.1568 | -0.1801 | -0.1338 | 0.0000 | 0.0000 | True |
| gpt-oss | avg_prob: iglb - platt | -0.1612 | -0.1839 | -0.1388 | 0.0000 | 0.0000 | True |
| gpt-oss | avg_prob: bmc - platt | -0.0774 | -0.0945 | -0.0595 | 0.0000 | 0.0000 | True |
| gpt-oss | B2: linr - platt | 0.0010 | -0.0005 | 0.0026 | 0.2220 | 1.0000 | False |
| gpt-oss | B2: iglb - platt | 0.0008 | -0.0009 | 0.0027 | 0.3910 | 1.0000 | False |
| gpt-oss | B2: bmc - platt | 0.0006 | -0.0006 | 0.0018 | 0.3480 | 1.0000 | False |
| gpt-oss | B4: linr - platt | 0.0003 | -0.0011 | 0.0017 | 0.7090 | 1.0000 | False |
| gpt-oss | B4: iglb - platt | 0.0007 | 0.0001 | 0.0012 | 0.0150 | 0.3750 | False |
| gpt-oss | B4: bmc - platt | 0.0000 | -0.0009 | 0.0011 | 0.9790 | 1.0000 | False |
| gpt-oss | B7: linr - platt | 0.0007 | -0.0008 | 0.0024 | 0.3880 | 1.0000 | False |
| gpt-oss | B7: iglb - platt | -0.0002 | -0.0004 | -0.0000 | 0.0340 | 0.7820 | False |
| gpt-oss | B7: bmc - platt | -0.0005 | -0.0015 | 0.0004 | 0.2800 | 1.0000 | False |
| gpt-oss | B4 - B2 (uncalibrated) | -0.0014 | -0.0043 | 0.0017 | 0.3690 | 1.0000 | False |
| gpt-oss | B7 - B4 (uncalibrated) | 0.0001 | -0.0030 | 0.0035 | 0.9880 | 1.0000 | False |

## Conformal risk control (Qwen3 Coder primary, GPT OSS fallback)

| calibrator | target | threshold | realized risk | escalation | system pass | oracle at same budget | regret | regret, rank based raw P4 |
|---|---|---|---|---|---|---|---|---|
| uncalibrated | 0.050 | 0.602 | 0.045 | 0.631 | 0.506 | 0.606 | 0.100 | 0.100 |
| uncalibrated | 0.100 | 0.407 | 0.084 | 0.522 | 0.523 | 0.606 | 0.083 | 0.083 |
| uncalibrated | 0.150 | 0.259 | 0.133 | 0.436 | 0.530 | 0.606 | 0.076 | 0.076 |
| platt | 0.050 | 0.645 | 0.045 | 0.631 | 0.506 | 0.606 | 0.100 | 0.100 |
| platt | 0.100 | 0.415 | 0.084 | 0.522 | 0.523 | 0.606 | 0.083 | 0.083 |
| platt | 0.150 | 0.240 | 0.133 | 0.436 | 0.530 | 0.606 | 0.076 | 0.076 |
| iglb | 0.050 | 0.602 | 0.045 | 0.631 | 0.506 | 0.606 | 0.100 | 0.100 |
| iglb | 0.100 | 0.407 | 0.084 | 0.522 | 0.523 | 0.606 | 0.083 | 0.083 |
| iglb | 0.150 | 0.259 | 0.133 | 0.436 | 0.530 | 0.606 | 0.076 | 0.076 |
| bmc | 0.050 | 0.636 | 0.047 | 0.626 | 0.508 | 0.606 | 0.098 | 0.100 |
| bmc | 0.100 | 0.458 | 0.084 | 0.522 | 0.523 | 0.606 | 0.083 | 0.083 |
| bmc | 0.150 | 0.227 | 0.133 | 0.436 | 0.530 | 0.606 | 0.076 | 0.076 |

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

## Distribution shift (optional V7)

| base | fit on | evaluated on | calibrator | BSS | ECE | risk at target 0.10 | coverage at 0.10 |
|---|---|---|---|---|---|---|---|
| B2 | qwen3 | qwen3 | uncalibrated | 0.545 | 0.063 | 0.084 | 0.478 |
| B2 | qwen3 | qwen3 | platt | 0.549 | 0.053 | 0.084 | 0.478 |
| B2 | qwen3 | qwen3 | iglb | 0.545 | 0.063 | 0.084 | 0.478 |
| B2 | qwen3 | qwen3 | bmc | 0.549 | 0.056 | 0.084 | 0.478 |
| B2 | gpt-oss | gpt-oss | uncalibrated | 0.901 | 0.030 | 0.084 | 0.571 |
| B2 | gpt-oss | gpt-oss | platt | 0.907 | 0.027 | 0.084 | 0.571 |
| B2 | gpt-oss | gpt-oss | iglb | 0.904 | 0.022 | 0.084 | 0.571 |
| B2 | gpt-oss | gpt-oss | bmc | 0.905 | 0.028 | 0.089 | 0.577 |
| B2 | qwen3 | gpt-oss | uncalibrated | -0.056 | 0.346 | 0.000 | 0.113 |
| B2 | qwen3 | gpt-oss | platt | -0.091 | 0.352 | 0.000 | 0.113 |
| B2 | qwen3 | gpt-oss | iglb | -0.056 | 0.346 | 0.000 | 0.113 |
| B2 | qwen3 | gpt-oss | bmc | -0.057 | 0.346 | 0.000 | 0.113 |
| B2 | gpt-oss | qwen3 | uncalibrated | 0.288 | 0.181 | 0.369 | 0.836 |
| B2 | gpt-oss | qwen3 | platt | 0.281 | 0.186 | 0.369 | 0.836 |
| B2 | gpt-oss | qwen3 | iglb | 0.276 | 0.185 | 0.369 | 0.836 |
| B2 | gpt-oss | qwen3 | bmc | 0.301 | 0.176 | 0.370 | 0.837 |
| B4 | qwen3 | qwen3 | uncalibrated | 0.514 | 0.089 | 0.091 | 0.486 |
| B4 | qwen3 | qwen3 | platt | 0.525 | 0.083 | 0.091 | 0.486 |
| B4 | qwen3 | qwen3 | iglb | 0.542 | 0.074 | 0.092 | 0.495 |
| B4 | qwen3 | qwen3 | bmc | 0.529 | 0.075 | 0.090 | 0.486 |
| B4 | gpt-oss | gpt-oss | uncalibrated | 0.906 | 0.036 | 0.067 | 0.554 |
| B4 | gpt-oss | gpt-oss | platt | 0.909 | 0.030 | 0.067 | 0.554 |
| B4 | gpt-oss | gpt-oss | iglb | 0.906 | 0.036 | 0.067 | 0.554 |
| B4 | gpt-oss | gpt-oss | bmc | 0.909 | 0.027 | 0.068 | 0.555 |
| B4 | qwen3 | gpt-oss | uncalibrated | 0.216 | 0.279 | 0.000 | 0.213 |
| B4 | qwen3 | gpt-oss | platt | 0.242 | 0.270 | 0.000 | 0.213 |
| B4 | qwen3 | gpt-oss | iglb | 0.226 | 0.276 | 0.000 | 0.193 |
| B4 | qwen3 | gpt-oss | bmc | 0.247 | 0.273 | 0.000 | 0.212 |
| B4 | gpt-oss | qwen3 | uncalibrated | 0.353 | 0.154 | 0.364 | 0.831 |
| B4 | gpt-oss | qwen3 | platt | 0.339 | 0.160 | 0.364 | 0.831 |
| B4 | gpt-oss | qwen3 | iglb | 0.353 | 0.154 | 0.364 | 0.831 |
| B4 | gpt-oss | qwen3 | bmc | 0.340 | 0.157 | 0.364 | 0.831 |
