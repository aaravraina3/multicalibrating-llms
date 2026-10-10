# Conformal risk control: the test miss

Post-run diagnosis. No setting was changed and nothing was refit. Cascade: Qwen3 Coder answers, escalation to GPT OSS when the calibrated B2 score is below a threshold chosen on `val_conformal` (132 problems). Loss: an answer that is accepted and fails, averaged per problem.

## Size of the miss

| target | threshold (Platt on B2) | risk on val_conformal (SE) | bound on val_conformal | realized risk on val_tune | realized risk on test (SE) |
|---|---|---|---|---|---|
| 0.05 | 0.645 | 0.042 (0.014) | 0.0496 | 0.045 | 0.063 (0.012) |
| 0.10 | 0.415 | 0.093 (0.021) | 0.1000 | 0.084 | 0.141 (0.017) |
| 0.15 | 0.240 | 0.143 (0.024) | 0.1496 | 0.133 | 0.195 (0.020) |

SE: standard error of the mean per problem loss, problems as units. At target 0.10 the test excess over val_conformal is 0.048, about 1.8 combined standard errors. Uncalibrated, Platt, and IGLB scores pick the same accepted set at every target (the threshold is chosen on the ranking), so the miss is the same for all three.

## What differs on test

| split | problems | pass rate | mean B2 | accepted at target 0.10: fail rate | mean B2 of accepted |
|---|---|---|---|---|---|
| val_conformal | 132 | 0.428 | 0.414 | 0.197 | 0.773 |
| val_tune | 132 | 0.467 | 0.405 | 0.177 | 0.744 |
| test | 264 | 0.458 | 0.464 | 0.251 | 0.747 |

At the same threshold, test's accepted answers fail 25% of the time vs 20% on val_conformal, at similar average scores. B2 is more overconfident on test's high scoring answers. This matches v1, where B2's BSS was lower on test (0.560) than on validation (0.585).

## Are the splits exchangeable?

The guarantee assumes calibration and test problems are exchangeable. Nothing points to a systematic split, such as by date:

* Median contest number by split (a proxy for date): AtCoder 346 (val_conformal), 358 (val_tune), 349 (test), 349 (base_train); LeetCode question number 3416, 3353, 3396, 3386; Codeforces 1886 (val_tune), 1883 (test).
* Platform mix: AtCoder 54% / 61% / 56% of problems in val_conformal / val_tune / test; LeetCode 46% / 37% / 43%.
* Difficulty mix: hard 38.6% (val_conformal), 33.3% (val_tune), 31.1% (test). With 132 problems the standard error of such a share is about 0.04, so this is ordinary sampling variation.

The splits look like random draws, so exchangeability is plausible and the miss looks like sampling luck plus a mildly different test split.

## Why one miss can happen

The guarantee is an average over draws of the calibration set, not a promise for every draw. In a simulation with known pass probabilities (500 fresh draws of 132 problems, seed 1), mean realized risk was 0.043, 0.094, 0.145 for targets 0.05, 0.10, 0.15, and single draws exceeded the target 20%, 29%, and 37% of the time.

## A disclosure that points the same way

`val_conformal` was never used for any v2 fit, tuning, or early stopping. But v1 tuned a few settings on the full validation set, which includes what is now `val_conformal`, and the v2 plan said to keep them: B2's regularization strength (C = 0.01, picked from 4 values by validation log loss), B2's Platt input (logit p, from 3 options), IGHB's alpha and IGLB's epsilon. That's a mild reuse of the conformal half. It would make val_conformal look slightly better than truly new data, which pushes the threshold the same way as the miss. The effect of choosing among 4 and 3 options is likely small, but it should be disclosed.

## Suggested wording

"Conformal risk control targeting an accepted failure rate" rather than "bounding": it held on validation (0.084 at a 0.10 target) and missed on test (0.141), with one 132 problem calibration set.
