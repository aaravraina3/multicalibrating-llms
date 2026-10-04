"""Phase 1: data sanity checks. Test split is used for IDs and counts only, never labels."""

import pandas as pd

from calib.data import load_raw, load_rows, load_test_ids

rows = load_rows()
test = load_test_ids()
out = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out.append(line)


say("## rows per model per split (expect 5270 / 2640 / 2640)")
counts = pd.concat([rows[["model", "split"]], test.assign(split="test")[["model", "split"]]])
say(counts.groupby(["model", "split"]).size().unstack().to_string())

say("\n## problems per split (expect 527 / 264 / 264)")
ids = pd.concat([rows[["task_id", "model", "split"]], test.assign(split="test")[["task_id", "model", "split"]]])
say(ids.groupby(["model", "split"]).task_id.nunique().unstack().to_string())

say("\n## no problem in two splits")
per_task = ids.drop_duplicates(["task_id", "model", "split"]).groupby(["model", "task_id"]).split.nunique()
say("max splits per problem:", per_task.max())

say("\n## same problem IDs and split for both models")
assign = ids.drop_duplicates(["task_id", "model"]).pivot(index="task_id", columns="model", values="split")
say("problems in only one model:", assign.isna().any(axis=1).sum())
say("problems with different split across models:", (assign["qwen3"] != assign["gpt-oss"]).sum())

say("\n## same prompt and difficulty for both models")
meta = rows.drop_duplicates(["task_id", "model"]).pivot(index="task_id", columns="model", values=["prompt", "difficulty"])
say("prompt mismatches:", (meta["prompt"]["qwen3"] != meta["prompt"]["gpt-oss"]).sum())
say("difficulty mismatches:", (meta["difficulty"]["qwen3"] != meta["difficulty"]["gpt-oss"]).sum())

say("\n## pass rate per model per split (train and validation only)")
say(rows.groupby(["model", "split"]).y.mean().unstack().round(3).to_string())

say("\n## pass rate per model per difficulty (train only)")
rows["is_empty"] = rows.code.str.strip() == ""
train = rows[rows.split == "train"]
say(train.groupby(["model", "difficulty"]).y.mean().unstack().round(3).to_string())
say("\nproblems per difficulty (train):")
say(train.drop_duplicates(["task_id", "model"]).groupby(["model", "difficulty"]).size().unstack().to_string())

say("\n## empty program rate per model per split")
say(rows.groupby(["model", "split"]).is_empty.mean().unstack().round(3).to_string())
say("\npass rate of empty programs (train + validation):", rows[rows.is_empty].y.mean())

say("\n## generations with no logprobs")
rows["n_tokens"] = rows.logprobs.map(len)
say("count:", (rows.n_tokens == 0).sum())
say("tokens per output, median by model:")
say(rows.groupby("model").n_tokens.median().to_string())

say("\n## code span checks")
span_empty = (rows.code_start == 0) & (rows.code_end == 0)
say("span [0,0] but program non empty:", (span_empty & ~rows.is_empty).sum())
say("program empty but span not [0,0]:", (~span_empty & rows.is_empty).sum())
say("span end beyond token count:", (rows.code_end > rows.n_tokens).sum())

say("\n## logprob range")
allv = pd.Series([v for a in rows.logprobs.sample(500, random_state=0) for v in a])
say("max:", allv.max(), "min:", allv.min(), "share exactly 0:", round((allv == 0).mean(), 3))

with open("runs/phase1/checks.txt", "w") as f:
    f.write("\n".join(out) + "\n")


# Manual examples from train: one pass, one fail with code, one empty program.
def show(model, task_id, k):
    raw = load_raw(model, "train")
    rec = raw[[i for i, t in enumerate(raw["id"]) if t == task_id][0]]
    toks = [t for _, t in rec["token_logprobs"][k]]
    s, e = rec["code_token_idx"][k]
    print(f"\n### {model} {task_id} sample {k} passed={rec['is_correct'][k]} span=[{s},{e}] tokens={len(toks)}")
    print("first 30 tokens:", "".join(toks[:30]).replace("Ġ", " ").replace("Ċ", "\\n"))
    print("code span tokens:", "".join(toks[s:e])[:600].replace("Ġ", " ").replace("Ċ", "\n"))
    print("program field:", (rec["program"][k] or "")[:300])


if __name__ == "__main__":
    for model in ["qwen3", "gpt-oss"]:
        t = train[train.model == model]
        p = t[t.y == 1].iloc[0]
        f = t[(t.y == 0) & ~t.is_empty].iloc[0]
        e = t[t.is_empty].iloc[0] if t.is_empty.any() else None
        for r in [p, f] + ([e] if e is not None else []):
            show(model, r.task_id, r.sample_idx)
