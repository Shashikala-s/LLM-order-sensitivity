from datasets import load_dataset
import json, os

os.makedirs("../../data", exist_ok=True)
ds = load_dataset("commonsense_qa", )


# print(ds)

def dump(split, path):
    keep = skip = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds[split]:
            # print(ex)
            # print('----' * 50)
            labs = ex["choices"]["label"]  # e.g. ["A","B","C","D"]
            texts = ex["choices"]["text"]
            if len(labs) != 5:
                skip += 1;
                continue
            m = dict(zip(labs, texts))
            try:
                choices = {k: m[k] for k in "ABCD"}
            except KeyError:
                skip += 1;
                continue
            obj = {"question": ex["question"], "choices": choices, "answer": ex["answerKey"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n");
            keep += 1
    print(f"{split}: wrote {keep}, skipped {skip}")


dump("train", "../../data/commonsense_qa_train.jsonl")
dump("test", "../../data/commonsense_qa_test.jsonl")
