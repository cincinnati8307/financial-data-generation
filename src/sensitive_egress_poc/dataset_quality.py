from __future__ import annotations
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import sacrebleu


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def sentence_bleu(candidate, references):
    """
    Calculate BLEU score of one sentence against other sentences.
    """

    if len(references) == 0:
        return 0.0

    bleu = sacrebleu.sentence_bleu(
        candidate,
        references
    )

    return bleu.score / 100.0


def compute_self_bleu(samples):
    """
    samples:
        list of strings

    return:
        average Self-BLEU
    """

    scores = []

    for i, sent in enumerate(tqdm(samples, leave=False)):

        references = samples[:i] + samples[i+1:]

        score = sentence_bleu(
            sent,
            references
        )

        scores.append(score)

    return sum(scores) / len(scores)



def group_by(data, key):

    groups = defaultdict(list)

    for item in data:

        groups[item.get(key, "unknown")].append(
            item["text"]
        )

    return groups



def evaluate_group(data, group_key):

    groups = group_by(data, group_key)

    results = {}

    for name, texts in groups.items():

        # Self-BLEU needs multiple samples
        if len(texts) < 5:
            continue

        score = compute_self_bleu(texts)

        results[name] = {
            "count": len(texts),
            "self_bleu": round(score, 4)
        }

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    args = parser.parse_args()


    data = load_jsonl(args.input)


    print("\n=== Overall Self-BLEU ===")

    overall = compute_self_bleu(
        [x["text"] for x in data]
    )

    print(
        f"Self-BLEU: {overall:.4f}"
    )


    for key in [
        "label",
        "subtype",
        "style",
        "region"
    ]:

        print(
            f"\n=== Self-BLEU grouped by {key} ==="
        )

        results = evaluate_group(
            data,
            key
        )

        for group, result in results.items():

            print(
                group,
                result
            )