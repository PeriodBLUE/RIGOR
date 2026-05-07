import argparse
import json
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="MATH")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--num2sample", type=int, default=50000)
args = parser.parse_args()

save_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    f"{args.dataset_name}.json",
)


def get_load_dataset():
    from datasets import load_dataset
    return load_dataset


def shuffle_and_sample(data_list, num2sample):
    random.seed(2024)
    random.shuffle(data_list)
    return data_list[:num2sample]


def deduplicate(data_list):
    seen_queries = set()
    unique_data = []
    for item in data_list:
        if item["query"] not in seen_queries:
            unique_data.append(item)
            seen_queries.add(item["query"])
    if len(unique_data) < len(data_list):
        print(f">> Duplicate samples removed: {len(data_list) - len(unique_data)}")
    return unique_data


if args.dataset_name == "MATH":
    load_dataset = get_load_dataset()
    load_dataset_path = args.dataset_path if args.dataset_path else "HuggingFaceH4/MATH-500"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'=' * 50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": example["answer"],
            "tag": [
                args.dataset_name,
                "math",
                example["subject"],
                f"Level {example['level']}",
            ],
            "source": args.dataset_name,
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
elif args.dataset_name == "MedMCQA":
    load_dataset = get_load_dataset()
    load_dataset_path = args.dataset_path if args.dataset_path else "openlifescienceai/medmcqa"
    dataset = load_dataset(load_dataset_path, split="validation", trust_remote_code=True)
    filtered_dataset = dataset.filter(lambda example: example["choice_type"] != "multi")
    print(f"{'=' * 50}\n", filtered_dataset)

    def format_medmcqa_query(example):
        query = example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['opa']}"
        query += f"\n(B) {example['opb']}"
        query += f"\n(C) {example['opc']}"
        query += f"\n(D) {example['opd']}"
        return query

    def format_medmcqa_gt(example):
        answer_list = [
            f"(A) {example['opa']}",
            f"(B) {example['opb']}",
            f"(C) {example['opc']}",
            f"(D) {example['opd']}",
        ]
        return f"The correct answer is: {answer_list[example['cop']]}"

    data_list = [
        {
            "query": format_medmcqa_query(example),
            "gt": format_medmcqa_gt(example),
            "tag": ["medical", example["subject_name"], example["topic_name"]],
            "source": "MedMCQA",
        }
        for example in filtered_dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
else:
    raise ValueError("Only MATH and MedMCQA dataset builders are supported.")

sample_pool = deduplicate(data_list)
print(f">> A data sample from the pool:\n{sample_pool[0]}")
print(f"{'=' * 50}\n There are {len(sample_pool)} queries in the pool.")

with open(save_path, "w", encoding="utf-8") as output_json:
    json.dump(sample_pool, output_json, indent=4)
