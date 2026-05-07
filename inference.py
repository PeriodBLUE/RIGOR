import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback

from methods import get_method_class
from utils import (
    auto_generate_val_dataset,
    load_model_api_config,
    redacted_model_api_config,
    reserve_unprocessed_queries,
    write_to_jsonl,
)

MIN_VAL_SIZE = 10
MAX_VAL_SIZE = 20
DATA_DIR = "./datasets/data"


def _resolve_dataset_path(dataset_name, val=False):
    suffix = "_val" if val else ""
    filename = f"{dataset_name}{suffix}.json"
    candidate = os.path.join(DATA_DIR, filename)
    if os.path.exists(candidate):
        return candidate

    lower_filename = filename.lower()
    if os.path.isdir(DATA_DIR):
        for existing in os.listdir(DATA_DIR):
            if existing.lower() == lower_filename:
                return os.path.join(DATA_DIR, existing)
    return candidate


def _expected_inference_artifact(mas):
    candidate = getattr(mas, "inference_artifact_path", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    state_dir = getattr(mas, "state_dir", None)
    if isinstance(state_dir, str) and state_dir.strip():
        guessed = os.path.join(state_dir, "best_workflow.npy")
        if os.path.exists(guessed):
            return guessed
    return None


def _load_or_create_val_dataset(test_dataset, general_config, dataset_name):
    val_dataset_path = _resolve_dataset_path(dataset_name, val=True)
    if not os.path.exists(val_dataset_path):
        print(">> ...")
        val_dataset, test_dataset = auto_generate_val_dataset(
            full_dataset=test_dataset,
            general_config=general_config,
            dataset_name=dataset_name,
            num_runs=5,
            candidate_ratio=0.2,
            select_ratio=0.05,
            seed=42,
        )
    else:
        with open(val_dataset_path, "r") as f:
            val_dataset = json.load(f)

    val_dataset = normalize_val_dataset(test_dataset, val_dataset)
    with open(val_dataset_path, "w", encoding="utf-8") as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)
    print(f">> Normalized val set size: {len(val_dataset)} (range [{MIN_VAL_SIZE}, {MAX_VAL_SIZE}])")

    filtered_test_dataset = exclude_val_from_test(test_dataset, val_dataset)
    print(f">> Excluded val samples from test set: {len(filtered_test_dataset)} remaining")
    return val_dataset, filtered_test_dataset

def _dedup_by_query(dataset):
    seen = set()
    out = []
    for sample in dataset or []:
        if not isinstance(sample, dict):
            continue
        q = str(sample.get("query", ""))
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(sample)
    return out

def normalize_val_dataset(full_dataset, val_dataset):
    """Ensure val size in [MIN_VAL_SIZE, MAX_VAL_SIZE] and deduplicate by query."""
    full_dataset = _dedup_by_query(full_dataset)
    val_dataset = _dedup_by_query(val_dataset)

    # Cap upper bound first.
    if len(val_dataset) > MAX_VAL_SIZE:
        val_dataset = val_dataset[:MAX_VAL_SIZE]

    # Top up to lower bound using samples from full dataset not already in val.
    if len(val_dataset) < MIN_VAL_SIZE:
        val_queries = {str(s.get("query", "")) for s in val_dataset}
        for sample in full_dataset:
            q = str(sample.get("query", ""))
            if q and q not in val_queries:
                val_dataset.append(sample)
                val_queries.add(q)
                if len(val_dataset) >= MIN_VAL_SIZE:
                    break

    # Keep upper bound after top-up.
    if len(val_dataset) > MAX_VAL_SIZE:
        val_dataset = val_dataset[:MAX_VAL_SIZE]
    return val_dataset

def exclude_val_from_test(test_dataset, val_dataset):
    """Remove validation samples from test split by query key."""
    if not val_dataset:
        return test_dataset
    val_queries = {
        str(sample.get("query", ""))
        for sample in val_dataset
        if isinstance(sample, dict) and sample.get("query") is not None
    }
    if not val_queries:
        return test_dataset
    return [
        sample for sample in test_dataset
        if str(sample.get("query", "")) not in val_queries
    ]

def process_sample(args, general_config, sample, output_path, lock):
    MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
    mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
    save_data = sample.copy()
    try:
        mas_output = mas.inference(sample)
        if "response" not in mas_output:    # ensure that there is a key named "response"
            raise ValueError(f"The key 'response' is not found in the MAS output: {mas_output}")
        save_data.update(mas_output)
    except Exception as e:
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"
    save_data.update({"token_stats": mas.get_token_stats()})
    write_to_jsonl(lock, output_path, save_data)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the method
    parser.add_argument("--method_name", type=str, default="rigor", help="MAS name.")
    parser.add_argument("--method_config_name", type=str, default=None, help="RIGOR config name. Defaults to the config matching the dataset.")

    # args related to the model
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=49152, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to dataset
    parser.add_argument("--test_dataset_name", type=str, default="MATH", help="The dataset to be used for testing.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optionally run only the first N unprocessed samples.")
    parser.add_argument("--require_val", action="store_true")
    parser.add_argument(
        "--no_training",
        action="store_true",
        help="With --require_val: skip optimizing() and load prior checkpoint (load-saved-optimization-state forced on for RIGOR).",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    
    general_config = vars(args)
    if args.require_val and args.no_training:
        general_config["method_config_overrides"] = {"load-saved-optimization-state": True}

    # Load model config
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})
    print("-"*50, f"\n>> Model config: {redacted_model_api_config(model_api_config[args.model_name])}")
    
    if args.debug:
        # MAS inference
        sample = {"query": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."}
        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)

        response = mas.inference(sample)
        
        print(json.dumps(response, indent=4))
        print(f"\n>> Token stats: {json.dumps(mas.get_token_stats(), indent=4)}")
    
    else:
        print(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")

        # load dataset
        test_dataset_path = _resolve_dataset_path(args.test_dataset_name)
        with open(test_dataset_path, "r") as f:
            test_dataset = json.load(f)
        
        val_dataset = None
        if args.require_val:
            val_dataset, test_dataset = _load_or_create_val_dataset(
                test_dataset,
                general_config,
                args.test_dataset_name,
            )
        
        # get output path
        if args.output_path is not None:
            output_path = args.output_path
        else:
            # Create a temporary MAS instance to check adversarial configuration
            MAS_METHOD_TEMP = get_method_class(args.method_name, args.test_dataset_name)
            mas_temp = MAS_METHOD_TEMP(general_config, method_config_name=args.method_config_name)
            
            # Check if adversarial attack is enabled and get configuration
            adversarial_enabled = mas_temp.method_config.get("adversarial_attack_enabled", False)
            if adversarial_enabled:
                num_truthful = mas_temp.method_config.get("num-truthful-agent", 0)
                exact_num_adversarial = mas_temp.method_config.get("adversarial_attack_num_agents")
                if exact_num_adversarial is not None:
                    num_adversarial = max(0, min(int(num_truthful), int(exact_num_adversarial)))
                else:
                    adversarial_ratio = mas_temp.method_config.get("adversarial_attack_ratio", 0.3)
                    num_adversarial = int(num_truthful * adversarial_ratio)
                    if num_adversarial == 0 and num_truthful > 0:
                        num_adversarial = 1
                output_path = f"./results/{args.test_dataset_name}/{args.model_name}/{args.method_name}_agents{num_truthful}_adv{num_adversarial}_infer.jsonl"
            else:
                output_path = f"./results/{args.test_dataset_name}/{args.model_name}/{args.method_name}_infer.jsonl"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
        artifact = _expected_inference_artifact(mas)
        if artifact and not os.path.exists(artifact) and not args.require_val:
            print(">> required inference artifact is missing:", artifact, flush=True)
            val_dataset, test_dataset = _load_or_create_val_dataset(
                test_dataset,
                general_config,
                args.test_dataset_name,
            )
            print(">> Running optimizing() to generate the missing artifact.", flush=True)
            mas.optimizing(val_dataset)
        
        # reserve unprocessed samples
        test_dataset = reserve_unprocessed_queries(output_path, test_dataset)
        if args.max_samples is not None and args.max_samples > 0:
            test_dataset = test_dataset[: args.max_samples]
        print(f">> After filtering: {len(test_dataset)} samples")
        
        # Optimize RIGOR if requested.
        if args.require_val:
            if args.no_training:
                artifact = _expected_inference_artifact(mas)
                if artifact and not os.path.exists(artifact):
                    print(
                        ">> no_training requested but required inference artifact is missing:",
                        artifact,
                        flush=True,
                    )
                    print(">> Running optimizing() to generate the missing artifact.", flush=True)
                    mas.optimizing(val_dataset)
                else:
                    ckpt = getattr(mas, "state_dir", None)
                    if artifact:
                        expected_path = artifact
                    elif ckpt:
                        expected_path = os.path.join(ckpt, "hypernetwork.pt")
                    else:
                        expected_path = "(unknown state_dir)"
                    print(
                        ">> no_training: skipping optimizing(); expect checkpoint at",
                        expected_path,
                        flush=True,
                    )
            else:
                mas.optimizing(val_dataset)
        
        # inference the mas
        lock = threading.Lock()
        if args.sequential:
            for sample in test_dataset:
                process_sample(args, general_config, sample, output_path, lock)
        else:
            max_workers = model_api_config[args.model_name]["max_workers"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(lambda sample: process_sample(args, general_config, sample, output_path, lock), test_dataset), total=len(test_dataset), desc="Processing queries"):
                    pass
