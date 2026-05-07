import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback
from pathlib import Path

from methods.mas_base import MAS
from utils import (
    load_model_api_config,
    read_valid_jsonl,
    redacted_model_api_config,
    reserve_unprocessed_queries,
    write_to_jsonl,
)
from evaluations import get_eval_func

def evaluate_sample(args, item, save_eval_path, lock=None, llm=None):
    eval_func = get_eval_func(args.eval_protocol, args.tested_dataset_name)

    if 'response' in item:
        eval_content, eval_score = eval_func(item, llm)
    else:
        eval_content, eval_score = "Infer Error", None
        
    save_data = item.copy()
    save_data["eval_content"] = eval_content
    save_data["eval_score"] = eval_score
    if args.debug:
        print(json.dumps(save_data, indent=4))
    else:
        write_to_jsonl(lock, save_eval_path, save_data)


def _derive_eval_path(tested_infer_path: str, eval_protocol: str) -> str:
    path = Path(tested_infer_path)
    stem = path.stem
    suffix = path.suffix or ".jsonl"

    if "infer" in stem:
        eval_stem = stem.replace("infer", f"{eval_protocol}_eval")
    else:
        eval_stem = f"{stem}_{eval_protocol}_eval"

    candidate = str(path.with_name(f"{eval_stem}{suffix}"))
    if candidate == tested_infer_path:
        candidate = str(path.with_name(f"{stem}_{eval_protocol}_eval{suffix}"))
    return candidate

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the evaluation
    parser.add_argument("--eval_protocol", type=str, default="xverify", help="The evaluation protocol to be used.")

    # args related to the model
    parser.add_argument("--model_name", type=str, default="xverify-9b-c", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to evaluated objects
    parser.add_argument("--tested_dataset_name", type=str, default="MATH", help="The dataset to be used for testing.")
    parser.add_argument("--tested_method_name", type=str, default="rigor", help="MAS name.")
    parser.add_argument("--tested_method_config_name", type=str, default=None, help="The config name for the method.")
    parser.add_argument("--tested_mas_model_name", type=str, default="gpt-4o-mini", help="The agent backend to be used for inference.")
    parser.add_argument("--tested_infer_path", type=str, default=None, help="Path to the output file.")
    

    parser.add_argument("--debug", action="store_true", help="Turn this on to run one defined sample for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Turn this on to overwrite the existing output file.")
    parser.add_argument("--sequential", action="store_true", help="Turn this on to run the evaluation sequentially.")
    args = parser.parse_args()
    
    general_config = vars(args)

    print("="*50, f"\nEvaluating {args.tested_method_name} on {args.tested_dataset_name} with {args.tested_mas_model_name} as MAS model using {args.model_name} as LLM")
    print(json.dumps(general_config, indent=4))

    llm = None
    max_workers = 8
    llm_required_protocols = {"xverify"}
    if args.eval_protocol in llm_required_protocols:
        model_api_config = load_model_api_config(args.model_api_config, args.model_name)
        general_config.update({"model_api_config": model_api_config})
        print("-"*50, f"\n>> Model config: {redacted_model_api_config(model_api_config[args.model_name])}")
        llm = MAS(general_config)
        max_workers = model_api_config[args.model_name]["max_workers"]
    else:
        print("-"*50, f"\n>> Eval protocol `{args.eval_protocol}` does not require LLM backend.")
    
    # Load evaluation data
    tested_infer_path = args.tested_infer_path if args.tested_infer_path is not None else f"./results/{args.tested_dataset_name}/{args.tested_mas_model_name}/{args.tested_method_name}_infer.jsonl"
    tested_infer_path = tested_infer_path.replace("_infer.jsonl", f"_{args.tested_method_config_name}_infer.jsonl") if (args.tested_method_config_name is not None and args.tested_infer_path is None) else tested_infer_path
    save_eval_path = _derive_eval_path(tested_infer_path, args.eval_protocol)
    
    if args.debug:
        sample = {"query": "1+3=?", "gt": "4", "response": "\\boxed{4}"}
        evaluate_sample(args, sample, save_eval_path, lock=None, llm=llm)
    else:
        eval_data = read_valid_jsonl(tested_infer_path)
        print(f">> Before filtering: {len(eval_data)} samples")

        if args.overwrite and os.path.exists(save_eval_path):
            os.remove(save_eval_path)
            print(f">> {save_eval_path} exists, remove it.")
        else:
            eval_data = reserve_unprocessed_queries(save_eval_path, eval_data)
        print(f">> After filtering: {len(eval_data)} samples")

        lock = threading.Lock()
        if args.sequential:
            for sample in eval_data:
                evaluate_sample(args, sample, save_eval_path, lock, llm)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(lambda sample: evaluate_sample(args, sample, save_eval_path, lock, llm), eval_data), total=len(eval_data), desc="Evaluating MAS"):
                    pass

        if not os.path.exists(save_eval_path):
            print(">> No evaluation output file found yet. Nothing to summarize.")
            raise SystemExit(0)

        # Load evaluation results and print the statistics
        saved_data = read_valid_jsonl(save_eval_path)

        sample_num = len(saved_data)
        valid_eval_score_list = [
            sample.get("eval_score")
            for sample in saved_data
            if sample.get("eval_score") is not None
        ]
        valid_correct_num = sum([1 for score in valid_eval_score_list if score==1])
        num_valid = len(valid_eval_score_list)
        num_exclude_eval_error = len([
            sample for sample in saved_data
            if not str(sample.get("eval_content", "Eval Error: missing")).startswith("Eval Error")
        ])
        valid_acc = (valid_correct_num / num_valid * 100) if num_valid > 0 else 0.0
        non_error_acc = (valid_correct_num / num_exclude_eval_error * 100) if num_exclude_eval_error > 0 else 0.0
        print(f">> Evaluation Finished:\n{sample_num} samples in total\n{num_valid} valid samples | {valid_correct_num} correct samples | accuracy: {valid_acc:.2f}%\n{num_exclude_eval_error} samples excluding eval error | accuracy: {non_error_acc:.2f}%")
        
