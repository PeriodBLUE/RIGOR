import os
import re
import json
import random
import threading
import concurrent.futures
import numpy as np
from tqdm import tqdm
from math import isclose

MIN_VAL_SIZE = 10
MAX_VAL_SIZE = 20

def _expand_env_placeholders(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    return value


def load_model_api_config(model_api_config, model_name):
    config_path = model_api_config
    with open(config_path, "r") as f:
        model_api_config = _expand_env_placeholders(json.load(f))
    if model_name not in model_api_config:
        available = ", ".join(sorted(model_api_config.keys()))
        raise KeyError(f"Model alias `{model_name}` is not in {config_path}. Available aliases: {available}")
    for model in model_api_config[model_name].get("model_list", []):
        api_key = str(model.get("api_key", ""))
        if "${" in api_key:
            raise EnvironmentError(
                f"Unresolved API key placeholder for model alias `{model_name}`. "
                "Export the referenced environment variable or edit model_api_configs/model_api_config.json."
            )
    for model_name in model_api_config:
        actual_max_workers = model_api_config[model_name]["max_workers_per_model"] * len(model_api_config[model_name]["model_list"])
        model_api_config[model_name]["max_workers"] = actual_max_workers
    return model_api_config


def redacted_model_api_config(model_cfg):
    """Return a copy of a model config that is safe to print."""
    if isinstance(model_cfg, list):
        return [redacted_model_api_config(item) for item in model_cfg]
    if isinstance(model_cfg, dict):
        redacted = {}
        for key, value in model_cfg.items():
            if key.lower() in {"api_key", "authorization", "token"}:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redacted_model_api_config(value)
        return redacted
    return model_cfg


# ================== LLM ==================

def extract_boxed_answer(text: str) -> str:
    """Extract the final LaTeX boxed answer if present."""
    if not isinstance(text, str):
        return ""
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()

def simple_math_equal(prediction: str, reference: str) -> bool:
    """Best-effort answer equality for math-style outputs."""
    if not prediction or not reference:
        return False
    
    pred = extract_boxed_answer(str(prediction))
    ref = extract_boxed_answer(str(reference))
    
    if pred == ref:
        return True
    
    try:
        pred_clean = re.sub(r",", "", pred)
        ref_clean = re.sub(r",", "", ref)
        
        if pred_clean.endswith("%"):
            pred_clean = pred_clean.rstrip("%").rstrip("\\")
            pred_val = float(pred_clean) / 100
        else:
            pred_val = float(pred_clean)
            
        if ref_clean.endswith("%"):
            ref_clean = ref_clean.rstrip("%").rstrip("\\")
            ref_val = float(ref_clean) / 100
        else:
            ref_val = float(ref_clean)
            
        return isclose(pred_val, ref_val, abs_tol=1e-3)
    except:
        pass
    
    try:
        from sympy import simplify, N
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
        
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s
        
        a = _parse(pred)
        b = _parse(ref)
        
        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass
        
        try:
            if isclose(float(N(a)), float(N(b)), abs_tol=1e-3):
                return True
        except:
            pass
    except:
        pass
    
    return False


# ================== Validation Set Selection ==================

def auto_generate_val_dataset(
    full_dataset: list,
    general_config: dict,
    dataset_name: str,
    num_runs: int = 5,
    candidate_ratio: float = 0.2,
    select_ratio: float = 0.05,
    seed: int = 42
) -> tuple:
    """
    Select a validation split from high-variance samples under repeated single-agent runs.

    Args:
        full_dataset: Complete task split.
        general_config: Runtime configuration with model settings.
        dataset_name: Benchmark name.
        num_runs: Number of repeated LLM runs per candidate.
        candidate_ratio: Fraction sampled as validation candidates.
        select_ratio: Fraction retained as final validation samples.
        seed: Random seed.
    
    Returns:
        (val_dataset, test_dataset).
    """
    from methods.mas_base import MAS
    
    print(f"\n{'='*50}")
    print(">> Selecting validation samples")
    print(f">> Full dataset size: {len(full_dataset)}")
    print(f">> Candidate ratio: {candidate_ratio*100}%, selection ratio: {select_ratio*100}%")
    print(f">> Repeated single-agent runs: {num_runs}")
    
    # Step 1: sample validation candidates.
    random.seed(seed)
    shuffled_dataset = full_dataset.copy()
    random.shuffle(shuffled_dataset)
    
    candidate_size = int(len(shuffled_dataset) * candidate_ratio)
    candidate_val_dataset = shuffled_dataset[:candidate_size]
    test_dataset = shuffled_dataset[candidate_size:]
    
    print(f">> Candidate validation size: {len(candidate_val_dataset)}")
    print(f">> Remaining test size: {len(test_dataset)}")
    
    # Step 2: estimate per-sample variance with repeated single-agent runs.
    print(f"\n>> Running single-agent baseline {num_runs} times per candidate")
    
    sample_scores = {i: [] for i in range(len(candidate_val_dataset))}
    max_workers = general_config["model_api_config"][general_config["model_name"]]["max_workers"]
    
    def run_single_sample(sample_idx_tuple):
        sample_idx, sample, run_idx = sample_idx_tuple
        try:
            mas = MAS(general_config)
            mas_output = mas.inference(sample)
            
            if "response" not in mas_output:
                return sample_idx, 0
            
            response = mas_output["response"]
            ground_truth = sample.get("gt", "")
            
            is_correct = simple_math_equal(response, ground_truth)
            score = 1 if is_correct else 0
            
            return sample_idx, score
        except Exception as e:
            print(f"Sample {sample_idx} run {run_idx} failed: {e}")
            return sample_idx, 0
    
    for run_idx in range(num_runs):
        print(f"\n>> Run {run_idx + 1}/{num_runs}")
        
        tasks = [(i, sample, run_idx) for i, sample in enumerate(candidate_val_dataset)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(run_single_sample, tasks),
                total=len(tasks),
                desc=f"Run {run_idx + 1}"
            ))
        
        for sample_idx, score in results:
            sample_scores[sample_idx].append(score)
    
    # Step 3: rank samples by variance.
    print("\n>> Ranking validation candidates")
    sample_variances = []
    for sample_idx, scores in sample_scores.items():
        variance = np.std(scores) if len(scores) > 1 else 0
        mean_score = np.mean(scores)
        sample_variances.append({
            "idx": sample_idx,
            "scores": scores,
            "variance": variance,
            "mean": mean_score,
            "sample": candidate_val_dataset[sample_idx]
        })
    
    sample_variances.sort(key=lambda x: x["variance"], reverse=True)
    
    final_val_size = int(len(full_dataset) * select_ratio)
    final_val_size = max(MIN_VAL_SIZE, min(MAX_VAL_SIZE, final_val_size))
    final_val_size = min(final_val_size, len(sample_variances))
    final_val_samples = sample_variances[:final_val_size]
    
    print("\n>> Selected validation variance stats:")
    if final_val_samples:
        print(f"   max: {final_val_samples[0]['variance']:.4f}")
        print(f"   min: {final_val_samples[-1]['variance']:.4f}")
        print(f"   mean: {np.mean([s['variance'] for s in final_val_samples]):.4f}")
    
    val_dataset = [s["sample"] for s in final_val_samples]
    
    # Step 5: persist selected validation samples and diagnostics.
    val_dataset_path = f"./datasets/data/{dataset_name}_val.json"
    os.makedirs(os.path.dirname(val_dataset_path), exist_ok=True)
    with open(val_dataset_path, "w", encoding="utf-8") as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)
    
    variance_stats_path = f"./datasets/data/{dataset_name}_val_stats.json"
    variance_stats = [{
        "idx": s["idx"],
        "variance": s["variance"],
        "mean": s["mean"],
        "scores": s["scores"],
        "query": s["sample"].get("query", "")[:100],
    } for s in final_val_samples]
    with open(variance_stats_path, "w", encoding="utf-8") as f:
        json.dump(variance_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n>> Validation set saved to: {val_dataset_path}")
    print(f">> Validation diagnostics saved to: {variance_stats_path}")
    print(f">> Validation set size: {len(val_dataset)}")
    print(f"{'='*50}\n")
    
    return val_dataset, test_dataset

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')


def append_to_json(lock, file_name, data):
    with lock:
        existing_data = []
        if os.path.exists(file_name):
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing_data = loaded
            except (json.JSONDecodeError, OSError):
                existing_data = []

        existing_data.append(data)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

def read_valid_jsonl(file_name):
    all_data = []
    with open(file_name, "r") as f:
        tmp = f.readlines()
    for line in tmp:
        try:
            all_data.append(json.loads(line))
        except Exception as e:
            print(line)
            print(f"{e}")
    return all_data

def reserve_unprocessed_queries(output_path, test_dataset):
    processed_queries = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset
