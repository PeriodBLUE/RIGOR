def get_eval_func(eval_protocol, tested_dataset_name):
    if eval_protocol == "xverify":
        from .evaluate_xverify import eval_func_xverify
        return eval_func_xverify
    else:
        raise ValueError(f"Unsupported evaluation protocol: {eval_protocol}")
