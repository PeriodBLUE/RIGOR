"""Method registry for RIGOR."""

import importlib


METHOD_CLASS_PATHS = {
    "rigor_math": (".rigor", "RIGOR_MATH"),
    "rigor_medmcqa": (".rigor", "RIGOR_MEDMCQA"),
}


CLASS_NAME_TO_METHOD = {}
for _method_name, (_, _class_name) in METHOD_CLASS_PATHS.items():
    CLASS_NAME_TO_METHOD.setdefault(_class_name, _method_name)


def _load_attr(module_path, attr_name):
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, attr_name)


def get_method_class(method_name, dataset_name=None):
    method_name = method_name.lower().replace(" ", "_").replace("-", "_")

    all_method_names = METHOD_CLASS_PATHS.keys()
    matched_method_names = [
        sample_method_name
        for sample_method_name in all_method_names
        if method_name in sample_method_name
    ]

    if not matched_method_names:
        raise ValueError(f"[ERROR] No method found matching {method_name}.")

    if dataset_name is not None:
        raw_dataset_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        if raw_dataset_name == "medmcqa":
            dataset_alias = "medmcqa"
        elif raw_dataset_name == "math":
            dataset_alias = "math"
        else:
            raise ValueError(
                f"[ERROR] RIGOR supports only MATH and MedMCQA in this release; got {dataset_name}."
            )

        matched_by_dataset = [
            sample_method_name
            for sample_method_name in matched_method_names
            if sample_method_name.split("_")[-1] == dataset_alias
        ]
        if not matched_by_dataset:
            supported = ", ".join(sorted(METHOD_CLASS_PATHS))
            raise ValueError(
                f"[ERROR] Method `{method_name}` does not match dataset `{dataset_name}`. "
                f"Supported RIGOR modes: {supported}."
            )
        matched_method_names = matched_by_dataset

    selected = matched_method_names[0]
    module_path, class_name = METHOD_CLASS_PATHS[selected]
    return _load_attr(module_path, class_name)


def __getattr__(name):
    if name in CLASS_NAME_TO_METHOD:
        method_name = CLASS_NAME_TO_METHOD[name]
        module_path, class_name = METHOD_CLASS_PATHS[method_name]
        return _load_attr(module_path, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
