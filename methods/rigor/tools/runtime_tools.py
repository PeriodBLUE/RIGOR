"""Runtime tools used by MATH and MedMCQA role-aware agents."""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Callable, Dict

import requests


ToolFn = Callable[[Dict[str, Any], Dict[str, Any]], str]


def _wiki_search_tool(args: Dict[str, Any], _context: Dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "No query provided for wiki_search."
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return f"wiki_search failed with status {resp.status_code}."
        data = resp.json() if resp.content else {}
        extract = str(data.get("extract", "")).strip()
        if not extract:
            return "wiki_search found no summary."
        return extract[:1000]
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return f"wiki_search error: {exc}"


def _python_eval_tool(args: Dict[str, Any], _context: Dict[str, Any]) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return "No expression provided for python_eval."
    try:
        node = ast.parse(expression, mode="eval")
    except Exception:
        return "python_eval parse error."
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            return "python_eval only supports pure arithmetic expressions."
    try:
        result = eval(compile(node, "<python_eval>", "eval"), {"__builtins__": {}}, {})
        return repr(result)
    except Exception as exc:
        return f"python_eval runtime error: {exc}"


def _math_helper_tool(args: Dict[str, Any], _context: Dict[str, Any]) -> str:
    op = str(args.get("op", "")).strip().lower()
    if not op:
        return "No op provided for math_helper."
    try:
        if op == "gcd":
            return repr(math.gcd(int(args.get("a")), int(args.get("b"))))
        if op == "lcm":
            a = int(args.get("a"))
            b = int(args.get("b"))
            return repr(abs(a * b) // math.gcd(a, b) if a and b else 0)
        if op == "is_prime":
            n = int(args.get("n"))
            if n < 2:
                return "False"
            if n % 2 == 0:
                return "True" if n == 2 else "False"
            d = 3
            while d * d <= n:
                if n % d == 0:
                    return "False"
                d += 2
            return "True"
        if op == "factorint":
            n = int(args.get("n"))
            if n == 0:
                return json.dumps({"0": 1}, ensure_ascii=False)
            sign = -1 if n < 0 else 1
            n = abs(n)
            factors: Dict[str, int] = {}
            if sign < 0:
                factors["-1"] = 1
            p = 2
            while p * p <= n:
                while n % p == 0:
                    key = str(p)
                    factors[key] = factors.get(key, 0) + 1
                    n //= p
                p = 3 if p == 2 else p + 2
            if n > 1:
                factors[str(n)] = factors.get(str(n), 0) + 1
            return json.dumps(factors, ensure_ascii=False)
        if op == "divisors_count":
            n = abs(int(args.get("n")))
            if n == 0:
                return "0"
            count = 1
            p = 2
            while p * p <= n:
                e = 0
                while n % p == 0:
                    e += 1
                    n //= p
                if e:
                    count *= e + 1
                p = 3 if p == 2 else p + 2
            if n > 1:
                count *= 2
            return repr(count)
        if op == "ncr":
            n = int(args.get("n"))
            r = int(args.get("r"))
            return repr(math.comb(n, r) if 0 <= r <= n else 0)
        if op == "npr":
            n = int(args.get("n"))
            r = int(args.get("r"))
            return repr(math.perm(n, r) if 0 <= r <= n else 0)
        if op == "pow_mod":
            base = int(args.get("base"))
            exp = int(args.get("exp"))
            mod = int(args.get("mod"))
            if mod == 0:
                return "math_helper error: mod must be non-zero."
            return repr(pow(base, exp, mod))
        return (
            "math_helper unsupported op. "
            "Supported ops: gcd, lcm, is_prime, factorint, divisors_count, ncr, npr, pow_mod."
        )
    except Exception as exc:
        return f"math_helper error: {exc}"


TOOL_REGISTRY: Dict[str, ToolFn] = {
    "wiki_search": _wiki_search_tool,
    "python_eval": _python_eval_tool,
    "math_helper": _math_helper_tool,
}

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "wiki_search": "Search a Wikipedia page summary. Args JSON: {\"query\": \"entity\"}",
    "python_eval": "Evaluate a pure arithmetic expression. Args JSON: {\"expression\": \"(12+3)/5\"}",
    "math_helper": "Competition-math helper for integer checks. Args JSON: {\"op\":\"gcd|lcm|is_prime|factorint|divisors_count|ncr|npr|pow_mod\", ...}",
}
