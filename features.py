# features.py
import pandas as pd

# 🔴 PASTE EXACT LIST FROM TRAINING HERE
FEATURE_COLUMNS = [
    "word_count",
    "char_count",
    "num_count",
    "max_number",
    "math_symbol_count",
    "bracket_count",
    "kw_dp",
    "kw_graph",
    "kw_greedy",
    "kw_math",
    "kw_string",
    "kw_tree",
    "kw_sort",
    "kw_search",
    "kw_recursion",
    "kw_bit",
    "kw_array",
    "kw_matrix",
    "kw_mod",
    "kw_probability",
    "kw_game",
    "kw_geometry",
    "kw_flow",
    "kw_dfs",
    "kw_bfs"
]

KEYWORDS = {
    "dp": ["dp", "dynamic programming"],
    "graph": ["graph"],
    "greedy": ["greedy"],
    "math": ["math"],
    "string": ["string"],
    "tree": ["tree"],
    "sort": ["sort"],
    "search": ["search", "binary search"],
    "recursion": ["recursion", "recursive"],
    "bit": ["bit", "bitmask"],
    "array": ["array"],
    "matrix": ["matrix"],
    "mod": ["mod", "modulo"],
    "probability": ["probability"],
    "game": ["game"],
    "geometry": ["geometry"],
    "flow": ["flow"],
    "dfs": ["dfs"],
    "bfs": ["bfs"]
}

def extract_features(text: str) -> pd.DataFrame:
    feats = {col: 0 for col in FEATURE_COLUMNS}

    feats["word_count"] = len(text.split())
    feats["char_count"] = len(text)
    feats["num_count"] = sum(c.isdigit() for c in text)
    feats["max_number"] = max(
        [int(s) for s in text.split() if s.isdigit()],
        default=0
    )

    feats["math_symbol_count"] = sum(text.count(sym) for sym in "+-*/=%")
    feats["bracket_count"] = text.count("(") + text.count(")")

    for k, kws in KEYWORDS.items():
        feats[f"kw_{k}"] = int(any(kw in text for kw in kws))

    return pd.DataFrame([feats])[FEATURE_COLUMNS]
