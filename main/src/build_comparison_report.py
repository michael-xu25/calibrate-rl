"""
Build HTML comparison report: baseline greedy vs checkpoint greedy.

Produces a rich side-by-side report with:
  - MathJax-rendered LaTeX in responses
  - Reward hacking analysis (format vs accuracy gains)
  - Per-problem annotations for "still wrong" cases
  - Entity-count breakdown
  - Token count deltas

Usage:
    python src/build_comparison_report.py [baseline_path] [checkpoint_path]
"""

import json
import html
import os
import re
import sys
from statistics import median


# ── Inline entity extraction (avoids importing datasets library) ─────────

DAYS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
}
MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}
COMMON_NON_NAMES = {
    "If", "How", "What", "When", "Where", "Who", "Why", "Which",
    "The", "He", "She", "They", "His", "Her", "It", "Its", "Their",
    "Each", "Every", "Some", "All", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "This", "That", "These", "Those", "There", "Here",
    "After", "Before", "During", "Since", "While", "Until",
    "In", "On", "At", "For", "From", "By", "To", "Of", "With",
    "But", "And", "Or", "So", "Yet", "Then", "Than", "Also",
    "An", "A", "As", "Be", "Do", "Is", "Was", "Are", "Were",
    "Has", "Had", "Have", "Can", "Could", "Would", "Should",
    "Not", "No", "Yes", "Both", "Many", "Most", "Only",
    "Step", "Answer", "Total", "Calculate", "Find", "Determine",
    "Given", "Let", "Solve", "Note", "Problem", "Solution",
    "Half", "Twice", "Double", "Triple", "Quarter",
    "UV", "Art", "TV", "DVD", "GPS", "ATM", "GPA", "SAT", "ACT",
    "NFL", "NBA", "MLB", "USA", "US", "UK", "PM", "AM",
    "North", "South", "East", "West", "Central",
    "City", "Town", "Street", "Avenue", "Road", "Park", "Drive",
    "Lake", "River", "Mountain", "Island", "Valley", "Beach", "Bay",
    "York", "Angeles", "Francisco", "Diego",
    "America", "American", "European", "African", "Asian",
    "German", "French", "Chinese", "Japanese", "Italian", "Spanish",
    "English", "British", "Mexican", "Canadian",
    "Facebook", "Instagram", "Twitter", "YouTube", "Amazon", "Google",
    "Metropolitan", "Museum", "Library", "University", "College",
    "Hospital", "Church", "Airport", "Station",
    "National", "International", "Local", "Royal",
    "Walmart", "Target", "Costco",
    "Student", "Students", "Teacher", "Teachers", "Farmer", "Class",
    "Children", "People", "Workers", "Company", "Store", "School",
    "Family", "Friends", "Team", "Group", "Member", "Members",
    "Boy", "Girl", "Boys", "Girls", "Man", "Woman", "Men", "Women",
    "Brother", "Sister", "Mother", "Father", "Uncle", "Aunt",
    "Son", "Daughter", "Husband", "Wife", "Baby",
    "Mom", "Dad", "Grandma", "Grandpa", "Grandad",
    "Grandmother", "Grandfather", "Cousin", "Neighbor",
    "Day", "Week", "Month", "Year", "Hour", "Minute",
    "Morning", "Afternoon", "Evening", "Night",
    "Western", "Eastern", "Northern", "Southern",
    "Lego", "Legos", "Scrabble", "Pokemon",
    "Junior", "Senior", "Little", "Big", "Great",
    "Christmas", "Easter", "Halloween", "Thanksgiving",
    "First", "Second", "Third", "Fourth", "Fifth", "Last", "Next",
    "New", "Old", "Once", "Twice", "Per", "Plus",
    "Currently", "Originally", "Recently", "Finally",
    "However", "Therefore", "Meanwhile", "Although",
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sir",
}
ALL_EXCLUSIONS = DAYS | MONTHS | COMMON_NON_NAMES


def _strip_possessive(name: str) -> str:
    if name.endswith("'s") and len(name) > 3:
        return name[:-2]
    if (name.endswith("s") and not name.endswith("ss") and len(name) > 3):
        return name[:-1]
    return name


def extract_entity_names(question: str) -> list[str]:
    sentences = re.split(r'[.?!\n]+', question)
    candidates = []
    for sent in sentences:
        words = sent.strip().split()
        if not words:
            continue
        for word in words:
            clean = re.sub(r"[^a-zA-Z']", "", word)
            if not clean or len(clean) < 2:
                continue
            if not clean[0].isupper():
                continue
            if clean.isupper() and len(clean) > 2:
                continue
            base = _strip_possessive(clean)
            if base in ALL_EXCLUSIONS or clean in ALL_EXCLUSIONS:
                continue
            if len(base) < 2:
                continue
            candidates.append(base)
    seen = set()
    unique = []
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


# ── Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def key_by_idx(records: list[dict]) -> dict[int, dict]:
    return {r["idx"]: r for r in records}



def escape_for_mathjax(text: str) -> str:
    """HTML-escape text while preserving LaTeX for MathJax rendering.

    Splits on LaTeX display-math delimiters so <br> tags are never
    injected inside \\[ ... \\] blocks (which would break MathJax).
    """
    parts = re.split(r'(\\\[.*?\\\])', text, flags=re.DOTALL)
    out = []
    for part in parts:
        escaped = html.escape(part)
        if part.startswith("\\[") and part.endswith("\\]"):
            out.append(escaped)
        else:
            out.append(escaped.replace("\n", "<br>"))
    return "".join(out)


# ── Analysis computation ─────────────────────────────────────────────────

def compute_analysis(rows):
    """Compute reward hacking and quality metrics from row data."""
    a = {}

    # Extraction method counts
    b_methods = {}
    c_methods = {}
    for r in rows:
        bm = r["b_method"]
        cm = r["c_method"]
        b_methods[bm] = b_methods.get(bm, 0) + 1
        c_methods[cm] = c_methods.get(cm, 0) + 1

    a["b_methods"] = b_methods
    a["c_methods"] = c_methods

    # boxed usage specifically
    a["b_boxed"] = sum(1 for r in rows if r["b_method"] == "boxed")
    a["c_boxed"] = sum(1 for r in rows if r["c_method"] == "boxed")

    # Accuracy by extraction method
    b_acc_by_method = {}
    c_acc_by_method = {}
    for r in rows:
        bm = r["b_method"]
        cm = r["c_method"]
        b_acc_by_method.setdefault(bm, []).append(r["b_correct"])
        c_acc_by_method.setdefault(cm, []).append(r["c_correct"])
    a["b_acc_by_method"] = {m: sum(v)/len(v) for m, v in b_acc_by_method.items()}
    a["c_acc_by_method"] = {m: sum(v)/len(v) for m, v in c_acc_by_method.items()}

    # Format-only wins: checkpoint uses boxed, baseline didn't, but answer still wrong
    a["format_only_wins"] = sum(
        1 for r in rows
        if r["c_method"] == "boxed" and r["b_method"] != "boxed"
        and not r["c_correct"]
    )
    # Format + correct: checkpoint uses boxed AND got it right, baseline didn't use boxed
    a["format_and_correct"] = sum(
        1 for r in rows
        if r["c_method"] == "boxed" and r["b_method"] != "boxed"
        and r["c_correct"]
    )

    # Extraction artifact check: "fixed" problems where the baseline used a
    # fallback extraction method AND the gold answer appeared in the baseline
    # response text — meaning the model may have had the right answer but the
    # extractor grabbed the wrong number.
    fixed = [r for r in rows if r["category"] == "fixed"]
    extraction_artifacts = []
    for r in fixed:
        if r["b_method"] == "boxed":
            continue
        gold = str(r["gold_answer"])
        if gold and re.search(r'(?<!\d)' + re.escape(gold) + r'(?!\d)', r["b_response"]):
            extraction_artifacts.append(r["idx"])
    a["extraction_artifacts"] = extraction_artifacts
    a["n_fixed"] = len(fixed)

    # Wrong-answer \boxed{} rate: did the model learn to always emit \boxed{}
    # even when wrong? This is the real reward hacking vector.
    b_wrong = [r for r in rows if not r["b_correct"]]
    c_wrong = [r for r in rows if not r["c_correct"]]
    a["b_wrong_boxed"] = sum(1 for r in b_wrong if r["b_method"] == "boxed")
    a["b_wrong_total"] = len(b_wrong)
    a["c_wrong_boxed"] = sum(1 for r in c_wrong if r["c_method"] == "boxed")
    a["c_wrong_total"] = len(c_wrong)

    # Token stats
    b_tokens = [r["b_tokens"] for r in rows]
    c_tokens = [r["c_tokens"] for r in rows]
    a["b_token_avg"] = sum(b_tokens) / len(b_tokens)
    a["c_token_avg"] = sum(c_tokens) / len(c_tokens)
    a["b_token_med"] = median(b_tokens)
    a["c_token_med"] = median(c_tokens)

    # Entity breakdown
    entity_buckets = {"0": [], "1-2": [], "3+": []}
    for r in rows:
        ne = r["n_entities"]
        if ne == 0:
            entity_buckets["0"].append(r)
        elif ne <= 2:
            entity_buckets["1-2"].append(r)
        else:
            entity_buckets["3+"].append(r)
    a["entity_breakdown"] = {}
    for bucket, rr in entity_buckets.items():
        if not rr:
            a["entity_breakdown"][bucket] = {"n": 0, "b_acc": 0, "c_acc": 0}
            continue
        a["entity_breakdown"][bucket] = {
            "n": len(rr),
            "b_acc": sum(1 for r in rr if r["b_correct"]) / len(rr),
            "c_acc": sum(1 for r in rr if r["c_correct"]) / len(rr),
        }

    # Still-wrong analysis
    still_wrong = [r for r in rows if r["category"] == "still_wrong"]
    a["sw_same_answer"] = sum(1 for r in still_wrong if str(r["b_pred"]) == str(r["c_pred"]))
    a["sw_diff_answer"] = len(still_wrong) - a["sw_same_answer"]

    return a


# ── HTML generation ──────────────────────────────────────────────────────

def build_html(baseline_records, checkpoint_records):
    bg = key_by_idx(baseline_records)
    cp = key_by_idx(checkpoint_records)
    common = sorted(set(bg) & set(cp))

    rows = []
    for idx in common:
        b = bg[idx]
        c = cp[idx]
        entities = extract_entity_names(b["question"])

        b_correct = b["is_correct"]
        c_correct = c["is_correct"]

        if not b_correct and c_correct:
            cat = "fixed"
        elif b_correct and not c_correct:
            cat = "broken"
        elif b_correct and c_correct:
            cat = "still_right"
        elif not b_correct and not c_correct:
            cat = "still_wrong"
        else:
            cat = "unchanged"

        rows.append({
            "idx": idx,
            "question": b["question"],
            "gold_answer": b["gold_answer"],
            "gold_label": b["gold_label"],
            "entities": entities,
            "n_entities": len(entities),
            "b_correct": b_correct,
            "b_pred": b.get("predicted_answer", ""),
            "b_method": b.get("extraction_method", ""),
            "b_response": b.get("response", ""),
            "b_error": b.get("error_type", ""),
            "b_tokens": b.get("n_tokens", 0),
            "c_correct": c_correct,
            "c_pred": c.get("predicted_answer", ""),
            "c_method": c.get("extraction_method", ""),
            "c_response": c.get("response", ""),
            "c_error": c.get("error_type", ""),
            "c_tokens": c.get("n_tokens", 0),
            "category": cat,
        })

    cat_order = {"fixed": 0, "broken": 1, "still_wrong": 2, "still_right": 3}
    rows.sort(key=lambda r: (cat_order.get(r["category"], 9), r["idx"]))

    n_b = sum(1 for r in rows if r["b_correct"])
    n_c = sum(1 for r in rows if r["c_correct"])
    cat_counts = {}
    for r in rows:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    analysis = compute_analysis(rows)

    cat_colors = {
        "fixed": "#22c55e",
        "broken": "#ef4444",
        "still_right": "#60a5fa",
        "still_wrong": "#6b7280",
    }
    cat_labels = {
        "fixed": "Fixed (was wrong, now correct)",
        "still_right": "Still correct",
        "broken": "Broken (was correct, now wrong)",
        "still_wrong": "Still wrong",
    }

    parts = []

    # ── Head with MathJax ────────────────────────────────────────────────
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Baseline vs Checkpoint: Greedy Comparison</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'code'],
    ignoreHtmlClass: 'no-mathjax',
  },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady();
    }
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; line-height: 1.6; }
  .container { max-width: 1500px; margin: 0 auto; }
  h1 { font-size: 1.8em; margin-bottom: 4px; color: #f1f5f9; }
  h2 { font-size: 1.2em; color: #cbd5e1; margin: 24px 0 12px; }
  .subtitle { color: #94a3b8; margin-bottom: 20px; font-size: 0.95em; }

  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                   gap: 10px; margin-bottom: 20px; }
  .stat-card { background: #1e293b; border-radius: 10px; padding: 14px; text-align: center; }
  .stat-card .value { font-size: 1.8em; font-weight: 700; }
  .stat-card .label { font-size: 0.82em; color: #94a3b8; margin-top: 2px; }
  .stat-card .sublabel { font-size: 0.75em; color: #64748b; }

  .analysis-section { background: #1e293b; border-radius: 12px; padding: 20px;
                       margin-bottom: 20px; }
  .analysis-section h2 { margin-top: 0; font-size: 1.1em; }
  .analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
  @media (max-width: 800px) { .analysis-grid { grid-template-columns: 1fr; } }
  .analysis-table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
  .analysis-table th { text-align: left; color: #94a3b8; font-weight: 600;
                        padding: 6px 10px; border-bottom: 1px solid #334155; }
  .analysis-table td { padding: 5px 10px; border-bottom: 1px solid #1e293b; }
  .analysis-table .num { text-align: right; font-variant-numeric: tabular-nums; }
  .good { color: #4ade80; }
  .bad { color: #f87171; }
  .neutral { color: #94a3b8; }
  .verdict-box { background: #0f172a; border-radius: 8px; padding: 12px 16px;
                  margin-top: 12px; font-size: 0.9em; line-height: 1.7; }

  .cat-bar { display: flex; height: 32px; border-radius: 8px; overflow: hidden; margin-bottom: 20px; }
  .cat-segment { display: flex; align-items: center; justify-content: center;
                  font-size: 0.75em; font-weight: 600; color: #0f172a; min-width: 30px; }

  .filter-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .filter-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
                 padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85em;
                 transition: all 0.15s; }
  .filter-btn:hover { border-color: #60a5fa; color: #60a5fa; }
  .filter-btn.active { background: #1e40af; border-color: #3b82f6; color: #fff; }

  .problem { background: #1e293b; border-radius: 12px; margin-bottom: 14px;
              border-left: 4px solid #334155; overflow: hidden; }
  .problem.cat-fixed { border-left-color: #22c55e; }
  .problem.cat-broken { border-left-color: #ef4444; }
  .problem.cat-still_right { border-left-color: #60a5fa; }
  .problem.cat-still_wrong { border-left-color: #6b7280; }

  .problem-header { padding: 12px 16px; cursor: pointer; display: flex;
                     justify-content: space-between; align-items: center;
                     gap: 12px; }
  .problem-header:hover { background: #253348; }
  .header-left { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }
  .header-left .idx { font-weight: 700; color: #60a5fa; min-width: 55px; flex-shrink: 0;
                       font-size: 0.9em; }
  .header-left .q-preview { color: #cbd5e1; font-size: 0.88em; overflow: hidden;
                              text-overflow: ellipsis; white-space: nowrap; }
  .header-right { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
  .badge { padding: 2px 9px; border-radius: 12px; font-size: 0.76em; font-weight: 600;
           white-space: nowrap; }
  .badge-correct { background: #166534; color: #86efac; }
  .badge-wrong { background: #7f1d1d; color: #fca5a5; }
  .badge-entities { background: #312e81; color: #a5b4fc; }
  .badge-cat { border-radius: 12px; font-size: 0.76em; font-weight: 600; padding: 2px 9px; }
  .arrow { color: #64748b; transition: transform 0.15s; font-size: 1.1em; margin-left: 4px; }
  .problem.open .arrow { transform: rotate(90deg); }

  .problem-body { display: none; padding: 0 16px 16px; }
  .problem.open .problem-body { display: block; }

  .section-label { font-size: 0.78em; font-weight: 600; color: #64748b;
                    text-transform: uppercase; letter-spacing: 0.05em; margin: 10px 0 5px; }

  .question-text { background: #0f172a; padding: 12px; border-radius: 8px;
                    font-size: 0.9em; line-height: 1.7; color: #e2e8f0; }
  .gold-box { background: #14532d; padding: 10px 14px; border-radius: 8px;
               font-size: 0.86em; margin-top: 8px; }
  .gold-box .gold-answer { font-size: 1.2em; font-weight: 700; color: #4ade80; }
  .gold-steps { color: #86efac; font-size: 0.82em; margin-top: 4px; white-space: pre-wrap; }

  .annotation { background: #1a1a2e; border: 1px solid #334155; border-radius: 8px;
                 padding: 10px 14px; margin-top: 8px; font-size: 0.85em; }

  .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  @media (max-width: 900px) { .comparison { grid-template-columns: 1fr; } }

  .panel { background: #0f172a; border-radius: 8px; padding: 12px; }
  .panel h3 { font-size: 0.88em; margin-bottom: 6px; }
  .panel.baseline h3 { color: #94a3b8; }
  .panel.checkpoint h3 { color: #60a5fa; }
  .pred-line { font-size: 0.88em; margin-bottom: 8px; }

  .response-text { font-size: 0.84em; line-height: 1.7; color: #cbd5e1;
                    max-height: 600px; overflow-y: auto; word-break: break-word;
                    padding: 4px 0; }
  .response-text br + br { display: none; }
  .meta { display: flex; gap: 14px; margin-top: 6px; font-size: 0.78em; color: #64748b; }

  .entity-tags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
  .entity-tag { background: #312e81; color: #c4b5fd; padding: 1px 8px;
                 border-radius: 10px; font-size: 0.76em; }

  .toolbar { display: flex; gap: 10px; margin-bottom: 14px; }
  .toolbar button { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
                     padding: 7px 14px; border-radius: 8px; cursor: pointer; font-size: 0.88em; }
  .toolbar button:hover { border-color: #60a5fa; color: #60a5fa; }

  mjx-container { color: #e2e8f0 !important; }
</style>
</head>
<body>
<div class="container">
""")

    # ── Title + summary cards ────────────────────────────────────────────
    delta = n_c - n_b
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    delta_color = "#4ade80" if delta > 0 else "#f87171" if delta < 0 else "#94a3b8"

    parts.append(f"""
<h1>Baseline vs Checkpoint: Greedy Comparison</h1>
<p class="subtitle">
  Both evaluated with greedy decoding (temp=0, single response) on {len(rows)} GSM8K <b>test</b> problems (seed=42), no system prompt.
</p>
<details style="margin-bottom:16px;">
  <summary style="color:#94a3b8;cursor:pointer;font-size:0.85em;">Provenance &amp; Reproducibility</summary>
  <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-top:6px;font-size:0.82em;color:#94a3b8;line-height:1.8;">
    <b>Base model:</b> Qwen/Qwen2.5-1.5B-Instruct<br>
    <b>Checkpoint:</b> LoRA adapter, GRPO step 325 (run_20260215_202141/checkpoint-50)<br>
    <b>Training data:</b> GSM8K train split, filtered to 3+ named entities (~240 problems)<br>
    <b>Eval data:</b> GSM8K test split, 100 random problems (seed=42) &mdash; no overlap with training<br>
    <b>Baseline script:</b> <code>src/eval_baseline.py</code> (greedy, no system prompt, no LoRA)<br>
    <b>Checkpoint script:</b> <code>src/eval_checkpoint.py --greedy --no-system-prompt</code><br>
    <b>Reward design:</b> correctness (1.0) + format bonus (0.1 for \\boxed{{}})<br>
    <b>Training config:</b> GRPO, LoRA rank 16, lr=5e-5, beta=0.04, grad_accum=16, 500 max steps<br>
    <b>Note:</b> Checkpoint was trained WITH a system prompt ("Think step by step...") but evaluated WITHOUT it.
    The improvement held without the prompt, suggesting the model internalized the reasoning behavior.
  </div>
</details>

<div class="summary-grid">
  <div class="stat-card">
    <div class="value">{n_b}%</div>
    <div class="label">Baseline</div>
    <div class="sublabel">greedy pass@1</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#60a5fa">{n_c}%</div>
    <div class="label">Checkpoint</div>
    <div class="sublabel">greedy pass@1</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:{delta_color}">{delta_str}</div>
    <div class="label">Delta</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#22c55e">{cat_counts.get('fixed', 0)}</div>
    <div class="label">Fixed</div>
    <div class="sublabel">wrong &rarr; right</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#ef4444">{cat_counts.get('broken', 0)}</div>
    <div class="label">Broken</div>
    <div class="sublabel">right &rarr; wrong</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#60a5fa">{cat_counts.get('still_right', 0)}</div>
    <div class="label">Still Right</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:#6b7280">{cat_counts.get('still_wrong', 0)}</div>
    <div class="label">Still Wrong</div>
  </div>
</div>
""")

    # ── Category bar ─────────────────────────────────────────────────────
    parts.append('<div class="cat-bar">')
    for cat in ["fixed", "still_right", "still_wrong", "broken"]:
        n = cat_counts.get(cat, 0)
        if n == 0:
            continue
        pct = n / len(rows) * 100
        parts.append(f'<div class="cat-segment" style="width:{pct}%;background:{cat_colors[cat]}" '
                      f'title="{cat_labels[cat]}: {n}">{n}</div>')
    parts.append('</div>')

    # ── Reward Hacking Analysis ──────────────────────────────────────────
    a = analysis
    sw_total = cat_counts.get("still_wrong", 0)

    # Pre-compute wrong-boxed rates for display
    b_wrong_boxed_pct = (a['b_wrong_boxed'] / a['b_wrong_total'] * 100) if a['b_wrong_total'] else 0
    c_wrong_boxed_pct = (a['c_wrong_boxed'] / a['c_wrong_total'] * 100) if a['c_wrong_total'] else 0
    n_artifacts = len(a['extraction_artifacts'])

    parts.append(f"""
<div class="analysis-section">
  <h2>Reward Hacking Analysis</h2>
  <p style="color:#94a3b8;font-size:0.88em;margin-bottom:12px;">
    Training used correctness reward (1.0) + format reward (0.1 for \\boxed{{}}).
    Did the model learn real math or just better formatting?
  </p>
  <div class="analysis-grid">
    <div>
      <table class="analysis-table">
        <tr><th colspan="3">\\boxed{{}} Usage (All Answers)</th></tr>
        <tr><td>Baseline uses \\boxed{{}}</td><td class="num">{a['b_boxed']}</td><td class="num">/ {len(rows)}</td></tr>
        <tr><td>Checkpoint uses \\boxed{{}}</td><td class="num">{a['c_boxed']}</td><td class="num">/ {len(rows)}</td></tr>
        <tr><th colspan="3" style="padding-top:10px">\\boxed{{}} on Wrong Answers (reward hack vector)</th></tr>
        <tr><td>Baseline wrong answers using \\boxed{{}}</td>
            <td class="num">{a['b_wrong_boxed']}/{a['b_wrong_total']}</td>
            <td class="num">{b_wrong_boxed_pct:.0f}%</td></tr>
        <tr><td>Checkpoint wrong answers using \\boxed{{}}</td>
            <td class="num {'bad' if c_wrong_boxed_pct > b_wrong_boxed_pct + 20 else 'neutral'}">{a['c_wrong_boxed']}/{a['c_wrong_total']}</td>
            <td class="num {'bad' if c_wrong_boxed_pct > b_wrong_boxed_pct + 20 else 'neutral'}">{c_wrong_boxed_pct:.0f}%</td></tr>
        <tr><td>Gained \\boxed{{}} but still wrong</td>
            <td class="num {'bad' if a['format_only_wins'] > 3 else 'neutral'}">{a['format_only_wins']}</td>
            <td></td></tr>
        <tr><td>Gained \\boxed{{}} and now correct</td>
            <td class="num good">{a['format_and_correct']}</td>
            <td></td></tr>
      </table>
    </div>
    <div>
      <table class="analysis-table">
        <tr><th colspan="3">Token Length</th></tr>
        <tr><td>Baseline avg / median</td>
            <td class="num">{a['b_token_avg']:.0f}</td>
            <td class="num">{a['b_token_med']:.0f}</td></tr>
        <tr><td>Checkpoint avg / median</td>
            <td class="num">{a['c_token_avg']:.0f}</td>
            <td class="num">{a['c_token_med']:.0f}</td></tr>
        <tr><td>Delta (avg)</td>
            <td class="num {('good' if a['c_token_avg'] < a['b_token_avg'] else 'neutral')}">{a['c_token_avg'] - a['b_token_avg']:+.0f}</td>
            <td></td></tr>
        <tr><th colspan="3" style="padding-top:10px">Extraction Artifact Check</th></tr>
        <tr><td>"Fixed" problems where gold answer<br>was already in baseline text</td>
            <td class="num {'bad' if n_artifacts > 0 else 'good'}">{n_artifacts}</td>
            <td class="num">/ {a['n_fixed']}</td></tr>
      </table>
      <p style="color:#64748b;font-size:0.78em;margin-top:6px;">
        If high, some "fixes" may be the extractor catching answers that were always there
        but not in \\boxed{{}}.
      </p>
    </div>
  </div>

  <div class="analysis-grid" style="margin-top:16px;">
    <div>
      <table class="analysis-table">
        <tr><th>Entities</th><th class="num">N</th><th class="num">Base Acc</th><th class="num">Ckpt Acc</th><th class="num">Delta</th></tr>
""")
    for bucket in ["0", "1-2", "3+"]:
        eb = a["entity_breakdown"][bucket]
        if eb["n"] == 0:
            continue
        d = eb["c_acc"] - eb["b_acc"]
        small_n = eb["n"] < 15
        dc = "neutral" if small_n else ("good" if d > 0.02 else "bad" if d < -0.02 else "neutral")
        caveat = ' <span style="color:#64748b" title="Too few samples for reliable comparison">*</span>' if small_n else ""
        parts.append(f'        <tr><td>{bucket} entities{caveat}</td>'
                      f'<td class="num">{eb["n"]}</td>'
                      f'<td class="num">{eb["b_acc"]*100:.0f}%</td>'
                      f'<td class="num">{eb["c_acc"]*100:.0f}%</td>'
                      f'<td class="num {dc}">{d*100:+.0f}%</td></tr>\n')

    parts.append(f"""
      </table>
      <p style="color:#64748b;font-size:0.78em;margin-top:6px;">
        * Buckets with fewer than 15 problems &mdash; treat deltas as noise, not signal.
      </p>
    </div>
    <div>
      <table class="analysis-table">
        <tr><th colspan="2">"Still Wrong" Breakdown ({sw_total} problems)</th></tr>
        <tr><td>Same wrong answer</td><td class="num">{a['sw_same_answer']}</td></tr>
        <tr><td>Different wrong answer</td><td class="num">{a['sw_diff_answer']}</td></tr>
      </table>
    </div>
  </div>

  <div class="verdict-box">
""")

    # Verdict logic — proportional to magnitude of change
    hack_score = a["format_only_wins"]
    n_fixed = cat_counts.get("fixed", 0)
    n_broken = cat_counts.get("broken", 0)
    real_gain = n_fixed - n_broken
    hack_fraction = hack_score / max(n_fixed, 1)
    wrong_boxed_jump = c_wrong_boxed_pct - b_wrong_boxed_pct

    if real_gain > 0 and hack_fraction < 0.3 and wrong_boxed_jump < 30:
        parts.append(f'<span class="good"><b>Verdict: Likely real improvement.</b></span> '
                      f'Net {real_gain} problems fixed ({n_fixed} fixed, {n_broken} broken). '
                      f'{hack_score} format-only gains ({hack_fraction*100:.0f}% of fixes). '
                      f'Wrong-answer \\boxed{{}} rate: {b_wrong_boxed_pct:.0f}% &rarr; {c_wrong_boxed_pct:.0f}%.')
    elif hack_fraction >= 0.5 or (real_gain <= 0 and hack_score > 0):
        parts.append(f'<span class="bad"><b>Verdict: Likely reward hacking.</b></span> '
                      f'{hack_score} format-only gains ({hack_fraction*100:.0f}% of fixes). '
                      f'Net accuracy change: {real_gain:+d}. '
                      f'Wrong-answer \\boxed{{}} rate: {b_wrong_boxed_pct:.0f}% &rarr; {c_wrong_boxed_pct:.0f}%.')
    else:
        parts.append(f'<span class="neutral"><b>Verdict: Mixed signal.</b></span> '
                      f'Net {real_gain} fixed, {hack_score} format-only gains ({hack_fraction*100:.0f}% of fixes). '
                      f'Wrong-answer \\boxed{{}} rate: {b_wrong_boxed_pct:.0f}% &rarr; {c_wrong_boxed_pct:.0f}%.')

    parts.append("""
  </div>
</div>
""")

    # ── Extraction method comparison ─────────────────────────────────────
    parts.append("""
<div class="analysis-section">
  <h2>Extraction Method Comparison</h2>
  <p style="color:#94a3b8;font-size:0.88em;margin-bottom:12px;">
    How the model formats its final answer, and accuracy for each method.
  </p>
  <div class="analysis-grid">
    <div>
      <table class="analysis-table">
        <tr><th>Method</th><th class="num">Baseline Count</th><th class="num">Accuracy</th></tr>
""")
    for m in sorted(a["b_methods"], key=lambda x: -a["b_methods"][x]):
        acc = a["b_acc_by_method"].get(m, 0)
        parts.append(f'        <tr><td><code>{m}</code></td>'
                      f'<td class="num">{a["b_methods"][m]}</td>'
                      f'<td class="num">{acc*100:.0f}%</td></tr>\n')
    parts.append("""
      </table>
    </div>
    <div>
      <table class="analysis-table">
        <tr><th>Method</th><th class="num">Checkpoint Count</th><th class="num">Accuracy</th></tr>
""")
    for m in sorted(a["c_methods"], key=lambda x: -a["c_methods"][x]):
        acc = a["c_acc_by_method"].get(m, 0)
        parts.append(f'        <tr><td><code>{m}</code></td>'
                      f'<td class="num">{a["c_methods"][m]}</td>'
                      f'<td class="num">{acc*100:.0f}%</td></tr>\n')
    parts.append("""
      </table>
    </div>
  </div>
</div>
""")

    # ── Filter bar + toolbar ─────────────────────────────────────────────
    parts.append('<div class="filter-bar">')
    parts.append('<button class="filter-btn active" onclick="filterProblems(\'all\')">All</button>')
    for cat in ["fixed", "broken", "still_right", "still_wrong"]:
        n = cat_counts.get(cat, 0)
        if n == 0:
            continue
        parts.append(f'<button class="filter-btn" onclick="filterProblems(\'{cat}\')">'
                      f'{cat_labels[cat]} ({n})</button>')
    parts.append('</div>')
    parts.append('<div class="toolbar">')
    parts.append('<button onclick="toggleAll()">Expand / Collapse All</button>')
    parts.append('</div>')

    # ── Problem cards ────────────────────────────────────────────────────
    for r in rows:
        q_preview = r["question"][:110] + ("..." if len(r["question"]) > 110 else "")
        b_badge = "badge-correct" if r["b_correct"] else "badge-wrong"
        b_label = "Correct" if r["b_correct"] else "Wrong"
        c_badge = "badge-correct" if r["c_correct"] else "badge-wrong"
        c_label = "Correct" if r["c_correct"] else "Wrong"
        cat_color = cat_colors.get(r["category"], "#9ca3af")

        entity_tags = "".join(
            f'<span class="entity-tag">{html.escape(e)}</span>' for e in r["entities"]
        )
        if not entity_tags:
            entity_tags = '<span style="color:#64748b;font-size:0.82em">no named entities</span>'

        token_delta = r["c_tokens"] - r["b_tokens"]
        token_delta_str = f"{token_delta:+d} tokens" if token_delta != 0 else "same length"

        # Build annotation for still_wrong
        annotation_html = ""
        if r["category"] == "still_wrong":
            b_pred_s = str(r["b_pred"]) if r["b_pred"] else "none"
            c_pred_s = str(r["c_pred"]) if r["c_pred"] else "none"
            if b_pred_s == c_pred_s:
                ann = f"Same wrong answer: <b>{html.escape(b_pred_s)}</b> (gold: {html.escape(r['gold_answer'])})"
            else:
                ann = (f"Answer changed: <b>{html.escape(b_pred_s)}</b> &rarr; "
                       f"<b>{html.escape(c_pred_s)}</b> (gold: {html.escape(r['gold_answer'])})")

            method_change = ""
            if r["b_method"] != r["c_method"]:
                method_change = (f" | Method: <code>{r['b_method']}</code> &rarr; "
                                  f"<code>{r['c_method']}</code>")

            annotation_html = f'<div class="annotation">{ann}{method_change}</div>'

        elif r["category"] == "fixed":
            b_pred_s = str(r["b_pred"]) if r["b_pred"] else "none"
            is_artifact = r["idx"] in a["extraction_artifacts"]
            artifact_note = (' <span class="bad" title="The correct answer appeared in the baseline '
                             'response but was not extracted — this may be a formatting improvement, '
                             'not a reasoning one.">&#9888; extraction artifact?</span>') if is_artifact else ""
            annotation_html = (
                f'<div class="annotation">'
                f'<span class="good">Now correct.</span> '
                f'Baseline predicted <b>{html.escape(b_pred_s)}</b> (via <code>{r["b_method"]}</code>), '
                f'checkpoint gets <b>{html.escape(r["gold_answer"])}</b> (via <code>{r["c_method"]}</code>).'
                f'{artifact_note}'
                f'</div>'
            )

        elif r["category"] == "broken":
            c_pred_s = str(r["c_pred"]) if r["c_pred"] else "none"
            annotation_html = (
                f'<div class="annotation">'
                f'<span class="bad">Regression.</span> '
                f'Baseline was correct ({html.escape(r["gold_answer"])}), '
                f'checkpoint predicts <b>{html.escape(c_pred_s)}</b>.'
                f'</div>'
            )

        escaped_b = escape_for_mathjax(r["b_response"])
        escaped_c = escape_for_mathjax(r["c_response"])
        escaped_gold = html.escape(r["gold_label"])

        parts.append(f"""
<div class="problem cat-{r['category']}" data-cat="{r['category']}">
  <div class="problem-header" onclick="this.parentElement.classList.toggle('open')">
    <div class="header-left">
      <span class="idx">#{r['idx']}</span>
      <span class="q-preview">{html.escape(q_preview)}</span>
    </div>
    <div class="header-right">
      <span class="badge badge-entities">{r['n_entities']} ent</span>
      <span class="badge {b_badge}">Base: {b_label}</span>
      <span class="badge {c_badge}">Ckpt: {c_label}</span>
      <span class="badge-cat" style="background:{cat_color}30;color:{cat_color}">
        {r['category'].replace('_', ' ').title()}
      </span>
      <span class="arrow">&#9654;</span>
    </div>
  </div>
  <div class="problem-body">
    <div class="section-label">Question</div>
    <div class="question-text">{html.escape(r['question'])}</div>
    <div class="entity-tags" style="margin-top:5px">{entity_tags}</div>

    <div class="gold-box">
      <div>Gold: <span class="gold-answer">{html.escape(r['gold_answer'])}</span></div>
      <div class="gold-steps">{escaped_gold}</div>
    </div>

    {annotation_html}

    <div class="comparison">
      <div class="panel baseline">
        <h3>Baseline (Greedy)</h3>
        <div class="pred-line">
          Predicted: <b>{html.escape(str(r['b_pred']))}</b>
          <span class="badge {b_badge}" style="margin-left:6px">{b_label}</span>
          <span style="color:#64748b;font-size:0.8em;margin-left:6px">via {r['b_method']}</span>
        </div>
        <div class="response-text">{escaped_b}</div>
        <div class="meta"><span>{r['b_tokens']} tokens</span></div>
      </div>
      <div class="panel checkpoint">
        <h3>Checkpoint (Greedy)</h3>
        <div class="pred-line">
          Predicted: <b>{html.escape(str(r['c_pred']))}</b>
          <span class="badge {c_badge}" style="margin-left:6px">{c_label}</span>
          <span style="color:#64748b;font-size:0.8em;margin-left:6px">via {r['c_method']}</span>
          <span style="color:#64748b;font-size:0.78em;margin-left:8px">({token_delta_str})</span>
        </div>
        <div class="response-text">{escaped_c}</div>
        <div class="meta"><span>{r['c_tokens']} tokens</span></div>
      </div>
    </div>
  </div>
</div>
""")

    # ── Footer scripts ───────────────────────────────────────────────────
    parts.append("""
</div>
<script>
function toggleAll() {
  const problems = document.querySelectorAll('.problem');
  const anyOpen = [...problems].some(p => p.classList.contains('open') && p.style.display !== 'none');
  problems.forEach(p => {
    if (anyOpen) p.classList.remove('open');
    else { p.classList.add('open'); }
  });
  if (typeof MathJax !== 'undefined') MathJax.typeset();
}
function filterProblems(cat) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.problem').forEach(p => {
    p.style.display = (cat === 'all' || p.dataset.cat === cat) ? '' : 'none';
  });
}
document.addEventListener('click', function(e) {
  const header = e.target.closest('.problem-header');
  if (!header) return;
  setTimeout(() => { if (typeof MathJax !== 'undefined') MathJax.typeset(); }, 50);
});
</script>
</body>
</html>
""")

    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/mxu/Downloads/baseline_eval.jsonl"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/mxu/Downloads/checkpoint_eval.jsonl"

    print(f"Loading baseline:   {baseline_path}")
    baseline = load_jsonl(baseline_path)
    print(f"  {len(baseline)} records")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_jsonl(checkpoint_path)
    print(f"  {len(checkpoint)} records")

    html_content = build_html(baseline, checkpoint)

    os.makedirs("analysis", exist_ok=True)
    output_path = "analysis/baseline_vs_checkpoint.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"\nReport saved to {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
