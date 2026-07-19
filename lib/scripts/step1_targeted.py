"""
step1_targeted.py — Sample real LLM responses for targeted failure-mode questions.

Unlike _append_synthetic_data.py (which wrote hand-crafted responses), this
script feeds the targeted questions through the ACTUAL model at temperature=1.0,
collects 10 sampled responses per question, judges them with Gemini, and writes
results in the same format as step1_collect_data.py.

WHY this matters:
  The hallucination probe is trained on CETT features extracted from a FORWARD
  PASS of the model on the stored response text.  A model reading a hand-written
  sentence produces different FFN activations than a model that GENERATED that
  sentence during autoregressive decoding.  Only model-generated text will place
  the CETT vectors in the correct region of the 458k-dim feature space.

Targeted question categories:
  A. Exact constants / formulas — should almost always be true.
     Goal: teach the probe that precise numerical answers are NOT hallucinations.
  B. Non-existent API / methods — correct disclaimer = true, confabulation = false.
     Goal: teach the probe that honest "this doesn't exist" language is faithful.
  C. Warranted hedging — correct hedged answers = true.
     Goal: break the spurious hedge-language ↔ hallucination correlation.
  D. False-premise / trap questions — correct premise rejection = true, acceptance = false.
     Goal: fix false negatives on questions with a hidden wrong assumption.
  E. Time-stable historical facts — should always be true.
     Goal: anchor the probe on settled facts so temporal uncertainty is not over-weighted.

Output appended to:  data/consistency_samples.jsonl  (same file as step 1)

Run:
  python scripts/step1_targeted.py

  # If you need to skip SSL verification:
  #   DISABLE_SSL_VERIFY=1 python scripts/step1_targeted.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from src.data_pipeline import HFSampler
from src.judge import GeminiJudge
from src.utils import load_existing_qids


# ── Targeted questions ────────────────────────────────────────────────────────
# Each entry:
#   qid          — unique ID (must not conflict with existing data)
#   question     — the question string (WITHOUT the suffix; it is appended below)
#   ground_truth — list of correct answer strings for the Gemini judge
#
# NOTE: The suffix "Respond with the answer only, without any explanation."
#       is appended automatically during sampling, matching step 1 behaviour.

TARGETED_QUESTIONS = [

    # ── GROUP A: Exact constants / formulas ──────────────────────────────────
    # The model at temperature=1.0 will write these in varied formats
    # (e.g. "299,792,458 m/s" vs "~3×10⁸ m/s"), triggering spurious
    # inconsistency labels.  Sampling real responses captures that variation
    # and lets the judge label all variants as "true".

    {"qid": "tgt_const_001", "question": "What is the speed of light in a vacuum?",
     "ground_truth": ["299792458 m/s", "299,792,458 m/s", "3×10⁸ m/s", "3 × 10^8 m/s",
                      "approximately 300000 km/s", "186282 miles per second"]},
    {"qid": "tgt_const_002", "question": "What is the boiling point of water at sea level in Celsius?",
     "ground_truth": ["100", "100°C", "100 degrees Celsius", "100 Celsius"]},
    {"qid": "tgt_const_003", "question": "What is the chemical formula for water?",
     "ground_truth": ["H2O", "H₂O"]},
    {"qid": "tgt_const_004", "question": "What is the value of pi to two decimal places?",
     "ground_truth": ["3.14", "3.14159"]},
    {"qid": "tgt_const_005", "question": "What is the square root of 144?",
     "ground_truth": ["12", "twelve"]},
    {"qid": "tgt_const_006", "question": "How many sides does a hexagon have?",
     "ground_truth": ["6", "six"]},
    {"qid": "tgt_const_007", "question": "What is Earth's gravitational acceleration at the surface?",
     "ground_truth": ["9.8 m/s²", "9.81 m/s²", "9.8", "9.81",
                      "approximately 9.8 metres per second squared"]},
    {"qid": "tgt_const_008", "question": "What is the chemical symbol for gold?",
     "ground_truth": ["Au", "au"]},
    {"qid": "tgt_const_009", "question": "What is the chemical symbol for oxygen?",
     "ground_truth": ["O", "o"]},
    {"qid": "tgt_const_010", "question": "What is the atomic number of carbon?",
     "ground_truth": ["6", "six"]},
    {"qid": "tgt_const_011", "question": "What is the chemical formula for carbon dioxide?",
     "ground_truth": ["CO2", "CO₂"]},
    {"qid": "tgt_const_012", "question": "How many degrees are in a right angle?",
     "ground_truth": ["90", "90 degrees", "90°"]},
    {"qid": "tgt_const_013", "question": "What is the chemical symbol for sodium?",
     "ground_truth": ["Na", "na"]},
    {"qid": "tgt_const_014", "question": "How many planets are in our Solar System?",
     "ground_truth": ["8", "eight"]},
    {"qid": "tgt_const_015", "question": "What is the atomic number of hydrogen?",
     "ground_truth": ["1", "one"]},

    # ── GROUP B: Non-existent API / methods ──────────────────────────────────
    # At temperature=1.0 the model will sometimes confabulate plausible-sounding
    # descriptions of these non-existent functions (false), and sometimes
    # correctly disclaim them (true).  Both are needed for contrastive training.

    {"qid": "tgt_noexist_001", "question": "What does the Python built-in method list.flatten() do?",
     "ground_truth": ["does not exist", "not a built-in", "no such method",
                      "Python has no list.flatten", "there is no list.flatten"]},
    {"qid": "tgt_noexist_002", "question": "What does the Python function torch.optimize_memory() do?",
     "ground_truth": ["does not exist", "not a PyTorch function",
                      "no such function in PyTorch", "I couldn't find",
                      "this function does not exist"]},
    {"qid": "tgt_noexist_003", "question": "What does the JavaScript Array method array.squish() do?",
     "ground_truth": ["does not exist", "not a standard method", "no such method",
                      "there is no array.squish"]},
    {"qid": "tgt_noexist_004", "question": "What does the JavaScript method String.truncate() do?",
     "ground_truth": ["does not exist", "not a native method", "not in JavaScript",
                      "there is no String.truncate"]},
    {"qid": "tgt_noexist_005", "question": "What does the Python built-in dict.sort() method do?",
     "ground_truth": ["does not exist", "use sorted()", "no dict.sort",
                      "Python dict has no sort method"]},
    {"qid": "tgt_noexist_006", "question": "What does the Python string method str.reverse() do?",
     "ground_truth": ["does not exist", "use s[::-1]", "no str.reverse",
                      "Python strings have no reverse method"]},
    {"qid": "tgt_noexist_007", "question": "What does the Python method set.append() do?",
     "ground_truth": ["does not exist", "use set.add()", "no set.append",
                      "Python sets have no append method"]},
    {"qid": "tgt_noexist_008", "question": "What does the JavaScript Array.prototype.shuffle() method do?",
     "ground_truth": ["does not exist", "not a standard method",
                      "there is no Array.prototype.shuffle"]},
    {"qid": "tgt_noexist_009", "question": "What does numpy.matrix_inverse() do?",
     "ground_truth": ["does not exist", "use numpy.linalg.inv()",
                      "there is no numpy.matrix_inverse"]},
    {"qid": "tgt_noexist_010", "question": "What does the CSS property text-decoration-bold do?",
     "ground_truth": ["does not exist", "not a valid CSS property",
                      "there is no text-decoration-bold"]},
    {"qid": "tgt_noexist_011", "question": "What does the JavaScript method array.flattenDeep() do?",
     "ground_truth": ["Lodash function", "not a native JavaScript method",
                      "_.flattenDeep", "not built-in", "Lodash"]},
    {"qid": "tgt_noexist_012", "question": "What does the Python function os.file_exists() do?",
     "ground_truth": ["does not exist", "use os.path.exists()",
                      "there is no os.file_exists"]},
    {"qid": "tgt_noexist_013", "question": "What does the JavaScript method Object.freeze_deep() do?",
     "ground_truth": ["does not exist", "not a standard JavaScript method",
                      "there is no Object.freeze_deep"]},

    # ── GROUP C: Warranted hedging ────────────────────────────────────────────
    # For these questions the CORRECT answer involves "it depends", "consult a
    # professional", or domain-specific caveats.  The model uses hedging language
    # not because it is hallucinating but because the truth IS hedged.

    {"qid": "tgt_hedge_001",
     "question": "Is it legal to record a phone call without telling the other person?",
     "ground_truth": ["it depends", "depends on jurisdiction",
                      "varies by location", "one-party consent", "two-party consent"]},
    {"qid": "tgt_hedge_002",
     "question": "What is the safe dosage of aspirin for a child?",
     "ground_truth": ["not recommended", "Reye's syndrome", "consult a doctor",
                      "not safe for children", "aspirin is not"]},
    {"qid": "tgt_hedge_003",
     "question": "What are the tax implications of selling cryptocurrency?",
     "ground_truth": ["it depends", "varies by jurisdiction",
                      "consult a tax professional", "capital gains tax",
                      "depends on your country"]},
    {"qid": "tgt_hedge_004",
     "question": "Should I invest in index funds or individual stocks?",
     "ground_truth": ["it depends", "depends on your risk", "financial advisor",
                      "individual circumstances", "risk tolerance"]},
    {"qid": "tgt_hedge_005",
     "question": "Is it safe to drive after taking antihistamines?",
     "ground_truth": ["it depends", "some cause drowsiness",
                      "first-generation", "check the label",
                      "sedating antihistamines"]},
    {"qid": "tgt_hedge_006",
     "question": "What is the right amount of exercise per week for weight loss?",
     "ground_truth": ["it depends", "150", "300 minutes",
                      "varies by individual", "moderate intensity"]},
    {"qid": "tgt_hedge_007",
     "question": "Is Python always better than C++ for performance-critical applications?",
     "ground_truth": ["no", "C++ is faster", "C++ generally outperforms",
                      "Python is slower", "not always"]},
    {"qid": "tgt_hedge_008",
     "question": "Does drinking more water always help with weight loss?",
     "ground_truth": ["not necessarily", "not always", "it can help",
                      "depends on diet", "calorie deficit"]},
    {"qid": "tgt_hedge_009",
     "question": "Can you copyright an idea?",
     "ground_truth": ["no", "copyright protects expression",
                      "ideas cannot be copyrighted",
                      "idea-expression dichotomy"]},
    {"qid": "tgt_hedge_010",
     "question": "Is a SQL injection attack possible on a parameterised query?",
     "ground_truth": ["no", "parameterised queries prevent",
                      "not if correctly implemented",
                      "prevents SQL injection"]},
    {"qid": "tgt_hedge_011",
     "question": "Does correlation imply causation?",
     "ground_truth": ["no", "correlation does not imply causation",
                      "does not mean one causes"]},
    {"qid": "tgt_hedge_012",
     "question": "Is it safe to eat raw chicken?",
     "ground_truth": ["no", "Salmonella", "Campylobacter",
                      "not safe", "harmful bacteria"]},
    {"qid": "tgt_hedge_013",
     "question": "Is it safe to mix bleach and vinegar for cleaning?",
     "ground_truth": ["no", "chlorine gas", "toxic", "dangerous",
                      "do not mix"]},

    # ── GROUP D: False-premise / trap questions ──────────────────────────────
    # The model at temperature=1.0 will sometimes accept the false premise
    # (hallucination), and sometimes reject it (faithful).  This gives the probe
    # genuine contrastive examples: confident-but-wrong vs. corrective.

    {"qid": "tgt_trap_001",
     "question": "Why does the Sun orbit the Earth?",
     "ground_truth": ["does not orbit", "Earth orbits the Sun",
                      "incorrect premise", "false premise",
                      "the Sun does not"]},
    {"qid": "tgt_trap_002",
     "question": "Is the capital of Australia Sydney or Melbourne?",
     "ground_truth": ["neither", "Canberra",
                      "Canberra is the capital", "not Sydney or Melbourne"]},
    {"qid": "tgt_trap_003",
     "question": "What is the default port of MongoDB when accessed over HTTPS?",
     "ground_truth": ["MongoDB does not use HTTPS",
                      "27017 over TCP", "does not natively support HTTPS",
                      "no HTTPS default"]},
    {"qid": "tgt_trap_004",
     "question": "Why did Einstein fail mathematics at school?",
     "ground_truth": ["he did not fail", "this is a myth",
                      "Einstein excelled at mathematics",
                      "did not fail mathematics"]},
    {"qid": "tgt_trap_005",
     "question": "Is the Great Wall of China visible from space with the naked eye?",
     "ground_truth": ["no", "not visible from space",
                      "this is a myth", "common misconception",
                      "too narrow"]},
    {"qid": "tgt_trap_006",
     "question": "How many moons does the Sun have?",
     "ground_truth": ["zero", "none", "the Sun does not have moons",
                      "stars do not have moons"]},
    {"qid": "tgt_trap_007",
     "question": "Does lightning never strike the same place twice?",
     "ground_truth": ["false", "this is a myth", "lightning can strike",
                      "frequently strikes the same place"]},
    {"qid": "tgt_trap_008",
     "question": "Does vitamin C prevent the common cold?",
     "ground_truth": ["no", "not reliably", "not supported by evidence",
                      "does not significantly prevent"]},
    {"qid": "tgt_trap_009",
     "question": "Is sugar the primary cause of hyperactivity in children?",
     "ground_truth": ["no", "this is a myth",
                      "not supported by evidence", "not supported by research",
                      "studies have found no link"]},
    {"qid": "tgt_trap_010",
     "question": "How much of the brain does a human typically use?",
     "ground_truth": ["all", "virtually all", "the 10% myth",
                      "10% is a myth", "essentially 100%"]},
    {"qid": "tgt_trap_011",
     "question": "Did Christopher Columbus prove the Earth was round?",
     "ground_truth": ["no", "this is a myth",
                      "already known", "already accepted",
                      "knew the Earth was spherical"]},
    {"qid": "tgt_trap_012",
     "question": "What colour is a polar bear's fur?",
     "ground_truth": ["transparent", "colourless", "appears white",
                      "not actually white", "hollow"]},

    # ── GROUP E: Time-stable historical facts ────────────────────────────────
    # These are settled facts where the model should answer consistently.
    # At temperature=1.0 most responses will be correct; varied phrasings
    # teach the probe that numerical/named-entity precision is faithful.

    {"qid": "tgt_hist_001", "question": "In which year did World War II end?",
     "ground_truth": ["1945", "nineteen forty-five"]},
    {"qid": "tgt_hist_002", "question": "Who was the first person to walk on the Moon?",
     "ground_truth": ["Neil Armstrong", "Neil Alden Armstrong"]},
    {"qid": "tgt_hist_003", "question": "Which country won the 2022 FIFA World Cup?",
     "ground_truth": ["Argentina", "Argentine Republic"]},
    {"qid": "tgt_hist_004", "question": "In which year did the Berlin Wall fall?",
     "ground_truth": ["1989", "nineteen eighty-nine"]},
    {"qid": "tgt_hist_005", "question": "Who invented the telephone?",
     "ground_truth": ["Alexander Graham Bell", "Bell"]},
    {"qid": "tgt_hist_006", "question": "Which war ended in 1918?",
     "ground_truth": ["World War I", "World War One", "WWI", "First World War", "The Great War"]},
    {"qid": "tgt_hist_007", "question": "Who wrote the play Hamlet?",
     "ground_truth": ["William Shakespeare", "Shakespeare"]},
    {"qid": "tgt_hist_008", "question": "What is the capital of France?",
     "ground_truth": ["Paris", "Paris France"]},
    {"qid": "tgt_hist_009", "question": "In which year did humans first land on the Moon?",
     "ground_truth": ["1969", "nineteen sixty-nine"]},
    {"qid": "tgt_hist_010", "question": "Who was the first President of the United States?",
     "ground_truth": ["George Washington", "Washington"]},
    {"qid": "tgt_hist_011", "question": "In which year did the Titanic sink?",
     "ground_truth": ["1912", "nineteen twelve"]},
    {"qid": "tgt_hist_012", "question": "Which scientist developed the theory of general relativity?",
     "ground_truth": ["Albert Einstein", "Einstein"]},
    {"qid": "tgt_hist_013", "question": "What is the capital of Japan?",
     "ground_truth": ["Tokyo", "Tokyo Japan"]},
    {"qid": "tgt_hist_014", "question": "In which year did the Soviet Union dissolve?",
     "ground_truth": ["1991", "nineteen ninety-one"]},
    {"qid": "tgt_hist_015", "question": "Which company created the iPhone?",
     "ground_truth": ["Apple", "Apple Inc"]},
    {"qid": "tgt_hist_016", "question": "What is the tallest mountain in the world?",
     "ground_truth": ["Mount Everest", "Everest"]},
    {"qid": "tgt_hist_017", "question": "Who invented the World Wide Web?",
     "ground_truth": ["Tim Berners-Lee", "Timothy Berners-Lee", "Berners-Lee"]},
    {"qid": "tgt_hist_018", "question": "Which country was Nelson Mandela president of?",
     "ground_truth": ["South Africa", "Republic of South Africa"]},
    {"qid": "tgt_hist_019", "question": "Who sang the song Thriller?",
     "ground_truth": ["Michael Jackson", "MJ"]},
    {"qid": "tgt_hist_020", "question": "What element did Marie Curie discover?",
     "ground_truth": ["polonium", "radium", "polonium and radium"]},
]


# ── Helper — rule-based judge for cases where Gemini is unavailable ──────────

def _rule_judge(response: str, ground_truths: list[str]) -> str:
    """
    Simple substring-match fallback judge.
    Returns "true" if any ground-truth string appears in the response (case-insensitive).
    """
    resp_lower = response.lower()
    for gt in ground_truths:
        if gt.lower() in resp_lower:
            return "true"
    return "false"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load already-processed QIDs to allow safe resume
    processed_qids = load_existing_qids(config.OUTPUT_PATH)
    remaining = [q for q in TARGETED_QUESTIONS if q["qid"] not in processed_qids]

    if not remaining:
        print("All targeted questions already processed.")
        return

    print(f"Targeted questions to process: {len(remaining)} of {len(TARGETED_QUESTIONS)}")

    # Initialise sampler (loads the LLM)
    sampler = HFSampler()

    # Initialise judge
    use_llm_judge = config.JUDGE_TYPE == "llm" and bool(config.GEMINI_API_KEY)
    judge = GeminiJudge(
        api_key=config.GEMINI_API_KEY,
        model_name=config.JUDGE_MODEL,
    ) if use_llm_judge else None
    if not use_llm_judge:
        print("Using rule-based judge (set JUDGE_TYPE='llm' and GEMINI_API_KEY for Gemini).")

    suffix = "Respond with the answer only, without any explanation."
    written = 0

    for entry in remaining:
        qid   = entry["qid"]
        q     = entry["question"]
        gts   = entry["ground_truth"]
        full_q = f"{q} {suffix}"

        messages = [{"role": "user", "content": full_q}]
        print(f"\n[{qid}] {q}")

        responses = sampler.sample(messages, n=config.SAMPLE_NUM)

        # Judge
        if use_llm_judge:
            items = [{"question": q, "response": r, "answers": ", ".join(gts)}
                     for r in responses]
            judges = judge.judge_batch(items)
        else:
            judges = [_rule_judge(r, gts) for r in responses]

        # Skip uncertain/error labels only in LLM-judge mode (rule judge never returns them)
        judges = [j if j in ("true", "false") else "uncertain" for j in judges]

        true_count  = judges.count("true")
        false_count = judges.count("false")
        print(f"  → true: {true_count}  false: {false_count}  other: {judges.count('uncertain')}")

        record = {
            qid: {
                "question":     full_q,
                "responses":    responses,
                "judges":       judges,
                "ground_truth": gts,
            }
        }

        with open(config.OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        written += 1

    print(f"\nDone. Wrote {written} targeted entries to {config.OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. Re-run step2 to extract CETT activations (including these new entries)")
    print("  2. Re-run step3 to retrain the probe with the expanded dataset")
    print("  3. Re-run step4 demo to verify improved behaviour")


if __name__ == "__main__":
    main()
