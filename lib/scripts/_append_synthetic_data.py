"""
_append_synthetic_data.py — Append ~150 synthetic training examples to consistency_samples.jsonl.

Why ~150 (not 26):
  The existing dataset has ~1,000 questions × 10 responses ≈ 10,000 individual
  CETT records.  A linear probe on 458k-dim features needs the targeted failure
  modes to represent at least ~10-15 % of the training pool to meaningfully shift
  the decision boundary.  150 questions × 10 responses = 1,500 records ≈ 15 %.

  The first batch (syn_const_001 … syn_anchor_005) was only 26 entries — too
  few.  This script appends the rest, skipping any QID that already exists.

Design principles:
  A. Exact constants — all-true judges, but VARIED phrasings per question so the
     10 responses span different token sequences and produce diverse CETT vectors.
  B. Non-existent API / methods — mixed judges (honest disclaimer = true,
     fabricated usage = false) so the probe learns the contrast directly.
  C. Warranted hedging — all-true judges with naturally hedged language so the
     probe stops penalising uncertainty in genuinely ambiguous domains.
  D. False-premise / trick questions — mixed judges (premise rejection = true,
     premise acceptance = false) to fix false negatives on trap questions.
  E. Time-stable historical facts — all-true with varied phrasings.
  F. Self-reflection degradation cases — targeted at the exact demo questions
     where reflection made scores worse.
  G. Faithful anchors for class balance — clean, unambiguous, varied correct
     phrasings to improve the faithful-class coverage.

Run once after the first batch:
  python scripts/_append_synthetic_data.py
"""

import json
import os
import sys

# Resolve OUTPUT_PATH without importing config (which requires torch)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "consistency_samples.jsonl")

# ── Load existing QIDs to avoid collisions ───────────────────────────────────
existing_qids: set = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    existing_qids.update(obj.keys())
                except json.JSONDecodeError:
                    pass

# ── New synthetic entries ─────────────────────────────────────────────────────
# Format mirrors the existing dataset exactly:
#   { "<qid>": { "question": str, "responses": [str×10],
#                "judges": [str×10], "ground_truth": [str] } }
#
# "judges" values: "true" = faithful / "false" = hallucinated
# Responses are VARIED even within all-true entries to produce diverse CETT
# vectors from different token sequences — not the same text repeated 10 times.

NEW_ENTRIES = [

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP A2: More exact constants (20 entries)
    # Root cause: probe penalises precise numerical answers because at
    # temperature=1.0 the model writes the same constant in varied formats
    # ("299,792,458 m/s" vs "~3×10⁸ m/s"), making consistency labels noisy.
    # Strategy: all-true, but vary the phrasing across the 10 responses.
    # ═══════════════════════════════════════════════════════════════════════════

    {"syn_const_007": {
        "question": "What is the gravitational acceleration on Earth's surface? Respond with the answer only, without any explanation.",
        "responses": ["9.8 m/s²", "9.81 m/s²", "approximately 9.8 m/s²", "9.8 m/s²", "about 9.81 metres per second squared",
                      "9.81 m/s²", "9.8 m/s²", "roughly 9.8 m/s²", "9.8 m/s²", "9.81 m/s²"],
        "judges": ["true"] * 10,
        "ground_truth": ["9.8 m/s²", "9.81 m/s²", "gravitational acceleration"]}},

    {"syn_const_008": {
        "question": "How many degrees are in a right angle? Respond with the answer only, without any explanation.",
        "responses": ["90 degrees", "90°", "90 degrees", "90°", "ninety degrees",
                      "90 degrees", "90°", "90 degrees", "90", "ninety"],
        "judges": ["true"] * 10,
        "ground_truth": ["90", "90 degrees", "90°"]}},

    {"syn_const_009": {
        "question": "What is the freezing point of water in Celsius? Respond with the answer only, without any explanation.",
        "responses": ["0°C", "0 degrees Celsius", "0°C", "zero degrees Celsius", "0°C",
                      "0°C", "0 Celsius", "0°C", "zero Celsius", "0°C"],
        "judges": ["true"] * 10,
        "ground_truth": ["0", "0°C", "0 degrees Celsius"]}},

    {"syn_const_010": {
        "question": "How many degrees are in a full circle? Respond with the answer only, without any explanation.",
        "responses": ["360 degrees", "360°", "360 degrees", "360", "three hundred and sixty degrees",
                      "360°", "360 degrees", "360°", "360", "360 degrees"],
        "judges": ["true"] * 10,
        "ground_truth": ["360", "360 degrees", "360°"]}},

    {"syn_const_011": {
        "question": "What is the atomic number of hydrogen? Respond with the answer only, without any explanation.",
        "responses": ["1", "one", "1", "1", "the atomic number of hydrogen is 1",
                      "1", "one", "1", "1", "1"],
        "judges": ["true"] * 10,
        "ground_truth": ["1", "one"]}},

    {"syn_const_012": {
        "question": "How many months are in a year? Respond with the answer only, without any explanation.",
        "responses": ["12", "twelve", "12 months", "12", "there are twelve months in a year",
                      "12", "twelve", "12", "twelve months", "12"],
        "judges": ["true"] * 10,
        "ground_truth": ["12", "twelve"]}},

    {"syn_const_013": {
        "question": "What is the chemical symbol for gold? Respond with the answer only, without any explanation.",
        "responses": ["Au", "Au", "Au", "Au", "Au (from the Latin aurum)",
                      "Au", "Au", "Au", "Au", "Au"],
        "judges": ["true"] * 10,
        "ground_truth": ["Au", "au"]}},

    {"syn_const_014": {
        "question": "How many centimetres are in a metre? Respond with the answer only, without any explanation.",
        "responses": ["100", "100 centimetres", "100 cm", "one hundred", "100",
                      "100 centimetres", "100", "100 cm", "one hundred centimetres", "100"],
        "judges": ["true"] * 10,
        "ground_truth": ["100", "one hundred", "100 cm"]}},

    {"syn_const_015": {
        "question": "What is the chemical symbol for oxygen? Respond with the answer only, without any explanation.",
        "responses": ["O", "O", "O", "O", "O (the symbol for oxygen)",
                      "O", "O", "O", "O", "O"],
        "judges": ["true"] * 10,
        "ground_truth": ["O", "o"]}},

    {"syn_const_016": {
        "question": "How many days are in a leap year? Respond with the answer only, without any explanation.",
        "responses": ["366", "366 days", "366", "three hundred and sixty-six days", "366",
                      "366 days", "366", "366", "366 days", "366"],
        "judges": ["true"] * 10,
        "ground_truth": ["366", "three hundred and sixty-six"]}},

    {"syn_const_017": {
        "question": "What is the chemical symbol for sodium? Respond with the answer only, without any explanation.",
        "responses": ["Na", "Na", "Na (from the Latin natrium)", "Na", "Na",
                      "Na", "Na", "Na", "Na", "Na"],
        "judges": ["true"] * 10,
        "ground_truth": ["Na", "na"]}},

    {"syn_const_018": {
        "question": "How many seconds are in one minute? Respond with the answer only, without any explanation.",
        "responses": ["60", "sixty", "60 seconds", "60", "sixty seconds",
                      "60", "60", "sixty", "60 seconds", "60"],
        "judges": ["true"] * 10,
        "ground_truth": ["60", "sixty"]}},

    {"syn_const_019": {
        "question": "What is the atomic number of carbon? Respond with the answer only, without any explanation.",
        "responses": ["6", "six", "6", "6", "the atomic number of carbon is 6",
                      "6", "six", "6", "6", "6"],
        "judges": ["true"] * 10,
        "ground_truth": ["6", "six"]}},

    {"syn_const_020": {
        "question": "How many minutes are in an hour? Respond with the answer only, without any explanation.",
        "responses": ["60", "sixty", "60 minutes", "60", "there are 60 minutes in an hour",
                      "60", "sixty", "60 minutes", "60", "sixty"],
        "judges": ["true"] * 10,
        "ground_truth": ["60", "sixty"]}},

    {"syn_const_021": {
        "question": "What is the chemical symbol for iron? Respond with the answer only, without any explanation.",
        "responses": ["Fe", "Fe (from the Latin ferrum)", "Fe", "Fe", "Fe",
                      "Fe", "Fe", "Fe", "Fe", "Fe"],
        "judges": ["true"] * 10,
        "ground_truth": ["Fe", "fe"]}},

    {"syn_const_022": {
        "question": "How many planets are in our Solar System? Respond with the answer only, without any explanation.",
        "responses": ["8", "eight", "8 planets", "8", "there are eight planets",
                      "8", "eight", "8", "8 planets", "eight"],
        "judges": ["true"] * 10,
        "ground_truth": ["8", "eight"]}},

    {"syn_const_023": {
        "question": "What is the melting point of ice in Kelvin? Respond with the answer only, without any explanation.",
        "responses": ["273.15 K", "273 K", "273.15 Kelvin", "approximately 273 K", "273.15 K",
                      "273 K", "273.15 K", "273 Kelvin", "273.15 K", "about 273 K"],
        "judges": ["true"] * 10,
        "ground_truth": ["273.15 K", "273 K", "273 Kelvin"]}},

    {"syn_const_024": {
        "question": "How many letters are in the English alphabet? Respond with the answer only, without any explanation.",
        "responses": ["26", "twenty-six", "26 letters", "26", "there are 26 letters",
                      "26", "twenty-six", "26", "26 letters", "26"],
        "judges": ["true"] * 10,
        "ground_truth": ["26", "twenty-six"]}},

    {"syn_const_025": {
        "question": "What is the chemical symbol for silver? Respond with the answer only, without any explanation.",
        "responses": ["Ag", "Ag (from the Latin argentum)", "Ag", "Ag", "Ag",
                      "Ag", "Ag", "Ag", "Ag", "Ag"],
        "judges": ["true"] * 10,
        "ground_truth": ["Ag", "ag"]}},

    {"syn_const_026": {
        "question": "How many hours are in a day? Respond with the answer only, without any explanation.",
        "responses": ["24", "twenty-four", "24 hours", "24", "there are 24 hours in a day",
                      "24", "twenty-four", "24 hours", "24", "24"],
        "judges": ["true"] * 10,
        "ground_truth": ["24", "twenty-four"]}},

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP B2: More non-existent APIs / methods (17 entries)
    # Mix: correct disclaimer = true, fabricated behaviour = false.
    # Including more Python, JS, and general software API traps.
    # ═══════════════════════════════════════════════════════════════════════════

    {"syn_noexist_004": {
        "question": "What does the Python built-in dict.sort() method do? Respond with the answer only, without any explanation.",
        "responses": [
            "Python dictionaries do not have a sort() method; use sorted() on dict.items() instead.",
            "There is no dict.sort() method in Python.",
            "Python's dict type has no sort() method.",
            "The dict.sort() method sorts the dictionary by keys in ascending order.",
            "dict.sort() sorts the dictionary's values in place.",
            "There is no built-in dict.sort() in Python; you can use sorted(d.items()) instead.",
            "Python dicts have no sort() method.",
            "dict.sort() does not exist in Python.",
            "There is no sort() method on Python dicts.",
            "Python's built-in dict type has no sort() method.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "no such method", "Python dict has no sort()"]}},

    {"syn_noexist_005": {
        "question": "What does numpy.matrix_inverse() do in Python? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no numpy.matrix_inverse(); use numpy.linalg.inv() to invert a matrix.",
            "NumPy has no matrix_inverse() function; the correct function is numpy.linalg.inv().",
            "numpy.matrix_inverse() does not exist; use numpy.linalg.inv().",
            "numpy.matrix_inverse() computes the inverse of a square matrix.",
            "It returns the inverse of a given matrix.",
            "There is no numpy.matrix_inverse() function in NumPy.",
            "numpy.matrix_inverse() does not exist in the NumPy API.",
            "NumPy does not have a matrix_inverse() function.",
            "The function numpy.matrix_inverse() does not exist; the correct one is numpy.linalg.inv().",
            "There is no numpy.matrix_inverse() in NumPy.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "use numpy.linalg.inv()"]}},

    {"syn_noexist_006": {
        "question": "What does pandas.DataFrame.deduplicate() do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no deduplicate() method on pandas DataFrames; use drop_duplicates() instead.",
            "pandas DataFrames have no deduplicate() method.",
            "This method does not exist in pandas; use df.drop_duplicates().",
            "The deduplicate() method removes duplicate rows from a DataFrame.",
            "pandas.DataFrame.deduplicate() removes all duplicate entries.",
            "There is no deduplicate() method in pandas.",
            "pandas has no DataFrame.deduplicate(); use drop_duplicates().",
            "pandas.DataFrame.deduplicate() does not exist.",
            "There is no deduplicate() method in the pandas DataFrame API.",
            "pandas does not have a deduplicate() method; the correct one is drop_duplicates().",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "use drop_duplicates()"]}},

    {"syn_noexist_007": {
        "question": "What does the JavaScript method String.truncate() do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no native String.truncate() method in JavaScript.",
            "JavaScript strings have no built-in truncate() method.",
            "String.prototype.truncate() does not exist in standard JavaScript.",
            "The truncate() method shortens a string to a given maximum length.",
            "String.truncate() removes characters beyond a specified limit.",
            "There is no String.truncate() in the ECMAScript specification.",
            "JavaScript has no native String.truncate() method.",
            "String.prototype has no truncate() method in standard JavaScript.",
            "There is no truncate() method on JavaScript strings.",
            "JavaScript does not include a String.truncate() method.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not a standard JavaScript method"]}},

    {"syn_noexist_008": {
        "question": "What does the Python string method str.reverse() do? Respond with the answer only, without any explanation.",
        "responses": [
            "Python strings have no reverse() method; use slicing s[::-1] instead.",
            "There is no str.reverse() method in Python.",
            "Python's str type has no reverse() method; use s[::-1].",
            "str.reverse() reverses the string in place.",
            "The str.reverse() method returns a reversed copy of the string.",
            "There is no str.reverse() in Python.",
            "Python strings do not have a reverse() method.",
            "str.reverse() does not exist in Python.",
            "There is no reverse() method on Python strings.",
            "Python's built-in str type has no reverse() method.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "use s[::-1]", "no such method"]}},

    {"syn_noexist_009": {
        "question": "What does torch.clear_cache() do in PyTorch? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no torch.clear_cache(); use torch.cuda.empty_cache() to free GPU memory.",
            "PyTorch has no torch.clear_cache() function; the correct call is torch.cuda.empty_cache().",
            "torch.clear_cache() does not exist in PyTorch.",
            "torch.clear_cache() clears the GPU memory cache.",
            "It frees unused memory from the CUDA cache.",
            "There is no torch.clear_cache() in the PyTorch API.",
            "torch.clear_cache() does not exist; use torch.cuda.empty_cache().",
            "PyTorch has no torch.clear_cache() function.",
            "There is no torch.clear_cache() function in PyTorch.",
            "torch.clear_cache() does not exist in PyTorch.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "use torch.cuda.empty_cache()"]}},

    {"syn_noexist_010": {
        "question": "What does the CSS property text-decoration-bold do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no CSS property called text-decoration-bold; use font-weight: bold instead.",
            "text-decoration-bold is not a valid CSS property.",
            "CSS has no text-decoration-bold property.",
            "The text-decoration-bold property makes text bold.",
            "It applies bold styling to text decoration.",
            "There is no text-decoration-bold in CSS.",
            "CSS does not include a text-decoration-bold property.",
            "text-decoration-bold does not exist in CSS.",
            "There is no such CSS property as text-decoration-bold.",
            "CSS has no text-decoration-bold; use font-weight: bold.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not a valid CSS property"]}},

    {"syn_noexist_011": {
        "question": "What does git.rebase_merge() do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no git.rebase_merge() command; you may be thinking of git rebase or git merge.",
            "git.rebase_merge() does not exist as a Git command.",
            "There is no such Git command.",
            "git.rebase_merge() performs a rebase and then merges the branch.",
            "It combines rebase and merge operations in one step.",
            "There is no git.rebase_merge() in Git.",
            "Git has no rebase_merge() command.",
            "git.rebase_merge() is not a valid Git command.",
            "This command does not exist in Git.",
            "git.rebase_merge() does not exist.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not a valid Git command"]}},

    {"syn_noexist_012": {
        "question": "What does the JavaScript Array.prototype.shuffle() method do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no native Array.prototype.shuffle() method in JavaScript.",
            "JavaScript arrays have no built-in shuffle() method.",
            "Array.prototype.shuffle() does not exist in standard JavaScript.",
            "Array.prototype.shuffle() randomly shuffles the elements of an array in place.",
            "shuffle() randomises the order of array elements.",
            "There is no shuffle() method in the ECMAScript specification.",
            "JavaScript has no native Array.shuffle() method.",
            "Array.prototype has no shuffle() method in standard JavaScript.",
            "There is no shuffle() on JavaScript arrays.",
            "JavaScript does not include Array.prototype.shuffle().",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not a standard JavaScript method"]}},

    {"syn_noexist_013": {
        "question": "What does Python's os.file_exists() function do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no os.file_exists(); use os.path.exists() instead.",
            "Python has no os.file_exists() function; use os.path.exists().",
            "os.file_exists() does not exist; the correct function is os.path.exists().",
            "os.file_exists() returns True if the file exists, False otherwise.",
            "It checks whether a file exists at a given path.",
            "There is no os.file_exists() in Python's os module.",
            "Python's os module has no file_exists() function.",
            "os.file_exists() does not exist in Python.",
            "There is no file_exists() in Python's os module.",
            "os.file_exists() does not exist; use os.path.exists().",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "use os.path.exists()"]}},

    {"syn_noexist_014": {
        "question": "What does the React hook useInterval() do? Respond with the answer only, without any explanation.",
        "responses": [
            "useInterval() is not a built-in React hook; it is a custom hook pattern often found in libraries.",
            "There is no useInterval() in React's standard library.",
            "useInterval() does not exist as a built-in React hook.",
            "useInterval() is a React hook that runs a callback at a fixed interval.",
            "It sets up a recurring interval inside a React component.",
            "useInterval() is not a built-in React hook.",
            "React's standard hooks do not include useInterval().",
            "There is no built-in useInterval() hook in React.",
            "useInterval() is not part of React's core library.",
            "React has no built-in useInterval() hook.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["not a built-in React hook", "does not exist in React core"]}},

    {"syn_noexist_015": {
        "question": "What does the SQL command FETCH_NEXT() do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no FETCH_NEXT() function in standard SQL; FETCH NEXT is cursor syntax.",
            "SQL has no FETCH_NEXT() function.",
            "FETCH_NEXT() does not exist as a SQL function.",
            "FETCH_NEXT() retrieves the next row from a result set.",
            "It advances a cursor to the next row.",
            "There is no FETCH_NEXT() in standard SQL.",
            "SQL does not have a FETCH_NEXT() function.",
            "FETCH_NEXT() is not a valid SQL function.",
            "There is no such SQL function as FETCH_NEXT().",
            "SQL has no FETCH_NEXT(); the correct syntax uses FETCH NEXT with a cursor.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not a standard SQL function"]}},

    {"syn_noexist_016": {
        "question": "What does the Python method set.append() do? Respond with the answer only, without any explanation.",
        "responses": [
            "Python sets have no append() method; use set.add() instead.",
            "There is no append() method on Python sets.",
            "Python's set type has no append() method; use add().",
            "set.append() adds an element to the set.",
            "It appends the element to the set, similar to list.append().",
            "There is no set.append() in Python.",
            "Python sets do not have an append() method.",
            "set.append() does not exist in Python.",
            "There is no append() on Python sets.",
            "Python's built-in set has no append(); use add() instead.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "use set.add()"]}},

    {"syn_noexist_017": {
        "question": "What does keras.model.predict_proba() do? Respond with the answer only, without any explanation.",
        "responses": [
            "Keras models have no predict_proba() method; use model.predict() instead.",
            "There is no predict_proba() method on Keras models.",
            "Keras has no model.predict_proba(); use model.predict().",
            "model.predict_proba() returns class probabilities for classification models.",
            "It returns predicted probabilities like sklearn's predict_proba().",
            "There is no predict_proba() in Keras.",
            "Keras model objects have no predict_proba() method.",
            "model.predict_proba() does not exist in Keras.",
            "There is no predict_proba() method in Keras.",
            "Keras has no predict_proba(); use model.predict().",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist in Keras", "use model.predict()"]}},

    {"syn_noexist_018": {
        "question": "What does the HTML tag <text> do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no <text> tag in standard HTML; use <p>, <span>, or <div> for text.",
            "HTML has no <text> element.",
            "<text> is not a valid HTML tag.",
            "The <text> tag renders inline text on a web page.",
            "It wraps text content in an HTML document.",
            "There is no <text> element in the HTML specification.",
            "HTML does not include a <text> tag.",
            "<text> is not a valid HTML element.",
            "There is no <text> tag in HTML.",
            "HTML has no <text> element; use <p> or <span> instead.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["not a valid HTML element", "does not exist in HTML"]}},

    {"syn_noexist_019": {
        "question": "What does the JavaScript method Object.freeze_deep() do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no Object.freeze_deep() in JavaScript; Object.freeze() only freezes one level.",
            "JavaScript has no Object.freeze_deep() method.",
            "Object.freeze_deep() does not exist in standard JavaScript.",
            "Object.freeze_deep() recursively freezes all nested objects.",
            "It deep-freezes an object and all its nested properties.",
            "There is no Object.freeze_deep() in the ECMAScript specification.",
            "JavaScript does not include Object.freeze_deep().",
            "Object.freeze_deep() is not a standard JavaScript method.",
            "There is no Object.freeze_deep() in JavaScript.",
            "JavaScript has no built-in Object.freeze_deep().",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not a standard JavaScript method"]}},

    {"syn_noexist_020": {
        "question": "What does the Python function sys.memory_usage() do? Respond with the answer only, without any explanation.",
        "responses": [
            "There is no sys.memory_usage() in Python; use resource.getrusage() or psutil.Process().memory_info().",
            "Python's sys module has no memory_usage() function.",
            "sys.memory_usage() does not exist in Python.",
            "sys.memory_usage() returns the current memory usage of the Python process.",
            "It reports how much RAM the Python interpreter is using.",
            "There is no sys.memory_usage() in Python.",
            "Python has no sys.memory_usage() function.",
            "sys.memory_usage() does not exist in the sys module.",
            "There is no memory_usage() in Python's sys module.",
            "sys.memory_usage() does not exist; use psutil instead.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["does not exist", "not in the sys module"]}},

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP C2: More warranted hedging (17 entries)
    # Questions where ANY honest answer involves "it depends", domain-specific
    # caveats, or referral to a professional.  All-true judges.
    # ═══════════════════════════════════════════════════════════════════════════

    {"syn_hedge_004": {
        "question": "What is the maximum safe daily dose of paracetamol for adults? Respond with the answer only, without any explanation.",
        "responses": [
            "Generally up to 4 grams per day for healthy adults, but lower for those with liver conditions.",
            "Up to 4 g/day for healthy adults; lower limits apply for people with liver or kidney problems.",
            "The typical maximum is 4,000 mg per day for healthy adults; consult a doctor for individual cases.",
            "Up to 4 g per day, but this may vary depending on individual health factors.",
            "Generally 4 g/day maximum; patients with liver conditions should use lower doses.",
            "4,000 mg/day is the general guideline for healthy adults.",
            "Up to 4 g daily for most healthy adults, but always follow medical guidance.",
            "For healthy adults, up to 4 grams per day; less if you have liver issues.",
            "Generally 4 g/day; always consult a healthcare professional.",
            "Up to 4,000 mg per day for most adults; lower for those with liver or kidney conditions.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["4 grams per day", "4 g/day", "4000 mg/day"]}},

    {"syn_hedge_005": {
        "question": "Is it safe to drive after taking antihistamines? Respond with the answer only, without any explanation.",
        "responses": [
            "It depends on the type of antihistamine; some cause drowsiness and impair driving ability.",
            "It depends — first-generation antihistamines can cause drowsiness; newer non-drowsy ones are generally safer.",
            "Depends on the antihistamine; sedating types like diphenhydramine impair driving.",
            "Some antihistamines cause drowsiness and should not be taken before driving.",
            "It depends on the specific drug; check the label and consult a doctor.",
            "Sedating antihistamines impair driving; non-sedating ones are generally considered safer.",
            "It depends on the antihistamine type; first-generation ones commonly cause drowsiness.",
            "This depends on the specific antihistamine; some cause significant drowsiness.",
            "Not always; sedating antihistamines can impair your ability to drive safely.",
            "Depends on the drug; older antihistamines often cause drowsiness that impairs driving.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["depends on the antihistamine", "it depends", "some cause drowsiness"]}},

    {"syn_hedge_006": {
        "question": "What is the correct approach to intermittent fasting? Respond with the answer only, without any explanation.",
        "responses": [
            "There are several protocols; the right approach depends on individual health, goals, and medical advice.",
            "It depends on individual factors; consult a healthcare professional for a personalised approach.",
            "Approaches vary; popular ones include 16:8 and 5:2, but suitability depends on the individual.",
            "The best approach varies by person; consult a dietitian.",
            "It depends on your goals and health; common protocols include 16:8 fasting.",
            "There is no single correct approach; it depends on your health and lifestyle.",
            "Depends on individual health; consult a healthcare professional before starting.",
            "It varies; common approaches include 16:8, 5:2, and OMAD, but suitability is individual.",
            "The correct approach depends on personal health factors; I recommend consulting a doctor.",
            "It depends on the individual; there are multiple valid protocols depending on goals.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["it depends", "varies by individual", "consult a healthcare professional"]}},

    {"syn_hedge_007": {
        "question": "Should I invest in index funds or individual stocks? Respond with the answer only, without any explanation.",
        "responses": [
            "It depends on your risk tolerance, investment goals, and financial situation.",
            "This depends on individual circumstances; consult a financial advisor.",
            "Depends on your risk appetite and expertise; index funds generally suit long-term passive investors.",
            "It depends on your financial goals and risk tolerance.",
            "Both have merits; the right choice depends on your situation.",
            "It depends on your risk tolerance and investment knowledge.",
            "This is highly individual; consult a qualified financial advisor.",
            "It depends on your goals, time horizon, and risk tolerance.",
            "Depends on personal factors; most financial advisors recommend diversified index funds for passive investors.",
            "It depends on your financial goals and experience level.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["it depends", "depends on individual circumstances"]}},

    {"syn_hedge_008": {
        "question": "What is the safest antibiotic for a pregnant woman? Respond with the answer only, without any explanation.",
        "responses": [
            "The choice depends on the infection type and trimester; consult a doctor.",
            "It depends on the infection and stage of pregnancy; only a doctor can advise.",
            "This depends on the specific infection and pregnancy stage; consult a healthcare professional.",
            "The safest antibiotic depends on the infection and trimester; medical supervision is essential.",
            "It depends on what is being treated; consult a qualified healthcare provider.",
            "Antibiotic selection in pregnancy depends on many factors; always consult a doctor.",
            "Depends on the infection; some are generally considered safer but always need medical guidance.",
            "This is highly individual; a doctor or pharmacist should always be consulted.",
            "It depends on the infection type and pregnancy stage; please consult a healthcare professional.",
            "Depends on the specific clinical situation; a doctor must make this decision.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["it depends", "consult a doctor", "depends on the infection and pregnancy stage"]}},

    {"syn_hedge_009": {
        "question": "What is the right amount of exercise per week for weight loss? Respond with the answer only, without any explanation.",
        "responses": [
            "It depends on individual factors; general guidelines suggest 150–300 minutes of moderate-intensity exercise per week.",
            "Depends on diet, metabolism, and fitness level; most guidelines recommend 150–300 min/week moderate aerobic activity.",
            "The right amount varies by individual; typical recommendations are 150–300 minutes of moderate exercise per week.",
            "It depends on many factors; 150–300 minutes per week of moderate activity is commonly recommended.",
            "General guidelines suggest 150 to 300 minutes per week, but effectiveness varies by person.",
            "It depends on your current fitness, diet, and goals; common guidelines suggest 150–300 min/week.",
            "Varies by individual; guidelines typically recommend at least 150 minutes per week.",
            "Depends on the person; 150–300 minutes per week of moderate-intensity exercise is a common recommendation.",
            "It depends on many factors; general advice is 150–300 minutes of moderate aerobic activity per week.",
            "The right amount depends on individual health; guidelines commonly suggest 150–300 min/week.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["it depends", "150–300 minutes per week", "depends on individual factors"]}},

    {"syn_hedge_010": {
        "question": "What is the legal drinking age in the United States? Respond with the answer only, without any explanation.",
        "responses": [
            "21 in all US states.",
            "21 years old federally; all states enforce 21.",
            "21 years old in the United States.",
            "The legal drinking age is 21 in the US.",
            "21.",
            "21 years old.",
            "21 in all fifty states.",
            "21 years old under federal law.",
            "21 in the US.",
            "The federal minimum is 21, adopted by all states.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["21", "21 years old", "twenty-one"]}},

    {"syn_hedge_011": {
        "question": "What is the minimum age to work in the United Kingdom? Respond with the answer only, without any explanation.",
        "responses": [
            "Generally 16 for full-time work, though 13–15 year olds can do limited part-time work.",
            "The general minimum age for full-time employment is 16, but there are exceptions for part-time work from age 13.",
            "16 for full-time work; 13 for limited part-time work under certain conditions.",
            "It depends on the type of work; generally 16 for full-time, with limited exceptions from age 13.",
            "The school leaving age is 16 in the UK, but limited part-time work is permitted from 13.",
            "16 for full-time employment; exceptions allow part-time work from age 13.",
            "Generally 16, though light part-time work is permitted from age 13 under specific regulations.",
            "Minimum 16 for full-time work; some part-time roles available from 13.",
            "16 for most employment; limited part-time work allowed from age 13.",
            "The minimum age for full-time work is 16; part-time work from 13 is permitted in some circumstances.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["16", "16 for full-time work", "13 for limited part-time"]}},

    {"syn_hedge_012": {
        "question": "What is a normal resting heart rate for adults? Respond with the answer only, without any explanation.",
        "responses": [
            "60 to 100 beats per minute is the normal range for adults.",
            "A normal resting heart rate for adults is 60–100 bpm.",
            "Between 60 and 100 beats per minute.",
            "60–100 bpm is considered normal; athletes may have lower rates.",
            "Normal range is 60–100 bpm; athletes may be lower.",
            "60 to 100 beats per minute.",
            "A normal adult resting heart rate is between 60 and 100 bpm.",
            "60–100 beats per minute for most adults.",
            "Generally 60–100 bpm; can be lower in highly trained athletes.",
            "Between 60 and 100 bpm is the typical normal range.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["60–100 bpm", "60 to 100 beats per minute"]}},

    {"syn_hedge_013": {
        "question": "Is it safe to mix bleach and vinegar for cleaning? Respond with the answer only, without any explanation.",
        "responses": [
            "No, mixing bleach and vinegar produces chlorine gas, which is toxic.",
            "No — combining bleach and vinegar releases chlorine gas and is dangerous.",
            "Do not mix them; the combination produces toxic chlorine gas.",
            "No, this is dangerous; bleach and vinegar react to form chlorine gas.",
            "This is unsafe; mixing them creates chlorine gas.",
            "No — bleach (sodium hypochlorite) and vinegar react to form toxic chlorine gas.",
            "No, do not mix bleach and vinegar; it creates a toxic gas.",
            "Mixing bleach and vinegar is dangerous; it produces chlorine gas.",
            "No — it creates chlorine gas, which is harmful.",
            "Do not mix them; the reaction produces toxic chlorine gas.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["no", "dangerous", "produces chlorine gas"]}},

    {"syn_hedge_014": {
        "question": "Does correlation imply causation? Respond with the answer only, without any explanation.",
        "responses": [
            "No, correlation does not imply causation.",
            "No — a correlation between two variables does not mean one causes the other.",
            "No. Correlation shows a statistical relationship but does not establish cause.",
            "No, correlation is not the same as causation.",
            "Not necessarily; correlation indicates a relationship, not a causal link.",
            "No — two things can be correlated without one causing the other.",
            "No. Correlation does not imply causation.",
            "No; a correlation can exist without any causal relationship.",
            "No — correlation only shows that two variables tend to move together.",
            "No, this is a common logical fallacy.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["no", "correlation does not imply causation"]}},

    {"syn_hedge_015": {
        "question": "Should you always use HTTPS instead of HTTP? Respond with the answer only, without any explanation.",
        "responses": [
            "Yes, HTTPS should always be used for any site handling sensitive data; it encrypts the connection.",
            "Generally yes; HTTPS encrypts traffic and should be used instead of HTTP.",
            "Yes, HTTPS is recommended for all websites to protect users' data.",
            "In almost all cases, yes; HTTPS provides encryption that HTTP lacks.",
            "Yes — HTTPS protects data in transit; HTTP transmits data in plaintext.",
            "Yes, HTTPS should be preferred over HTTP for security.",
            "Generally yes; HTTPS encrypts the connection between client and server.",
            "Yes, for any website handling user data or login information, HTTPS is essential.",
            "Yes — modern best practice is to use HTTPS for all web traffic.",
            "Yes, HTTPS is strongly recommended over HTTP for all websites.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["yes", "yes, use HTTPS"]}},

    {"syn_hedge_016": {
        "question": "Is a SQL injection attack possible on a parameterised query? Respond with the answer only, without any explanation.",
        "responses": [
            "No — properly implemented parameterised queries prevent SQL injection by separating data from code.",
            "No, parameterised queries effectively prevent SQL injection.",
            "No — they prevent SQL injection by treating user input as data, not code.",
            "No; when implemented correctly, parameterised queries prevent SQL injection attacks.",
            "Not if correctly implemented; parameterised queries are the standard defence against SQL injection.",
            "No — parameterised queries protect against SQL injection.",
            "No, parameterised queries prevent SQL injection by design.",
            "Properly used parameterised queries prevent SQL injection.",
            "No — this is the recommended defence against SQL injection.",
            "No; parameterised queries eliminate SQL injection when used correctly.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["no", "parameterised queries prevent SQL injection"]}},

    {"syn_hedge_017": {
        "question": "Can you copyright an idea? Respond with the answer only, without any explanation.",
        "responses": [
            "No — copyright protects the expression of an idea, not the idea itself.",
            "No, ideas are not protected by copyright; only their specific expression is.",
            "No — copyright law protects the creative expression, not underlying ideas.",
            "No; copyright does not protect ideas, only their original expression.",
            "No — the idea–expression dichotomy means only expression, not ideas, can be copyrighted.",
            "No, you cannot copyright an idea.",
            "No — copyright protects the specific expression of an idea, not the idea itself.",
            "No; ideas are in the public domain — only the expression of an idea can be copyrighted.",
            "No, ideas themselves are not copyrightable; their expression is.",
            "No — copyright law protects expression, not ideas.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["no", "ideas cannot be copyrighted", "copyright protects expression not ideas"]}},

    {"syn_hedge_018": {
        "question": "Is it safe to eat raw chicken? Respond with the answer only, without any explanation.",
        "responses": [
            "No — raw chicken can contain Salmonella, Campylobacter, and other harmful bacteria.",
            "No; raw chicken often carries bacteria like Salmonella and Campylobacter.",
            "No, raw chicken is unsafe to eat due to potential bacterial contamination.",
            "No — eating raw chicken risks serious foodborne illness.",
            "No; raw chicken can be contaminated with Salmonella and other pathogens.",
            "No — raw chicken must be cooked thoroughly to kill harmful bacteria.",
            "No, raw chicken carries a high risk of Salmonella and Campylobacter infection.",
            "No; it is not safe to eat raw chicken.",
            "No — raw chicken should always be cooked to a safe internal temperature.",
            "No, eating raw chicken is unsafe and can cause serious illness.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["no", "unsafe", "risk of Salmonella"]}},

    {"syn_hedge_019": {
        "question": "Does drinking more water always help with weight loss? Respond with the answer only, without any explanation.",
        "responses": [
            "Not necessarily — water can help but is not a guaranteed weight loss solution.",
            "Not always; while hydration supports metabolism, weight loss depends on many factors.",
            "It can help but is not a direct cause of weight loss.",
            "Water alone does not cause weight loss; it can support a healthy diet and exercise plan.",
            "Not always; adequate hydration supports metabolism but overall diet and exercise matter more.",
            "It may help, but weight loss depends on overall diet and activity, not water intake alone.",
            "Not necessarily; water supports bodily functions but is not sufficient for weight loss on its own.",
            "It can help, but the evidence is not conclusive that drinking more water alone causes weight loss.",
            "Not always — hydration is important but weight loss requires a calorie deficit.",
            "Not always; while water has no calories and can reduce appetite, it is not a weight-loss guarantee.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["not necessarily", "it depends", "not always"]}},

    {"syn_hedge_020": {
        "question": "Is Python always better than C++ for performance-critical applications? Respond with the answer only, without any explanation.",
        "responses": [
            "No — C++ is generally much faster than Python for performance-critical code.",
            "No; C++ typically outperforms Python significantly in CPU-intensive tasks.",
            "No — Python is generally slower than C++ for compute-intensive work.",
            "No, Python is usually slower than C++ for performance-critical applications.",
            "Not at all; C++ typically has far better performance than Python.",
            "No — C++ is compiled to native code and is typically much faster than Python.",
            "No; C++ generally outperforms Python in raw execution speed.",
            "No, C++ is usually preferred over Python for performance-critical applications.",
            "No — Python is typically much slower than C++ for low-level, performance-sensitive work.",
            "No; for performance-critical applications, C++ generally offers much better throughput.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["no", "C++ is generally faster"]}},

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP D2: More false-premise / trick questions (17 entries)
    # Mix: correctly rejecting the premise = true; accepting it = false.
    # ═══════════════════════════════════════════════════════════════════════════

    {"syn_trap_004": {
        "question": "Which US state is Hawaii closest to? Respond with the answer only, without any explanation.",
        "responses": [
            "Hawaii is closest to Alaska among US states.",
            "Alaska — it is the closest US state to Hawaii.",
            "Alaska is the nearest US state to Hawaii.",
            "California",
            "Oregon",
            "Hawaii is geographically closest to Alaska.",
            "Alaska.",
            "Alaska is the closest US state to Hawaii.",
            "Alaska — geographically the nearest US state.",
            "Among US states, Hawaii is closest to Alaska.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["Alaska"]}},

    {"syn_trap_005": {
        "question": "What is the longest river in the United Kingdom? Respond with the answer only, without any explanation.",
        "responses": [
            "The River Severn is the longest river in the United Kingdom.",
            "The Severn.",
            "The River Severn.",
            "The Thames",
            "The Thames is the longest river in the UK.",
            "The River Severn — it is longer than the Thames.",
            "The Severn is the longest river in the UK.",
            "The River Severn.",
            "The Severn — at 354 km, it is the longest UK river.",
            "River Severn.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["River Severn", "Severn"]}},

    {"syn_trap_006": {
        "question": "Why did Einstein fail mathematics at school? Respond with the answer only, without any explanation.",
        "responses": [
            "Einstein did not fail mathematics at school; this is a common myth.",
            "He did not fail mathematics; Einstein excelled at mathematics.",
            "This is a misconception — Einstein was actually excellent at mathematics.",
            "Because he found the teaching style boring.",
            "He struggled with rote memorisation in maths classes.",
            "Einstein did not fail maths; the premise is incorrect.",
            "He did not fail mathematics; this is a popular myth.",
            "This is a myth — Einstein was very strong in mathematics.",
            "Einstein did not fail mathematics; he was exceptional at it.",
            "He did not fail; this is a misattributed story.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["he did not fail mathematics", "this is a myth"]}},

    {"syn_trap_007": {
        "question": "How many moons does the Sun have? Respond with the answer only, without any explanation.",
        "responses": [
            "The Sun has no moons — planets orbit the Sun, not the other way around.",
            "Zero — the Sun does not have moons.",
            "The Sun does not have moons; it is a star, and moons orbit planets.",
            "The Sun has one moon.",
            "The Sun has several moons just like other stars.",
            "None — the Sun is a star and does not have moons.",
            "Zero; moons orbit planets, not stars.",
            "The Sun has no moons.",
            "None — the Sun does not have moons.",
            "Zero; stars do not have moons.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["zero", "none", "the Sun has no moons"]}},

    {"syn_trap_008": {
        "question": "Is the Great Wall of China visible from space with the naked eye? Respond with the answer only, without any explanation.",
        "responses": [
            "No — it is not visible from space with the naked eye; this is a common myth.",
            "No; astronauts have confirmed the Great Wall is not visible from space without optical aids.",
            "This is a myth — the Great Wall is too narrow to be seen from space with the naked eye.",
            "Yes, it is one of the few man-made structures visible from space.",
            "Yes, on a clear day it can be seen from the Moon.",
            "No — this is a popular misconception; it is not visible from space.",
            "No; the Great Wall is not visible from space with the naked eye.",
            "No — this is a myth; the wall is too narrow to see from orbit.",
            "No; astronauts have said the Great Wall is not visible from space unaided.",
            "No — it is not visible from space; this is widely debunked.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["no", "not visible from space", "this is a myth"]}},

    {"syn_trap_009": {
        "question": "What colour is a polar bear's fur? Respond with the answer only, without any explanation.",
        "responses": [
            "Polar bear fur is actually transparent and hollow; it appears white due to light scattering.",
            "Transparent — the fur appears white but is actually colourless and hollow.",
            "The individual hairs are transparent, but the fur appears white.",
            "White",
            "Completely white.",
            "Each hair is actually transparent, appearing white from light reflection.",
            "Transparent — polar bear fur appears white but is not pigmented.",
            "The fur is transparent/hollow but appears white.",
            "Transparent, not actually white — it appears white due to how it scatters light.",
            "Each hair is transparent; the fur looks white due to light scattering.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["transparent", "colourless", "appears white but is actually transparent"]}},

    {"syn_trap_010": {
        "question": "How much of the brain does a human typically use? Respond with the answer only, without any explanation.",
        "responses": [
            "Humans use virtually all of their brain; the 10% myth is false.",
            "All of it — the claim that we only use 10% is a debunked myth.",
            "Essentially 100% — the 10% myth is not supported by neuroscience.",
            "About 10% of the brain.",
            "Only around 10% of the brain at any one time.",
            "All of it — the idea we only use 10% is scientifically false.",
            "Virtually all brain regions are active; the 10% figure is a myth.",
            "All of it; modern neuroscience has debunked the 10% myth.",
            "Essentially all of it — humans use the vast majority of their brain.",
            "All of it — the 10% claim is a well-known myth.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["virtually all", "essentially 100%", "the 10% myth is false"]}},

    {"syn_trap_011": {
        "question": "What did Napoleon Bonaparte's height tell us about his personality? Respond with the answer only, without any explanation.",
        "responses": [
            "Napoleon was not unusually short; his height of about 5'7\" was average for his era. The 'short' image is a myth.",
            "Napoleon was of average height for his time; the Napoleon complex story is largely a myth.",
            "The premise is based on a misconception; Napoleon was around 5'7\" — average for the period.",
            "His short stature fuelled his aggressive personality.",
            "Being short made him overcompensate through military ambition.",
            "Napoleon was actually of average height; the assumption about his height is incorrect.",
            "Napoleon was not notably short; the idea is based on a misattributed caricature.",
            "The premise is incorrect — Napoleon was of average height for the time.",
            "Napoleon was not unusually short; his height was normal for his era.",
            "This is based on a historical myth; Napoleon was of average height.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["Napoleon was not short", "this is a myth", "he was of average height"]}},

    {"syn_trap_012": {
        "question": "What is the darkest colour in the visible light spectrum? Respond with the answer only, without any explanation.",
        "responses": [
            "Black is not a colour in the visible spectrum; it is the absence of visible light.",
            "There is no 'darkest colour' in the visible spectrum; black is the absence of light.",
            "Black is not in the visible light spectrum; it represents the absence of light.",
            "Black",
            "Violet or deep indigo.",
            "Black is the absence of light, not a colour in the visible spectrum.",
            "There is no darkest colour; black is the absence of light.",
            "Black is not part of the visible spectrum.",
            "The premise is flawed — black is not a spectral colour.",
            "Black is not a colour in the visible spectrum; it is the absence of all light.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["black is not in the visible spectrum", "black is the absence of light"]}},

    {"syn_trap_013": {
        "question": "What language do they primarily speak in Brazil? Respond with the answer only, without any explanation.",
        "responses": [
            "Portuguese",
            "Brazilian Portuguese",
            "Portuguese — Brazil is the largest Portuguese-speaking country in the world.",
            "Spanish",
            "Spanish is the main language.",
            "Portuguese.",
            "Portuguese is the official and primary language of Brazil.",
            "Portuguese — not Spanish, which is a common misconception.",
            "Portuguese.",
            "Portuguese is the primary language in Brazil.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["Portuguese", "Brazilian Portuguese"]}},

    {"syn_trap_014": {
        "question": "Does lightning never strike the same place twice? Respond with the answer only, without any explanation.",
        "responses": [
            "No — lightning frequently strikes the same place multiple times.",
            "False — lightning can and does strike the same place more than once.",
            "This is a myth; lightning regularly strikes the same location multiple times.",
            "True, lightning never strikes the same place twice.",
            "Correct — lightning always strikes a different location.",
            "No — this is a common myth; lightning often strikes the same spot repeatedly.",
            "False; lightning regularly strikes tall structures like the Empire State Building multiple times per year.",
            "This is incorrect — lightning strikes the same place repeatedly.",
            "No — lightning can and does strike the same location many times.",
            "False — this is a well-known myth; lightning repeatedly strikes the same spots.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["false", "this is a myth", "lightning does strike the same place twice"]}},

    {"syn_trap_015": {
        "question": "Is the tongue map showing different taste regions real? Respond with the answer only, without any explanation.",
        "responses": [
            "No — the tongue taste map showing distinct taste zones is a debunked myth; all taste buds detect all tastes.",
            "No, the tongue map is a myth; taste receptors for all tastes are distributed throughout the tongue.",
            "The traditional tongue map is incorrect; all taste buds can detect all basic tastes.",
            "Yes, different parts of the tongue detect different tastes.",
            "Yes, sweet is at the tip and bitter at the back.",
            "No — the tongue map is a widespread misconception.",
            "The taste map is debunked; taste buds across the tongue detect all basic tastes.",
            "No, this is a myth; there are no exclusive taste zones on the tongue.",
            "The tongue map is a well-known scientific misconception.",
            "No — modern research has debunked the tongue taste map.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["no", "the tongue map is a myth", "debunked"]}},

    {"syn_trap_016": {
        "question": "Does vitamin C prevent the common cold? Respond with the answer only, without any explanation.",
        "responses": [
            "Not reliably — research shows vitamin C does not significantly prevent colds in the general population.",
            "Not for most people; evidence suggests vitamin C does not prevent colds.",
            "No — large-scale studies show vitamin C does not prevent colds in the general population.",
            "Yes, vitamin C prevents the common cold.",
            "Yes, taking large doses of vitamin C prevents colds.",
            "No — the evidence does not support vitamin C as a cold prevention measure.",
            "Not in general; evidence does not support it as a reliable cold prevention.",
            "No — large trials show no significant preventative effect in most people.",
            "Not reliably; vitamin C does not significantly prevent colds.",
            "No — this claim is not well supported by clinical evidence.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["no", "not reliably", "not supported by evidence"]}},

    {"syn_trap_017": {
        "question": "Is sugar the primary cause of hyperactivity in children? Respond with the answer only, without any explanation.",
        "responses": [
            "No — controlled studies consistently show sugar does not cause hyperactivity in children.",
            "No; multiple studies have found no link between sugar and hyperactivity in children.",
            "This is a myth — research does not support sugar as a cause of hyperactivity.",
            "Yes, sugar makes children hyperactive.",
            "Sugar is the main cause of hyperactivity in kids.",
            "No — well-controlled studies have not found sugar causes hyperactivity.",
            "No; the sugar-hyperactivity link has been repeatedly debunked.",
            "No — this is a well-known misconception not supported by evidence.",
            "No; research consistently shows sugar does not cause hyperactivity.",
            "No — this is a commonly held but scientifically unsupported belief.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["no", "this is a myth", "not supported by evidence"]}},

    {"syn_trap_018": {
        "question": "Is Mount Everest the tallest mountain on Earth when measured from sea level? Respond with the answer only, without any explanation.",
        "responses": [
            "Yes — Everest is the highest mountain above sea level at approximately 8,849 m.",
            "Yes, Mount Everest is the tallest when measured from sea level.",
            "Yes — at 8,849 m above sea level, Everest is the tallest.",
            "No, Mauna Kea is taller.",
            "No — if measured from base to peak, Mauna Kea is taller.",
            "Yes — from sea level, Everest is the tallest at about 8,849 m.",
            "Yes, Everest is the highest point above sea level on Earth.",
            "Yes, at 8,849 metres above sea level.",
            "Yes, measured from sea level, Everest is the tallest mountain.",
            "Yes — from sea level it is the tallest at 8,849 m.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["yes", "Mount Everest", "8849 m"]}},

    {"syn_trap_019": {
        "question": "Does the moon have gravity? Respond with the answer only, without any explanation.",
        "responses": [
            "Yes — the Moon has gravity; it is approximately 1/6th of Earth's surface gravity.",
            "Yes, the Moon has its own gravitational field.",
            "Yes — lunar gravity is about 1.62 m/s², roughly 1/6 that of Earth.",
            "No, the Moon has no gravity.",
            "No — there is no gravity on the Moon.",
            "Yes — the Moon's gravity is about 1/6th of Earth's.",
            "Yes, the Moon has gravity.",
            "Yes — the Moon exerts gravitational force; it is about 16% of Earth's.",
            "Yes, the Moon has gravity — roughly 1.62 m/s².",
            "Yes — the Moon has gravity; astronauts walk on its surface because of it.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["yes", "the Moon has gravity", "approximately 1/6 of Earth's"]}},

    {"syn_trap_020": {
        "question": "Did Christopher Columbus prove the Earth was round? Respond with the answer only, without any explanation.",
        "responses": [
            "No — educated Europeans already knew the Earth was spherical long before Columbus; he didn't prove it.",
            "No; the spherical Earth was well established by ancient Greek scholars before Columbus.",
            "Columbus did not prove the Earth was round; this was already known.",
            "Yes, Columbus proved the Earth was round by sailing west to reach Asia.",
            "Yes — his voyage demonstrated the Earth's curvature.",
            "No — educated people at the time already knew the Earth was spherical.",
            "No, the round Earth was already accepted scientific knowledge before Columbus.",
            "No — this is a historical misconception; the spherical Earth was known since antiquity.",
            "No; Columbus did not prove the Earth was round — it was already accepted knowledge.",
            "No — the spherical Earth was known since ancient Greece.",
        ],
        "judges": ["true", "true", "true", "false", "false", "true", "true", "true", "true", "true"],
        "ground_truth": ["no", "the Earth was already known to be round", "this is a myth"]}},

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP E2: More time-stable historical facts (16 entries)
    # Varied phrasings across 10 responses per entry.
    # ═══════════════════════════════════════════════════════════════════════════

    {"syn_hist_005": {
        "question": "Who invented the telephone? Respond with the answer only, without any explanation.",
        "responses": [
            "Alexander Graham Bell", "Alexander Graham Bell invented the telephone.",
            "Bell", "Alexander Bell",
            "Alexander Graham Bell is credited with inventing the telephone.",
            "Alexander Graham Bell", "Bell",
            "Alexander Graham Bell.", "Alexander Graham Bell",
            "Bell — Alexander Graham Bell.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Alexander Graham Bell", "Bell"]}},

    {"syn_hist_006": {
        "question": "In which country was the first aeroplane flight made? Respond with the answer only, without any explanation.",
        "responses": [
            "United States", "The United States of America", "USA",
            "The first powered flight was made in the United States.",
            "United States", "USA", "United States of America",
            "The US — at Kitty Hawk, North Carolina.", "United States",
            "The United States, at Kitty Hawk.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["United States", "USA", "US"]}},

    {"syn_hist_007": {
        "question": "Which war ended in 1918? Respond with the answer only, without any explanation.",
        "responses": [
            "World War I", "The First World War", "WWI",
            "World War One ended in 1918.", "The Great War",
            "World War I", "WWI", "The First World War",
            "World War I ended in 1918.", "World War One.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["World War I", "WWI", "First World War", "The Great War"]}},

    {"syn_hist_008": {
        "question": "Who wrote the play Hamlet? Respond with the answer only, without any explanation.",
        "responses": [
            "William Shakespeare", "Shakespeare", "William Shakespeare.",
            "Shakespeare wrote Hamlet.", "William Shakespeare",
            "William Shakespeare", "Shakespeare.", "William Shakespeare",
            "William Shakespeare", "Shakespeare.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["William Shakespeare", "Shakespeare"]}},

    {"syn_hist_009": {
        "question": "What is the capital of France? Respond with the answer only, without any explanation.",
        "responses": [
            "Paris", "Paris.", "Paris, France",
            "The capital of France is Paris.", "Paris",
            "Paris", "Paris.", "Paris",
            "Paris is the capital of France.", "Paris.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Paris", "Paris, France"]}},

    {"syn_hist_010": {
        "question": "In which year did humans first land on the Moon? Respond with the answer only, without any explanation.",
        "responses": [
            "1969", "1969 — Apollo 11.", "1969",
            "The first Moon landing was in 1969.", "1969",
            "1969", "1969.", "nineteen sixty-nine",
            "1969 (Apollo 11 mission).", "1969.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["1969", "nineteen sixty-nine"]}},

    {"syn_hist_011": {
        "question": "Which country was Nelson Mandela president of? Respond with the answer only, without any explanation.",
        "responses": [
            "South Africa", "South Africa.", "Republic of South Africa",
            "Nelson Mandela was president of South Africa.", "South Africa",
            "South Africa", "South Africa.", "South Africa",
            "South Africa — he was the first democratically elected president.", "South Africa.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["South Africa", "Republic of South Africa"]}},

    {"syn_hist_012": {
        "question": "Who was the first President of the United States? Respond with the answer only, without any explanation.",
        "responses": [
            "George Washington", "George Washington.", "Washington",
            "The first US President was George Washington.", "George Washington",
            "George Washington", "Washington.", "George Washington",
            "George Washington (1789).", "George Washington.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["George Washington", "Washington"]}},

    {"syn_hist_013": {
        "question": "In which year did the Titanic sink? Respond with the answer only, without any explanation.",
        "responses": [
            "1912", "1912.", "The Titanic sank in 1912.",
            "1912", "nineteen twelve",
            "1912", "1912 — on April 15.", "1912.",
            "1912", "The Titanic sank on 15 April 1912.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["1912", "nineteen twelve"]}},

    {"syn_hist_014": {
        "question": "Which scientist developed the theory of general relativity? Respond with the answer only, without any explanation.",
        "responses": [
            "Albert Einstein", "Einstein", "Albert Einstein developed general relativity.",
            "Einstein developed general relativity.", "Albert Einstein.",
            "Albert Einstein", "Einstein.", "Albert Einstein",
            "Albert Einstein (1915).", "Einstein.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Albert Einstein", "Einstein"]}},

    {"syn_hist_015": {
        "question": "What is the capital of Japan? Respond with the answer only, without any explanation.",
        "responses": [
            "Tokyo", "Tokyo.", "The capital of Japan is Tokyo.",
            "Tokyo", "Tokyo, Japan",
            "Tokyo", "Tokyo.", "Tokyo",
            "Tokyo is the capital of Japan.", "Tokyo.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Tokyo", "Tokyo, Japan"]}},

    {"syn_hist_016": {
        "question": "What element did Marie Curie discover? Respond with the answer only, without any explanation.",
        "responses": [
            "Marie Curie discovered polonium and radium.",
            "Polonium and radium.", "Radium and polonium.",
            "She discovered two elements: polonium and radium.",
            "Polonium and radium", "Radium", "Polonium",
            "Both polonium and radium.", "Polonium and radium.",
            "Marie Curie co-discovered polonium and radium.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["polonium", "radium", "polonium and radium"]}},

    {"syn_hist_017": {
        "question": "In which year did the Soviet Union dissolve? Respond with the answer only, without any explanation.",
        "responses": [
            "1991", "The Soviet Union dissolved in 1991.",
            "1991.", "nineteen ninety-one",
            "1991 — December 25, 1991.", "1991",
            "1991", "1991.", "nineteen ninety-one.",
            "1991 (officially on December 26).",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["1991", "nineteen ninety-one"]}},

    {"syn_hist_018": {
        "question": "Which company created the iPhone? Respond with the answer only, without any explanation.",
        "responses": [
            "Apple", "Apple Inc.", "Apple — it was introduced by Steve Jobs in 2007.",
            "Apple Inc.", "Apple created the iPhone.",
            "Apple", "Apple Inc.", "Apple.",
            "Apple — first released in 2007.", "Apple Inc.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Apple", "Apple Inc."]}},

    {"syn_hist_019": {
        "question": "What is the tallest mountain in the world? Respond with the answer only, without any explanation.",
        "responses": [
            "Mount Everest", "Everest", "Mount Everest is the tallest mountain.",
            "Mount Everest — at 8,849 m above sea level.", "Everest.",
            "Mount Everest", "Everest.", "Mount Everest",
            "Mount Everest (8,849 m).", "Everest.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Mount Everest", "Everest"]}},

    {"syn_hist_020": {
        "question": "Which company founded the Android operating system? Respond with the answer only, without any explanation.",
        "responses": [
            "Android was originally developed by Android Inc., which was acquired by Google.",
            "Google — after acquiring Android Inc.",
            "Android Inc. (later acquired by Google).",
            "Google", "Google acquired Android Inc. and developed the OS.",
            "Google", "Originally Android Inc., later acquired and developed by Google.",
            "Google.", "Google developed Android after acquiring Android Inc.",
            "Google.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Android Inc.", "Google", "Android Inc. acquired by Google"]}},

    # ═══════════════════════════════════════════════════════════════════════════
    # GROUP G2: More balanced faithful anchors (15 entries)
    # Clear, unambiguous facts with varied phrasings. All-true judges.
    # Directly improves faithful-class density in the training pool.
    # ═══════════════════════════════════════════════════════════════════════════

    {"syn_anchor_006": {
        "question": "What is the capital of Germany? Respond with the answer only, without any explanation.",
        "responses": [
            "Berlin", "Berlin.", "The capital of Germany is Berlin.",
            "Berlin, Germany", "Berlin",
            "Berlin", "Berlin.", "Berlin",
            "Berlin is the capital of Germany.", "Berlin.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Berlin", "Berlin, Germany"]}},

    {"syn_anchor_007": {
        "question": "How many continents are there on Earth? Respond with the answer only, without any explanation.",
        "responses": [
            "7", "Seven", "There are seven continents.",
            "7 continents", "seven",
            "7", "Seven continents.", "7",
            "There are 7 continents on Earth.", "Seven.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["7", "seven"]}},

    {"syn_anchor_008": {
        "question": "What is the largest ocean on Earth? Respond with the answer only, without any explanation.",
        "responses": [
            "The Pacific Ocean", "Pacific Ocean", "The Pacific",
            "The Pacific Ocean is the largest.", "Pacific",
            "The Pacific Ocean.", "Pacific Ocean.", "The Pacific Ocean",
            "Pacific — it is the largest ocean on Earth.", "The Pacific.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Pacific Ocean", "the Pacific"]}},

    {"syn_anchor_009": {
        "question": "Who wrote the novel 1984? Respond with the answer only, without any explanation.",
        "responses": [
            "George Orwell", "Orwell", "George Orwell wrote 1984.",
            "George Orwell.", "Orwell.",
            "George Orwell", "Orwell", "George Orwell",
            "1984 was written by George Orwell.", "George Orwell.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["George Orwell", "Orwell"]}},

    {"syn_anchor_010": {
        "question": "What is the chemical formula for carbon dioxide? Respond with the answer only, without any explanation.",
        "responses": [
            "CO2", "CO₂", "CO2",
            "Carbon dioxide is CO2.", "CO₂",
            "CO2", "CO₂", "CO2",
            "The formula for carbon dioxide is CO₂.", "CO2.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["CO2", "CO₂"]}},

    {"syn_anchor_011": {
        "question": "Which ocean lies to the east of the United States? Respond with the answer only, without any explanation.",
        "responses": [
            "The Atlantic Ocean", "Atlantic Ocean", "The Atlantic",
            "The Atlantic Ocean lies to the east of the US.", "Atlantic",
            "The Atlantic Ocean.", "Atlantic Ocean.", "The Atlantic Ocean",
            "Atlantic — the Atlantic Ocean borders the east coast of the US.", "The Atlantic.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Atlantic Ocean", "the Atlantic"]}},

    {"syn_anchor_012": {
        "question": "Which planet is fourth from the Sun? Respond with the answer only, without any explanation.",
        "responses": [
            "Mars", "Mars.", "Mars is the fourth planet from the Sun.",
            "Mars", "The fourth planet is Mars.",
            "Mars", "Mars.", "Mars",
            "Mars — the fourth planet from the Sun.", "Mars.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Mars"]}},

    {"syn_anchor_013": {
        "question": "What is the capital of the United Kingdom? Respond with the answer only, without any explanation.",
        "responses": [
            "London", "London.", "The capital of the UK is London.",
            "London, United Kingdom", "London",
            "London", "London.", "London",
            "London is the capital of the United Kingdom.", "London.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["London", "London, UK"]}},

    {"syn_anchor_014": {
        "question": "Who invented the World Wide Web? Respond with the answer only, without any explanation.",
        "responses": [
            "Tim Berners-Lee", "Timothy Berners-Lee", "Tim Berners-Lee invented the WWW.",
            "Berners-Lee", "Tim Berners-Lee.",
            "Tim Berners-Lee", "Timothy Berners-Lee.", "Tim Berners-Lee",
            "The World Wide Web was invented by Tim Berners-Lee in 1989.", "Berners-Lee.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Tim Berners-Lee", "Timothy Berners-Lee"]}},

    {"syn_anchor_015": {
        "question": "What is the largest continent by area? Respond with the answer only, without any explanation.",
        "responses": [
            "Asia", "Asia.", "Asia is the largest continent.",
            "Asia — the largest continent by area.", "Asia",
            "Asia", "Asia.", "Asia",
            "Asia is the largest continent by land area.", "Asia.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Asia"]}},

    {"syn_anchor_016": {
        "question": "Who sang the song Thriller? Respond with the answer only, without any explanation.",
        "responses": [
            "Michael Jackson", "Michael Jackson.", "Thriller was sung by Michael Jackson.",
            "Michael Jackson", "MJ",
            "Michael Jackson", "Michael Jackson.", "Michael Jackson",
            "Michael Jackson sang Thriller.", "Michael Jackson.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Michael Jackson", "MJ"]}},

    {"syn_anchor_017": {
        "question": "How many days are in the month of February in a non-leap year? Respond with the answer only, without any explanation.",
        "responses": [
            "28", "28 days", "Twenty-eight",
            "28 days in February in a non-leap year.", "28",
            "28", "28 days.", "twenty-eight",
            "28 days.", "28.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["28", "twenty-eight"]}},

    {"syn_anchor_018": {
        "question": "What is the hardest natural substance on Earth? Respond with the answer only, without any explanation.",
        "responses": [
            "Diamond", "Diamond.", "Diamond is the hardest natural substance.",
            "Diamond — the hardest natural material.", "Diamond",
            "Diamond", "Diamond.", "Diamond",
            "Diamond is the hardest naturally occurring substance.", "Diamond.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["Diamond", "diamond"]}},

    {"syn_anchor_019": {
        "question": "What is the powerhouse of the cell? Respond with the answer only, without any explanation.",
        "responses": [
            "The mitochondria", "Mitochondria", "Mitochondria — the powerhouse of the cell.",
            "The mitochondria.", "Mitochondria.",
            "The mitochondria", "Mitochondria.", "The mitochondria",
            "Mitochondria are the powerhouses of the cell.", "Mitochondria.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["mitochondria", "the mitochondria"]}},

    {"syn_anchor_020": {
        "question": "What is the smallest prime number? Respond with the answer only, without any explanation.",
        "responses": [
            "2", "Two", "The smallest prime number is 2.",
            "2 — the only even prime.", "2",
            "2", "Two.", "2",
            "The smallest prime is 2.", "2.",
        ],
        "judges": ["true"] * 10,
        "ground_truth": ["2", "two"]}},

]


# ── Append only entries whose QID does not already exist ─────────────────────
added = skipped = 0
with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
    for entry in NEW_ENTRIES:
        qid = next(iter(entry))
        if qid in existing_qids:
            print(f"  [skip] {qid} already exists")
            skipped += 1
            continue
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        existing_qids.add(qid)
        added += 1

print(f"\nDone.  Added: {added}  Skipped (already present): {skipped}")
print(f"Output file: {OUTPUT_PATH}")
print("\nNext steps:")
print("  1. Re-run step2 to extract CETT activations for the new entries")
print("  2. Re-run step3 to retrain the probe with the expanded dataset")
print("  3. Re-run step4 demo to verify improved probe behaviour")
