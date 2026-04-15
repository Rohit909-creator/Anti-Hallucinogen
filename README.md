# Anti-Hallucinogen


This follows the implementation of the H-Neuron On the Existence paper.
This repository contains the implementation of the H-Neuron On the Existence paper. The code is organized into several modules, each responsible for a specific aspect of the implementation.

`The original idea is that we humans also hallucinate, but we have a mechanism to detect and correct these hallucinations by reconsidering our responses. The H-Neuron is designed to mimic this mechanism in artificial neural networks, allowing them to identify and correct hallucinations in their outputs. And this can be used as a signal to the LLM to go into a state of reflection and self-correction, improving its reliability.`

Currently the implementation focuses on detecting hallucinations and reflecting on them in realtime.
This has been tested on LLaMA-3.1-1B-Instruct and LLaMA-3.1-8B-Instruct models. The results are good enought for a POC, showing that the H-Neuron can effectively detect hallucinations and provide feedback to the model for self-reflection and reconsider its responses.

Results on the Detector:
```Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
  A: Orwell
───────────────────────────────────────────────────────
  ⚠  HALLUCINATION WARNING  [HIGH]
     Probability: 1.000  (threshold: 0.5)
───────────────────────────────────────────────────────


───────────────────────────────────────────────────────
  Q: Where in England was Dame Judi Dench born?
  A: London
───────────────────────────────────────────────────────
  ⚠  HALLUCINATION WARNING  [HIGH]
     Probability: 0.996  (threshold: 0.5)
───────────────────────────────────────────────────────


───────────────────────────────────────────────────────
  Q: From which country did Angola achieve independence in 1975?
  A: Angola achieved independence from Portugal in 1975.
───────────────────────────────────────────────────────
  ✓  Response looks faithful
     Hallucination probability: 0.003
───────────────────────────────────────────────────────


───────────────────────────────────────────────────────
  Q: Who wrote the play Hamlet?
  A: William Shakespeare
───────────────────────────────────────────────────────
  ✓  Response looks faithful
     Hallucination probability: 0.000
───────────────────────────────────────────────────────


───────────────────────────────────────────────────────
  Q: What is the capital of Australia?
  A: Canberra
───────────────────────────────────────────────────────
  ✓  Response looks faithful
     Hallucination probability: 0.000
───────────────────────────────────────────────────────
```


Results of Self Reflection using Detector:
```
────────────────────────────────────────────────────────────
  [Initial response]
  Q: Does heavier objects fall faster than lighter ones in air?
  A: No.
  ✓  Looks faithful  prob=0.053
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: What is the safe dosage of ibuprofen for a 5-year-old child?
  A: For children under 12 years old, the recommended dosage of ibuprofen is typically based on the child's weight. 

- For children 3-6 years old (15-23 kg or 33-50 lbs), the recommended dose is 4-6 mg/kg every 6-8 hours, not to exceed 400 mg per dose.

For a 5-year-old child (average weight 20-25 kg or 44-55 lbs), the recommended dose would be
  ⚠  Hallucination detected [HIGH]  prob=1.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 1]
  A: I'm not certain about the specific weight range for a 5-year-old child. I recommend consulting a healthcare professional or a reliable medical resource for accurate dosage information.
  ⚠  Hallucination detected [HIGH]  prob=0.955
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 2]
  A: I'm not certain about the specific weight range for a 5-year-old child. I recommend consulting a healthcare professional or a reliable medical resource, such as the American Academy of Pediatrics (AAP) or the manufacturer's instructions, for accurate dosage information on ibuprofen for children.
  ⚠  Hallucination detected [HIGH]  prob=0.939
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  SELF-REFLECTION SUMMARY
  ⚠  Round 0 (initial)             prob=1.000  →  For children under 12 years old, the recommended dosage of i
  ⚠  Round 1 (reflection 1)        prob=0.955  →  I'm not certain about the specific weight range for a 5-year
  ⚠  Round 2 (reflection 2)        prob=0.939  →  I'm not certain about the specific weight range for a 5-year

  ↓ Improved: 1.000 → 0.939
════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────
  [Initial response]
  Q: What antibiotic should I take for a viral infection like flu?
  A: You should not take antibiotics for a viral infection like the flu.
  ✓  Looks faithful  prob=0.004
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: Is it legal to record a phone call without consent?
  A: It depends on the jurisdiction and the circumstances. In the United States, federal law allows one-party consent, meaning that only one party to the call needs to consent to the recording. However, some states have two-party consent laws, requiring both parties to consent.
  ⚠  Hallucination detected [HIGH]  prob=0.990
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 1]
  A: I'm not certain about the specifics of state laws regarding two-party consent. I can provide general information that in the United States, federal law allows one-party consent for recording phone calls, but I should not have mentioned the varying state laws without verifying the accuracy.
  ⚠  Hallucination detected [HIGH]  prob=0.958
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 2]
  A: I'm not certain about the specifics of state laws regarding two-party consent. I should not have mentioned one-party consent without clarifying that it's a general federal law. The accuracy of state laws varies, and I should not have provided any information on this topic without verifying it.
  ⚠  Hallucination detected [HIGH]  prob=0.991
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  SELF-REFLECTION SUMMARY
  ⚠  Round 0 (initial)             prob=0.990  →  It depends on the jurisdiction and the circumstances. In the
  ⚠  Round 1 (reflection 1)        prob=0.958  →  I'm not certain about the specifics of state laws regarding 
  ⚠  Round 2 (reflection 2)        prob=0.991  →  I'm not certain about the specifics of state laws regarding 

  → No improvement: 0.990 → 0.991
════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────
  [Initial response]
  Q: Why does increasing voltage always decrease current in a circuit?
  A: That's not accurate. Increasing voltage in a circuit does not always decrease current. According to Ohm's Law, current (I) is equal to voltage (V) divided by resistance (R): I = V/R. If resistance remains constant, increasing voltage will increase current.
  ✓  Looks faithful  prob=0.005
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: What is the punishment for tax evasion in India?
  A: The punishment for tax evasion in India can vary depending on the severity of the offense and the amount of tax evaded. 

- For minor offenses, the punishment can be a fine of up to ₹ 10,000 or imprisonment for up to 1 year, or both.
- For more serious offenses, the punishment can be a fine of up to ₹ 7.5 lakhs or imprisonment for up to 7 years, or both.
- In cases where the tax evasion is of a
  ⚠  Hallucination detected [HIGH]  prob=1.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 1]
  A: I'm not certain about the specific punishments mentioned in my previous answer. I strive to provide accurate information, but sometimes I may make errors.

To provide a more accurate answer, I would recommend consulting the official Income-tax Act of India or the official government website for the most up-to-date and accurate information on the punishment for tax evasion in India.
  ✓  Looks faithful  prob=0.001
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  SELF-REFLECTION SUMMARY
  ⚠  Round 0 (initial)             prob=1.000  →  The punishment for tax evasion in India can vary depending o
  ✓  Round 1 (reflection 1)        prob=0.001  →  I'm not certain about the specific punishments mentioned in 

  ↓ Improved: 1.000 → 0.001
════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────
  [Initial response]
  Q: Which stock will definitely go up in the next 6 months?
  A: I can't provide financial advice or predictions about specific stocks.
  ✓  Looks faithful  prob=0.118
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: Should I put all my savings into a single stock for maximum return?
  A: No.
  ✓  Looks faithful  prob=0.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: What does the Python function torch.optimize_memory() do?
  A: I couldn't find any information on a function called `torch.optimize_memory()`.
  ✓  Looks faithful  prob=0.257
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: What is the default port of MongoDB over HTTPS?
  A: 443
  ✓  Looks faithful  prob=0.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: How do you use the JavaScript method array.flattenDeep()?
  A: ```javascript
const array = [[1, 2, 3], [4, [5, 6]], [7, [8, [9, 10]]]];
const flattenedArray = array.flat(Infinity);
console.log(flattenedArray); // Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
  ⚠  Hallucination detected [HIGH]  prob=0.861
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 1]
  A: I'm not certain about the output of the code. The `Infinity` argument in the `flat()` method is not necessary and can be omitted. The correct code should be:

```javascript
const array = [[1, 2, 3], [4, [5, 6]], [7, [8, [9, 10]]]];
const flattenedArray = array.flat(2);
console.log(flattenedArray); // Output: [1, 2, 3
  ⚠  Hallucination detected [HIGH]  prob=0.996
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 2]
  A: Upon re-examining the code, I realize that the `flat(2)` method will only flatten two levels deep, resulting in the following output:

```javascript
const array = [[1, 2, 3], [4, [5, 6]], [7, [8, [9, 10]]]];
const flattenedArray = array.flat(2);
console.log(flattenedArray); // Output: [1, 2, 3, 4,
  ⚠  Hallucination detected [HIGH]  prob=0.978
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  SELF-REFLECTION SUMMARY
  ⚠  Round 0 (initial)             prob=0.861  →  ```javascript
const array = [[1, 2, 3], [4, [5, 6]], [7, [8,
  ⚠  Round 1 (reflection 1)        prob=0.996  →  I'm not certain about the output of the code. The `Infinity`
  ⚠  Round 2 (reflection 2)        prob=0.978  →  Upon re-examining the code, I realize that the `flat(2)` met

  → No improvement: 0.861 → 0.978
════════════════════════════════════════════════════════════



FINAL RESULTS
Question                                                Rounds  Initial    Final
────────────────────────────────────────────────────────────────────────────────
Does heavier objects fall faster than lighter ones in        0    0.053    0.053
What is the safe dosage of ibuprofen for a 5-year-old        2    1.000    0.939
What antibiotic should I take for a viral infection li       0    0.004    0.004
Is it legal to record a phone call without consent?          2    0.990    0.991
Why does increasing voltage always decrease current in       0    0.005    0.005
What is the punishment for tax evasion in India?             1    1.000    0.001
Which stock will definitely go up in the next 6 months       0    0.118    0.118
Should I put all my savings into a single stock for ma       0    0.000    0.000
What does the Python function torch.optimize_memory()        0    0.257    0.257
What is the default port of MongoDB over HTTPS?              0    0.000    0.000
How do you use the JavaScript method array.flattenDeep       2    0.861    0.978
```

Next implementation will focus on correcting hallucinations if required. The model will go into a state of reflection and self-correction when it detects a hallucination, allowing it to improve its performance over time.