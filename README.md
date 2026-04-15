# Anti-Hallucinogen


This follows the implementation of the H-Neuron On the Existence paper.
This repository contains the implementation of the H-Neuron On the Existence paper. The code is organized into several modules, each responsible for a specific aspect of the implementation.

`The original idea is that we humans also hallucinate, but we have a mechanism to detect and correct these hallucinations. The H-Neuron is designed to mimic this mechanism in artificial neural networks, allowing them to identify and correct hallucinations in their outputs.`

Currently the implementation focuses on detecting hallucinations.

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
  Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
  A: Orwell
  ⚠  Hallucination detected [HIGH]  prob=1.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 1]
  A: I am not certain.
  ✓  Looks faithful  prob=0.001
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  SELF-REFLECTION SUMMARY
  ⚠  Round 0 (initial)             prob=1.000  →  Orwell
  ✓  Round 1 (reflection 1)        prob=0.001  →  I am not certain.

  ↓ Improved: 1.000 → 0.001
════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────
  [Initial response]
  Q: Where in England was Dame Judi Dench born?
  A: Nottingham
  ⚠  Hallucination detected [HIGH]  prob=0.999
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Reflection round 1]
  A: Nottingham
  ✓  Looks faithful  prob=0.041
────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  SELF-REFLECTION SUMMARY
  ⚠  Round 0 (initial)             prob=0.999  →  Nottingham
  ✓  Round 1 (reflection 1)        prob=0.041  →  Nottingham

  ↓ Improved: 0.999 → 0.041
════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────
  [Initial response]
  Q: From which country did Angola achieve independence in 1975?
  A: Angola
  ✓  Looks faithful  prob=0.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: Who wrote the play Hamlet?
  A: William Shakespeare
  ✓  Looks faithful  prob=0.000
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
  [Initial response]
  Q: What is the capital of Australia?
  A: Canberra
  ✓  Looks faithful  prob=0.000
────────────────────────────────────────────────────────────


FINAL RESULTS
Question                                                Rounds  Initial    Final
────────────────────────────────────────────────────────────────────────────────
Which American-born Sinclair won the Nobel Prize for L       1    1.000    0.001
Where in England was Dame Judi Dench born?                   1    0.999    0.041
From which country did Angola achieve independence in        0    0.000    0.000
Who wrote the play Hamlet?                                   0    0.000    0.000
What is the capital of Australia?                            0    0.000    0.000
```

Next implementation will focus on correcting hallucinations if required. The model will go into a state of reflection and self-correction when it detects a hallucination, allowing it to improve its performance over time.