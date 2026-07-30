# Adversarial Review

This review treats generated data, model messages, and recorded trajectories as
untrusted inputs. It covers the Sudoku example itself, not the entire AReno
training stack.

## Threats and mitigations

| Threat | Mitigation | Verification |
| --- | --- | --- |
| Hidden solution leaks into the policy context | The prompt and tool results are constructed only from the puzzle and live board. Candidate inspection applies row, column, and box rules without reading the stored solution. | Loader rejects a prompt containing the complete solution; CPU tests verify that candidates are not reduced to the hidden digit. |
| Generator produces only permutations of one canonical grid | Completed boards now use randomized MRV backtracking rather than transforming one Latin-square pattern. | Deterministic tests require distinct puzzle and solution hashes in a sample. |
| Train/test exact leakage | Every record declares a split and contains puzzle and solution hashes. `audit_dataset.py` rejects duplicate IDs, puzzles, or solutions across all supplied files. | The audit test intentionally reuses a seed across train and test and must fail. |
| Invalid or non-unique puzzles enter training | The generator removes a clue only when uniqueness remains. The loader independently checks the complete solution, givens, and solution count. | Generator output is also cross-checked against an independent Algorithm X implementation in CPU tests. |
| Difficulty label is based only on clue count | The v2 label combines fixed clue count with deterministic full uniqueness-proof search overhead. Metadata is recalculated by the loader. | Tampered metadata and out-of-band effort are rejected. |
| Search difficulty changes with a random solver path | Difficulty analysis uses deterministic MRV and ascending candidate order, and counts the full proof required for a unique puzzle rather than only the first solution path. | Stored counters must exactly match recalculation. |
| Model claims that a puzzle is solved | Reward ignores free-form claims and replays parsed tool calls from the original puzzle. | Solved status requires the replayed board to equal the unique stored solution. |
| Model stops early to collect progress reward | Every unfinished trajectory is capped at a non-positive base outcome; exhaustion adds another penalty. | A correct partial trajectory is asserted to score at most zero. |
| Model farms observations by repeating calls | Repeating the same tool and arguments against the same board state is counted and penalized. | A repeated inspection must score below distinct inspections. |
| Model batches many calls into one turn | The environment executes every emitted call in order so the transcript stays valid, while reward counts and penalizes extra calls in a single assistant message. | Multi-call and one-call trajectories are compared in CPU tests. |
| Malformed calls bypass the action budget | Unknown tools, malformed JSON, invalid coordinates, forbidden edits, and empty undo all consume an action and count as invalid. Calls after terminal state cannot mutate the board. | Boundary and exhaustion tests cover these paths. |
| Reported metrics trust model-provided state | `evaluate.py` invokes the same deterministic replay used by reward. | Episode metrics are derived from source puzzle and parsed calls. |

## Residual risks

1. **Difficulty is computational, not human.** The three bands measure one
   deterministic solver's proof effort. They do not guarantee a human-style
   technique such as hidden singles, X-Wing, or Swordfish.
2. **The leakage audit is exact, not symmetry-aware.** It catches identical
   puzzles and completed grids but not every isomorphic board produced through
   row, column, digit, rotation, or reflection transformations.
3. **Reward weights still require empirical calibration.** The ordering and
   bounds have CPU tests, but model behavior can expose new reward-hacking
   strategies. Inspect action/redundancy/solve metrics on Kaggle.
4. **The model path is not locally GPU-validated.** CPU tests establish game,
   generator, loader, replay, and agent transcript behavior. They do not prove
   that a Kaggle CUDA environment can complete an AReno optimization step.
5. **Evaluation export is a separate integration boundary.** `evaluate.py`
   consumes replayable episode JSONL; it does not currently launch a checkpoint
   against the frozen test split by itself.

## Kaggle acceptance gates

Before scaling a run:

1. Audit all three generated splits with `--verify-solutions`.
2. Inspect the normalized training dataset and require `"ok": true`.
3. Complete one actual trainer step and retain `areno env --json`, metrics, and
   the exact command.
4. Manually inspect several replayable trajectories for solution leakage,
   malformed call handling, repeated calls, and reward ordering.
5. Keep validation and test splits out of training, then compare solve rate by
   difficulty against an untrained checkpoint using the same sampling settings.
