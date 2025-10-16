"""Run LLM-as-judge evaluations and write results to a JSON file.

This module exposes a single high-level function, :func:`judge`, which
is intended to be the canonical entrypoint for batch LLM-as-judge
evaluations in this repository. The function takes an OpenAI client,
loads a JSON file containing preference pairs, judges each pair with
the configured judge model in both original and swapped order, and
writes a timestamped JSON results file to disk.

Design and responsibilities
---------------------------
- The module focuses on data collection: it records raw judge responses
  (including swapped evaluations) and basic summary statistics. It does
  not attempt to perform higher-level analysis of the judge responses
  beyond extracting the final "A"/"B" preference for each evaluation.
  This separation keeps evaluation (data collection) and analysis
  concerns separate.

Input file format (required)
---------------------------
The input must be a JSON array where each element is an object with one
of the following accepted shapes. The runner normalizes either shape to
the canonical ``control``/``experimental`` pair used internally.

Canonical structure:

[
  {
    "control": "User: <query>\n\nAssistant: <response>",
    "experimental": "User: <query>\n\nAssistant: <response>"
  },
  ...
]


Output format (what judge() writes)
-------------------------------------
The runner writes a timestamped JSON file named ``eval_results_<ts>.json``
in ``output_dir``. The file has the following top-level keys:

- ``summary``: High-level statistics
  - ``total_pairs``: number of input pairs
  - ``orig_prefs`` / ``swap_prefs``: counts of preferences in each
    evaluation ordering (control vs experimental)
  - ``agreement_rate``: fraction of pairs where the judge's orig/swap
    responses were consistent (A<->B flip)
  - ``valid_evaluations``: number of pairs with valid both-direction
    preferences

- ``individual_results``: A list with two entries per input pair (the
  original order and the swapped order). Each entry is an object with
  these fields:
  - ``pair_id``: integer index of the pair in the input list
  - ``order``: ``"orig"`` or ``"swap"`` to indicate ordering
  - ``prompt``: the full prompt sent to the judge model
  - ``raw_resp``: the raw string returned by the judge model
  - ``pref``: extracted preference ``"A"`` or ``"B"`` (raises if
    unparseable)

- ``metadata``: runtime metadata
  - ``total_pairs``: number of unique pairs
  - ``evaluation_date``: timestamp
  - ``model``: the judge model id constant used for evaluation

Important behavioral note
-------------------------
All judge responses (including the swapped-order responses) are
recorded verbatim in the output under ``individual_results``. The
runner intentionally does not collapse, reconcile, or otherwise
post-process the raw swapped responses beyond extracting the single
preference label. Higher-level analysis (for example, aggregating
natural-language comparisons or filtering inconsistent answers) should
be performed by downstream analysis tooling so that the data collection
step remains auditable and reproducible.

CLI usage
---------
The module exposes a small CLI wrapper (``main()``) that accepts these
arguments:

- ``--data-file``: path to the input JSON file (default: ``pref_pairs.json``)
- ``--output-dir``: destination directory for results (default: current directory)

Example:

``python -m src.llm_judge.eval --data-file pref_pairs.json --output-dir results/``

"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Final, List, Literal, Optional, Tuple

import openai

# Global constants
PROMPT_TEMPLATE: Final = (
  "For the following query to a chatbot, which resp is more helpful?\n"
  "\n"
  "Query: {user_query}\n"
  "\n"
  "resp A:\n"
  "{resp_a}\n"
  "\n"
  "resp B:\n"
  "{resp_b}\n"
  "\n"
  "FIRST provide a one-sentence comparison of the two summaries and explain which "
  'you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your '
  "choice. Your response should use the format:\n"
  "Comparison: <one-sentence comparison and explanation>\n"
  'More helpful: <"A" or "B">'
)

MODEL_ID = "gpt-4.1-nano-2025-04-14"


def get_openai_client() -> openai.OpenAI:
  """Get OpenAI client instance."""
  api_key = os.environ.get("OAIKEY")
  if not api_key:
    raise ValueError("OAIKEY environment variable not set")
  return openai.OpenAI(api_key=api_key)


def parse_dialog(dialog: str) -> Tuple[str, str]:
  """Parse dialog format to extract query and response.

  Args:
    dialog: Dialog string in the format "User: <query>\n\nAssistant: <response>"

  Returns:
    A tuple of (query, response)
  """
  match = re.match(r"^User: (.+)\n\nAssistant: (.+)$", dialog, re.DOTALL)
  if not match:
    raise ValueError(f"dialog malformed. Got: {dialog}")

  query = match.group(1).strip()
  resp = match.group(2).strip()
  return query, resp


def call_judge(client: openai.OpenAI, prompt: str) -> Optional[str]:
  """Call OpenAI API to judge between two responses."""
  resp = client.chat.completions.create(
    model=MODEL_ID,
    # max_completion_tokens=2000,
    messages=[{"role": "user", "content": prompt}],
  )
  resp = resp.choices[0].message.content
  if len(resp) == 0:
    raise ValueError(f"Judge returned empty response: {resp=}")
  return resp


def parse_judge_resp(resp_text: str) -> Literal["A", "B"]:
  """Parse the judge response to extract the preference."""
  if not resp_text:
    raise ValueError("No response text to parse.")

  # LLMs will not match the requested format exactly.
  lines = resp_text.strip().split("\n")
  line = lines[-1].strip()  # Last line should contain the preference
  line = line[-10:]  # Should be in the last 10 characters.
  if "A" in line and "B" not in line:
    pref = "A"
  elif "B" in line and "A" not in line:
    pref = "B"
  else:
    raise ValueError(
      "Malformed judge response: could not extract preference 'A' or 'B'."
      f" Got: {resp_text!r}"
    )
  return pref


def judge_pair(
  client: openai.OpenAI,
  control_dialog: str,
  experimental_dialog: str,
  pair_id: int,
) -> List[Dict[str, Any]]:
  """judge a single preference pair."""
  query_control, resp_control = parse_dialog(control_dialog)
  query_exp, resp_exp = parse_dialog(experimental_dialog)
  if query_control != query_exp:
    raise ValueError(
      f"Queries do not match {pair_id}: '{query_control}' vs '{query_exp}'"
    )
  query = query_control  # Both are the same
  del query_control, query_exp

  results = []

  # Original evaluation (control vs experimental)
  orig_prompt = PROMPT_TEMPLATE.format(
    user_query=query,
    resp_a=resp_control,
    resp_b=resp_exp,
  )

  orig_resp = call_judge(client, orig_prompt)
  orig_pref = parse_judge_resp(orig_resp)

  results.append(
    {
      "pair_id": pair_id,
      "order": "orig",
      "prompt": orig_prompt,
      "raw_resp": orig_resp,
      "pref": orig_pref,
    }
  )

  # Swapped evaluation (experimental vs control)
  swap_prompt = PROMPT_TEMPLATE.format(
    user_query=query,
    resp_a=resp_exp,  # Swapped
    resp_b=resp_control,  # Swapped
  )

  swap_resp = call_judge(client, swap_prompt)
  swap_pref = parse_judge_resp(swap_resp)

  results.append(
    {
      "pair_id": pair_id,
      "order": "swap",
      "prompt": swap_prompt,
      "raw_resp": swap_resp,
      "pref": swap_pref,
    }
  )

  return results


def judge(
  client: openai.OpenAI,
  control_file: str,
  experimental_file: str,
  output_dir: str,
) -> str:
  """LLMjudge decides whether control or experimental responses are preferred.

  This version accepts two input files: control and experimental. 
  
  Each input file must contain a JSON array of dialog strings in the required 
  chat format (e.g. ``User: <query>\n\nAssistant: <response>``),
  and both arrays must have the same length. The function writes two JSON
  files into ``output_dir``: ``summary.json`` and ``results.json``. Returns the
  tuple ``(summary_path, results_path)``.

  Args:
    client: OpenAI client instance.
    control_file: Path to JSON file containing control dialog strings.
    experimental_file: Path to JSON file containing experimental dialog strings.
    output_dir: Directory where results JSON files will be saved.

  Returns:
    Path to the summary JSON file.
    Path to the results JSON file.
  """

  if not os.path.exists(control_file):
    raise FileNotFoundError(f"Control file '{control_file}' not found.")
  if not os.path.exists(experimental_file):
    raise FileNotFoundError(f"Experimental file '{experimental_file}' not found.")

  with open(control_file, "r") as f:
    control_list = json.load(f)
  with open(experimental_file, "r") as f:
    experimental_list = json.load(f)

  if not isinstance(control_list, list) or not isinstance(experimental_list, list):
    raise ValueError("Both control and experimental files must contain a JSON array of dialog strings.")

  if len(control_list) != len(experimental_list):
    raise ValueError(
      f"Control/experimental files differ in length: {len(control_list)} vs {len(experimental_list)}"
    )

  pref_pairs_count = len(control_list)
  print(f"Loaded {pref_pairs_count} pref pairs (control/experimental)")

  all_results = []
  for i, (control_dialog, experimental_dialog) in enumerate(zip(control_list, experimental_list)):
    pair_results = judge_pair(
      client,
      control_dialog,
      experimental_dialog,
      i,
    )
    all_results.extend(pair_results)
    time.sleep(1)

  # Calculate summary statistics
  orig_prefs = [r["pref"] for r in all_results if r["order"] == "orig"]
  swap_prefs = [r["pref"] for r in all_results if r["order"] == "swap"]

  orig_control_wins = orig_prefs.count("A")
  orig_experimental_wins = orig_prefs.count("B")
  swap_control_wins = swap_prefs.count("B")
  swap_experimental_wins = swap_prefs.count("A")

  agreements = 0
  total_valid_pairs = 0
  for i in range(len(orig_prefs)):
    if orig_prefs[i] and swap_prefs[i]:
      total_valid_pairs += 1
      # When swapped, consistent judge should give opposite preference
      # A in original should become B in swapped (both prefer same model)
      if (orig_prefs[i] == "A" and swap_prefs[i] == "B") or (
        orig_prefs[i] == "B" and swap_prefs[i] == "A"
      ):
        agreements += 1

  agreement_rate = (
    agreements / total_valid_pairs if total_valid_pairs > 0 else 0
  )

  summary = {
    "total_pairs": pref_pairs_count,
    "orig_prefs": {
      "control": orig_control_wins,
      "experimental": orig_experimental_wins,
    },
    "swap_prefs": {
      "control": swap_control_wins,
      "experimental": swap_experimental_wins,
    },
    "agreement_rate": agreement_rate,
    "valid_evaluations": total_valid_pairs,
  }

  # Ensure output dir exists
  os.makedirs(output_dir, exist_ok=True)
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
  results_path = os.path.join(output_dir, f"results_{timestamp}.json")

  with open(summary_path, "w") as sf:
    json.dump(summary, sf, indent=2)

  with open(results_path, "w") as rf:
    json.dump(all_results, rf, indent=2)

  return summary_path, results_path

def main() -> int:
  """CLI entry point for running LLM evaluation."""
  parser = argparse.ArgumentParser(
    description="LLM-as-judge evaluation for model comparison"
  )
  parser.add_argument(
    "--control-file",
    default="control_pref_pairs.json",
    help="Path to JSON file containing control dialog strings",
  )
  parser.add_argument(
    "--experimental-file",
    default="experimental_pref_pairs.json",
    help="Path to JSON file containing experimental dialog strings",
  )
  parser.add_argument(
    "--output-dir",
    default=".",
    help="Directory where results JSON files will be saved",
  )

  args = parser.parse_args()

  try:
    client = get_openai_client()
    summary_path, results_path = judge(
      client=client,
      control_file=args.control_file,
      experimental_file=args.experimental_file,
      output_dir=args.output_dir,
    )

    # Re-open the summary file.
    with open(summary_path, "r") as sf:
      summary = json.load(sf)
      
    # Display summary results
    print("\n" + "=" * 50)
    print("LLMJUDGE SUMMARY")
    print("=" * 50)
    print(f"Total pairs judged: {summary['total_pairs']}")
    print(
      f"Forward prefs - Control: {summary['orig_prefs']['control']}, "
      f"Experimental: {summary['orig_prefs']['experimental']}"
    )
    print(
      f"Swap prefs - Control: {summary['swap_prefs']['control']}, "
      f"Experimental: {summary['swap_prefs']['experimental']}"
    )
    print(f"Agreement rate: {summary['agreement_rate']:.2%}")

    print("Evaluation completed successfully!")
    return 0
  except Exception as e:
    print(f"Evaluation failed: {e}")
    return 1


if __name__ == "__main__":
  sys.exit(main())
