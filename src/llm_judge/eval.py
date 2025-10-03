"""LLM-as-judge evaluation for model comparison."""

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


def evaluate_pair(
  client: openai.OpenAI,
  control_dialog: str,
  experimental_dialog: str,
  pair_id: int,
) -> List[Dict[str, Any]]:
  """Evaluate a single preference pair."""
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


def evaluate(
  client: openai.OpenAI,
  data_file: str,
  output_dir: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
  """Main evaluation function."""
  if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file '{data_file}' not found.")

  with open(data_file, "r") as f:
    pref_pairs = json.load(f)
  print(f"Loaded {len(pref_pairs)} pref pairs")

  all_results = []
  for i, pair in enumerate(pref_pairs):
    pair_results = evaluate_pair(
      client,
      pair["control"],
      pair["experimental"],
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
    "total_pairs": len(pref_pairs),
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

  output_file = os.path.join(
    output_dir, f"eval_results_{int(time.time())}.json"
  )

  # Save results inline
  output = {
    "summary": summary,
    "individual_results": all_results,
    "metadata": {
      "total_pairs": len(all_results) // 2,
      "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
      "model": MODEL_ID,
    },
  }

  with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

  print(f"Results saved to {output_file}")

  return summary, all_results


def main() -> int:
  """CLI entry point for running LLM evaluation."""
  parser = argparse.ArgumentParser(
    description="LLM-as-judge evaluation for model comparison"
  )
  parser.add_argument(
    "--data-file",
    default="pref_pairs.json",
    help="Path to JSON file containing preference pairs",
  )
  parser.add_argument(
    "--output-dir",
    default=".",
    help="Directory where results JSON file will be saved",
  )

  args = parser.parse_args()

  try:
    client = get_openai_client()
    summary, results = evaluate(
      client=client,
      data_file=args.data_file,
      output_dir=args.output_dir,
    )

    # Display summary results
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total pairs evaluated: {summary['total_pairs']}")
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
