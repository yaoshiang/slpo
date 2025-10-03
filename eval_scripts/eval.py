"""This eval script follows the original DPO llm-as-judge evaluation.py

INPUT FORMAT:
Expected input file: preference_pairs.json
Structure (foll    # Reverse evaluation (rejected vs chosen) - swap responses
    reverse_prompt = prompt_template.format(
        user_query=query,
        response_a=rejected_response,  # Swapped
        response_b=chosen_response     # Swapped
    )Anthropic HH format):
[
    {
        "chosen": "Human: User query/question string\n\nAssistant: Preferred response text",
        "rejected": "Human: User query/question string\n\nAssistant: Less preferred response text"
    },
    ...
]

OUTPUT FORMAT:
Creates timestamped JSON file: eval_results_<timestamp>.json
Structure:
{
    "summary": {
        "total_pairs": <int>,
        "forward_preferences": {"A": <count>, "B": <count>},
        "reverse_preferences": {"A": <count>, "B": <count>},
        "agreement_rate": <float 0-1>,
        "valid_evaluations": <int>
    },
    "individual_results": [
        {
            "pair_id": <int>,
            "order": "forward" | "reverse",
            "prompt": "Full prompt sent to judge",
            "raw_response": "Raw LLM judge response",
            "comparison": "Extracted comparison explanation",
            "preference": "A" | "B" | null,
            "preference_original_order": "A" | "B" | null  // Only for reverse order
        },
        ...
    ],
    "metadata": {
        "total_pairs": <int>,
        "evaluation_date": "YYYY-MM-DD HH:MM:SS",
        "model": "gpt-4o"
    }
}

BIAS MITIGATION:
- Each pair is evaluated twice: A vs B (forward) and B vs A (reverse)
- Reverse evaluation preferences are mapped back to original ordering
- Agreement rate measures consistency between forward/reverse evaluations
"""

import json
import os
import time

from openai import OpenAI

# Initialize OpenAI client
key = os.environ.get("OAIKEY")
if not key:
    raise ValueError("OAIKEY environment variable not set")

client = OpenAI(api_key=key)

prompt_template = (
    "For the following query to a chatbot, which response is more helpful?\n\n"
    "Query: {user_query}\n\n"
    "Response A:\n"
    "{response_a}\n\n"
    "Response B:\n"
    "{response_b}\n\n"
    "FIRST provide a one-sentence comparison of the two responses and explain "
    'which you feel is more helpful. SECOND, on a new line, state only "A" or '
    '"B" to indicate which response is more helpful. Your response should use '
    "the format:\n"
    "Comparison: <one-sentence comparison and explanation>\n"
    'More helpful: <"A" or "B">'
)


def call_openai_judge(prompt):
    """Call OpenAI API to judge between two responses."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o as gpt-5-nano might not be available yet
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator comparing chatbot responses.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Deterministic for consistency
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def parse_hh_conversation(conversation_text):
    """Parse Anthropic HH format conversation to extract query and response."""
    # Split on "\n\nAssistant:" to separate human and assistant parts
    parts = conversation_text.split("\n\nAssistant:")
    if len(parts) != 2:
        raise ValueError(f"Invalid conversation format: {conversation_text}")

    # Extract query (remove "Human: " prefix)
    query = parts[0].replace("Human: ", "").strip()
    # Extract response
    response = parts[1].strip()

    return query, response


def parse_judge_response(response_text):
    """Parse the judge response to extract the preference."""
    if not response_text:
        return None, None

    lines = response_text.strip().split("\n")
    comparison = None
    preference = None

    for line in lines:
        if line.startswith("Comparison:"):
            comparison = line.replace("Comparison:", "").strip()
        elif line.startswith("More helpful:"):
            pref_text = line.replace("More helpful:", "").strip()
            if pref_text in ["A", "B"]:
                preference = pref_text

    return comparison, preference


def evaluate_pair(control_conversation, experimental_conversation, pair_id):
    """Evaluate a single preference pair with ordering bias mitigation."""
    # Parse the conversation format
    query, control_response = parse_hh_conversation(control_conversation)
    _, experimental_response = parse_hh_conversation(experimental_conversation)

    # Verify queries match (they should be identical)
    query_experimental, _ = parse_hh_conversation(experimental_conversation)
    if query != query_experimental:
        print(f"Warning: Queries don't match for pair {pair_id}")

    results = []

    # Forward evaluation (control vs experimental)
    forward_prompt = prompt_template.format(
        user_query=query,
        response_a=control_response,
        response_b=experimental_response,
    )

    print(f"Evaluating pair {pair_id} - Forward (A vs B)...")
    forward_response = call_openai_judge(forward_prompt)
    forward_comparison, forward_preference = parse_judge_response(
        forward_response
    )

    results.append(
        {
            "pair_id": pair_id,
            "order": "forward",
            "prompt": forward_prompt,
            "raw_response": forward_response,
            "comparison": forward_comparison,
            "preference": forward_preference,
        }
    )

    # Reverse evaluation (experimental vs control) - swap responses
    reverse_prompt = prompt_template.format(
        user_query=query,
        response_a=experimental_response,  # Swapped
        response_b=control_response,  # Swapped
    )

    print(f"Evaluating pair {pair_id} - Reverse (B vs A)...")
    reverse_response = call_openai_judge(reverse_prompt)
    reverse_comparison, reverse_preference = parse_judge_response(
        reverse_response
    )

    # Flip the preference back to original ordering
    if reverse_preference == "A":
        reverse_preference_original = "B"
    elif reverse_preference == "B":
        reverse_preference_original = "A"
    else:
        reverse_preference_original = None

    results.append(
        {
            "pair_id": pair_id,
            "order": "reverse",
            "prompt": reverse_prompt,
            "raw_response": reverse_response,
            "comparison": reverse_comparison,
            "preference": reverse_preference,  # In swapped context
            "preference_original_order": reverse_preference_original,  # Flipped back
        }
    )

    return results


def load_preference_data(filepath):
    """Load preference pairs from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_results(results, summary, output_path):
    """Save both individual results and summary statistics."""
    output = {
        "summary": summary,
        "individual_results": results,
        "metadata": {
            "total_pairs": len(results) // 2,  # Each pair has 2 evaluations
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "gpt-4o",
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    # Load preference pairs
    data_file = "preference_pairs.json"  # Expected format: [{"query": "...", "response_a": "...", "response_b": "..."}]

    if not os.path.exists(data_file):
        print(f"Creating example data file: {data_file}")
        example_data = [
            {
                "control": "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris.",
                "experimental": "Human: What is the capital of France?\n\nAssistant: Paris is the capital and largest city of France.",
            }
        ]
        with open(data_file, "w") as f:
            json.dump(example_data, f, indent=2)
        print(
            f"Please populate {data_file} with your model comparison pairs (control vs experimental) and run again."
        )
        return

    preference_pairs = load_preference_data(data_file)
    print(f"Loaded {len(preference_pairs)} preference pairs")

    all_results = []

    for i, pair in enumerate(preference_pairs):
        pair_results = evaluate_pair(pair["control"], pair["experimental"], i)
        all_results.extend(pair_results)

        # Small delay to be respectful to API
        time.sleep(1)

    # Calculate summary statistics
    forward_prefs = [
        r["preference"] for r in all_results if r["order"] == "forward"
    ]
    reverse_prefs = [
        r["preference_original_order"]
        for r in all_results
        if r["order"] == "reverse"
    ]

    # Count preferences
    forward_a_wins = forward_prefs.count("A")
    forward_b_wins = forward_prefs.count("B")
    reverse_a_wins = reverse_prefs.count("A")
    reverse_b_wins = reverse_prefs.count("B")

    # Agreement between forward and reverse
    agreements = 0
    total_valid_pairs = 0
    for i in range(len(forward_prefs)):
        if forward_prefs[i] and reverse_prefs[i]:
            total_valid_pairs += 1
            if forward_prefs[i] == reverse_prefs[i]:
                agreements += 1

    agreement_rate = (
        agreements / total_valid_pairs if total_valid_pairs > 0 else 0
    )

    summary = {
        "total_pairs": len(preference_pairs),
        "forward_preferences": {"A": forward_a_wins, "B": forward_b_wins},
        "reverse_preferences": {"A": reverse_a_wins, "B": reverse_b_wins},
        "agreement_rate": agreement_rate,
        "valid_evaluations": total_valid_pairs,
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total pairs evaluated: {len(preference_pairs)}")
    print(f"Forward preferences - A: {forward_a_wins}, B: {forward_b_wins}")
    print(f"Reverse preferences - A: {reverse_a_wins}, B: {reverse_b_wins}")
    print(f"Agreement rate: {agreement_rate:.2%}")

    # Save results
    output_file = f"eval_results_{int(time.time())}.json"
    save_results(all_results, summary, output_file)


if __name__ == "__main__":
    main()
