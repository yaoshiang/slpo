"""Unit tests for LLM judge module."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import openai
import pytest

from llm_judge import eval


class TestSetup:
  """Test setup and configuration."""

  def setup_method(self):
    """Setup method run before each test."""
    # Assert OAIKEY exists - we won't run tests without a real key
    assert "OAIKEY" in os.environ, (
      "OAIKEY environment variable must be set to run these tests"
    )
    self.api_key = os.environ["OAIKEY"]


class TestGetOpenAIClient(TestSetup):
  """Test OpenAI client creation."""

  def test_get_client_from_env(self):
    """Test getting client from environment variable."""
    client = eval.get_openai_client()
    assert isinstance(client, openai.OpenAI)
    assert client.api_key == self.api_key

  def test_get_client_no_key_raises_error(self):
    """Test that missing API key raises ValueError."""
    with patch.dict(os.environ, {}, clear=True):
      with pytest.raises(
        ValueError, match="OAIKEY environment variable not set"
      ):
        eval.get_openai_client()


class TestParseDialog(TestSetup):
  """Test dialog parsing functionality."""

  def test_parse_valid_dialog(self):
    """Test parsing a valid dialog format."""
    dialog = "User: What is 2+2?\n\nAssistant: 2+2 equals 4."
    query, resp = eval.parse_dialog(dialog)
    assert query == "What is 2+2?", query
    assert resp == "2+2 equals 4.", resp

  def test_parse_dialog_with_multiline_query(self):
    """Test parsing dialog with multiline query."""
    dialog = "User: What is 2+2?\nPlease explain step by step.\n\nAssistant: 2+2 equals 4 because you add two and two together."
    query, resp = eval.parse_dialog(dialog)
    assert query == "What is 2+2?\nPlease explain step by step."
    assert resp == "2+2 equals 4 because you add two and two together."

  def test_parse_dialog_with_multiline_response(self):
    """Test parsing dialog with multiline response."""
    dialog = "User: Explain addition\n\nAssistant: Addition is a basic arithmetic operation.\nIt combines two numbers to get a sum."
    query, resp = eval.parse_dialog(dialog)
    assert query == "Explain addition"
    assert (
      resp
      == "Addition is a basic arithmetic operation.\nIt combines two numbers to get a sum."
    )

  def test_parse_malformed_dialog_raises_error(self):
    """Test that malformed dialog raises ValueError."""
    invalid_dialogs = [
      "User: What is 2+2?",  # Missing Assistant part
      "Assistant: 2+2 equals 4.",  # Missing Human part
      "User: What is 2+2?\nAssistant: 2+2 equals 4.",  # Wrong separator
      "Random text",  # Completely wrong format
      "",  # Empty string
    ]

    for invalid_dialog in invalid_dialogs:
      with pytest.raises(ValueError, match="dialog malformed"):
        eval.parse_dialog(invalid_dialog)


class TestParseJudgeResponse(TestSetup):
  """Test judge response parsing."""

  def test_parse_valid_judge_response_a(self):
    """Test parsing valid judge response preferring A."""
    resp = 'comp: Response A is more detailed and helpful.\nMore helpful: "A"'
    result = eval.parse_judge_resp(resp)
    assert result == "A"

  def test_parse_valid_judge_response_b(self):
    """Test parsing valid judge response preferring B."""
    resp = 'comp: Response B provides better examples and clearer explanation.\nMore helpful: "B"'
    result = eval.parse_judge_resp(resp)
    assert result == "B"

  def test_parse_judge_response_with_multiline_comp(self):
    """Test parsing judge response with multiline comparison."""
    resp = 'comp: Response A is more detailed.\nIt provides better context and examples.\nMore helpful: "A"'
    result = eval.parse_judge_resp(resp)
    assert result == "A"

  def test_parse_empty_response_raises_error(self):
    """Test that empty response raises ValueError."""
    with pytest.raises(ValueError, match="No response text to parse"):
      eval.parse_judge_resp("")

  def test_parse_malformed_judge_response_raises_error(self):
    """Test that malformed judge response raises ValueError."""
    invalid_responses = ["\nAB", "\na", "\nb", "\n"]

    for invalid_resp in invalid_responses:
      with pytest.raises(ValueError, match="Malformed judge response"):
        eval.parse_judge_resp(invalid_resp)


class TestCallJudge(TestSetup):
  """Test OpenAI API calls."""

  def fix_success(self):
    """Test successful judge API call."""
    # Use real OpenAI client
    client = eval.get_openai_client()

    # Test with a simple prompt
    test_prompt = eval.PROMPT_TEMPLATE.format(
      user_query="What is 2+2?",
      resp_a="The answer is 4.",
      resp_b="2+2 equals 4 because you add two and two together.",
    )

    result = eval.call_judge(client, test_prompt)

    # Verify we got a response
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


class TestEvaluatePair(TestSetup):
  """Test pair evaluation functionality."""

  def test_evaluate_pair_success(self):
    """Test successful pair evaluation."""
    # Setup with real client
    client = eval.get_openai_client()
    control_dialog = "User: What is 2+2?\n\nAssistant: 2+2 is 4."
    experimental_dialog = (
      "User: What is 2+2?\n\nAssistant: The answer is 4, because 2+2=4."
    )

    # Test with real API
    results = eval.evaluate_pair(client, control_dialog, experimental_dialog, 0)

    # Verify structure
    assert len(results) == 2

    # Check original order result
    orig_result = results[0]
    assert orig_result["pair_id"] == 0
    assert orig_result["order"] == "orig"
    assert orig_result["pref"] in ["A", "B"]
    assert "2+2 is 4" in orig_result["prompt"]  # Control response as A
    assert (
      "The answer is 4" in orig_result["prompt"]
    )  # Experimental response as B

    # Check swapped order result
    swap_result = results[1]
    assert swap_result["pair_id"] == 0
    assert swap_result["order"] == "swap"
    assert swap_result["pref"] in ["A", "B"]
    assert (
      "The answer is 4" in swap_result["prompt"]
    )  # Experimental response as A
    assert "2+2 is 4" in swap_result["prompt"]  # Control response as B

  def test_evaluate_pair_mismatched_queries_raises_error(self):
    """Test that mismatched queries raise ValueError."""
    mock_client = Mock()
    control_dialog = "User: What is 2+2?\n\nAssistant: 4"
    experimental_dialog = "User: What is 3+3?\n\nAssistant: 6"

    with pytest.raises(ValueError, match="Queries do not match"):
      eval.evaluate_pair(
        mock_client, control_dialog, experimental_dialog, 0
      )


class TestEvaluate(TestSetup):
  """Test main evaluation function."""

  def test_evaluate_success(self):
    """Test successful evaluation with real API calls."""
    # Setup test data
    test_pairs = [
      {
        "control": "User: Hi\n\nAssistant: Hello!",
        "experimental": "User: Hi\n\nAssistant: Hello there!",
      }
    ]

    # Create temporary directory for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
      # Create real input file
      input_file = os.path.join(temp_dir, "test_data.json")
      with open(input_file, "w") as f:
        json.dump(test_pairs, f)

      # Test with real client and API
      client = eval.get_openai_client()
      summary, results = eval.evaluate(
        client, input_file, temp_dir
      )

      # Verify summary structure
      assert summary["total_pairs"] == 1
      assert "orig_prefs" in summary
      assert "swap_prefs" in summary
      assert "agreement_rate" in summary
      assert "valid_evaluations" in summary

      # Verify results structure
      assert len(results) == 2
      assert results[0]["order"] == "orig"
      assert results[1]["order"] == "swap"

  def test_evaluate_missing_file_raises_error(self):
    """Test that missing data file raises FileNotFoundError."""
    mock_client = Mock()

    with pytest.raises(
      FileNotFoundError, match="Data file 'nonexistent.json' not found"
    ):
      eval.evaluate(mock_client, "nonexistent.json", ".")

  def test_evaluate_agreement_rate_calculation(self):
    """Test agreement rate calculation with real API."""
    # Setup test data
    test_pairs = [
      {
        "control": "User: Hi\n\nAssistant: Hello!",
        "experimental": "User: Hi\n\nAssistant: Hi!",
      }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
      # Create real input file
      input_file = os.path.join(temp_dir, "test.json")
      with open(input_file, "w") as f:
        json.dump(test_pairs, f)

      client = eval.get_openai_client()
      summary, _ = eval.evaluate(
        client, input_file, temp_dir
      )

      # Verify agreement rate is calculated (between 0.0 and 1.0)
      assert 0.0 <= summary["agreement_rate"] <= 1.0


class TestMain(TestSetup):
  """Test CLI main function."""

  def test_main_success(self):
    """Test successful main execution with real API."""
    # Create temporary test data
    test_data = [
      {
        "control": "User: Hi\n\nAssistant: Hello!",
        "experimental": "User: Hi\n\nAssistant: Hello there!",
      }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
      # Create test input file
      input_file = os.path.join(temp_dir, "pref_pairs.json")
      with open(input_file, "w") as f:
        json.dump(test_data, f)

      # Test with real arguments
      with patch(
        "sys.argv",
        ["evaluator.py", "--data-file", input_file, "--output-dir", temp_dir],
      ):
        result = eval.main()

      assert result == 0

  def test_main_failure(self):
    """Test main function with missing file."""
    # Test with non-existent file
    with patch("sys.argv", ["evaluator.py", "--data-file", "nonexistent.json"]):
      result = eval.main()

    assert result == 1


class TestPromptTemplate(TestSetup):
  """Test prompt template formatting."""

  def test_prompt_template_formatting(self):
    """Test that PROMPT_TEMPLATE formats correctly."""
    formatted = eval.PROMPT_TEMPLATE.format(
      user_query="What is AI?",
      resp_a="AI is artificial intelligence.",
      resp_b="AI stands for artificial intelligence and refers to machine intelligence.",
    )

    assert "What is AI?" in formatted
    assert "resp A:" in formatted
    assert "AI is artificial intelligence." in formatted
    assert "resp B:" in formatted
    assert "AI stands for artificial intelligence" in formatted
    assert "Comparison:" in formatted
    assert "More helpful:" in formatted


class TestIntegration(TestSetup):
  """Integration tests with temporary files."""

  def test_file_output_format(self):
    """Test that output files are created with correct format."""
    with tempfile.TemporaryDirectory() as temp_dir:
      # Create test input file
      test_data = [
        {
          "control": "User: Test\n\nAssistant: Control response",
          "experimental": "User: Test\n\nAssistant: Experimental response",
        }
      ]

      input_file = os.path.join(temp_dir, "input.json")
      with open(input_file, "w") as f:
        json.dump(test_data, f)

      # Use real evaluation process
      client = eval.get_openai_client()
      summary, results = eval.evaluate(
        client, input_file, temp_dir
      )

      # Verify output file was created
      output_files = [
        f for f in os.listdir(temp_dir) if f.startswith("eval_results_")
      ]
      assert len(output_files) == 1

      # Verify output file content
      output_file = os.path.join(temp_dir, output_files[0])
      with open(output_file, "r") as f:
        output_data = json.load(f)

      assert "summary" in output_data
      assert "individual_results" in output_data
      assert "metadata" in output_data
      assert output_data["metadata"]["total_pairs"] == 1
      assert output_data["metadata"]["model"] == eval.MODEL_ID
      assert "evaluation_date" in output_data["metadata"]
