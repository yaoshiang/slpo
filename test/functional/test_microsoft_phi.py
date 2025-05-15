from transformers import AutoModelForCausalLM, AutoTokenizer
import pytest

@pytest.mark.skip
def test_load_phi():
    """Test loading the phi-2 model."""
    # Arrange
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    inputs = tokenizer('''def print_prime(n):
    """
    Print all primes between 1 and n
    """''', return_tensors="pt", return_attention_mask=False)

    # Act
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

    # Assert
    assert "def print_prime(n):" in text