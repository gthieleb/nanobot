"""Test env var directly."""
import os


def test_env_var_set():
    """Test that we can set and read env vars."""
    os.environ["TEST_VAR"] = "/tmp/test.json"
    
    result = os.environ.get("TEST_VAR")
    
    print(f"Set: /tmp/test.json")
    print(f"Got: {result}")
    
    assert result == "/tmp/test.json"
