"""
Unit tests for Streamlit formatting utilities
"""
import pytest
from streamlit_app.utils.formatting import format_metric, safe_get


class TestFormatMetric:
    """Tests for format_metric()"""
    
    def test_format_int(self):
        assert format_metric(1000, " px") == "1,000 px"
        assert format_metric(1234567, " px²") == "1,234,567 px²"
    
    def test_format_float(self):
        assert format_metric(3.14, " rad") == "3.14 rad"
        assert format_metric(1000.5, " ms") == "1,000.5 ms"
    
    def test_none_value(self):
        assert format_metric(None, " px") == "N/A"
    
    def test_string_value(self):
        assert format_metric("invalid", " px") == "N/A"
    
    def test_custom_default(self):
        assert format_metric(None, " px", default="---") == "---"
    
    def test_bool_value(self):
        assert format_metric(True, " px") == "N/A"
        assert format_metric(False, " px") == "N/A"
    
    def test_no_unit(self):
        assert format_metric(1000) == "1,000"
        assert format_metric(42) == "42"
    
    def test_zero(self):
        assert format_metric(0, " px") == "0 px"
    
    def test_negative(self):
        assert format_metric(-100, " px") == "-100 px"


class TestSafeGet:
    """Tests for safe_get()"""
    
    def test_simple_key(self):
        config = {'a': 1}
        assert safe_get(config, 'a') == 1
    
    def test_nested_keys(self):
        config = {'a': {'b': {'c': 42}}}
        assert safe_get(config, 'a', 'b', 'c') == 42
    
    def test_missing_key(self):
        config = {'a': 1}
        assert safe_get(config, 'b', default=0) == 0
    
    def test_partial_path(self):
        config = {'a': {'b': 1}}
        assert safe_get(config, 'a', 'c', 'd', default=None) is None
    
    def test_none_intermediate(self):
        config = {'a': None}
        assert safe_get(config, 'a', 'b', default="missing") == "missing"
    
    def test_non_dict_intermediate(self):
        config = {'a': 42}
        assert safe_get(config, 'a', 'b', default=0) == 0
    
    def test_empty_dict(self):
        config = {}
        assert safe_get(config, 'a', default="default") == "default"
    
    def test_no_default(self):
        config = {'a': 1}
        assert safe_get(config, 'b') is None
    
    def test_existing_none_value(self):
        config = {'a': {'b': None}}
        # When a key exists but has None value, it should return the default
        # This is the desired behavior for the Streamlit app
        assert safe_get(config, 'a', 'b', default="default") == "default"
