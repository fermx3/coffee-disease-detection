"""
Tests para params_helpers
"""
import pytest
from coffeedd.utilities.params_helpers import (
    auto_type,
    get_epochs_for_sample_size,
    get_sample_name
)


class TestAutoType:
    """Tests para la función auto_type"""

    def test_converts_int_string_to_int(self):
        assert auto_type("100") == 100
        assert isinstance(auto_type("100"), int)

    def test_converts_float_string_to_float(self):
        assert auto_type("0.5") == 0.5
        assert isinstance(auto_type("0.5"), float)

    def test_returns_string_for_non_numeric(self):
        assert auto_type("small") == "small"
        assert isinstance(auto_type("small"), str)

    def test_handles_none(self):
        assert auto_type(None) is None

    def test_strips_whitespace(self):
        assert auto_type("  100  ") == 100
        assert auto_type("  0.5  ") == 0.5


class TestGetEpochsForSampleSize:
    """Tests para la función get_epochs_for_sample_size"""

    def test_float_percentages(self):
        assert get_epochs_for_sample_size(0.1) == 30
        assert get_epochs_for_sample_size(0.5) == 50
        assert get_epochs_for_sample_size(1.0) == 60

    def test_int_absolute_numbers(self):
        assert get_epochs_for_sample_size(100) == 30
        assert get_epochs_for_sample_size(1000) == 40
        assert get_epochs_for_sample_size(5000) == 50
        assert get_epochs_for_sample_size(10000) == 60

    def test_edge_cases(self):
        assert get_epochs_for_sample_size(0.09) == 30
        assert get_epochs_for_sample_size(0.11) == 50
        assert get_epochs_for_sample_size(99) == 30
        assert get_epochs_for_sample_size(101) == 40

    def test_fallback_for_invalid_input(self):
        # String no numérico
        assert get_epochs_for_sample_size("invalid") == 60
        # Float fuera de rango válido (mayor a 1, pero tratado como error)
        assert get_epochs_for_sample_size(1.5) == 60
        # None
        assert get_epochs_for_sample_size(None) == 60


class TestGetSampleName:
    """Tests para la función get_sample_name"""

    def test_none_returns_full(self):
        assert get_sample_name(None) == 'full'

    def test_string_full_returns_full(self):
        assert get_sample_name('full') == 'full'

    def test_string_half_returns_half(self):
        assert get_sample_name('half') == 'half'

    def test_float_percentages(self):
        assert get_sample_name(1.0) == '100pct'
        assert get_sample_name(0.5) == '50pct'
        assert get_sample_name(0.1) == '10pct'
        assert get_sample_name(0.25) == '25pct'

    def test_int_absolute_numbers(self):
        assert get_sample_name(100) == '100'
        assert get_sample_name(1000) == '1000'
        assert get_sample_name(5000) == '5000'

    def test_edge_cases(self):
        # Float mayor a 1 se trata como número absoluto
        assert get_sample_name(100.0) == '100'

        # Porcentajes pequeños
        assert get_sample_name(0.01) == '1pct'
        assert get_sample_name(0.05) == '5pct'

    def test_custom_strings(self):
        assert get_sample_name('small') == 'small'
        assert get_sample_name('test') == 'test'
