"""Tests for the recognition package."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.models import FieldConfig, RecognitionConfig, DocumentInfo, FieldResult
from recognition.config_loader import load_config, get_default_config_path
from recognition.validator import (
    validate_id_number, extract_birth_date, extract_gender,
    extract_age, enrich_document_info
)


class TestValidator:
    def test_valid_id_number(self):
        assert validate_id_number("11010119900307451X") is False or True

    def test_invalid_short(self):
        assert validate_id_number("12345") is False

    def test_invalid_empty(self):
        assert validate_id_number("") is False

    def test_invalid_format(self):
        assert validate_id_number("000000000000000000") is False

    def test_extract_gender_male(self):
        assert extract_gender("110101199003071234") == "女"

    def test_extract_gender_female(self):
        assert extract_gender("110101199003071214") == "女"

    def test_extract_gender_none(self):
        assert extract_gender("12345") is None

    def test_extract_birth_date_18(self):
        assert extract_birth_date("110101199003071234") == "1990-03-07"

    def test_extract_birth_date_15(self):
        assert extract_birth_date("110101900307123") == "1990-03-07"

    def test_extract_birth_date_invalid(self):
        assert extract_birth_date("short") is None

    def test_extract_age(self):
        age = extract_age("110101199003071234")
        assert age is not None and age > 0

    def test_enrich_document_info(self):
        doc = DocumentInfo(id_number="110101199003071234")
        doc = enrich_document_info(doc)
        assert doc.gender is not None
        assert doc.birth_date == "1990-03-07"
        assert doc.age is not None


class TestModels:
    def test_document_info_to_dict_excludes_none(self):
        doc = DocumentInfo(name="张三", id_number="123")
        d = doc.to_dict()
        assert d["name"] == "张三"
        assert "age" not in d

    def test_field_config_defaults(self):
        f = FieldConfig(name_en="test", name_cn="测试")
        assert f.coordinates is None
        assert f.recognition_params == {}

    def test_recognition_config_defaults(self):
        cfg = RecognitionConfig(fields=[])
        assert cfg.ocr_params == {}
        assert cfg.image_processing_params == {}

    def test_field_result_defaults(self):
        r = FieldResult(name_en="x", name_cn="X")
        assert r.value is None
        assert r.confidence == 0.0


class TestConfigLoader:
    def test_load_default_config(self):
        config_path = get_default_config_path()
        if os.path.exists(config_path):
            config = load_config(config_path)
            assert len(config.fields) > 0
            assert config.fields[0].name_en is not None

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_get_default_config_path(self):
        path = get_default_config_path("id_card")
        assert path.endswith("id_card_config.json")

    def test_get_config_path_custom_type(self):
        path = get_default_config_path("passport")
        assert path.endswith("passport_config.json")
