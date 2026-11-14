from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any, Iterable, Mapping
import zipfile

import pandas as pd
from openpyxl import load_workbook

from routing_benchmark.v4.artifacts.manifest import validate_artifact_pack
from routing_benchmark.v4.artifacts.schema import ArtifactFile, ArtifactManifest, sha256_bytes


GENERATOR_VERSION = "routingbench-v4-five-domains-v1"
SOURCE_CACHE = Path("puppeteer/.cache/routingbench_v4/five_domains_20261101")
OUTPUT_ROOT = Path("puppeteer/data/RoutingBench/v4_five_domains_100_20261101")
ROLE_DESIGN = Path("puppeteer/docs/five_domain_agent_pool_role_card_design_vi.md")

SOURCE_LOCKS: dict[str, dict[str, Any]] = {
    "structured_data_processing": {
        "canonical_repository": "https://github.com/RUCKBReasoning/SpreadsheetBench",
        "canonical_revision": "49b73a94775fb489063f60ca1865e3a650079a79",
        "artifact_repository": "https://huggingface.co/datasets/KAKA22/SpreadsheetBench",
        "artifact_revision": "ab0b742b0fc95b946f212d80ac7771b5531272e4",
        "artifact_path": "spreadsheetbench_verified_400.tar.gz",
        "license": "CC-BY-SA-4.0",
    },
    "scientific_procedure": {
        "canonical_repository": "https://github.com/OSU-NLP-Group/ScienceAgentBench",
        "canonical_revision": "c26e151ed601ba109dc4d35e057ff8e73fec469d",
        "artifact_repository": "https://huggingface.co/datasets/osunlp/ScienceAgentBench",
        "artifact_revision": "9c6e96c9e74572e979b0930ee735041cef528cb7",
        "artifact_path": "data/verified-00000-of-00001.parquet",
        "license": "CC-BY-4.0 (selected IDs exclude upstream-license exceptions)",
    },
    "cross_app_workflow": {
        "canonical_repository": "https://github.com/StonyBrookNLP/appworld",
        "canonical_revision": "42b5bcf3cd334fee33f0c37c02070a9f5807add5",
        "package_version": "0.1.3.post1",
        "protected_bundle_url": "https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle",
        "protected_bundle_version": "0.1.0",
        "protected_bundle_etag": "9b077001e5e1feba23cbab10d04de597-2",
        "license": "Apache-2.0 with encrypted-public-redistribution requirement for protected material",
    },
    "multi_hop_knowledge": {
        "canonical_repository": "https://github.com/stonybrooknlp/musique",
        "canonical_revision": "922ac98f19a201998dbdae6d7f2887a5258dbdeb",
        "canonical_download_id": "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
        "acquisition_mirror": "https://huggingface.co/datasets/voidful/MuSiQue",
        "mirror_revision": "a7d9f9adf6191604fc67cde318ee1a86fcf7babc",
        "artifact_path": "musique_full_v1.0_dev.jsonl",
        "license": "CC-BY-4.0",
    },
    "document_media_processing": {
        "canonical_repository": "https://github.com/datalab-to/omni_parse_bench",
        "canonical_revision": "735b646f481fdde90f00f94510c9bee93bac7d2a",
        "artifact_repository": "https://huggingface.co/datasets/datalab-to/omni_parse_bench",
        "artifact_revision": "74f8d391683d69813a6c9bfbb5f2d53b273658e9",
        "license": "per-page upstream license; tests CC-BY-4.0 unless source-specific exception applies",
    },
}


SPREADSHEET_SELECTION = {
    "formula_dependency": ["183-8", "33722", "37456", "45738"],
    "reshape_join": ["13-1", "267-21", "531-18", "341-14"],
    "cleaning_normalization": ["120-24", "227-40", "585-41", "493-5"],
    "aggregation_calculation": ["41-47", "84-40", "250-20", "398-14"],
    "formatting_presentation": ["50631", "58484", "58032", "47827"],
}


SPREADSHEET_PYTHON_OJ_SPECS = {
    "183-8": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!J3:L6",
        "validation": {
            "kind": "weighted_average_formulas",
            "sheet": "Sheet1",
            "group_column": "B",
            "weight_column": "C",
            "value_columns": ["D", "E", "F"],
            "output_group_column": "H",
            "output_columns": ["J", "K", "L"],
            "row_range": [3, 6],
            "require_formulas": True,
            "preserve_existing_summary_row": 7,
            "preserve_unrelated_cells_and_styles": True,
        },
        "metadata_repairs": [
            "Narrow the editable answer range from upstream I3:L7 to J3:L6 because column I and row 7 already contain source formulas and must be preserved."
        ],
        "case1_difference_evidence": [
            "Verified-400 adds the missing public label FY in H2.",
            "The normalized evaluator excludes the unrelated pre-existing summary formulas in row 7, whose cached values differ between archive releases.",
            "The instruction-derived transform matches J3:L6 in all three full-archive source-native cases."
        ],
    },
    "33722": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!E2:E6",
        "validation": {
            "kind": "progressive_percentile_median_formulas",
            "sheet": "Sheet1",
            "value_column": "B",
            "percentile_column": "C",
            "output_column": "E",
            "row_range": [2, 6],
            "lower_exclusive": 0.0,
            "upper_exclusive": 0.01,
            "exclude_prior_rows_on_copy_down": True,
            "require_formulas": True,
            "style": {"bordered": True, "horizontal_alignment": "left", "number_format": "0"},
            "preserve_unrelated_cells_and_styles": True,
        },
        "metadata_repairs": ["Qualify the upstream unqualified answer range E2:E6 with worksheet Sheet1."],
    },
    "37456": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "Sample Data",
        "answer_position": "'Sample Data'!G12:J15",
        "validation": {
            "kind": "custom_day_sales_formulas",
            "sheet": "Sample Data",
            "source_date_column": "A",
            "source_time_column": "B",
            "source_product_column": "C",
            "source_sales_column": "D",
            "start_time_cell": "F1",
            "end_time_cell": "F2",
            "output_date_column": "F",
            "output_header_row": 11,
            "output_range": "G12:J15",
            "require_formulas": True,
            "number_format_family": "currency",
            "preserve_unrelated_cells_and_styles": True,
        },
        "metadata_repairs": ["Qualify the upstream unqualified answer range G12:J15 with worksheet Sample Data."],
    },
    "45738": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!E6:AB25",
        "validation": {
            "kind": "bond_cashflow_formulas",
            "sheet": "Sheet1",
            "principal_column": "A",
            "interest_column": "B",
            "frequency_column": "C",
            "maturity_column": "D",
            "date_header_row": 4,
            "output_range": "E6:AB25",
            "require_formulas": True,
            "horizontal_alignment": "center",
            "preserve_target_number_format": True,
            "preserve_unrelated_cells_and_styles": True,
        },
        "metadata_repairs": [
            "Qualify the upstream unqualified answer range E6:AB25 with worksheet Sheet1.",
            "Treat the dash wording as the existing zero-display number format; row 5 is outside the requested edit range and remains immutable."
        ],
    },
    "13-1": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "LISTS",
        "answer_sheet": "LISTS",
        "answer_position": "'LISTS'!A3:D32",
        "validation": {
            "kind": "section_aggregate_lists",
            "source_sheet": "RANGES",
            "output_sheet": "LISTS",
            "section_title_column": "C",
            "headers": ["S.N", "DATE", "BATCH", "REF", "AMOUNTS"],
            "group_by": ["DATE", "REF"],
            "sum_column": "AMOUNTS",
            "row_order": "ascending_date_then_reference_A_to_Z",
            "total_row": True,
            "clear_previous_section_rows_before_write": True,
            "preserve_section_titles_headers_styles_and_source_sheet": True,
        },
        "metadata_repairs": [
            "Use the Verified-400 corrected instruction and sorted golden behavior; the older full-archive answers preserve first occurrence instead."
        ],
        "case1_difference_evidence": [
            "Verified-400 corrects the task to sort each output section by date ascending and REF alphabetically.",
            "Verified-400 case 1 input and golden contain the corrected sorted STAGE example and sorted completed sections.",
            "Full-archive cases 2 and 3 are retained only as private input variants; the independent checker derives sorted answers from their inputs rather than trusting the older unsorted goldens."
        ],
        "case_verification": "Verified-400 sorted semantics are derived independently from each input; the older full-archive answer is retained for provenance but is not an evaluator oracle.",
    },
    "267-21": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "merging",
        "answer_position": "'merging'!C2:D11",
        "validation": {
            "kind": "monthly_sheet_lookup_merge",
            "output_sheet": "merging",
            "id_column": "B",
            "output_columns": {"C": "SH1", "D": "RP1"},
            "source_id_header": "ID",
            "source_value_header": "QTY",
            "row_range": [2, 11],
            "missing_value": "-",
            "preserve_unrelated_cells_sheets_and_styles": True,
        },
        "metadata_repairs": [
            "Repair malformed answer_sheet metadata to merging.",
            "Repair malformed answer_position metadata to 'merging'!C2:D11."
        ],
    },
    "531-18": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "Transpose",
        "answer_sheet": "Transpose",
        "answer_position": "'Transpose'!A3:C970",
        "validation": {
            "kind": "wide_to_long_unpivot",
            "source_sheet": "Data",
            "output_sheet": "Transpose",
            "source_header_row": 6,
            "item_column": "A",
            "first_date_column": "G",
            "output_start_row": 3,
            "output_columns": ["ITEM", "DATE", "QTY"],
            "include_zero_and_blank_quantities": True,
            "row_order": "item_major_then_source_date_column_order",
            "clear_previous_output_before_write": True,
            "preserve_source_sheet_and_unrelated_output_content": True,
        },
    },
    "341-14": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "result",
        "answer_sheet": "result",
        "answer_position": "'result'!A1:B30",
        "validation": {
            "kind": "split_multiline_rows",
            "source_sheet": "problem",
            "output_sheet": "result",
            "source_key_column": "A",
            "source_values_column": "B",
            "output_start_row": 1,
            "split_on_line_breaks": True,
            "blank_values_emit_one_blank_row": True,
            "coerce_numeric_tokens_to_numbers": True,
            "preserve_source_order_and_token_order": True,
            "clear_previous_output_before_write": True,
            "preserve_source_sheet_and_unrelated_output_content": True,
        },
        "metadata_repairs": ["Widen the normalized evaluator range to A1:B30 so it covers the largest trusted source-native case."],
    },
    "58484": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!H5:H26",
        "validation": {
            "kind": "last_transfer_count_formulas",
            "sheet": "Sheet1",
            "operator_column": "D",
            "staff_column": "E",
            "target_operator": 5551234,
            "output_column": "H",
            "row_range": [5, 26],
            "require_formulas": True,
            "preserve_recreated_sheet_formatting": True,
            "blank_unbordered_column": "G",
            "preserve_unrelated_cells_and_styles": True,
        },
        "metadata_repairs": ["Qualify the upstream unqualified answer range H5:H26 with worksheet Sheet1."],
    },
    "58032": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!A2:A35",
        "validation": {
            "kind": "two_key_lookup_formulas",
            "sheet": "Sheet1",
            "title_column": "B",
            "department_column": "C",
            "lookup_name_column": "H",
            "lookup_title_column": "I",
            "lookup_department_column": "J",
            "output_column": "A",
            "row_range": [2, 35],
            "require_formulas": True,
            "output_fill_rgb": "FCD5B4",
            "hide_gridlines": True,
            "preserve_recreated_sheet_formatting": True,
            "preserve_unrelated_cells_and_styles": True,
        },
        "metadata_repairs": ["Qualify the upstream unqualified answer range A2:A35 with worksheet Sheet1."],
    },
    "250-20": {
        "trusted_cases": [1, 3],
        "excluded_cases": [2],
        "example_sheet": "Desired outcome",
        "answer_position": "'RNM'!A1:J20",
        "validation": {
            "kind": "deduplicate_sum",
            "sheet": "RNM",
            "header_row": 1,
            "data_range": "A2:K20",
            "key_columns": ["B", "C"],
            "aggregate_columns": {"J": "sum"},
            "duplicate_policy": "retain_first_row_and_delete_later_rows",
            "row_order": "preserve_first_occurrence_order",
            "preserve_other_cell_values_in_retained_rows": True,
            "preserve_unrelated_sheets": True,
        },
        "exclusion_evidence": {
            "2": [
                "Instruction-consistent aggregation yields 14 data rows while the upstream answer contains 16.",
                "Key (70000009932150, 8) occurs four times with J=1 but the upstream answer retains J=1 instead of J=4.",
                "The upstream answer contains rows absent from case-2 input.",
            ]
        },
    },
    "398-14": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "COLLECTION",
        "answer_position": "'COLLECTION'!A2:G9",
        "validation": {
            "kind": "collection_aggregate",
            "output_sheet": "COLLECTION",
            "key_headers": ["TY", "OR"],
            "sum_headers": ["SALE", "RET"],
            "balance": "SALE - RET",
            "preserve_existing_key_order_then_append_new": True,
            "zero_display_number_format": "#,##0;[Red]\\-#,##0;\\-",
            "font": {"name": "Calibri", "size": 11, "bold": False},
            "alignment": {"right": ["A", "E", "F", "G"], "left": ["B", "C", "D"]},
            "preserve_non_output_sheets": True,
        },
    },
    "41-47": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "OUT CAS",
        "answer_position": "'OUT CAS'!A2:C1529,'OUT CAS'!E2:G586,'OUT CAS'!I2:K13,'OUT CAS'!L2:O8",
        "validation": {
            "kind": "account_bucket_aggregate",
            "output_sheet": "OUT CAS",
            "source_columns": {"account_reference": "B", "debit": "C", "credit": "D"},
            "buckets": [
                {"prefix": "PUR PURCHASE", "output_columns": ["A", "B", "C"]},
                {"prefix": "PUR PAID", "output_columns": ["E", "F", "G"]},
                {"prefix": "CASH SS", "output_columns": ["I", "J", "K"]},
                {"prefix": "EXPRR", "output_columns": ["M", "N", "O"]},
            ],
            "aggregate_duplicate_references": True,
            "sort": "ascending trailing item number",
            "clear_previous_output_before_write": True,
            "preserve_source_sheets": True,
        },
        "metadata_repairs": ["Add the missing leading quote to the first OUT CAS answer range."],
    },
    "84-40": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "LIST",
        "answer_position": "'LIST'!C2:E5",
        "validation": {
            "kind": "sheet_column_summary",
            "output_sheet": "LIST",
            "include_sheets": "all sheets before LIST",
            "sum_columns": {"C": "C", "D": "D"},
            "balance_column": {"E": "C - D"},
            "sheet_name_column": "B",
            "item_column": "A",
            "total_row": True,
            "clear_previous_output_before_write": True,
            "preserve_source_sheets": True,
        },
    },
    "120-24": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": None,
        "answer_position": "'Sheet1'!BN2:BN45",
        "validation": {
            "kind": "conditional_replace_and_fill",
            "sheet": "Sheet1",
            "row_range": [2, 45],
            "condition": {"BL": "LAG", "BG_in": ["OCOGS - Spares - Transfer Price Overhead", "OFSS - ORCL Consulting Prod Cost I/C"]},
            "branch": {
                "AY_in": ["BOA_033E", "BOA_011G"],
                "then_BN": "OCOGS - Spares - Transfer Price Overhead",
                "else_BN": "OFSS - ORCL Consulting Prod Cost I/C",
            },
            "qualified_BL_fill_rgb": "0070C0",
            "preserve_other_values": True,
        },
    },
    "227-40": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "Sheet1",
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!A14:E17",
        "validation": {
            "kind": "row_text_dedup_output",
            "sheet": "Sheet1",
            "source_range": "A1:E4",
            "output_range": "A14:E17",
            "label_column": "A",
            "value_columns": ["B", "C", "D", "E"],
            "duplicate_policy": "keep_first_occurrence_in_its_original_column_and_blank_later_occurrences",
            "preserve_source_and_unrelated_cells": True,
        },
    },
    "493-5": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": None,
        "answer_sheet": "Imported Data",
        "answer_position": "'Imported Data'!A1:F10",
        "validation": {
            "kind": "delete_reversing_row_pairs",
            "sheet": "Imported Data",
            "header_row": 1,
            "reference_column": "A",
            "debit_column": "C",
            "credit_column": "D",
            "net_formula_column": "E",
            "narrative_column": "F",
            "pair_policy": "same reference and narrative with debit/credit values reversed",
            "delete_both_rows": True,
            "preserve_remaining_row_order": True,
            "rebase_net_formulas_after_deletion": True,
        },
    },
    "585-41": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": None,
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!A1:H1",
        "validation": {
            "kind": "strip_non_numeric_text",
            "sheet": "Sheet1",
            "target_range": "A1:H1",
            "preserve_numeric_values": True,
            "preserve_decimal_point": True,
            "comma_policy": "accept commas while parsing and remove grouping separators in the numeric result",
            "output_type": "numeric",
            "preserve_unrelated_cells_and_styles": True,
        },
    },
    "47827": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": "3 straights to 3 boxed",
        "answer_sheet": "3 straights to 3 boxed",
        "answer_position": "'3 straights to 3 boxed'!G2:G16",
        "validation": {
            "kind": "sort_digits_and_format",
            "sheet": "3 straights to 3 boxed",
            "source_column": "E",
            "output_column": "G",
            "row_range": [2, 16],
            "digit_width": 3,
            "digit_order": "ascending",
            "output_representation": "three-character text preserving leading zeroes",
            "font_bold": True,
            "fill_rgb": "EBF1DE",
            "number_format": "000",
            "preserve_unrelated_cells_and_sheets": True,
        },
        "metadata_repairs": [
            "Use the Verified-400 instruction and source gold behavior: sort digits from smallest to largest; the older full-archive instruction says largest to smallest."
        ],
    },
    "50631": {
        "trusted_cases": [1, 2, 3],
        "excluded_cases": [],
        "example_sheet": None,
        "answer_sheet": "Sheet1",
        "answer_position": "'Sheet1'!H7:H38",
        "validation": {
            "kind": "workday_formula_series",
            "sheet": "Sheet1",
            "output_column": "H",
            "row_range": [7, 38],
            "start_date_cell": "B3",
            "end_date_cell": "F3",
            "holiday_range": "J5:J7",
            "sequence": "successive workdays beginning with the first workday on or after the start date",
            "blank_after_end_date": True,
            "preserve_all_other_values_formulas_styles_and_sheet_view": True,
            "preserve_target_cell_styles": True,
        },
        "metadata_repairs": ["Qualify the upstream unqualified answer range H7:H38 with worksheet Sheet1."],
    },
}

SCIENCE_SELECTION = {
    "direct_analysis": [16, 21, 29, 85],
    "method_comparison": [1, 18, 71, 76],
    "anomaly_diagnosis": [9, 69, 77, 59],
    "result_reproduction": [28, 27, 50, 42],
    "conclusion_under_uncertainty": [52, 70, 64, 36],
}

APPWORLD_SELECTION = [
    "6f4b9a5_1", "3d9a636_1", "634f342_1", "270f1ff_1", "042a9fc_1",
    "d194965_1", "b9c5c9a_1", "0d01c76_1", "652485c_1", "9dabbc9_1",
    "ffe6d5e_1", "83a7951_1", "f323bae_1", "9016950_1", "90adc3f_1",
    "b6d1104_1", "3aa1a22_1", "32616b5_1", "f861c32_1", "6b6ca61_1",
]

MUSIQUE_SELECTION = {
    "short_chain": [
        "2hop__231890_595414",
        "2hop__525242_52835",
        "2hop__169443_83769",
        "2hop__82910_75184",
    ],
    "long_chain": [
        "4hop1__436202_765799_282674_759393",
        "4hop1__17192_17130_70784_61381",
        "4hop1__56635_49925_13759_736921",
        "4hop1__33012_53706_795904_580996",
    ],
    "branching_evidence": [
        "3hop2__19639_90098_10557",
        "3hop2__326964_36852_7713",
        "3hop2__31398_2130_2232",
        "4hop3__323121_88460_30152_20999",
    ],
    "ambiguity_conflict": [
        "3hop1__857_846_7702",
        "3hop1__104237_214799_259594",
        "3hop1__103890_10659_23296",
        "3hop1__30390_36635_87083",
    ],
    "insufficient_evidence": [
        "2hop__153573_109006",
        "3hop1__539312_744503_34754",
        "3hop2__655849_223623_162182",
        "4hop1__129721_32392_823060_610794",
    ],
}

OMNI_SELECTION = {
    "ocr_reading_order": [
        "docparsing_images-images_99663c5b_images_legal_contract_00007-p1",
        "realdoc_pages-images_990322d4_images_760b6f57-61b5-4b3e-9507-4c81ea277213-p1",
        "multiling_finepdfs-images_aaz_Latn_38769501-p1",
        "olmo_old_scans-31-p1",
    ],
    "table": [
        "tables_derived-PMC10771002.1_doc-p6",
        "tables_derived-PMC6635801.1_doc-p5",
        "tables_derived-PMC11327485.1_doc-p4",
        "tables_derived-PMC10914731.1_doc-p9",
    ],
    "formula": [
        "olmo_arxiv_math-2503.07965_pg7-p1",
        "olmo_arxiv_math-2503.08614_pg8-p1",
        "olmo_arxiv_math-2503.08341_pg5-p1",
        "olmo_arxiv_math-2503.07708_pg3-p1",
    ],
    "form_key_value": [
        "forms_documentcloud-493872-obamacare-individual-mandate-tax-compliance-form-p1",
        "forms_federal-scan_osha_300-p1",
        "forms_filled-filled_irs_1040-p2",
        "forms_federal-scan_ntsb_6120_1-p1",
    ],
    "degraded_mixed_layout": [
        "olmo_long_tiny_text-17_pg2_pg1-p1",
        "handwriting_samples-catt_mss154040168_p8",
        "docparsing_images-images_00e18000_images_education_english_00083-p1",
        "olmo_old_scans_math-5_pg174-p1",
    ],
}

EXCLUDED_SCIENCE_IDS = {3, 32, 46, 53, 54, 84}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_type(path: str) -> str:
    overrides = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return overrides.get(Path(path).suffix.lower(), mimetypes.guess_type(path)[0] or "application/octet-stream")


def artifact_id(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", path).strip("-")


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def flatten(selection: Mapping[str, Iterable[Any]]) -> list[tuple[str, Any]]:
    return [(archetype, source_id) for archetype, ids in selection.items() for source_id in ids]


def clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    if hasattr(value, "tolist"):
        return clean_value(value.tolist())
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def workbook_values_equal(left: bytes, right: bytes) -> bool:
    """Compare workbook structure, formulas and values while ignoring ZIP/style serialization."""
    left_book = load_workbook(io.BytesIO(left), data_only=False, read_only=False)
    right_book = load_workbook(io.BytesIO(right), data_only=False, read_only=False)
    try:
        if left_book.sheetnames != right_book.sheetnames:
            return False
        for name in left_book.sheetnames:
            left_sheet, right_sheet = left_book[name], right_book[name]
            max_row = max(left_sheet.max_row, right_sheet.max_row)
            max_column = max(left_sheet.max_column, right_sheet.max_column)
            for row in range(1, max_row + 1):
                for column in range(1, max_column + 1):
                    left_value = left_sheet.cell(row, column).value
                    right_value = right_sheet.cell(row, column).value
                    if hasattr(left_value, "text"):
                        left_value = (left_value.text, getattr(left_value, "ref", None))
                    if hasattr(right_value, "text"):
                        right_value = (right_value.text, getattr(right_value, "ref", None))
                    if left_value != right_value:
                        return False
        return True
    finally:
        left_book.close()
        right_book.close()


class PacketWriter:
    def __init__(self, pack_dir: Path) -> None:
        self.pack_dir = pack_dir
        self.files: list[ArtifactFile] = []

    def add(self, relative_path: str, content: bytes | str, visibility: str) -> None:
        relative_path = Path(relative_path).as_posix()
        raw = content.encode("utf-8") if isinstance(content, str) else content
        destination = self.pack_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        self.files.append(
            ArtifactFile(
                artifact_id=artifact_id(relative_path),
                path=relative_path,
                media_type=media_type(relative_path),
                sha256=sha256_bytes(raw),
                visibility=visibility,
            )
        )

    def finish(
        self,
        *,
        packet_id: str,
        domain: str,
        task: str,
        adapter_id: str,
        source_type: str,
        license_name: str,
        metadata: Mapping[str, Any],
        generator_version: str = GENERATOR_VERSION,
    ) -> dict[str, Any]:
        manifest = ArtifactManifest(
            base_workflow_id=packet_id,
            domain=domain,
            task=task,
            adapter_id=adapter_id,
            public_files=tuple(item for item in self.files if item.visibility == "public"),
            private_files=tuple(item for item in self.files if item.visibility == "private"),
            fixture_files=tuple(item for item in self.files if item.visibility == "fixture"),
            source_type=source_type,
            generator_version=generator_version,
            license=license_name,
            human_review_status="pending",
            packet_review_status="pending",
            execution_readiness="pending_runtime_verification",
            metadata=dict(metadata),
        )
        (self.pack_dir / "manifest.json").write_bytes(json_bytes(manifest.to_dict()))
        validation = validate_artifact_pack(self.pack_dir)
        if not validation["valid"]:
            raise RuntimeError(f"Invalid packet {packet_id}: {validation['errors']}")
        return validation


def parse_role_cards(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    sections = {
        "structured_data_processing": ("5.3", "5.4"),
        "scientific_procedure": ("6.3", "6.4"),
        "cross_app_workflow": ("7.3", "7.4"),
        "multi_hop_knowledge": ("8.3", "8.4"),
        "document_media_processing": ("9.3", "9.4"),
    }
    pools: dict[str, dict[str, Any]] = {}
    for domain, (card_number, binding_number) in sections.items():
        card_match = re.search(
            rf"### {re.escape(card_number)} Role cards\s*(.*?)\s*### {re.escape(binding_number)} Executor bindings",
            text,
            flags=re.DOTALL,
        )
        binding_match = re.search(
            rf"### {re.escape(binding_number)} Executor bindings\s*(.*?)(?=\n## |\Z)",
            text,
            flags=re.DOTALL,
        )
        if not card_match or not binding_match:
            raise RuntimeError(f"Cannot parse role design for {domain}")
        cards: list[dict[str, Any]] = []
        chunks = re.split(r"(?m)^#### ", card_match.group(1))
        for chunk in chunks[1:]:
            role_name, body = chunk.split("\n", 1)
            card: dict[str, Any] = {
                "agent_id": role_name.strip(),
                "role_goal": "",
                "use_when": [],
                "avoid_when": [],
                "expected_outputs": [],
            }
            current: str | None = None
            for line in body.splitlines():
                field_match = re.match(r"^- `([^`]+)`: ?(.*)$", line)
                if field_match:
                    current = field_match.group(1)
                    inline = field_match.group(2).strip()
                    if current == "role_goal":
                        card[current] = inline
                    elif current in {"use_when", "avoid_when", "expected_outputs"} and inline:
                        card[current].append(inline)
                    continue
                nested = re.match(r"^  - (.+)$", line)
                if nested and current in {"use_when", "avoid_when", "expected_outputs"}:
                    card[current].append(nested.group(1).strip())
            if not card["role_goal"] or any(not card[key] for key in ("use_when", "avoid_when", "expected_outputs")):
                raise RuntimeError(f"Incomplete role card: {domain}/{role_name}")
            cards.append(card)
        bindings: dict[str, dict[str, Any]] = {}
        for line in binding_match.group(1).splitlines():
            if not line.startswith("|") or "---" in line or "Primary tools" in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != 3:
                continue
            role, tools, side_effects = cells
            bindings[role] = {
                "primary_tools": re.findall(r"`([^`]+)`", tools),
                "side_effects": side_effects,
            }
        for card in cards:
            if card["agent_id"] not in bindings:
                raise RuntimeError(f"Missing executor binding: {domain}/{card['agent_id']}")
            card["executor_binding"] = bindings[card["agent_id"]]
        if len(cards) != 10:
            raise RuntimeError(f"Expected 10 roles for {domain}, found {len(cards)}")
        pools[domain] = {
            "schema_version": "routingbench.v4.agent_pool.v1",
            "domain": domain,
            "guard_enabled": False,
            "router_actions": [card["agent_id"] for card in cards] + ["STOP"],
            "role_cards": cards,
            "stop_action": {
                "agent_id": "STOP",
                "role_goal": "End routing and submit the current public state for independent evaluation.",
            },
        }
    return pools


def spreadsheet_oj_role_contracts() -> list[dict[str, Any]]:
    read_artifact = {
        "name": "read_artifact",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "version": {"type": ["integer", "null"]},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8000},
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    }
    inspect_workbook = {
        "name": "inspect_workbook",
        "input_schema": {
            "type": "object",
            "properties": {
                "sheet": {"type": ["string", "null"]},
                "cell_range": {"type": ["string", "null"]},
                "include_formulas": {"type": "boolean"},
                "include_styles": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    }
    write_solution = {
        "name": "write_solution",
        "input_schema": {
            "type": "object",
            "properties": {
                "python_source": {"type": "string", "minLength": 1},
                "rationale": {"type": "string"},
                "input_artifact_refs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["python_source", "input_artifact_refs"],
            "additionalProperties": False,
        },
    }
    revise_solution = {
        "name": "revise_solution",
        "input_schema": {
            "type": "object",
            "properties": {
                "base_candidate_artifact_id": {"type": "string"},
                "python_source": {"type": "string", "minLength": 1},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["base_candidate_artifact_id", "python_source", "evidence_refs"],
            "additionalProperties": False,
        },
    }
    profile_workbook = {
        "name": "profile_workbook_data",
        "input_schema": {
            "type": "object",
            "properties": {
                "sheet": {"type": "string"},
                "cell_range": {"type": ["string", "null"]},
                "key_columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["sheet"],
            "additionalProperties": False,
        },
    }
    inspect_formulas = {
        "name": "inspect_formulas",
        "input_schema": {
            "type": "object",
            "properties": {
                "sheet": {"type": ["string", "null"]},
                "cell_range": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
    }
    run_public = {
        "name": "run_solution_public",
        "input_schema": {
            "type": "object",
            "properties": {"candidate_artifact_id": {"type": "string"}},
            "required": ["candidate_artifact_id"],
            "additionalProperties": False,
        },
    }
    validate_public = {
        "name": "validate_public_output",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_artifact_id": {"type": "string"},
                "execution_artifact_id": {"type": "string"},
            },
            "required": ["candidate_artifact_id", "execution_artifact_id"],
            "additionalProperties": False,
        },
    }

    definitions = [
        (
            "Workbook Surveyor",
            [inspect_workbook, read_artifact],
            ["public_case_1", "current runtime artifacts"],
            ["workbook_inventory"],
            [],
            "Read or schema failure creates a visible rejected event and no mutation.",
        ),
        (
            "Data Quality Profiler",
            [profile_workbook, read_artifact],
            ["public_case_1", "workbook_inventory"],
            ["data_profile"],
            [],
            "Profiling failure preserves prior artifacts and records bounded diagnostics.",
        ),
        (
            "Formula Dependency Analyst",
            [inspect_formulas, read_artifact],
            ["public_case_1", "workbook_inventory"],
            ["formula_analysis"],
            [],
            "Inspection failure is observable and cannot create validation evidence.",
        ),
        (
            "Table Reshaping Specialist",
            [write_solution, read_artifact],
            ["public_case_1", "analysis artifacts"],
            ["candidate_solution"],
            ["create a new immutable candidate solution version"],
            "Invalid source is retained as a rejected attempt; the applied candidate pointer is unchanged.",
        ),
        (
            "Formula Computation Specialist",
            [write_solution, read_artifact],
            ["public_case_1", "formula analysis"],
            ["candidate_solution"],
            ["create a new immutable candidate solution version"],
            "Invalid source is retained as a rejected attempt; the applied candidate pointer is unchanged.",
        ),
        (
            "Cross-sheet Reconciler",
            [write_solution, read_artifact],
            ["public_case_1", "cross-sheet evidence"],
            ["candidate_solution"],
            ["create a new immutable candidate solution version"],
            "Invalid source is retained as a rejected attempt; no workbook is silently repaired.",
        ),
        (
            "Formatting & Presentation Specialist",
            [write_solution, read_artifact],
            ["public_case_1", "presentation evidence"],
            ["candidate_solution"],
            ["create a new immutable candidate solution version"],
            "Invalid source is retained as a rejected attempt and does not overwrite a prior candidate.",
        ),
        (
            "Workbook Repair Specialist",
            [revise_solution, read_artifact],
            ["current candidate", "fresh public failure evidence"],
            ["candidate_solution"],
            ["create a revised immutable candidate solution version"],
            "Rejected repair remains in history while the prior applied candidate remains available.",
        ),
        (
            "Workbook Test Executor",
            [run_public],
            ["current candidate", "public_case_1"],
            ["public_output", "public_execution"],
            ["write only clean-workspace public outputs and execution evidence"],
            "Candidate failure or timeout is task evidence; missing trusted runtime is UNAVAILABLE.",
        ),
        (
            "Workbook Integrity Validator",
            [validate_public],
            ["current candidate", "fresh public execution"],
            ["public_validation"],
            ["write validation evidence only"],
            "Validation failure does not mutate the candidate or output and records exact dependencies.",
        ),
    ]
    return [
        {
            "agent_id": agent_id,
            "purpose": f"Fulfill the {agent_id} responsibility without choosing the next routing action.",
            "typed_tools": tools,
            "preconditions": [f"{item} is available and allowed by the current public state" for item in reads],
            "read_artifacts": reads,
            "write_artifacts": writes,
            "allowed_mutations": mutations,
            "failure_consequences": failure,
        }
        for agent_id, tools, reads, writes, mutations, failure in definitions
    ]


def build_spreadsheets(cache: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive_path = cache / "KAKA22__SpreadsheetBench" / "spreadsheetbench_verified_400.tar.gz"
    full_archive_path = cache / "KAKA22__SpreadsheetBench" / "spreadsheetbench_912_v0.1.tar.gz"
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    if not full_archive_path.is_file():
        raise FileNotFoundError(
            "SpreadsheetBench full archive is required for migrated Python-OJ evaluator cases: "
            f"{full_archive_path}"
        )
    with tarfile.open(archive_path, "r:gz") as archive, tarfile.open(full_archive_path, "r:gz") as full_archive:
        dataset = json.load(archive.extractfile("spreadsheetbench_verified_400/dataset.json"))
        by_id = {str(item["id"]): item for item in dataset}
        for archetype, source_id in flatten(SPREADSHEET_SELECTION):
            source = by_id[source_id]
            prefix = f"spreadsheetbench_verified_400/spreadsheet/{source_id}/"
            members = [item for item in archive.getmembers() if item.name.startswith(prefix) and item.isfile()]
            init_member = next(item for item in members if item.name.endswith("_init.xlsx"))
            gold_member = next(item for item in members if item.name.endswith("_golden.xlsx"))
            input_bytes = archive.extractfile(init_member).read()
            gold_bytes = archive.extractfile(gold_member).read()
            packet_id = f"sdp-spreadsheetbench-{source_id}"
            pack_dir = output / "packs" / "structured_data_processing" / packet_id
            writer = PacketWriter(pack_dir)
            task = str(source["instruction"])
            if source_id in SPREADSHEET_PYTHON_OJ_SPECS:
                oj_spec = SPREADSHEET_PYTHON_OJ_SPECS[source_id]
                task = str(oj_spec.get("instruction_override", task))
                trusted_cases = list(oj_spec["trusted_cases"])
                excluded_cases = list(oj_spec["excluded_cases"])
                private_cases = [case_id for case_id in trusted_cases if case_id != 1]
                full_prefix = f"all_data_912_v0.1/spreadsheet/{source_id}/"
                full_members = {
                    case_id: {
                        kind: f"{full_prefix}{case_id}_{source_id}_{kind}.xlsx"
                        for kind in ("input", "answer")
                    }
                    for case_id in (1, 2, 3)
                }
                full_payloads = {
                    (case_id, kind): full_archive.extractfile(path).read()
                    for case_id, paths in full_members.items()
                    for kind, path in paths.items()
                }
                case1_full_archive_equivalent = (
                    workbook_values_equal(full_payloads[(1, "input")], input_bytes)
                    and workbook_values_equal(full_payloads[(1, "answer")], gold_bytes)
                )
                if not case1_full_archive_equivalent and not oj_spec.get("case1_difference_evidence"):
                    raise RuntimeError(f"SpreadsheetBench {source_id} case 1 has undocumented semantic value/formula differences between Verified-400 and v0.1 full archive")

                adapter_id = "spreadsheetbench.python_oj.v1"
                candidate_contract = {
                    "artifact_key": "candidate_solution",
                    "path": "candidate/solution.py",
                    "type": "python_workbook_transformation",
                    "entrypoint": "solve(input_path: str, output_path: str) -> None",
                    "required_output": "a readable XLSX workbook at output_path",
                }
                public_task = {
                    "source_id": source_id,
                    "instruction": task,
                    "observable_goal": "Produce one reusable Python workbook transformation that solves every trusted evaluator case.",
                    "public_case": {
                        "case_id": 1,
                        "input_workbook": "public/cases/case_1/input.xlsx",
                    },
                    "candidate": candidate_contract,
                    "answer_sheet": oj_spec.get("answer_sheet", source.get("answer_sheet")),
                    "answer_position": oj_spec["answer_position"],
                    "allowed_edit_scope": "candidate solution and runtime outputs only; packet inputs are immutable",
                    "agent_pool_id": "structured_data_processing",
                }
                if oj_spec.get("example_sheet"):
                    public_task["source_visible_example"] = {
                        "present": True,
                        "sheet": oj_spec["example_sheet"],
                        "meaning": "Partially completed source output/template supplied upstream; it is not a substitute for solving evaluator cases.",
                    }
                execution_contract = {
                    "adapter_id": adapter_id,
                    "candidate": candidate_contract,
                    "public_development_case_ids": [1],
                    "trusted_evaluator_case_ids": trusted_cases,
                    "excluded_case_ids": excluded_cases,
                    "network_during_execution": False,
                    "original_inputs_are_immutable": True,
                    "private_evaluator_feedback_during_episode": False,
                    "same_candidate_hash_required_for_all_cases": True,
                    "recalculate_before_grading": True,
                    "case_timeout_seconds": 60,
                    "maximum_output_bytes": 52428800,
                    "scoring": {
                        "case_results": "PASS/FAIL/UNAVAILABLE per included case",
                        "soft_score": f"passed_cases / {len(trusted_cases)} when all included cases are available",
                        "hard_pass": "all included evaluator cases pass",
                        "primary_task_outcome": "hard_pass",
                    },
                    "terminal_condition": "the submitted candidate version independently passes every trusted evaluator case",
                }
                validation_contract = {
                    **oj_spec["validation"],
                    "public_validation_scope": "case_1_only",
                    "trusted_oj_answer_position": oj_spec["answer_position"],
                }
                packet_spec = {
                    "domain": "structured_data_processing",
                    "adapter_id": adapter_id,
                    "packet_id": packet_id,
                    "source": {
                        "repository_or_dataset": SOURCE_LOCKS["structured_data_processing"]["artifact_repository"],
                        "revision": SOURCE_LOCKS["structured_data_processing"]["artifact_revision"],
                        "record_id": source_id,
                        "split": "SpreadsheetBench Verified-400 case 1 plus source-native evaluator cases from the full v0.1 archive",
                        "license": "CC-BY-SA-4.0",
                        "adaptations": [
                            "Normalize the model-facing deliverable to a reusable Python solve(input_path, output_path) entrypoint.",
                            "Use Verified-400 case 1 as the public development case.",
                            (
                                "Verify case-1 sheet names, cell values and formulas are equivalent to the full archive; retain Verified-400 bytes when ZIP/style serialization differs."
                                if case1_full_archive_equivalent
                                else "Retain the Verified-400 public case after documenting bounded case-1 archive differences outside the normalized evaluator contract."
                            ),
                            f"Use full-archive cases {private_cases} as private evaluator cases.",
                            *([f"Exclude cases {excluded_cases} before live runs with source-quality evidence."] if excluded_cases else []),
                            *oj_spec.get("metadata_repairs", []),
                        ],
                    },
                    "task_contract": {
                        "goal": "Produce one reusable Python transformation that satisfies the task and output contract for every trusted evaluator case.",
                        "initial_state": "Original instruction, public case-1 workbook, normalized answer ranges, and any source-provided partial output/template.",
                        "editable_scope": ["candidate/solution.py", "versioned runtime outputs and evidence"],
                        "immutable_scope": ["all packet files", "all evaluator inputs", "all private answers"],
                        "invariants": [
                            "Every input workbook remains byte-identical.",
                            "Every produced output is a readable XLSX workbook.",
                            "Required output sheets remain present and unrelated workbook content is preserved.",
                            f"The identical candidate hash is executed for cases {trusted_cases}.",
                            "All private evaluator cases and answers are absent from router and executor episode context.",
                        ],
                        "candidate_acceptance": "Python source parses, defines solve(input_path, output_path), and launches in the pinned sandbox; acceptance is not task success.",
                        "terminal_validation": "Fresh independent evidence for the submitted candidate hash reports PASS on every trusted evaluator case.",
                        "independent_checker": "Trusted clean-workspace execution and OJ cell comparison returning PASS, FAIL, or UNAVAILABLE per case.",
                    },
                    "runtime": {
                        "versions_and_revisions": {
                            "python": "3.13.9",
                            "openpyxl": "3.1.5",
                            "pandas": "2.3.3",
                            "pywin32": "311",
                            "recalculation_backend": "Microsoft Excel 16.0.20527.20112",
                            "recalculation_binary_sha256": "1c51767e1136479c38b3e9ce4f0e4d59f8facfa782b9718ad73516539da2815c",
                        },
                        "commands": [
                            "trusted runner invokes solve(input_path, output_path) in an isolated subprocess",
                            "trusted recalculation backend opens and saves only the produced workbook copy",
                            f"trusted checker applies the semantic contract and source OJ ranges {oj_spec['answer_position']}",
                        ],
                        "timeouts_and_resource_limits": {
                            "wall_clock_seconds_per_case": 60,
                            "maximum_output_bytes": 52428800,
                            "network": "disabled",
                        },
                        "network_policy": "No network access during candidate execution or checking.",
                        "provisioning_and_preflight": "Runtime profile must verify Python dependencies, candidate sandbox, Excel automation, clean-workspace isolation and checker smoke before live use.",
                    },
                    "visibility": {
                        "public_files": [
                            "public/task.json",
                            "public/packet_spec.json",
                            "public/execution_contract.json",
                            "public/validation_contract.json",
                            "public/cases/case_1/input.xlsx",
                        ],
                        "private_oracle_files": [
                            "private/evaluation_target.json",
                            "private/cases/case_1/answer.xlsx",
                            *[item for case_id in private_cases for item in (f"private/cases/case_{case_id}/input.xlsx", f"private/cases/case_{case_id}/answer.xlsx")],
                        ],
                        "fixture_materialization_allowlist": [],
                        "router_projection": "Instruction, public task contract, runtime artifact summaries, freshness and observable public failures; exclude raw workbook bytes, selection_archetype, private cases and answers.",
                        "executor_projection_by_role": {
                            "inspection_roles": "Public case 1 plus bounded workbook/range readers and current public artifacts.",
                            "candidate_roles": "Public case 1, current candidate/evidence and typed solution writer; no private evaluator assets.",
                            "execution_roles": "Current candidate and public case 1 only.",
                            "validation_roles": "Public output and public contract only; final private evaluation remains trusted and post-episode.",
                        },
                        "artifact_reader_scope": "Only allowlisted public/runtime artifact IDs with containment and hash checks; latest or explicit immutable version; no arbitrary paths.",
                        "raw_trace_replay_to_router": False,
                    },
                    "role_contracts": spreadsheet_oj_role_contracts(),
                    "artifact_contracts": [
                        {
                            "logical_key": "candidate_solution",
                            "immutable_file_and_version": "_runtime/artifacts/{version}_candidate_solution.py",
                            "input_versions": ["public case-1 input sha256", "task contract sha256", "runtime profile sha256"],
                            "dependency_refs": [],
                            "freshness_rule": "Candidate identity is its exact source hash plus runtime profile hash.",
                            "reader_semantics": "Explicit immutable version selected by current state.",
                            "bounded_read_policy": "UTF-8 source preview plus offset/length reader; raw source retained.",
                            "normalization_policy": "Normalize line endings for display only; hash and execute retained raw bytes.",
                        },
                        {
                            "logical_key": "public_execution",
                            "immutable_file_and_version": "_runtime/artifacts/{version}_public_execution.json",
                            "input_versions": ["candidate_solution hash", "case-1 input hash", "runtime profile hash"],
                            "dependency_refs": ["candidate_solution explicit version"],
                            "freshness_rule": "Stale immediately when candidate, case input, or runtime profile changes.",
                            "reader_semantics": "Latest fresh evidence plus explicit dependency refs.",
                            "bounded_read_policy": "Structured summary plus bounded stdout/stderr fragments with total lengths.",
                            "normalization_policy": "Representation-only truncation; raw logs retained separately.",
                        },
                        {
                            "logical_key": "public_validation",
                            "immutable_file_and_version": "_runtime/artifacts/{version}_public_validation.json",
                            "input_versions": ["public output hash", "candidate_solution hash", "checker version"],
                            "dependency_refs": ["public_execution explicit version"],
                            "freshness_rule": "Cannot remain fresh after candidate, output, checker, or runtime changes.",
                            "reader_semantics": "Latest fresh evidence only in routing observation.",
                            "bounded_read_policy": "Structured issue list and bounded evidence refs.",
                            "normalization_policy": "No semantic repair or hidden-case feedback.",
                        },
                    ],
                    "automatic_side_effects": [
                        "Create clean per-run workspace copies.",
                        "Recalculate only produced workbook copies during trusted checking.",
                        "Retain rejected candidates and execution evidence as immutable history.",
                    ],
                    "router_selected_recovery_actions": ["Workbook Repair Specialist revises the candidate only when selected by the router."],
                    "error_classification_policy": "Parse/schema/precondition errors are rejected actions; candidate exceptions, timeouts, missing output and wrong cells are task FAIL; missing trusted runtime or checker infrastructure is UNAVAILABLE; transport retry metadata is not a routing action.",
                    "selection_and_difficulty_policy": "Packet and included evaluator cases are fixed before live outcomes; every exclusion requires source-quality evidence recorded before execution.",
                    "gates": {
                        "manifest_validation": "pending",
                        "runtime_preflight": "pending",
                        "positive_and_negative_checker_tests": "pending",
                        "artifact_handoff_and_freshness": "pending",
                        "visibility_isolation": "pending",
                        "role_tool_conformance": "pending",
                        "common_runner_integration": "pending",
                        "human_packet_review": "pending",
                    },
                }
                evaluation_target = {
                    "answer_sheet": oj_spec.get("answer_sheet", source.get("answer_sheet")),
                    "answer_position": oj_spec["answer_position"],
                    "upstream_answer_position": source.get("answer_position"),
                    "data_position": source.get("data_position"),
                    "public_case_ids": [1],
                    "trusted_evaluator_case_ids": trusted_cases,
                    "excluded_case_ids": excluded_cases,
                    "soft_score_denominator": len(trusted_cases),
                    "hard_pass": "all included evaluator cases pass",
                    "candidate_entrypoint": candidate_contract["entrypoint"],
                    "no_private_feedback_during_episode": True,
                }
                case_selection = {
                    "selection_fixed_before_live": True,
                    "included_cases": [
                        {
                            "case_id": case_id,
                            "role": "public_development_case" if case_id == 1 else "private_evaluator_case",
                            "verification": oj_spec.get("case_verification", "instruction-derived reference transform matches the source OJ ranges"),
                        }
                        for case_id in trusted_cases
                    ],
                    "excluded_cases": [
                        {
                            "case_id": case_id,
                            "reason": "upstream_answer_inconsistent_with_instruction",
                            "excluded_before_live_runs": True,
                            "input_sha256": sha256_bytes(full_payloads[(case_id, "input")]),
                            "answer_sha256": sha256_bytes(full_payloads[(case_id, "answer")]),
                            "evidence": oj_spec.get("exclusion_evidence", {}).get(str(case_id), []),
                        }
                        for case_id in excluded_cases
                    ],
                    "case_1_archive_equivalent": case1_full_archive_equivalent,
                    "case_1_difference_evidence": oj_spec.get("case1_difference_evidence", []),
                }
                provenance = {
                    "canonical_repository": SOURCE_LOCKS["structured_data_processing"]["canonical_repository"],
                    "canonical_revision": SOURCE_LOCKS["structured_data_processing"]["canonical_revision"],
                    "artifact_repository": SOURCE_LOCKS["structured_data_processing"]["artifact_repository"],
                    "artifact_revision": SOURCE_LOCKS["structured_data_processing"]["artifact_revision"],
                    "license": "CC-BY-SA-4.0",
                    "source_id": source_id,
                    "archives": [
                        {"path": archive_path.name, "sha256": sha256_file(archive_path), "purpose": "Verified case 1 and task metadata"},
                        {"path": full_archive_path.name, "sha256": sha256_file(full_archive_path), "purpose": "Original source-native OJ cases selected before live execution"},
                    ],
                    "members": [
                        {"case_id": 1, "kind": "input", "source_member": init_member.name, "sha256": sha256_bytes(input_bytes), "visibility": "public"},
                        {"case_id": 1, "kind": "answer", "source_member": gold_member.name, "sha256": sha256_bytes(gold_bytes), "visibility": "private"},
                        *[
                            {"case_id": case_id, "kind": kind, "source_member": full_members[case_id][kind], "sha256": sha256_bytes(full_payloads[(case_id, kind)]), "visibility": "private"}
                            for case_id in private_cases for kind in ("input", "answer")
                        ],
                    ],
                    "adaptations": packet_spec["source"]["adaptations"],
                }

                writer.add("public/task.json", json_bytes(public_task), "public")
                writer.add("public/packet_spec.json", json_bytes(packet_spec), "public")
                writer.add("public/execution_contract.json", json_bytes(execution_contract), "public")
                writer.add("public/validation_contract.json", json_bytes(validation_contract), "public")
                writer.add("public/cases/case_1/input.xlsx", input_bytes, "public")
                writer.add("private/evaluation_target.json", json_bytes(evaluation_target), "private")
                writer.add("private/cases/case_1/answer.xlsx", gold_bytes, "private")
                for case_id in private_cases:
                    writer.add(f"private/cases/case_{case_id}/input.xlsx", full_payloads[(case_id, "input")], "private")
                    writer.add(f"private/cases/case_{case_id}/answer.xlsx", full_payloads[(case_id, "answer")], "private")
                writer.add("fixtures/upstream_provenance.json", json_bytes(provenance), "fixture")
                writer.add("fixtures/case_selection.json", json_bytes(case_selection), "fixture")
                validation = writer.finish(
                    packet_id=packet_id,
                    domain="structured_data_processing",
                    task=task,
                    adapter_id=adapter_id,
                    source_type="spreadsheetbench_verified_400_plus_912_oj_cases",
                    license_name="CC-BY-SA-4.0",
                    metadata={
                        "source_id": source_id,
                        "selection_archetype": archetype,
                        "public_case_ids": [1],
                        "trusted_evaluator_case_ids": trusted_cases,
                        "excluded_case_ids": excluded_cases,
                    },
                    generator_version="routingbench-v4-five-domains-v5-spreadsheet-oj",
                )
                validations.append(validation)
                rows.append(
                    selection_row(
                        packet_id,
                        "structured_data_processing",
                        source_id,
                        archetype,
                        public_case_ids=[1],
                        trusted_evaluator_case_ids=trusted_cases,
                        excluded_case_ids=excluded_cases,
                    )
                )
                continue

            public_task = {
                "source_id": source_id,
                "instruction": task,
                "input_workbook": "public/input.xlsx",
                "candidate_output": "candidate/output.xlsx",
                "answer_sheet": source.get("answer_sheet"),
                "answer_position": source.get("answer_position"),
                "allowed_edit_scope": "workbook copy only; preserve unrelated content",
                "agent_pool_id": "structured_data_processing",
            }
            contract = {
                "adapter_id": "spreadsheetbench.workbook.v1",
                "network_during_execution": False,
                "original_is_immutable": True,
                "recalculate_before_grading": True,
                "terminal_condition": "candidate workbook passes OJ-style comparison and integrity checks",
                "private_reference_visible_to_router": False,
            }
            writer.add("public/task.json", json_bytes(public_task), "public")
            writer.add("public/input.xlsx", input_bytes, "public")
            writer.add("public/execution_contract.json", json_bytes(contract), "public")
            writer.add("private/golden.xlsx", gold_bytes, "private")
            writer.add(
                "private/evaluation_target.json",
                json_bytes({key: clean_value(source.get(key)) for key in ("answer_sheet", "answer_position", "data_position")}),
                "private",
            )
            writer.add(
                "fixtures/upstream_provenance.json",
                json_bytes({**SOURCE_LOCKS["structured_data_processing"], "source_id": source_id}),
                "fixture",
            )
            validation = writer.finish(
                packet_id=packet_id,
                domain="structured_data_processing",
                task=task,
                adapter_id="spreadsheetbench.workbook.v1",
                source_type="spreadsheetbench_verified_400",
                license_name="CC-BY-SA-4.0",
                metadata={"source_id": source_id, "selection_archetype": archetype},
            )
            validations.append(validation)
            rows.append(selection_row(packet_id, "structured_data_processing", source_id, archetype))
    SOURCE_LOCKS["structured_data_processing"]["artifact_sha256"] = sha256_file(archive_path)
    SOURCE_LOCKS["structured_data_processing"]["full_oj_artifact_path"] = full_archive_path.name
    SOURCE_LOCKS["structured_data_processing"]["full_oj_artifact_sha256"] = sha256_file(full_archive_path)
    return rows, validations


def build_science(cache: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parquet = cache / "osunlp__ScienceAgentBench" / "data" / "verified-00000-of-00001.parquet"
    dataframe = pd.read_parquet(parquet)
    by_id = {int(item["instance_id"]): clean_value(item) for item in dataframe.to_dict(orient="records")}
    selected_ids = {int(source_id) for _, source_id in flatten(SCIENCE_SELECTION)}
    if selected_ids & EXCLUDED_SCIENCE_IDS:
        raise RuntimeError("Science selection contains a source-license exception")
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for archetype, source_id in flatten(SCIENCE_SELECTION):
        source = by_id[int(source_id)]
        packet_id = f"sci-scienceagentbench-{int(source_id):03d}"
        writer = PacketWriter(output / "packs" / "scientific_procedure" / packet_id)
        task = str(source["task_inst"])
        public_task = {
            "source_id": str(source_id),
            "instruction": task,
            "scientific_domain": source["domain"],
            "subtask_categories": source["subtask_categories"],
            "domain_knowledge": source["domain_knowledge"],
            "dataset_folder_tree": source["dataset_folder_tree"],
            "dataset_preview": source["dataset_preview"],
            "output_path": source["output_fname"],
            "agent_pool_id": "scientific_procedure",
        }
        writer.add("public/task.json", json_bytes(public_task), "public")
        writer.add(
            "public/execution_contract.json",
            json_bytes(
                {
                    "adapter_id": "scienceagentbench.procedure.v1",
                    "network_during_execution": False,
                    "candidate_output": source["output_fname"],
                    "terminal_condition": "task-specific deterministic evaluator passes on an isolated output copy",
                    "source_assets_required_at_runtime": True,
                }
            ),
            "public",
        )
        writer.add(
            "private/evaluator_locator.json",
            json_bytes(
                {
                    "gold_program_name": source["gold_program_name"],
                    "eval_script_name": source["eval_script_name"],
                    "source_program_path": source["src_file_or_path"],
                }
            ),
            "private",
        )
        writer.add(
            "fixtures/upstream_provenance.json",
            json_bytes({**SOURCE_LOCKS["scientific_procedure"], "source_id": str(source_id), "github_name": source["github_name"]}),
            "fixture",
        )
        validation = writer.finish(
            packet_id=packet_id,
            domain="scientific_procedure",
            task=task,
            adapter_id="scienceagentbench.procedure.v1",
            source_type="scienceagentbench_verified_metadata",
            license_name="CC-BY-4.0",
            metadata={"source_id": str(source_id), "selection_archetype": archetype, "scientific_domain": source["domain"]},
        )
        validations.append(validation)
        rows.append(selection_row(packet_id, "scientific_procedure", str(source_id), archetype))
    SOURCE_LOCKS["scientific_procedure"]["artifact_sha256"] = sha256_file(parquet)
    SOURCE_LOCKS["scientific_procedure"]["excluded_source_ids"] = sorted(EXCLUDED_SCIENCE_IDS)
    return rows, validations


def decrypt_appworld_bundle(path: Path) -> zipfile.ZipFile:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    encrypted = path.read_bytes()
    key = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"Nvx#rYcYQ2%btf",
        iterations=100000,
        backend=default_backend(),
    ).derive(b"WEquKLy##9M@qu")
    decryptor = Cipher(algorithms.AES(key), modes.CFB(encrypted[:16]), backend=default_backend()).decryptor()
    return zipfile.ZipFile(io.BytesIO(decryptor.update(encrypted[16:]) + decryptor.finalize()))


def build_appworld(cache: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = cache / "appworld_wheel" / "data-0.1.0.bundle"
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    with decrypt_appworld_bundle(bundle) as archive:
        test_ids = set(archive.read("data/datasets/test_normal.txt").decode("utf-8").split())
        for source_id in APPWORLD_SELECTION:
            if source_id not in test_ids:
                raise RuntimeError(f"AppWorld task is not in test_normal: {source_id}")
            metadata = json.loads(archive.read(f"data/tasks/{source_id}/ground_truth/metadata.json"))
            if int(metadata["num_apps"]) < 2:
                raise RuntimeError(f"AppWorld task is not multi-app: {source_id}")
            packet_id = f"xapp-appworld-{source_id}"
            writer = PacketWriter(output / "packs" / "cross_app_workflow" / packet_id)
            public_task = {
                "source_task_id": source_id,
                "source_split": "test_normal",
                "protected_task_loader": "appworld.task.Task.load",
                "instruction_visibility": "loaded from protected bundle at runtime; not redistributed in plaintext",
                "agent_pool_id": "cross_app_workflow",
            }
            task = f"Execute protected AppWorld task {source_id} using the runtime-loaded instruction and state-based contract."
            writer.add("public/task_reference.json", json_bytes(public_task), "public")
            writer.add(
                "public/execution_contract.json",
                json_bytes(
                    {
                        "adapter_id": "appworld.cross_app.v1",
                        "protected_bundle_version": "0.1.0",
                        "network_during_execution": False,
                        "initial_state_is_reset_per_run": True,
                        "all_mutations_are_logged": True,
                        "auto_rollback": False,
                        "terminal_condition": "state-based task tests pass without collateral-damage failures",
                    }
                ),
                "public",
            )
            writer.add(
                "private/evaluator_locator.json",
                json_bytes(
                    {
                        "source_task_id": source_id,
                        "protected_paths": [
                            f"data/tasks/{source_id}/ground_truth/evaluation.py",
                            f"data/tasks/{source_id}/ground_truth/test_data.json",
                        ],
                        "plaintext_redistribution": False,
                    }
                ),
                "private",
            )
            writer.add(
                "fixtures/upstream_provenance.json",
                json_bytes({**SOURCE_LOCKS["cross_app_workflow"], "source_task_id": source_id, "source_split": "test_normal"}),
                "fixture",
            )
            validation = writer.finish(
                packet_id=packet_id,
                domain="cross_app_workflow",
                task=task,
                adapter_id="appworld.cross_app.v1",
                source_type="appworld_protected_reference",
                license_name=str(SOURCE_LOCKS["cross_app_workflow"]["license"]),
                metadata={
                    "source_id": source_id,
                    "source_split": "test_normal",
                    "selection_archetype": "protected_review_only",
                },
            )
            validations.append(validation)
            rows.append(
                selection_row(
                    packet_id,
                    "cross_app_workflow",
                    source_id,
                    "protected_review_only",
                )
            )
    SOURCE_LOCKS["cross_app_workflow"]["protected_bundle_sha256"] = sha256_file(bundle)
    wheel = cache / "appworld_wheel" / "appworld-0.1.3.post1-py3-none-any.whl"
    SOURCE_LOCKS["cross_app_workflow"]["package_wheel_sha256"] = sha256_file(wheel)
    return rows, validations


def build_musique(cache: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_path = cache / "voidful__MuSiQue" / "musique_full_v1.0_dev.jsonl"
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
    by_id: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        by_id.setdefault(item["id"], []).append(item)
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    answerable_count = 0
    for archetype, source_id in flatten(MUSIQUE_SELECTION):
        expected_answerable = archetype != "insufficient_evidence"
        matches = [item for item in by_id[source_id] if bool(item["answerable"]) == expected_answerable]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one MuSiQue variant for {source_id} answerable={expected_answerable}, found {len(matches)}"
            )
        source = matches[0]
        answerable = bool(source["answerable"])
        answerable_count += int(answerable)
        if archetype == "insufficient_evidence" and answerable:
            raise RuntimeError(f"Expected unanswerable MuSiQue sample: {source_id}")
        if archetype != "insufficient_evidence" and not answerable:
            raise RuntimeError(f"Expected answerable MuSiQue sample: {source_id}")
        variant_id = f"{source_id}:{'answerable' if answerable else 'unanswerable'}"
        packet_id = f"mhop-musique-{short_hash(variant_id)}"
        writer = PacketWriter(output / "packs" / "multi_hop_knowledge" / packet_id)
        public_paragraphs = [
            {key: paragraph[key] for key in ("idx", "title", "paragraph_text")}
            for paragraph in source["paragraphs"]
        ]
        task = str(source["question"])
        writer.add(
            "public/task.json",
            json_bytes(
                {
                    "source_id": source_id,
                    "source_variant": "answerable" if answerable else "unanswerable",
                    "question": task,
                    "corpus_path": "public/corpus.json",
                    "answer_format": "short answer with cited paragraph IDs, or INSUFFICIENT_EVIDENCE",
                    "agent_pool_id": "multi_hop_knowledge",
                }
            ),
            "public",
        )
        writer.add("public/corpus.json", json_bytes({"paragraphs": public_paragraphs}), "public")
        writer.add(
            "public/execution_contract.json",
            json_bytes(
                {
                    "adapter_id": "musique.frozen_corpus.v1",
                    "open_web_allowed": False,
                    "terminal_condition": "answer and evidence-support checks are reported separately",
                    "insufficiency_is_a_valid_terminal_answer": True,
                }
            ),
            "public",
        )
        supporting_ids = [paragraph["idx"] for paragraph in source["paragraphs"] if paragraph.get("is_supporting")]
        writer.add(
            "private/reference.json",
            json_bytes(
                {
                    "answerable": answerable,
                    "answer": source["answer"],
                    "answer_aliases": source["answer_aliases"],
                    "supporting_paragraph_ids": supporting_ids,
                    "question_decomposition": source["question_decomposition"],
                }
            ),
            "private",
        )
        writer.add(
            "fixtures/upstream_provenance.json",
            json_bytes(
                {
                    **SOURCE_LOCKS["multi_hop_knowledge"],
                    "source_id": source_id,
                    "source_variant": "answerable" if answerable else "unanswerable",
                    "source_split": "dev/full",
                }
            ),
            "fixture",
        )
        validation = writer.finish(
            packet_id=packet_id,
            domain="multi_hop_knowledge",
            task=task,
            adapter_id="musique.frozen_corpus.v1",
            source_type="musique_full_v1_dev",
            license_name="CC-BY-4.0",
            metadata={"source_id": source_id, "selection_archetype": archetype, "answerable": answerable},
        )
        validations.append(validation)
        rows.append(
            selection_row(
                packet_id,
                "multi_hop_knowledge",
                variant_id,
                archetype,
                answerable=answerable,
            )
        )
    if answerable_count != 16:
        raise RuntimeError(f"Expected 16 answerable MuSiQue samples, found {answerable_count}")
    SOURCE_LOCKS["multi_hop_knowledge"]["artifact_sha256"] = sha256_file(data_path)
    return rows, validations


OMNI_LICENSE_FILES = {
    "olmo_arxiv_math": ["olmocr-bench-LICENSE.txt"],
    "olmo_long_tiny_text": ["olmocr-bench-LICENSE.txt"],
    "olmo_old_scans": ["olmocr-bench-LICENSE.txt"],
    "olmo_old_scans_math": ["olmocr-bench-LICENSE.txt"],
    "tables_derived": ["pmc-LICENSE.txt"],
    "docparsing_images": ["docparsingbench-LICENSE.txt"],
    "realdoc_pages": ["realdoc-bench-LICENSE.txt"],
    "multiling_finepdfs": ["finepdfs-LICENSE.txt"],
    "forms_filled": ["us-federal-forms-LICENSE.txt", "cerfa-LICENSE.txt"],
    "forms_federal": ["us-federal-forms-LICENSE.txt"],
    "forms_documentcloud": ["documentcloud-LICENSE.txt"],
    "handwriting_samples": ["loc-catt-papers-LICENSE.txt", "nls-marjory-fleming-LICENSE.txt", "htr-united-LICENSE.txt"],
}


def build_omni(cache: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = cache / "datalab-to__omni_parse_bench"
    manifest = pd.read_parquet(root / "manifest.parquet").set_index("page_id")
    results = pd.read_parquet(root / "results.parquet")
    results = results[results["system"] == "Datalab accurate"]
    pmc = pd.read_csv(root / "licenses" / "pmc_articles.csv")
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for archetype, source_id in flatten(OMNI_SELECTION):
        source = clean_value(manifest.loc[source_id].to_dict())
        page_results = results[results["page_id"] == source_id]
        test_families = sorted(set(str(value) for value in page_results["test_type"].tolist()))
        if len(test_families) < 2:
            raise RuntimeError(f"Omni sample has fewer than two test families: {source_id}")
        page_path = root / str(source["doc_path"])
        gold_path = root / str(source["gt_path"])
        tests = json.loads(gold_path.read_text(encoding="utf-8"))
        packet_id = f"dmp-omni-{short_hash(source_id)}"
        writer = PacketWriter(output / "packs" / "document_media_processing" / packet_id)
        extension = page_path.suffix.lower()
        public_page = f"public/page{extension}"
        task = "Parse the supplied page into the requested structured outputs while preserving reading order, layout semantics, and extractable content."
        writer.add(
            "public/task.json",
            json_bytes(
                {
                    "source_id": source_id,
                    "page_path": public_page,
                    "required_outputs": source["outputs"],
                    "candidate_output_directory": "candidate/",
                    "agent_pool_id": "document_media_processing",
                }
            ),
            "public",
        )
        writer.add(public_page, page_path.read_bytes(), "public")
        writer.add(
            "public/execution_contract.json",
            json_bytes(
                {
                    "adapter_id": "omni_parse_bench.page.v1",
                    "network_during_execution": False,
                    "terminal_condition": "all source unit-test families are evaluated on the candidate output copy",
                    "model_inference_during_grading": False,
                }
            ),
            "public",
        )
        writer.add("private/tests.json", json_bytes(tests), "private")
        suite = str(source["suite"])
        license_files = OMNI_LICENSE_FILES[suite]
        provenance: dict[str, Any] = {
            **SOURCE_LOCKS["document_media_processing"],
            "source_id": source_id,
            "suite": suite,
            "document": source["document"],
            "license_files": license_files,
            "test_families": test_families,
        }
        if suite == "tables_derived":
            pmcid = re.search(r"PMC\d+", source_id)
            credit = pmc[pmc["pmcid"] == pmcid.group(0)].to_dict(orient="records") if pmcid else []
            provenance["page_credit"] = clean_value(credit[0]) if credit else None
        writer.add("fixtures/upstream_provenance.json", json_bytes(provenance), "fixture")
        for license_file in license_files:
            writer.add(f"fixtures/licenses/{license_file}", (root / "licenses" / license_file).read_bytes(), "fixture")
        validation = writer.finish(
            packet_id=packet_id,
            domain="document_media_processing",
            task=task,
            adapter_id="omni_parse_bench.page.v1",
            source_type="omni_parse_bench_page",
            license_name="per-page upstream license; evaluator tests CC-BY-4.0/source-specific",
            metadata={"source_id": source_id, "selection_archetype": archetype, "suite": suite, "test_families": test_families},
        )
        validations.append(validation)
        rows.append(selection_row(packet_id, "document_media_processing", source_id, archetype, test_families=test_families))
    SOURCE_LOCKS["document_media_processing"]["manifest_sha256"] = sha256_file(root / "manifest.parquet")
    SOURCE_LOCKS["document_media_processing"]["results_sha256"] = sha256_file(root / "results.parquet")
    return rows, validations


def selection_row(packet_id: str, domain: str, source_id: str, archetype: str, **extra: Any) -> dict[str, Any]:
    return {
        "packet_id": packet_id,
        "domain": domain,
        "source_id": str(source_id),
        "selection_archetype": archetype,
        "selection_status": "selected_before_live_trajectory",
        "human_review_status": "pending",
        **extra,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def readme(selection_rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in selection_rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    return f"""# RoutingBench V4 — five new domains / 100 packets

This package materializes the fixed pre-live selection for the five new domains.

## Contents

- Structured data processing: {counts.get('structured_data_processing', 0)} packets.
- Scientific procedure: {counts.get('scientific_procedure', 0)} packets.
- Cross-app workflow: {counts.get('cross_app_workflow', 0)} packets.
- Multi-hop knowledge: {counts.get('multi_hop_knowledge', 0)} packets.
- Document/media processing: {counts.get('document_media_processing', 0)} packets.
- Total: {len(selection_rows)} packets.

There is intentionally no dataset split or holdout. Each domain contains five selection archetypes
with four packets per archetype. Archetype labels live in the selection manifest and packet metadata,
not in executor-visible task files, so they cannot prescribe a route.

The absence of a global dataset split does not prohibit source-native evaluator cases inside one
packet. Those cases test a submitted reusable artifact and are never used to select packets or tune
the decision model after outcomes are observed.

## Visibility

- `public/`: task input and execution contract visible to executors.
- `private/`: references and deterministic evaluator material; never router-visible.
- `fixtures/`: provenance, source locators and license evidence; not a routing hint.

All manifests remain `pending_runtime_verification`. Packet construction validates file integrity and
visibility only; deterministic preflight and live trajectories are separate gates.

All twenty structured-data packets use the source-faithful Python OJ contract. Case 1 is public and
source-native cases 2 and 3 are private evaluator cases, except `250-20`, which uses case 3 privately
and excludes upstream case 2 because its answer is inconsistent with its input and instruction. The
adapter, restricted worker and independent semantic checker are implemented for these contracts; each
packet must still pass its recorded preflight, human review and live-run gates before its trajectory is
admitted as benchmark evidence.

AppWorld task text, evaluator code and state are not redistributed in plaintext. Its packets contain
task IDs and a protected-bundle locator; runtime preparation must obtain the official encrypted bundle.

ScienceAgentBench packets include verified public metadata but still require the official protected
task assets at runtime. The six upstream-license exception IDs are excluded from this selection.
"""


def build(source_cache: Path, output_root: Path, role_design: Path, overwrite: bool) -> dict[str, Any]:
    source_cache = source_cache.resolve()
    output_root = output_root.resolve()
    role_design = role_design.resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite to rebuild")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    pools = parse_role_cards(role_design)
    for domain, pool in pools.items():
        write_json(output_root / "agent_pools" / f"{domain}.json", pool)
    write_json(
        output_root / "routing_contract.json",
        {
            "schema_version": "routingbench.v4.dynamic_routing_contract.v1",
            "guard_enabled": False,
            "router_sees_full_agent_pool_plus_stop": True,
            "workflow_graph_provided": False,
            "valid_action_mask_provided": False,
            "automatic_rollback": False,
            "all_actions_and_side_effects_are_logged": True,
            "budget_exhaustion": "request final decision; record non-STOP as suppressed_by_budget and exclude it from executed routing metrics",
        },
    )

    selection_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for builder in (build_spreadsheets, build_science, build_appworld, build_musique, build_omni):
        rows, checks = builder(source_cache, output_root)
        selection_rows.extend(rows)
        validations.extend(checks)

    selection_rows.sort(key=lambda row: (row["domain"], row["selection_archetype"], row["packet_id"]))
    write_jsonl(output_root / "selection_manifest.jsonl", selection_rows)
    write_json(output_root / "source_lock.json", SOURCE_LOCKS)
    write_json(
        output_root / "selection_report.json",
        {
            "schema_version": "routingbench.v4.five_domain_selection.v1",
            "selected_total": len(selection_rows),
            "selected_by_domain": {
                domain: sum(row["domain"] == domain for row in selection_rows)
                for domain in sorted(SOURCE_LOCKS)
            },
            "archetype_quota": "5 archetypes x 4 packets per domain",
            "split_or_holdout": False,
            "selected_before_live_trajectory": True,
            "human_review_status": "pending",
        },
    )
    write_json(
        output_root / "validation_report.json",
        {
            "valid": len(validations) == 100 and all(item["valid"] for item in validations),
            "packet_count": len(validations),
            "packets": validations,
        },
    )
    (output_root / "README.md").write_text(readme(selection_rows), encoding="utf-8", newline="\n")
    if len(selection_rows) != 100 or len(validations) != 100:
        raise RuntimeError(f"Expected 100 packets, got rows={len(selection_rows)}, validations={len(validations)}")
    return {
        "output_root": str(output_root),
        "packet_count": len(validations),
        "valid": all(item["valid"] for item in validations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--role-design", type=Path, default=ROLE_DESIGN)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.source_cache, args.output, args.role_design, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
