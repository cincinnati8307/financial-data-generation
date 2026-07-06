from sensitive_egress_poc.filters import looks_like_full_card_or_account_number, validate_synthetic_example, validate_augmented_row
from sensitive_egress_poc.llm_augmenter import DryRunProvider, augment_rows
from sensitive_egress_poc.synthetic_generator import SyntheticFinancialGenerator


def test_generator_creates_valid_private_examples_without_full_numbers():
    gen=SyntheticFinancialGenerator(seed=1)
    rows=gen.generate_private(30)
    assert rows
    for row in rows:
        ok, reason = validate_synthetic_example(row)
        assert ok, reason
        assert row["label"] == "financial_private"
        assert not looks_like_full_card_or_account_number(row["text"])


def test_hard_negatives_non_private_labels():
    gen=SyntheticFinancialGenerator(seed=2)
    rows=gen.generate_hard_negatives(30)
    assert {r["label"] for r in rows} <= {"non_private_financial", "benign"}


def test_dry_run_augmentation_produces_valid_rows():
    gen=SyntheticFinancialGenerator(seed=3)
    source=gen.generate_private(1)[0]
    rows, reasons = augment_rows([source], DryRunProvider(), 1, 6, False, "dry-run")
    assert rows
    for row in rows:
        ok, reason = validate_augmented_row(row, source)
        assert ok, reason
