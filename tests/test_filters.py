from sensitive_egress_poc.filters import contains_disallowed_secret_like_content, is_masked_card_or_account_reference, looks_like_full_card_or_account_number


def test_masked_references_allowed():
    assert is_masked_card_or_account_reference("银行卡尾号 1234")
    assert is_masked_card_or_account_reference("card ending 1234")
    assert is_masked_card_or_account_reference("账号 ****5678")
    assert not contains_disallowed_secret_like_content("账号 ****5678 余额人民币 100")


def test_full_numbers_rejected():
    assert looks_like_full_card_or_account_number("card 4111111111111111")
    assert contains_disallowed_secret_like_content("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456")
