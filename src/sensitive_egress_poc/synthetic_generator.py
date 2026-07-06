from __future__ import annotations

import json
import random
from dataclasses import asdict

from .filters import validate_synthetic_example
from .schemas import MixedEgressExample, PRIVATE_SUBTYPES, SyntheticExample

MAINLAND_BANKS = ["中国银行", "工商银行", "建设银行", "招商银行", "交通银行", "农业银行"]
SINGAPORE_BANKS = ["DBS", "OCBC", "UOB", "星展银行", "华侨银行", "大华银行"]
PAYMENT_APPS = ["支付宝", "微信支付", "PayNow", "GrabPay", "云闪付"]
MERCHANTS = ["盒马", "美团", "滴滴", "京东", "淘宝", "饿了么", "FairPrice", "Grab", "Shopee", "Guardian", "Toast Box"]
CURRENCIES = ["人民币", "RMB", "CNY", "¥", "新币", "SGD", "S$"]
FORMATS = ["natural_sentence", "key_value", "csv_row", "json", "agent_summary"]
STYLES = ["zh_casual", "zh_formal", "zh_en_codeswitch", "agent_summary", "email_mixed"]

HARD_NEGATIVES = [
    ("中国人民银行宣布下调贷款市场报价利率。", "non_private_financial"),
    ("新加坡股市今天小幅上涨。", "non_private_financial"),
    ("这家公司第三季度营收增长 12%。", "non_private_financial"),
    ("The central bank raised interest rates this week.", "non_private_financial"),
    ("The database transaction was rolled back after the error.", "benign"),
    ("这个 account settings 页面打不开。", "benign"),
    ("手机 battery charge 只剩 20%。", "benign"),
    ("河岸 bank 在暴雨后塌陷了。", "benign"),
    ("会议讨论了预算规划，但没有涉及个人账户信息。", "non_private_financial"),
    ("财务部门会统一处理报销流程。", "non_private_financial"),
]
BENIGN = [
    "会议从下午两点开始。", "The document contains three sections.", "今天的天气比较热。",
    "The function returns a boolean value.", "请在周五之前 review 这个 draft。", "The article explains how ocean currents work.",
]

class SyntheticFinancialGenerator:
    def __init__(self, seed: int = 1337) -> None:
        self.rng = random.Random(seed)
        self.counter = 0

    def _id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:06d}"

    def _amount(self, subtype: str, sg: bool = False) -> str:
        ranges = {"loan_debt": (50000, 900000), "investment": (5000, 80000), "tax": (30000, 500000), "transaction": (20, 2000)}
        lo, hi = ranges.get(subtype, (500, 30000))
        n = self.rng.randint(lo, hi)
        cur = self.rng.choice(["SGD", "S$", "新币"] if sg else ["人民币", "RMB", "CNY", "¥"])
        return f"{cur} {n:,}"

    def _base_private(self, subtype: str) -> tuple[str, str, str]:
        sg = self.rng.random() < 0.35
        bank = self.rng.choice(SINGAPORE_BANKS if sg else MAINLAND_BANKS)
        app = self.rng.choice(PAYMENT_APPS)
        merchant = self.rng.choice(MERCHANTS)
        amt = self._amount(subtype, sg)
        templates = {
            "bank_balance": [f"今天查了一下，{bank} 账户里还剩 {amt}。", f"我的{bank}储蓄卡余额是 {amt}，银行卡尾号 1234。"],
            "transaction": [f"我昨天在{merchant}消费了 {amt}。", f"{bank} card ending 1234 显示一笔 {merchant} transaction：{amt}。"],
            "salary_income": [f"这个月工资到账 {amt}。", f"我的税后工资是 {amt}，入账到账号 ****5678。"],
            "card_payment": [f"这个月信用卡账单是 {amt}。", f"信用卡尾号 1234 的 card payment due 是 {amt}。"],
            "loan_debt": [f"我的房贷余额还有 {amt}。", f"贷款账户 ****5678 的剩余 debt 是 {amt}。"],
            "invoice_receipt": [f"这张发票金额是 {amt}，商户是{merchant}。", f"receipt summary: merchant={merchant}, amount={amt}, card ending 1234。"],
            "investment": [f"我的股票账户现在价值 {amt}。", f"投资 account ****5678 当前 portfolio value 是 {amt}。"],
            "tax": [f"我的个税申报显示年收入是 {amt}。", f"tax filing summary：年度应税收入 {amt}。"],
            "wallet_payment": [f"{app}扣款 {amt}，商户是{merchant}。", f"用 {app} paid {merchant}，金额 {amt}。"],
        }
        return self.rng.choice(templates[subtype]), ("singapore_cn" if sg else "mainland_cn"), ("zh_en" if re_has_en(self.rng.choice([bank, app, merchant])) else "zh")

    def _format_text(self, text: str, fmt: str, subtype: str) -> str:
        if fmt == "key_value":
            return f"subtype: {subtype}\n备注: {text}"
        if fmt == "csv_row":
            return f"类别,内容\n{subtype},{text}"
        if fmt == "json":
            return json.dumps({"subtype": subtype, "note": text}, ensure_ascii=False)
        if fmt == "agent_summary":
            return f"Agent summary: 用户文本包含个人财务信息。Evidence: {text}"
        return text

    def private_example(self, subtype: str | None = None) -> SyntheticExample:
        subtype = subtype or self.rng.choice(PRIVATE_SUBTYPES)
        base, region, language = self._base_private(subtype)
        fmt = self.rng.choice(FORMATS)
        row = SyntheticExample(self._id("fin_priv"), self._format_text(base, fmt, subtype), "financial_private", subtype, region, language, fmt, self.rng.choice(STYLES), "high", "synthetic_template", {})
        ok, reason = validate_synthetic_example(asdict(row))
        if not ok: raise ValueError(reason)
        return row

    def hard_negative_example(self) -> SyntheticExample:
        text, label = self.rng.choice(HARD_NEGATIVES)
        return SyntheticExample(self._id("hard_neg"), text, label, "*", "global", "zh_en" if any(c.isascii() and c.isalpha() for c in text) else "zh", "natural_sentence", "hard_negative", "none", "synthetic_template", {})

    def benign_example(self) -> SyntheticExample:
        text = self.rng.choice(BENIGN)
        return SyntheticExample(self._id("benign"), text, "benign", "*", "global", "zh_en" if any(c.isascii() and c.isalpha() for c in text) else "zh", "natural_sentence", "neutral", "none", "synthetic_template", {})

    def mixed_egress_example(self) -> MixedEgressExample:
        priv = self.private_example()
        intent = self.rng.choice(["把会议纪要发给老板。", "Send the sprint notes to the team.", "请把项目更新同步到工作群。"])
        work = "会议纪要：\n- 周五部署新版本\n- Alice review API\n\n"
        return MixedEgressExample(self._id("egress"), intent, work + "附带备注：" + priv.text, ["work"], ["work", "financial_private"], ["financial_private"], "request_approval", "mixed_egress", "synthetic_mixed", priv.subtype, priv.text)

    def generate_private(self, n: int) -> list[dict]:
        return [asdict(self.private_example(PRIVATE_SUBTYPES[i % len(PRIVATE_SUBTYPES)])) for i in range(n)]
    def generate_hard_negatives(self, n: int) -> list[dict]:
        return [asdict(self.hard_negative_example()) for _ in range(n)]
    def generate_benign(self, n: int) -> list[dict]:
        return [asdict(self.benign_example()) for _ in range(n)]
    def generate_mixed(self, n: int) -> list[dict]:
        return [asdict(self.mixed_egress_example()) for _ in range(n)]

def re_has_en(text: str) -> bool:
    return any(ch.isascii() and ch.isalpha() for ch in text)
