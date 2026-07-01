from sensitive_egress_poc.io_utils import read_json, read_jsonl, write_json, write_jsonl


def test_jsonl_and_json_utf8(tmp_path):
    rows=[{"text":"我 DBS 账户里还剩 SGD 4,200。"}]
    p=tmp_path/"x.jsonl"; write_jsonl(p, rows)
    assert read_jsonl(p)==rows
    raw=p.read_text(encoding="utf-8")
    assert "账户" in raw
    jp=tmp_path/"m.json"; write_json(jp,{"说明":"中文"})
    assert read_json(jp)["说明"]=="中文"
