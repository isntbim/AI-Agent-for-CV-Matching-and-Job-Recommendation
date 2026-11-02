from scripts.cv_external_pool import CopyIndex, classify


def test_classify_prefers_specific_domains_and_drops_others():
    assert classify("Summary  Front-end developer with React") == "frontend_web_developer"
    assert classify("Career Overview Data Analyst and software user") == "data_scientist_analyst"
    assert classify("CHEF   Summary  Line cook") is None
    assert classify("x " * 200 + "accountant") is None  # only the opening title/summary counts


def test_copy_index_finds_reformatted_copies_only():
    cv = " ".join(f"word{i}" for i in range(300))
    index = CopyIndex()
    index.add("tp_hr_001", cv)
    assert index.match(cv.upper().replace(" ", "\n ")) == "tp_hr_001"
    assert index.match(" ".join(f"other{i}" for i in range(300))) is None
