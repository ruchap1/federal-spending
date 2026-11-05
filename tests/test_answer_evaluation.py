from scripts.evaluate_answers import citation_checks, select_questions


def test_citation_labels_do_not_prove_support():
    checks = citation_checks("A claim [1] and another [4] [1] [0].", 3)
    assert checks["citation_labels"] == [0, 1, 4]
    assert checks["invalid_labels"] == [0, 4]
    assert "supported" not in checks


def test_sampling_preserves_categories():
    questions = [{"category": value} for value in ["a", "a", "b", "b"]]
    assert select_questions(questions, 1) == [{"category": "a"}, {"category": "b"}]
    assert len(select_questions(questions, 0)) == 4
