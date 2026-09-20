from agent.request_classifier import RequestClassifier, RequestMode


def test_classifier_marks_clear_greeting_as_direct() -> None:
    result = RequestClassifier().classify("こんにちは")

    assert result.mode == RequestMode.DIRECT


def test_classifier_marks_operational_request_as_task() -> None:
    result = RequestClassifier().classify("このフォルダに何があるか調べてください")

    assert result.mode == RequestMode.TASK


def test_classifier_marks_web_request_as_task() -> None:
    result = RequestClassifier().classify("今日の天気を教えてください")

    assert result.mode == RequestMode.TASK


def test_classifier_does_not_infer_a_capability() -> None:
    result = RequestClassifier().classify("Python 3.14について調べて")

    assert result == type(result)(RequestMode.TASK)


def test_classifier_keeps_ambiguous_advice_as_direct_conversation() -> None:
    result = RequestClassifier().classify("どうすればよいですか")

    assert result.mode == RequestMode.DIRECT


def test_classifier_keeps_plain_topics_as_direct_conversation() -> None:
    result = RequestClassifier().classify("Python 3.14")

    assert result.mode == RequestMode.DIRECT
