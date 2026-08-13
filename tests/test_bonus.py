from src.pipeline import ask_question


def test_empty_question_returns_helpful_message():
    result = ask_question(None, None, "   ")

    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []


def test_question_is_trimmed_before_search():
    class FakeDoc:
        page_content = "Growth package costs $5,500/month"

    class FakeVectorStore:
        def similarity_search(self, question, k=3):
            assert question == "How much is the Growth package?"
            assert k == 3
            return [FakeDoc()]

    def fake_llm(prompt):
        return [{"generated_text": "$5,500/month"}]

    result = ask_question(
        FakeVectorStore(),
        fake_llm,
        "   How much is the Growth package?   ",
    )

    assert result["answer"] == "$5,500/month"
    assert result["sources"] == [
        "Growth package costs $5,500/month"
    ]


def test_retrieval_requests_three_chunks():
    class FakeDoc:
        def __init__(self, text):
            self.page_content = text

    class FakeVectorStore:
        def similarity_search(self, question, k=3):
            assert k == 3

            return [
                FakeDoc("Source one"),
                FakeDoc("Source two"),
                FakeDoc("Source three"),
            ]

    def fake_llm(prompt):
        return [{"generated_text": "Test answer"}]

    result = ask_question(
        FakeVectorStore(),
        fake_llm,
        "Test question",
    )

    assert len(result["sources"]) == 3


def test_retrieved_context_is_sent_to_llm():
    class FakeDoc:
        def __init__(self, text):
            self.page_content = text

    class FakeVectorStore:
        def similarity_search(self, question, k=3):
            return [
                FakeDoc("Growth package costs $5,500/month"),
                FakeDoc("Minimum commitment is 6 months"),
                FakeDoc("Setup fee is $1,000"),
            ]

    captured_prompt = {}

    def fake_llm(prompt):
        captured_prompt["prompt"] = prompt

        return [
            {
                "generated_text":
                    "The Growth package costs $5,500/month."
            }
        ]

    result = ask_question(
        FakeVectorStore(),
        fake_llm,
        "How much does the Growth package cost?",
    )

    assert "Growth package costs $5,500/month" in captured_prompt["prompt"]
    assert "Minimum commitment is 6 months" in captured_prompt["prompt"]
    assert "Setup fee is $1,000" in captured_prompt["prompt"]
    assert "How much does the Growth package cost?" in captured_prompt["prompt"]

    assert result["answer"] == (
        "The Growth package costs $5,500/month."
    )