from rag_studio.query_router import QueryRouter


def test_router_sends_greeting_to_direct_answer() -> None:
    decision = QueryRouter().route("hello what can you do?")

    assert decision.route == "direct"
    assert decision.multi_query is False


def test_router_uses_precise_retrieval_for_exact_fact_question() -> None:
    decision = QueryRouter().route("How many specialized agents were used?")

    assert decision.route == "retrieve"
    assert decision.retriever == "hybrid"
    assert decision.parent_context is True
    assert decision.multi_query is False
    assert decision.hyde is False


def test_router_uses_broader_retrieval_for_job_fit_question() -> None:
    decision = QueryRouter().route("Which resume is the best fit for this job?")

    assert decision.route == "retrieve"
    assert decision.multi_query is True
    assert decision.hyde is True
