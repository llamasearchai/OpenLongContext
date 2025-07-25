from typing import Tuple

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load a real long-context model (Longformer QA)
MODEL_NAME = "allenai/longformer-base-4096"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def answer_question(document: str, question: str) -> Tuple[str, str]:
    """
    Given a document and a question, return (answer, context_span).
    Uses a real long-context QA model (Longformer).
    """
    result = qa_pipeline({"context": document, "question": question})
    answer = result["answer"]
    context_span = document[result["start"]:result["end"]]
    return answer, context_span
