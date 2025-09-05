MCQ_TEMPLATE = """You are a careful reasoner. Answer with the letter only.

Question: {question}

Options:
A) {A}
B) {B}
C) {C}
D) {D}

Answer:"""

DEMO_TEMPLATE = """Example:
Question: {question}
Options:
A) {A}
B) {B}
C) {C}
D) {D}
Correct answer: {answer}
"""

def build_demo(example: dict) -> str:
    c = example["choices"]
    return DEMO_TEMPLATE.format(
        question=example["question"], A=c["A"], B=c["B"], C=c["C"], D=c["D"], answer=example["answer"]
    )

def build_prompt(query: dict, demos: list[dict]) -> str:
    demo_block = "".join(build_demo(e) for e in demos)
    c = query["choices"]
    mcq = MCQ_TEMPLATE.format(question=query["question"], A=c["A"], B=c["B"], C=c["C"], D=c["D"])
    return demo_block + "\n" + mcq
