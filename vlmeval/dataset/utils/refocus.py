from ...smp import *
from typing import Optional

def build_extract_prompt(question, predicted_text):
    prompt = f"""
    Role: You are an “Answer Extraction Assistant.” You are given a question and a model's response. The response contains the final answer to the question.
    
    Task: Extract only the final answer from the response and output it. Do not include any extra words, punctuation, or units. If the final answer does not appear in the response, output: None.

    Rules:
        1. Output only the answer itself—no explanations, labels, or extra text.
        2. If the answer is numeric, remove units and extra symbols (e.g., %, currency); keep the minus sign and decimal point.

    Examples:
    [example_1]
    Question:
    What is the difference in value between mutton and corn?
    Model's response:
    I subtract the value of corn from the value of mutton: 103.7 - 103.13 = 0.57. Therefore, the difference in value between mutton and corn is 0.57.
    Your output:
    0.57

    [example_2]
    Question:
    Is the average of all bars in 55 to 64 age group greater than average of 25 to 64 age group?
    Model's response:
    No
    Your output:
    No
    
    [example_3]
    Question:
    How much does the value of Approve decrease from Jul 2015 to Sep 2015?
    Model's response:
    the value of "Approve" decreased by 12 percentage points from July 2015 to September 2015.
    Your output:
    12

    Question:\n{question}\n
    Model's response:\n {predicted_text.strip()}\n
    Your output:\n
    """
    
    return prompt

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def process_line(line, method='vqa_score'):
    ret = {}
    if istype(line['answer'], list):
        answers = eval(line['answer'])
    else:
        answers = [line['answer']]
    if method == 'relaxed_accuracy':
        ret['gt'] = answers
        ret['pred'] = line['extracted_ans'].strip()
        ret['match'] = [relaxed_correctness(ret['pred'], x) for x in ret['gt']]

    return ret

def Refocus_extract_answer(line, model):
    ret = {}
    predicted_text = line['prediction'].replace("Round_0:", " ").replace("Round_1:", " ")
    question = line['question']

    if "<answer>" in predicted_text and "</answer>" in predicted_text:
        extracted_ans = predicted_text.split("<answer>")[1].split("</answer>")[0]
    else:
        extract_prompt = build_extract_prompt(question, predicted_text)
        extracted_ans = model.generate(extract_prompt, temperature = 0)
    
    ret['extracted_ans'] = extracted_ans

    return ret