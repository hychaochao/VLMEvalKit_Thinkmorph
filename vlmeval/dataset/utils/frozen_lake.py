from ...smp import *
import re
import os

def extract_path_from_response(response_text):
    
    response_text = response_text.replace('\x08', '\\b')

    matches = re.findall(r'\\{1,2}boxed\{([RLUD](?:\s*,\s*[RLUD])*)\}', response_text)

    if matches:
        extracted_path = matches[-1].strip()
        cleaned_path = re.sub(r'\s+', '', extracted_path).strip(',')
        return cleaned_path
    print(f"No boxed path found in response.")
    return None

def str_to_maze_rows(maze_str):
    rows = re.findall(r"'([^']+)'", maze_str)
    return rows

def validate_path(maze_map, path_sequence, maze_size):
    if type(maze_map) is str:
        maze_map = str_to_maze_rows(maze_map)
    if not path_sequence:
        return False
    start_pos, goal_pos = None, None
    grid = maze_map.tolist() if hasattr(maze_map, 'tolist') else maze_map
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == 'S':
                start_pos = (r, c)
            elif char == 'G':
                goal_pos = (r, c)
    if not start_pos or not goal_pos:
        return False
        
    current_pos = list(start_pos)
    moves = path_sequence.upper().replace(',', '')
    
    for move in moves:
        r, c = current_pos
        
        # MODIFIED: Check boundaries before making a move. If the move is invalid, fail immediately.
        if move == 'U':
            if r == 0: return False
            current_pos[0] -= 1
        elif move == 'D':
            if r == maze_size - 1: return False
            current_pos[0] += 1
        elif move == 'L':
            if c == 0: return False
            current_pos[1] -= 1
        elif move == 'R':
            if c == maze_size - 1: return False
            current_pos[1] += 1
        else:
            return False # Invalid move character

        # Check if the new position is a hole
        if grid[current_pos[0]][current_pos[1]] == 'H':
            return False
            
    # Check if the final position is the goal
    return tuple(current_pos) == goal_pos

def build_gpt_extract_prompt(line):
    question = line.get('question', '')
    prediction = str(line.get('prediction', ''))
    prompt = f"""
        You are a path answer extraction assistant. You are given a FrozenLake-style question that expects a sequence of moves and a model's response.

        Question:
        {question}

        Model Response:
        {prediction.strip()}

        Task:
        Extract ONLY the move sequence exactly as explicitly provided by the model. Do NOT infer, guess, recompute, fix, or complete any sequence. If you cannot find a single clear valid sequence, return an empty answer.

        Valid moves: U,D,L,R (uppercase). Output must be a comma-separated list like D,D,R,R (no spaces). 

        Extraction rules:
        1. Primary target: a boxed form \\boxed{{...}} where ... is one or more moves separated by commas (e.g. \\boxed{{L,D,D,L}}). Extract only the inside moves.
        2. If multiple boxed candidates appear:
        - Keep only syntactically valid ones (only U,D,L,R separated by commas, optional spaces).
        - If only one valid remains, use it.
        - If several differ and no clear final preference (no later confirmation, no negation of earlier ones), return empty.
        - If a later candidate is explicitly negated (e.g. "that path fails/invalid"), discard it.
        3. If no boxed sequence: look for a final explicit declaration lines beginning with/containing keywords (Answer, Solution, Path, Moves, Sequence) followed immediately by a comma-separated list of ONLY U,D,L,R. Use the last such unnegated one.
        4. Truncated or partial fragments (e.g. \\boxed{{L,D  without closing brace, dangling commas, incomplete tokens) are invalid. Do NOT merge fragments.
        5. Ignore analytical text describing attempts or coordinates; only explicit final sequence counts.
        6. If the model states impossibility or gives only an empty box (\\boxed{{}}) or nothing valid, output empty.
        7. Output format MUST be exactly:
        <answer>MOVES</answer>
        If not extractable:
        <answer></answer>
        No extra text, no explanations, no additional tags or whitespace.

        Now output ONLY the required tag:
        """
    return prompt

def parse_gpt_extracted_answer(gpt_output):
    if not gpt_output:
        return None
    
    match = re.search(r'<answer>(.*?)</answer>', gpt_output.strip(), re.DOTALL)
    if not match:
        return None
    
    content = match.group(1).strip()
    if not content:
        return None

    if re.fullmatch(r'[UDLR](?:,[UDLR])*', content):
        return content
    return None


def Frozen_lake_auxeval(model, line):
    pred_text = line['prediction']
    maze_map = line['maze_map']
    maze_size = line['maze_size']

    # extract path
    path = extract_path_from_response(pred_text)

    # if path is None:
    #     prompt = build_gpt_extract_prompt(line)
    #     path = parse_gpt_extracted_answer(model.generate(prompt, temperature=0))

    if path is None:
        return dict(extracted_path = None, log = "extract_path_failed", hit = 0)
    
    is_correct = validate_path(maze_map=maze_map, path_sequence=path, maze_size=maze_size)

    if is_correct:
        log = "eval_succeed"
        hit = 1
    else:
        log = "eval_succeed"
        hit = 0

    return dict(extracted_path = path, log=log, hit=hit)


def Frozen_lake_acc(result_file):
    data = load(result_file)
    score = 0
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        hit = item['hit']
        score += hit

    res = {'Overall': [score / lt]}
    res = pd.DataFrame(res)
    return res





