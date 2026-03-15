"""
Based on https://github.com/Continual-Intelligence/SEAL. 
"""
import requests
import logging
import time
from typing import Any, Dict, List, Optional, Union, Generator
import re
import os
from openai import OpenAI
import io
import json
import concurrent.futures

# ---------------------------  CONFIG  ---------------------------------- #
# Qwen-base answering template
SQUAD_ANSWER_TEMPLATE_BASE = (
    "Let's answer a question directly and concisely.\n"
    "Question: {question}\n"
    "Answer:\n"
)

# Qwen-base answering template with chain of thought
SQUAD_ANSWER_TEMPLATE_BASE_COT = (
    "Let's think step by step and then answer the question directly and concisely. "
    "Let's first give reasoning under \"Reasoning:\" and then the final answer under \"Final answer:\".\n"
    "Question: {question}\n"
    "Reasoning:"
)

# Qwen-instruct answering (unused)
SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant to answer a question directly and concisely."
    "<|im_end|>\n"
    "<|im_start|>user\n{question}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


# Qwen grading (unused)
SQUAD_GRADE_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are a grading assistant. Your job is to determine whether a student's answer "
    "correctly answers the question based solely on the provided gold answer. Do not use any outside knowledge. "
    "The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.<|im_end|>\n"
    "<|im_start|>user\n{question}\n"
    "Gold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# OpenAI grading
SQUAD_GRADE_TEMPLATE = (
    "You are a grading assistant. Your job is to determine whether a student's answer correctly answers the question based solely on the provided gold answer. "
    "Do not use any outside knowledge. The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'."
)

TRAINING_SEQUENCE_TEMPLATE = "{title}\n{completion_text}"
TRAINING_SEQUENCE_TEMPLATE_NOTITLE = "{completion_text}"
# ----------------------------------------------------------------------- #

# vLLM API thin wrapper
API = requests.Session()
VLLM_API_URL: Optional[str] = None


def set_vllm_api_url(url: str):
    """Initialize the base URL for vLLM API calls."""
    global VLLM_API_URL
    VLLM_API_URL = url
    logging.info("vLLM API → %s", VLLM_API_URL)


def _api(endpoint: str, payload: Dict[str, Any], timeout: int = 300):
    assert VLLM_API_URL, "VLLM API URL not set"
    url = f"{VLLM_API_URL}/v1/{endpoint}"
    for attempt in range(300):
        try:
            logging.debug("POST %s try %d payload %s", endpoint, attempt + 1, payload)
            r = API.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                if r.headers.get("Content-Type", "").startswith("application/json"):
                    return r.json()
                return r.text or True
            r.raise_for_status()
        except Exception as e:
            logging.warning("API error %s - attempt %d/3", e, attempt + 1)
            time.sleep(2 * (attempt + 1))
    logging.error("API %s failed after retries", endpoint)
    return None


def load_adapter(path: str, name: str) -> bool:
    return _api("load_lora_adapter", {"lora_name": name, "lora_path": path}) is not None


def unload_adapter(name: str) -> bool:
    _api("unload_lora_adapter", {"lora_name": name}); return True


def generate(
    prompts: List[str], model: str, sampling: Dict[str, Any], stop_ids: List[int]
) -> Optional[List[Dict[str, Any]]]:
    payload = {"model": model, "prompt": prompts, **sampling, "stop_token_ids": stop_ids}
    res = _api("completions", payload, timeout=120*len(prompts))
    return res.get("choices") if isinstance(res, dict) else None


# -------------------  SQUAD HELPERS  ---------------------------------- #
def format_answer_prompts(q_batch: List[Dict[str, str]], instruct_model: bool, chain_of_thought: bool = False, eval_with_context: bool = False) -> List[str]:
    if chain_of_thought:
        SQUAD_ANSWER_TEMPLATE = SQUAD_ANSWER_TEMPLATE_BASE_COT
    else:
        SQUAD_ANSWER_TEMPLATE = SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT if instruct_model else SQUAD_ANSWER_TEMPLATE_BASE
    # eval_with_context
    if eval_with_context:
        def _ctx_prompt(q: Dict[str, str]) -> str:
            title = q.get("title", "").strip()
            context = q.get("context", "").strip()
            prefix = ""
            if title:
                prefix += f"Title: {title}\n"
            if context:
                prefix += f"Context:\n{context}\n\n"
            if instruct_model:
                #把prefix和question连在一起放入template
                return SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT.format(question=prefix + q["question"])
            else:
                return prefix + SQUAD_ANSWER_TEMPLATE.format(question=q["question"])
        return [_ctx_prompt(q) for q in q_batch]
    # eval_without_context
    return [SQUAD_ANSWER_TEMPLATE.format(question=q["question"]) for q in q_batch]


def format_grade_prompts(
    q_batch: List[Dict[str, str]], preds: List[str]
) -> List[str]:
    return [
        SQUAD_GRADE_TEMPLATE.format(
            question=q["question"],
            gold=q["answer"],
            pred=p.strip(),
        )
        for q, p in zip(q_batch, preds)
    ]

_yes_re = re.compile(r"\b(yes)\b", re.I)
_no_re  = re.compile(r"\b(no)\b",  re.I)

_gpt4: OpenAI | None = None

_final_ans_re = re.compile(
    r"(?:^|\n)\s*final\s*answer\s*[:\-]\s*(.*)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)

def extract_final_answer(text: str) -> str:
    """
    Return only the content after a 'Final answer:' (case-insensitive) marker.
    If no marker is present, return 'idk' (so it will be graded as incorrect).
    """
    if not text:
        return "idk"
    m = _final_ans_re.search(text.strip())
    return (m.group(1).strip() if m else "idk").strip()

def _client() -> OpenAI:
    """Return a singleton OpenAI client.
    Honors OPENAI_API_KEY. If OPENAI_BASE_URL or OPENAI_API_BASE is set, uses it as base_url.
    If the given base contains an endpoint path (e.g., '/v1/chat/completions' or '/v1/responses'),
    it will be normalized to the '/v1' root to support both Responses and Chat Completions APIs.
    """
    global _gpt4
    if _gpt4 is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if base_url:
            # normalize to '/v1' root if '/v1' exists in the path
            idx = base_url.find("/v1")
            if idx != -1:
                base_url = base_url[: idx + 3]
            logging.info("OpenAI base_url resolved to %s", base_url)
            _gpt4 = OpenAI(api_key=api_key, base_url=base_url)
        else:
            logging.info("OpenAI base_url using default (api.openai.com)")
            _gpt4 = OpenAI(api_key=api_key)
    return _gpt4

def grade_with_gpt4(prompts: List[str], max_workers: int = 10) -> List[bool]:
    """
    Take already-formatted grading prompts, send each to GPT-4.1,
    and return the yes/no verdicts as booleans.
    """
    client: OpenAI = _client()

    def process_single(p: str) -> bool:
        # Try Responses API; on 404/Not Found, fast-fallback to Chat Completions
        used_fallback = False
        verdict = False
        for attempt in range(300):
            try:
                r = client.responses.create(model="gpt-4.1", input=p)
                verdict = parse_yes_no(r.output_text)
                return verdict
            except Exception as e:
                msg = str(e)
                # Fast fallback if provider doesn't support /responses
                if "404" in msg or "Not Found" in msg:
                    used_fallback = True
                    break
                time.sleep(1.5 * (attempt + 1))
        else:
            used_fallback = True

        if used_fallback:
            try:
                chat = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": p}],
                )
                verdict = parse_yes_no(chat.choices[0].message.content)
            except Exception:
                verdict = False  # couldn't grade this prompt
        return verdict

    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            verdicts = list(executor.map(process_single, prompts))
    else:
        verdicts = [process_single(p) for p in prompts]

    return verdicts


def parse_yes_no(text: str) -> bool:
    """Return True for yes, False for no or ambiguous responses."""
    if _yes_re.search(text) and not _no_re.search(text):
        return True
    return False

def _split_segments(text: str) -> List[str]:
    return [seg.strip() for seg in text.split("---") if seg.strip()]

MAX_TRAIN_SEQS_PER_COMPLETION=30

def build_train_sequences(
    completion_raw: Union[str, List[str]],
    context: str,
    title: str,
    *,
    add_context: bool = True,
) -> List[str]:
    inputs = completion_raw if isinstance(completion_raw, list) else [completion_raw]
    all_segs = []

    for completion_raw in inputs:
        segs = [completion_raw]
        all_segs.extend([s for s in segs if s.strip()])
    if title.strip() == "":
        return [TRAINING_SEQUENCE_TEMPLATE_NOTITLE.format(completion_text=s) for s in all_segs]
    seqs = [TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=s) for s in all_segs]
    if add_context:
        seqs.append(TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=context.strip()))
    return seqs

# -------------------  PROXY GRADING  ---------------------------------- #
# This is a proxy used to evaluate the quality of synthetic data generated by the model.
# It is meant as an alternative to the full meta-learning inner loop, and can be applied
# in TTT_server.py by setting --reward_mode to "proxy" or "both". This rubric is not heavily tuned
# and likely could be improved further.
PROXY_SCORE_TEMPLATE = (
    "You are to evaluate a list of implications that are derived (directly or indirectly) from the provided document. Be critical and fair.\n"
    "For each of the four criteria below, rate on a 1-5 integer scale (higher is better). We want to reward implication lists that are longer (relative to the size of the original document), more diverse (statements should be less repetitive), higher in quality, and more correct (supported by the document).\n"
    "For each criterion, think briefly and then output exactly this format:\n"
    "Length: <1-5> - <one sentence rationale>\n"
    "Diversity: <1-5> - <one sentence rationale>\n"
    "Quality: <1-5> - <one sentence rationale>\n"
    "Correctness: <1-5> - <one sentence rationale>\n"
    "After that, output the sum of the 4 scores as: Final Score: <integer>\n\n"
    "Title: {title}\n"
    "Document:\n{context}\n\n"
    "Implications:\n{completion}\n"
)

_proxy_len_re = re.compile(r"^\s*Length:\s*(\d+)", re.I | re.M)
_proxy_div_re = re.compile(r"^\s*Diversity:\s*(\d+)", re.I | re.M)
_proxy_quality_re = re.compile(r"^\s*Quality:\s*(\d+)", re.I | re.M)
_proxy_correct_re = re.compile(r"^\s*Correct(ness)?:\s*(\d+)", re.I | re.M)
_proxy_final_re = re.compile(r"^\s*Final\s*Score\s*:\s*(\d+)", re.I | re.M)

def build_proxy_prompt(title: str, context: str, completion: str) -> str:
    return PROXY_SCORE_TEMPLATE.format(title=title, context=context, completion=completion)

def _to_int_clamped(x: str, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return 1

def parse_proxy_scores(text: str) -> Dict[str, int]:
    """Extract integer sub-scores and final score from GPT output."""
    length = _to_int_clamped((_proxy_len_re.search(text) or [None, "1"])[1])
    diversity = _to_int_clamped((_proxy_div_re.search(text) or [None, "1"])[1])
    quality = _to_int_clamped((_proxy_quality_re.search(text) or [None, "1"])[1])
    correctness = _to_int_clamped(((_proxy_correct_re.search(text) or [None, None, "1"])[2]))
    final_match = _proxy_final_re.search(text)
    if final_match:
        final_score = int(final_match.group(1))
    else:
        final_score = length + diversity + quality + correctness
    # clamp final into [4, 20]
    final_score = max(4, min(20, final_score))
    return {
        "length": length,
        "diversity": diversity,
        "quality": quality,
        "correctness": correctness,
        "final": final_score,
    }

def score_proxy_with_gpt4(title: str, context: str, completion: str) -> Dict[str, int]:
    """
    Ask GPT-4.1 to score the completion on four criteria (1-5) and a Final Score.
    Returns a dict with keys: length, diversity, quality, correctness, final.
    """
    client: OpenAI = _client()
    prompt = build_proxy_prompt(title, context, completion)
    text_out = ""
    for attempt in range(3):
        try:
            r = client.responses.create(model="gpt-4.1", input=prompt)
            text_out = r.output_text or ""
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if not text_out:
        return {"length": 1, "diversity": 1, "quality": 1, "correctness": 1, "final": 4}
    return parse_proxy_scores(text_out)


def set_openai_key():
    with open('data/dataset/openai.key', 'r') as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

def set_cohere_private_key():
    with open('data/dataset/cohere.key', 'r') as f:
        os.environ["COHERE_API_KEY"] = f.read().strip()

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f: str, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload_list(f, mode="r"):
    """Load multiple JSON objects from a file."""
    objects = []
    with open(f, mode) as file:
        for line in file:
            obj = json.loads(line)
            objects.append(obj)
    return objects

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def round_dict_values(input_dict: Dict, decimal_places: int = 4):
    def round_helper(value):
        if isinstance(value, dict):
            return {k: round_helper(v) for k, v in value.items()}
        elif isinstance(value, float):
            return round(value, decimal_places)
        else:
            return value
    return round_helper(input_dict)

def flatten_dict(nested_dict, parent_key='', sep='_'):
    flat_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, float):
            flat_dict[new_key] = f"{value:.4f}"
        else:
            flat_dict[new_key] = str(value)
    return flat_dict

def rm_file(file_path):
    """Delete Files

    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")

def save_list_to_json(lst, filename):
    """Save Files
    """
    with open(filename, 'w') as file:
        json.dump(lst, file)
