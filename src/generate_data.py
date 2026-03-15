from pathlib import Path
import argparse, json, random, time, datetime
from typing import Any, Dict, List
import os
import concurrent.futures
import threading
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from openai import OpenAI
from src.tasks.quality import QuALITY
from src.utils_tools.prompt_utils import MAKE_DATA_TEMPLATE_INSTRUCT, MAKE_DATA_TEMPLATE_OSS, MAKE_DATA_TEMPLATE_API, MAKE_DATA_TEMPLATES_BASE

def make_prompt(title: str, context: str, instruct_model: bool, prompt_key: str, model: str, use_api: bool = False) -> str:
    if use_api:
        MAKE_DATA_TEMPLATE = MAKE_DATA_TEMPLATE_API[prompt_key]
    elif "oss" in model.lower():
        MAKE_DATA_TEMPLATE = MAKE_DATA_TEMPLATE_OSS[prompt_key]
    else:
        MAKE_DATA_TEMPLATE = MAKE_DATA_TEMPLATE_INSTRUCT[prompt_key] if instruct_model else MAKE_DATA_TEMPLATES_BASE[prompt_key]
    return MAKE_DATA_TEMPLATE.format(
            title=title,
            context=context,
        )
    
    
def generate_bulk(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    tokenizer = None,
    model=None,
) -> List[str]:
    outputs = llm.generate(prompts, sampling_params)
    out = []
    if "oss" in model.lower():
        FINAL_MARKER = "<|start|>assistant<|channel|>final<|message|>"
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            idx = decoded.find(FINAL_MARKER)
            if idx == -1:
                content = decoded.strip()
            else:
                content = decoded[idx+len(FINAL_MARKER):].strip()
            out.append(content)
        return out
    else:
        out = [output.outputs[0].text.strip() for output in outputs]
        return out
    
def generate_bulk_api(
    client: OpenAI,
    model_name: str,
    prompts: List[str],
    temperature: float,
    workers: int = 1
) -> List[str]:
    lock = threading.Lock()

    def process_prompt(prompt: str) -> str:
        max_retries = 100
        system_content = "You are a helpful assistant."
        user_content = prompt
        
        if "<|SEP|>" in prompt:
            parts = prompt.split("<|SEP|>", 1)
            system_content = parts[0].strip()
            user_content = parts[1].strip()

        content = ""
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                )
                content = resp.choices[0].message.content       
                break
            except Exception as e:
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Max retries reached. Returning empty string.")
                    content = ""
        return content

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            out = list(executor.map(process_prompt, prompts))
    else:
        out = [process_prompt(prompt) for prompt in prompts]
            
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", choices=["quality", "mhrag", "squad"], required=True, help="Which dataset loader to use")
    p.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using instruction model")
    p.add_argument("--dataset_in", help="Path to the input dataset (required if dataset_name is mhrag)")
    p.add_argument("--dataset_out", required=True, help="Path to the output dataset")
    p.add_argument("--n", type=int, default=-1, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index for processing")
    p.add_argument('--k', type=int, default=5, help='Number of samples generated per article')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='top-p')
    p.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    p.add_argument('--min_p', type=float, default=0.0, help='min-p')
    p.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate")
    p.add_argument("--use_api", action="store_true", help="Use OpenAI API instead of vLLM")
    p.add_argument("--api_workers", type=int, default=16, help="Number of concurrent API workers (threads)")
    p.add_argument("--prompt_key", default="implications", choices=list(set(MAKE_DATA_TEMPLATES_BASE.keys())), help="Which prompt to use")
    args = p.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    if args.use_api:
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = None
    client = None

    if args.use_api:
        print(f"Initializing OpenAI API client for model {args.model}...")
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
        if not OPENAI_API_KEY:
            if not os.environ.get("OPENAI_API_KEY"):
                 print("Warning: OPENAI_API_KEY not found in environment.")
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
        )
    else:
        print(f"Loading vLLM model {args.model}...")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_num_batched_tokens=65536,
        )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
    )

    raw: List[Dict[str, Any]] = []
    if args.dataset_name == "quality":
        print("Loading QuALITY dataset (all)...")
        if args.dataset_in:
            task = QuALITY('all', base_path=args.dataset_in)
        else:
            task = QuALITY('all')
        for doc in task.documents:
            raw.append({
                "title": doc.title,
                "context": doc.text,
                "author": doc.author,
                "year": doc.year,
                "topic": doc.topic,
                "questions": [q.asdict() for q in doc.questions]
            })
    elif args.dataset_name == "mhrag":
        if not args.dataset_in:
            p.error("--dataset_in is required for mhrag dataset")
        
        print(f"Loading MHRAG dataset from {args.dataset_in}...")
        with open(args.dataset_in, "r", encoding="utf-8") as f:
            source_data = json.load(f)
        for doc in source_data:
            raw.append({
                "title": doc.get("title", ""),
                "context": doc.get("body", ""),
                "author": doc.get("author"),
                "published_at": doc.get("published_at"),
                "category": doc.get("category"),
                "url": doc.get("url"),
                "source": doc.get("source"),
            })
    elif args.dataset_name == "squad":
        if not args.dataset_in:
            p.error("--dataset_in is required for squad dataset")
        
        print(f"Loading SQuAD dataset from {args.dataset_in}...")
        with open(args.dataset_in, "r", encoding="utf-8") as f:
            source_data = json.load(f)
        for doc in source_data:
            raw.append({
                "title": doc.get("title", ""),
                "context": doc.get("context", ""),
            })
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    random.seed(42)
    random.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start:]
    print(f"Processing {len(subset)} articles (from index {args.start})...")
    
    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_data: List[Dict[str, Any]] = []
    processed_count = 0
    if out_path.exists():
        try:
            print(f"Found existing output at {out_path}, attempting to resume...")
            with open(out_path, "r", encoding="utf-8") as f:
                out_data = json.load(f)
            processed_count = len(out_data)
            print(f"Resuming from {processed_count} processed items.")
        except Exception as e:
            print(f"Could not load existing file: {e}. Starting from scratch.")
            out_data = []

    if processed_count < len(subset):
        subset_to_process = subset[processed_count:]
        print(f"Processing remaining {len(subset_to_process)} articles...")
        
        save_freq = 5
        for i, item in enumerate(subset_to_process):
            current_prompts = [] 
            prompt = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key,model=args.model, use_api=args.use_api)
            current_prompts.extend([prompt] * args.k)

            if args.use_api:
                item_completions = generate_bulk_api(client, args.model, current_prompts, args.temperature, workers=args.api_workers)
            else:
                item_completions = generate_bulk(llm, current_prompts, sampling_params, tokenizer=tokenizer,model=args.model)

            item_idx = 0 
            doc_completions = [[] for _ in range(args.k)]
            doc_prompts = [] 
            doc_prompts.append(make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key,model=args.model, use_api=args.use_api))
            
            chunk_completions = item_completions[item_idx : item_idx + args.k]
            item_idx += args.k
            
            for k in range(args.k):
                doc_completions[k].append(chunk_completions[k])

            new_item = dict(item)
            new_item["prompts"] = doc_prompts
            new_item["completions"] = doc_completions
                
            out_data.append(new_item)

            if (i + 1) % save_freq == 0 or (i + 1) == len(subset_to_process):
                 print(f"Saving progress: {len(out_data)}/{len(subset)} items...")
                 json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    else:
        print("All items already processed. Nothing to do.")

    print(f"Finished. Saved → {out_path}  ({len(out_data)} records)")

    meta = {
        "dataset_name": args.dataset_name,
        "model": args.model,
        "dataset_in": args.dataset_in,
        "dataset_out": args.dataset_out,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_tokens": args.max_tokens,
        "api_workers": args.api_workers,
        "n": len(subset),
        "k": args.k,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "start_index": args.start,
        "instruct_model": args.instruct_model,
        "prompt_key": args.prompt_key,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    main()