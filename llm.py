from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch, os

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        settings.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.MODEL_NAME,
        trust_remote_code=True,)
    return model, tokenizer

class StopOnNewline(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[-1] >= 2 and input_ids[0, -2:].tolist() == [tokenizer.eos_token_id]*2:
            return True
        return False

def get_gen_kwargs(tokenizer):
    return {
        "max_new_tokens": settings.MAX_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 0.8,
        "stopping_criteria": StoppingCriteriaList([StopOnNewline()]),}

def classify_page_blocks(block_texts, model, tokenizer, gen_kwargs):
    truncated = [t.replace("\n"," ")[:30] for t in block_texts]
    prompt = (
        "You are a PDF block classifier. This is a page from a book.  "
        "Given the truncated text of each block on a single page, "
        "return labels in order, one per block, chosen from: header, body, footer, quote, exclude.  "
        "Output only the comma-separated list.\n\n"
        f"Blocks ({len(truncated)}):\n"
        + "\n".join(f"{i+1}. “{txt}”" for i, txt in enumerate(truncated))
        + "\n\nLabels:")
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in enc: del enc["token_type_ids"]
    with torch.no_grad():
        out_ids = model.generate(**enc, **gen_kwargs)
    out = tokenizer.decode(out_ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    labels = out.splitlines()[0].split(",")
    return [lbl.strip() for lbl in labels]

def main_evaluate():
    model, tokenizer = load_model_and_tokenizer()
    gen_kwargs = get_gen_kwargs(tokenizer)
    evaluator = PDFEvaluator(...)
    while evaluator.current_page < evaluator.total_pages:
        blocks = evaluator.process_page(evaluator.current_page)
        texts = [b["text"] for b in blocks]
        preds = classify_page_blocks(texts, model, tokenizer, gen_kwargs)
        mapped = [settings.LABEL_MAP[p] for p in preds]
        evaluator.current_page += 1

if __name__ == "__main__":
    class settings: MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"; MAX_TOKENS=500; LABEL_MAP={"header": 0, "body": 1, "footer": 2, "quote": 3, "exclude": 4}
    example_blocks = ["Introduction:", "The Case for Political History", "Byung-Kook Kim", "F\new periods have changed South Korean history more than the\nPark era that began in May 1961 with a military"]
    model, tokenizer = load_model_and_tokenizer()
    gen_kwargs = get_gen_kwargs(tokenizer)
    labels = classify_page_blocks(example_blocks, model, tokenizer, gen_kwargs)
    print(f"Input blocks: {[b[:30]+'...' for b in example_blocks]}")
    print(f"Predicted labels: {labels}")

