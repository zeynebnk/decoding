import transformers as tr
import torch.nn.functional as F
import torch


def prep_prompt(tokenizer):
  
  if args.prompt_path is not None:
    with open(args.prompt_path, 'r') as f:
      user_message = f.read()
  
  else:
    user_message = "Once upon a time"

  prompt = tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': user_message}
    ],
    add_generation_prompt=True,
    tokenize=False)
  
  return prompt


def load_models(amateur_path, expert_path):

  ama_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path).to(device).eval()
  expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path).to(device).eval()
  tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

  return ama_model, expert_model, tokenizer

def contrastive_decode_step(input_ids, expert, amateur, temp, alpha):

  with torch.no_grad():
    exp_logits = expert(input_ids).logits[0, -1, :]
    ama_logits = amateur(input_ids).logits[0, -1, :] / temp

  exp_prob = F.softmax(exp_logits, dim=-1)
  ama_prob = F.softmax(ama_logits, dim=-1)
  exp_logprob = F.log_softmax(exp_logits, dim=-1)
  ama_logprob = F.log_softmax(ama_logits, dim=-1)

  L_cd = exp_logprob - ama_logprob
  V_head = exp_prob >= alpha * exp_prob.max()
  cd_scores = torch.where(V_head, L_cd, torch.tensor(-float('inf')).to(device))

  x_i = torch.argmax(cd_scores).item()
  new_input = torch.cat([input_ids, torch.tensor([[x_i]]).to(device)], dim=-1)

  return new_input, x_i

def contrastive_generation(amateur, expert, tokenizer, prompt, max_tokens) -> str:

  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
  output_ids = []
  x_i = None
  
  while len(output_ids) < max_tokens and x_i != tokenizer.eos_token_id:

    input_ids, x_i = contrastive_decode_step(input_ids, expert, amateur, args.temp, args.alpha)
    output_ids.append(x_i)

  return tokenizer.decode(output_ids)

def write_outputs(out):
  if args.output_path is not None:
    with open(args.output_path, 'w') as f:
      f.write(out)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
  
  ama_model, expert_model, tokenizer = load_models(args.amateur_model_path, args.expert_model_path)
  prompt = prep_prompt(tokenizer)
  out = contrastive_generation(ama_model, expert_model, tokenizer, prompt, args.max_gen_len)
  write_outputs(out)
  
  print(out)
  
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive decoding")
    parser.add_argument("--amateur_model_path", type=str, default='Qwen/Qwen2.5-Coder-0.5B-Instruct',help="HuggingFace model path to amateur model")
    parser.add_argument("--expert_model_path", type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct', help="HuggingFace model path to expert model")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha hyperparam for contrastive decoding")
    parser.add_argument("--temp", type=float, default=1.0, help="temp hyperparam for contrastive decoding")
    parser.add_argument("--max_gen_len", type=int, default=150, help="Maximum generation length")
    
    parser.add_argument("--prompt_path", required=False, help="file path to read input prompt")
    parser.add_argument("--output_path", required=False, help="file path to write outputs")
    
    args = parser.parse_args()
    main()

