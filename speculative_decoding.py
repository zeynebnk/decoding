import transformers as tr
import torch.nn.functional as F
import torch


def prep_prompt():
  
  if args.prompt_path is not None:
    with open(args.prompt_path, 'r') as f:
      user_message = f.read()
  else:
    user_message = "Once upon a time"

  return user_message

def load_models(mini_path, tgt_path):

  Mp = tr.AutoModelForCausalLM.from_pretrained(tgt_path).to(device).eval()
  Mq = tr.AutoModelForCausalLM.from_pretrained(mini_path).to(device).eval()
  Tp = tr.AutoTokenizer.from_pretrained(tgt_path)
  Tq = tr.AutoTokenizer.from_pretrained(mini_path)

  return Mp, Tp, Mq, Tq


def speculative_decoding_step(Mp, Tp, Mq, Tq, prefix, gamma):
  # approximation model
  q_is = []
  x_is = []
  mini_inputs = Tq(prefix, return_tensors='pt').to(device)['input_ids']
  for i in range(gamma):

    q_i = F.softmax(Mq.forward(mini_inputs).logits[:, -1, :], dim=-1) # probs distribution for next word
    q_is.append(q_i)
    x_i = torch.multinomial(q_i, num_samples=1)[0].item()
    x_is.append(x_i)
    mini_inputs = torch.cat([mini_inputs, torch.tensor([[x_i]]).to(device)], dim=1)

  print("Draft: " + Tq.decode(mini_inputs[0]))


  # validating (run in parallel, simplified to iterative)
  p_is = []
  tgt_inputs = Tp(prefix, return_tensors='pt').to(device)['input_ids']

  p_is = []
  for i in range(gamma + 1):
    if i == 0:
      inputs = tgt_inputs
    else:
      inputs = torch.cat([tgt_inputs, torch.tensor([x_is[:i]]).to(device)], dim=1)
    p_i = F.softmax(Mp.forward(inputs).logits[:, -1, :], dim=-1) # probs distribution for next word
    p_is.append(p_i)


  # keep n
  r_is = torch.rand(gamma)
  n = min([i for i in range(gamma) if r_is[i] > p_is[i][0][x_is[i]] / q_is[i][0][x_is[i]]] + [gamma])

  p_prime = p_is[n]

  # adjust
  if n < gamma:
    adjusted_probs = torch.maximum(torch.zeros_like(p_prime), p_prime - q_is[n])
    if adjusted_probs.sum() > 0:
        p_prime = adjusted_probs / adjusted_probs.sum()

  t = torch.multinomial(p_prime, num_samples=1)[0].to(device)

  if n > 0:
    output = torch.cat([tgt_inputs[0], torch.tensor(x_is[:n]).to(device) , t], dim=-1)
  else:
    output = torch.cat([tgt_inputs[0], t], dim=-1)

  return output


def speculative_generation(Mp, Tp, Mq, Tq, prompt, max_tokens) -> str:

  gen_len = 0

  prefix = prompt
  while gen_len < max_tokens:

    output = speculative_decoding_step(Mp, Tp, Mq, Tq, prefix, args.gamma)
    gen_len = len(output)
    prefix = Tp.decode(output)

  return prefix


def write_outputs(out):
  if args.output_path is not None:
    with open(args.output_path, 'w') as f:
      f.write(out)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
  Mp, Tp, Mq, Tq = load_models(args.mini_model_path, args.tgt_model_path)
  prompt = prep_prompt()
  out = speculative_generation(Mp, Tp, Mq, Tq, prompt, args.max_gen_len)
  write_outputs(out)
  
  print(out)
  
  
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative decoding")
    parser.add_argument("--mini_model_path", type=str, default='Qwen/Qwen2.5-Coder-0.5B-Instruct',help="HuggingFace model path to approximation model")
    parser.add_argument("--tgt_model_path", type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct', help="HuggingFace model path to target model")
    parser.add_argument("--gamma", type=int, default=5, help="gamma hyperparam for speculative decoding")
    parser.add_argument("--max_gen_len", type=int, default=150, help="Maximum generation length")
    
    parser.add_argument("--prompt_path", required=False, help="file path to read input prompt")
    parser.add_argument("--output_path", required=False, help="file path to write outputs")
    
    args = parser.parse_args()
    main()

