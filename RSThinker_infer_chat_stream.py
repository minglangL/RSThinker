"""
A command-line interface for chatting with RSThinker.
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Glm4vForConditionalGeneration, TextIteratorStreamer
from PIL import Image
from threading import Thread


def build_content(image_path, text):
    content = []
    if image_path:
        content.append({"type": "image", "url": image_path})
    content.append({"type": "text", "text": text})
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="minglanga/RSThinker"
    )
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--top_p", type=float, default=0.00001)
    parser.add_argument("--top_k", type=int, default=1)

    args = parser.parse_args()

    processor_infer = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    model_infer = Glm4vForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    def infer_single_image(image_path, question):
        messages = []
        content = build_content(image_path, question)
        messages.append({"role": "user", "content": content})
        inputs = processor_infer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model_infer.device)
        
        model_infer.eval()
        with torch.no_grad():
          streamer = TextIteratorStreamer(processor_infer)
          generation_kwargs = dict(inputs,
                                   streamer=streamer,
                                   max_new_tokens=args.max_tokens,
                                   repetition_penalty=args.repetition_penalty,
                                   do_sample=args.temperature > 0,
                                   top_k=args.top_k,
                                   top_p=args.top_p,
                                   temperature=args.temperature if args.temperature > 0 else None
                                   )
          thread = Thread(target=model_infer.generate, kwargs=generation_kwargs)
          thread.start()
          
          out_start = False
          
          generated_text = ""
          for new_text in streamer:
              output = new_text
              if '<think>' in output:
                  out_start = True
                  idx = output.find("<think>")
                  if idx != -1:
                      output = output[idx:]
                  else:
                    output = '<think>'
              if output and out_start:
                  generated_text += output
                  print(output, end="")
    
        raw = generated_text
        try:
            answer = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
            answer = answer.group(1)
            think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            think = think.group(1)
    
        except Exception as e:
            print(f"Error: {e}")
            think, answer = raw, raw
        # print(f"Assistant: {raw},\nThink: {think},\nAnswer: {answer}")
        return_item = {
            'thinker': think,
            'answer': answer
        }
        return return_item


    while True:
        question = input("\nUser: ").strip()
        if question.lower() == "exit":
            break
        answer_item = infer_single_image(args.image_path, question)
        print(f'\n\n{answer_item}\n\n')

