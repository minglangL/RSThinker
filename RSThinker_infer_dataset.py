"""
A command-line interface for chatting with GLM-4.1V-9B-Thinkng model supporting images and videos.

Examples:
    # Text-only chat
    python trans_infer_cli.py
    # Chat with single image
    python trans_infer_cli.py --image_paths /path/to/image.jpg
    # Chat with multiple images
    python trans_infer_cli.py --image_paths /path/to/img1.jpg /path/to/img2.png /path/to/img3.png
    # Chat with single video
    python trans_infer_cli.py --video_path /path/to/video.mp4
    # Custom generation parameters
    python trans_infer_cli.py --temperature 0.8 --top_k 5 --max_tokens 4096

Notes:
    - Media files are loaded once at startup and persist throughout the conversation
    - Type 'exit' to quit the chat
    - Chat with images and video is NOT allowed
    - The model will remember the conversation history and can reference uploaded media in subsequent turns
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

import nltk
nltk.download('wordnet')


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
            output = model_infer.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.temperature > 0,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature if args.temperature > 0 else None
            )
            raw = processor_infer.decode(
                output[0][inputs["input_ids"].shape[1]: -1], skip_special_tokens=False
            )
        
        # print(raw)
        
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
    
    import sys
    import importlib.util
    
    tester_module_path = "./RS_Tester"
    
    if tester_module_path not in sys.path:
        sys.path.insert(0, tester_module_path)

    from RS_Tester.worker import WorkerFlower
    
    config_path = os.path.join(tester_module_path, "./configs/default.yaml")
    worker = WorkerFlower(config_path, infer_single_image, model_name='rs_thinker_grpo')
    worker.run()
    

