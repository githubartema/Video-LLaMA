import argparse
import time
from PIL import Image
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import os
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC,ToUint8, load_video, load_video_direct
from video_llama.processors import Blip2ImageEvalProcessor
            
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class VideoProcessor:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()

    def run_caption(
        self, 
        prompt: str,
        max_new_tokens=300, 
        num_beams=8, 
        min_length=1, 
        top_p=0.9,
        repetition_penalty=1.0, 
        length_penalty=0, 
        temperature=1.0, 
        max_length=2000
    ):

        embs = self.get_context_emb(prompt)

        current_max_len = embs.shape[1] + max_new_tokens
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        stop_words_ids = [torch.tensor([2]).to(self.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # stopping_criteria
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        output_token = outputs[0]

        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]

        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]

        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)

        output_text = output_text.split("</s>")[0]  # remove the stop sign '###'
        output_text = output_text.split("ASSISTANT"+':')[-1].strip()

        return output_text

    def upload_video_without_audio(self, frames, av_fps=25):

        self.VIDEO_MSG = ""
        self.video_list = []

        video, self.VIDEO_MSG = load_video_direct(
            frames,
            av_fps,
            n_frms=8,
            height=224,
            width=224,
            sampling ="uniform", 
            return_msg = True
        )
        video = self.vis_processor.transform(video)
        video = video.unsqueeze(0).to(self.device)

        image_emb, _ = self.model.encode_videoQformer_visual(video)
        self.video_list.append(image_emb)

        return "Received."

    def get_prompt(self, prompt: str):

        prompt = f"""[INST] <<SYS>>
You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail.
<</SYS>>

<Video><ImageHere></Video> {self.VIDEO_MSG} {prompt} [/INST]"""

        return prompt


    def get_context_emb(self, prompt: str) -> torch.Tensor:

        prompt = self.get_prompt(prompt)
        prompt_segs = prompt.split('<ImageHere>')

        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], self.video_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        return mixed_embs