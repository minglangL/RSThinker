# RSThinker (Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded GeoSpatial Chain-of-Thought for Vision-Language Models)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![hf_model](https://img.shields.io/badge/🤗-Model-blue.svg)](https://huggingface.co/minglanga/RSThinker)

---

## 📢 Latest Updates
- 🔥 **News**: `2025/09/27`: We released **RSThinker** checkpoint [huggingface](https://huggingface.co/minglanga/RSThinker). 

## TODO
- Geo-CoT380k, the first large-scale dataset of structured Geo-CoT rationales. 
- RSThinker's GRPO training code and the Remote Sensing reward model.
- RSTest, the evaluation dataset of Geo-CoT. 
---

## Quick Start

### Environment Installation

```bash
conda create -n rsthinker python=3.9
conda activate rsthinker
pip install -r requirements.txt
```

### Chat with RSThinker

You can use `RSThinker_infer_chat_stream.py`，a CLI for chat with RSThinker.

```bash 
python RSThinker_infer_chat_stream.py --model_path minglanga/RSThinker --image_path /path/to/image_path
```

### Evaluation

To validate the versatility and robustness of RSThinker, we evaluate its performance on a diverse set of canonical remote sensing tasks. These tasks are selected to span the full spectrum from fine-grained perception to holistic scene understanding. To showcase the model's core strengths in systematic, object-level analysis, we first evaluate on object counting using the **HRRSD, RSOD, DOTAv2-val, and NWPU-VHR datasets**, and on object detection across benchmarks such as **DOTAv2-val and HRRSD**. This precise object-level localization is further tested through visual grounding on the **VRSBench-VG, DIOR-RSVG, RRSIS-D and RSVG** benchmarks. Moving from object-centric analysis to holistic scene interpretation, we assess performance on scene classification with the **NWPU-RESISC45, AID, WHU-RS19, SIRI-WHU and UCMerced** datasets, and on descriptive image captioning using benchmarks like **UCM-Captions, RSICD, RSITMD, NWPU-captions, Sydney-Captions and VRSBench-cap**. Finally, to evaluate the model's ability to handle complex, open-ended queries, we use the challenging **VRSBench-VQA and RSVQA-HR** benchmarks.

We have developed a testing project `RSTester`, which can be used via the script `RSThinker_infer_dataset.py`. You can also organize your own training data by modifying the configuration.

```bash 
python RSThinker_infer_dataset.py --model_path minglanga/RSThinker
```

### Train
#### Fine-tuning
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune **RSThinker**. Below is an example of **Geo-CoT380k** dataset. You can download the Geo-CoT380k dataset(will be released soon), or organize your dataset into `finetune.json` in the following format.
```json
[
    {
        "messages": [
            {
                "content": "<image>Detect all baseball diamond in the image.",
                "role": "user"
            },
            {
                "content": "<think>To detect all baseball diamonds in the image, I first analyze the overall layout and visual characteristics of the scene. The image appears to be an aerial view of a recreational area with grassy fields. Notably, baseball diamonds typically have a distinct layout featuring a diamond shape with bases at the corners and an infield area that is often dirt or lighter grass compared to the surrounding outfield. \n\n\nI observe that there are two areas within the image that exhibit attributes consistent with baseball diamonds. The first diamond shape is located towards the center of the image and presents clear delineation of bases and the infield, which is formatted geometrically in the expected diamond layout. The color contrast of the infield and surrounding grass supports the identification.\n\n\nMoving on to the second area, I notice another similar structure towards the lower part of the image. This area also shows a baseball field configuration with a recognizable diamond layout, distinguishable by its internal patterns and base placements. \n\n\nAfter reviewing these observations, I can confidently list two specific diamond areas and translate their positions into coordinates relative to the dimensions of the image. The first baseball diamond is located with the bounding coordinates [244,290,710,638], which encapsulates the full extent of its shape. The second diamond is found at [242,646,718,980], similarly outlining its features.\n\n\nThus, the image contains two distinct baseball diamonds located at the specified coordinates, marking them clearly for any analysis or follow-up scrutiny.</think><answer>[[244, 290, 710, 638], [242, 646, 718, 980]]</answer>",
                "role": "assistant"
            }
        ],
        "images": [
            "HRRSD/resize_images/07907.jpg"
        ]
    }
]
```

#### GRPO training
RSThinker's GRPO training project is based on [EasyR1](https://github.com/hiyouga/EasyR1),  a clean fork of the original [veRL](https://github.com/volcengine/verl) project to support vision language models. We thank all the authors for providing such high-performance RL training frameworks. 
We have redesigned the training code, adapted it for the **RSThinker** model, and developed a remote sensing reward model. The training code will be released soon.


## RSThinker: Overview
Vision-Language Models (VLMs) in remote sensing often fail at complex analytical tasks, a limitation stemming from their end-to-end training paradigm that bypasses crucial reasoning steps and leads to unverifiable outputs. To address this limitation, we introduce the Perceptually-Grounded Geospatial Chain-of-Thought (Geo-CoT), a framework that models remote sensing analysis as a verifiable, multi-step process. We instill this analytical process through a two-stage alignment strategy, leveraging Geo-CoT380k, the first large-scale dataset of structured Geo-CoT rationales. This strategy first employs supervised fine-tuning (SFT) to instill the foundational cognitive architecture, then leverages Group Reward Policy Optimization (GRPO) to refine the model's reasoning policy towards factual correctness. The resulting model, RSThinker, outputs both a final answer and its justifying, verifiable analytical trace. This capability yields dominant performance, significantly outperforming state-of-the-art models across a comprehensive range of tasks. The public release of our Geo-CoT380k dataset and RSThinker model upon publication serves as a concrete pathway from opaque perception towards structured, verifiable reasoning for Earth Observation.
![Description](abs.pdf)

## Citation

```bibtex
```

## Acknowledgement

The initial weights of **RSThinker** are initialized from [GLM-4.1V-9B-Base](https://github.com/zai-org/GLM-V), a general-domain pre-trained model. 
The training code of **RSThinker** benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [veRL](https://github.com/volcengine/verl), and [EasyR1](https://github.com/hiyouga/EasyR1). 
We sincerely thank their wonderful open-source works.
