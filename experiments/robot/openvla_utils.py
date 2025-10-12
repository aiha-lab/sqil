"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from experiments.robot.utils_quant import QuantizeLinear

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)
######QUANT##############
#    quantizer = "fake"
#    if quantizer == "fake":
#        wrapped_modules={}
#        module_dict={}
#        it=[(name,m) for name,m in vla.named_modules()]
#        for i, (name,m) in enumerate(it):
#            if i==len(it)-1: break # except last layer
#            module_dict[name]=m
#            idx=name.rfind('.')
#            if idx==-1:
#                idx=0
#            father_name=name[:idx]
#            if father_name in module_dict:
#                father_module=module_dict[father_name]
#            else:
#                raise RuntimeError(f"father module {father_name} not found")
#
#            #if 'LMHead' not in name:
#            if 'lm_head' not in name:
#                if 'vision_backbone' in name:
#                    if isinstance(m,torch.nn.Linear):
#                        idx = idx+1 if idx != 0 else idx
#                        new_m = QuantizeLinear(m.in_features, m.out_features, args=32, bias=m.bias is not None)
#                        new_m.weight.data=m.weight.data
#                        new_m.bias=m.bias
#                        replace_m=new_m
#                        wrapped_modules[name] = new_m
#                        setattr(father_module,name[idx:],replace_m)
#                else :
#                    if isinstance(m,torch.nn.Linear):
#                        idx = idx+1 if idx != 0 else idx
#                        new_m = QuantizeLinear(m.in_features, m.out_features, args=4, bias=m.bias is not None)
#                        new_m.weight.data=m.weight.data
#                        new_m.bias=m.bias
#                        replace_m=new_m
#                        wrapped_modules[name] = new_m
#                        setattr(father_module,name[idx:],replace_m)
#        print(f"Quantized modules: {list(wrapped_modules.keys())}")
##################


    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

import torch.nn.functional as F  # 파일 상단 import 구역에 추가

def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
    inspect_dist: bool = False,
    top_k: int = 2,
):
    """Generates an action with the VLA policy (optionally dumps token distributions)."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize(image_tf, crop_scale, batch_size)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image = Image.fromarray(image_tf.numpy()).convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # === (A) 분포까지 보고 싶은 경우: generate를 직접 호출해 scores(=step별 logits) 확보 ===
    dist_info = None
    if inspect_dist:
        # predict_action과 동일하게 빈 토큰(29871)을 콜론 뒤에 강제로 붙여주기
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)), dim=1
            )

        # pixel_values/attention_mask 유지
        pixel_values = inputs.get("pixel_values", None)
        attention_mask = inputs.get("attention_mask", None)

        # action 차원 수
        action_dim = vla.get_action_dim(unnorm_key)

        # logits 추출을 위한 generate
        gen = vla.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=action_dim,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # step별 logits -> 확률 -> top-k
        scores = gen.scores  # List[len = action_dim], each: (1, vocab_size)
        dist_list = []
        for t, step_logits in enumerate(scores):
            # (1, V) -> (V,)
            probs = F.softmax(step_logits[0], dim=-1)
            top = torch.topk(probs, k=top_k)
            top_ids = top.indices.tolist()
            top_probs = [float(x) for x in top.values]

            # 토큰 텍스트 (참고용; action 해석은 bin center를 보세요)
            top_tokens = [processor.tokenizer.decode([tid]) for tid in top_ids]

            # OpenVLA의 action mapping: token_id -> bin index -> bin center(연속값)
            # 참고: predict_action과 동일한 규칙 적용
            # discretized_idx = clip( vocab_size - token_id - 1, 0, len(bin_centers)-1 )
            bin_idx = np.clip(
                vla.vocab_size - np.array(top_ids) - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1
            )
            top_bin_centers = [float(vla.bin_centers[i]) for i in bin_idx]

            dist_list.append(
                {
                    "step": t,                      # 0..ACTION_DIM-1 (각 action 차원에 대응)
                    "top_ids": top_ids,             # 토큰 ID
                    "top_probs": top_probs,         # 확률
                    "top_tokens": top_tokens,       # 디코드된 텍스트 토큰(참고)
                    "top_bin_centers": top_bin_centers,  # 이 토큰이 의미하는 연속 action 값
                }
            )

        dist_info = {
            "action_dim": action_dim,
            "distributions": dist_list,
        }

        # 참고: 원한다면 여기서 바로 출력/로그 가능
        # for d in dist_list:
        #     print(f"[dim {d['step']}]",
        #           list(zip(d["top_ids"], d["top_probs"], d["top_tokens"], d["top_bin_centers"])) )

    # === (B) 실제 액션(연속값) 계산은 기존 경로로 ===
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # 분포를 같이 보고 싶다면 콘솔에 간단히 찍어주거나 호출부에서 활용
    if inspect_dist:
        # 여기서는 최소 요약만 출력 (원하면 주석 해제하고 자세히 찍어도 됨)
        summary = []
        for d in dist_info["distributions"]:
            # top-1만 요약
            summary.append(
                {
                    "dim": d["step"],
                    "id": d["top_ids"][0],
                    "prob": d["top_probs"][0],
                    "token": d["top_tokens"][0],
                    "bin_center": d["top_bin_centers"][0],
                }
            )
        print("[OpenVLA] Top-1 per dim (id/prob/token/bin):", summary)

        # 필요하면 dist_info를 반환하는 형태로 바꿀 수도 있음
        # (기존 호출부와의 호환성을 위해 여기서는 action만 반환)
        # return action, dist_info

    return action

#def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
#    """Generates an action with the VLA policy."""
#    image = Image.fromarray(obs["full_image"])
#    image = image.convert("RGB")
#
#    # (If trained with image augmentations) Center crop image and then resize back up to original size.
#    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
#    #            the original height and width by sqrt(0.9) -- not 0.9!
#    if center_crop:
#        batch_size = 1
#        crop_scale = 0.9
#
#        # Convert to TF Tensor and record original data type (should be tf.uint8)
#        image = tf.convert_to_tensor(np.array(image))
#        orig_dtype = image.dtype
#
#        # Convert to data type tf.float32 and values between [0,1]
#        image = tf.image.convert_image_dtype(image, tf.float32)
#
#        # Crop and then resize back to original size
#        image = crop_and_resize(image, crop_scale, batch_size)
#
#        # Convert back to original data type
#        image = tf.clip_by_value(image, 0, 1)
#        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
#
#        # Convert back to PIL Image
#        image = Image.fromarray(image.numpy())
#        image = image.convert("RGB")
#
#    # Build VLA prompt
#    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
#        prompt = (
#            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
#        )
#    else:  # OpenVLA
#        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
#
#    # Process inputs.
#    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
#    # Get action.
#    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
#    return action
