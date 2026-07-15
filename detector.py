import re
import time
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel

class LocateAnythingDetector:
    def __init__(self, model_id="nvidia/LocateAnything-3B", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.dtype = torch.bfloat16 if os.getenv("DTYPE", "bfloat16") == "bfloat16" else torch.float16
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", "2500"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device).eval()

    def _resize_image(self, image):
        w, h = image.size
        scale = self.max_image_size / max(w, h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h))
        return image

    def _to_text(self, output_ids):
        first = output_ids[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, list):
            flat = []
            for item in first:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            try:
                flat = [int(x) for x in flat]
                return self.tokenizer.decode(flat, skip_special_tokens=True)
            except (ValueError, TypeError):
                return "".join(str(x) for x in flat)
        return str(first)

    def _query(
        self,
        image,
        prompt_text,
        generation_mode="hybrid",
        max_new_tokens=8192,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
    ):
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
        }]
        if hasattr(self.processor, "py_apply_chat_template"):
            prompt = self.processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if hasattr(self.processor, "process_vision_info"):
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(text=[prompt], images=images, videos=videos, return_tensors="pt")
        else:
            inputs = self.processor(images=[image], text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = dict(
            tokenizer=self.tokenizer,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=None if top_k <= 0 else top_k,
            repetition_penalty=repetition_penalty,
            verbose=True,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        result = self._to_text(output_ids)

        del inputs, output_ids
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    def _parse_boxes(self, text, img_w, img_h):
        pattern = r"(?:<ref>(.*?)</ref>)?\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        matches = re.findall(pattern, text)
        boxes = []
        for label, x1, y1, x2, y2 in matches:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes.append({
                "class": label.strip() or "unknown",
                "box": (
                    x1 / 1000 * img_w,
                    y1 / 1000 * img_h,
                    x2 / 1000 * img_w,
                    y2 / 1000 * img_h,
                ),
            })
        return boxes

    def detect(
        self,
        image,
        classes,
        generation_mode="hybrid",
        max_new_tokens=8192,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
    ):
        original_w, original_h = image.size
        resized_image = self._resize_image(image)
        resized_w, resized_h = resized_image.size

        results = []
        timing_info = []

        prompt_classes = "</c>".join(classes)
        prompt_text = f"Locate all the instances that matches the following description: {prompt_classes}."
        start = time.time()
        raw = self._query(
            resized_image,
            prompt_text,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        elapsed = time.time() - start

        boxes = self._parse_boxes(raw, resized_w, resized_h)
        scale_x = original_w / resized_w
        scale_y = original_h / resized_h
        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            results.append({
                "class": item["class"],
                "box": (
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                ),
            })

        timing_info.append({
            "classes": classes,
            "seconds": round(elapsed, 2),
            "num_detections": len(results),
            "raw_output": raw,
        })

        return results, timing_info

