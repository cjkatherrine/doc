import re
import time
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel

MODEL_ID = "nvidia/LocateAnything-3B"

class LocateAnythingDetector:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device

    def _resize_image(self, image, max_side=1024):
        w, h = image.size
        scale = max_side / max(w, h)
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

    def _query(self, image, prompt_text, max_new_tokens=64, temperature=0.0):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = dict(
            tokenizer=self.tokenizer,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        result = self._to_text(output_ids)

        del inputs, output_ids
        torch.cuda.empty_cache()

        return result

    def _parse_boxes(self, text, img_w, img_h):
        matches = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text)
        boxes = []
        for x1, y1, x2, y2 in matches:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes.append((
                x1 / 1000 * img_w,
                y1 / 1000 * img_h,
                x2 / 1000 * img_w,
                y2 / 1000 * img_h,
            ))
        return boxes

    def detect(self, image, classes, temperature=0.0):
        original_w, original_h = image.size
        resized_image = self._resize_image(image, max_side=1024)
        resized_w, resized_h = resized_image.size

        results = []
        timing_info = []

        for cls in classes:
            prompt_text = "Locate all " + cls + " regions in this document image."
            start = time.time()
            raw = self._query(resized_image, prompt_text, temperature=temperature)
            elapsed = time.time() - start

            boxes = self._parse_boxes(raw, resized_w, resized_h)

            scale_x = original_w / resized_w
            scale_y = original_h / resized_h
            scaled_boxes = [
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                for x1, y1, x2, y2 in boxes
            ]

            for box in scaled_boxes:
                results.append({"class": cls, "box": box})

            timing_info.append({
                "class": cls,
                "seconds": round(elapsed, 2),
                "num_detections": len(scaled_boxes),
            })

        return results, timing_info

