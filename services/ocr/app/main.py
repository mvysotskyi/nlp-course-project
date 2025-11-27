# app/main.py
import io
from typing import List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from mmocr.apis import TextDetInferencer


MODEL_NAME = "cyrillic-trocr/trocr-handwritten-cyrillic"

app = FastAPI(
    title="DBNet + TrOCR OCR Service",
    description="OCR API using DBNet-style detection (PaddleOCR) + Microsoft TrOCR recognition",
    version="1.1.0",
)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class OCRModel:
    def __init__(self, model_name: str = "stepfun-ai/GOT-OCR-2.0-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # GOT-OCR2 uses the standard HF processor + ImageTextToText model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map=self.device,  # "cuda" or "cpu"
        )

    def recognize(self, image: Image.Image) -> str:
        """
        Run OCR on a single cropped text region (PIL image) and return text.
        GOT-OCR2 is multilingual; we just tell it to transcribe and keep original language.
        """
        prompt = (
            "Transcribe all the text in this image exactly as it appears, "
            "in the original language. Output only plain text."
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        # Decode and clean
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return text.strip()



class MMOCRDetector:
    """
    Text detector using MMOCR's TextDetInferencer (DBNet / DBNet++).
    Returns boxes in the same [[x1,y1],...,[x4,y4]] format you already use.
    """

    def __init__(self, model_name: str = "DBNet"):
        # model_name can be 'DBNet' or a specific config name from MMOCR's model zoo
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.inferencer = TextDetInferencer(model=model_name, device=device)

    def detect_boxes(self, image: Image.Image) -> List[List[List[float]]]:
        """
        Returns a list of quadrilateral boxes:
        each box = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # MMOCR expects BGR ndarray
        img_np = np.array(image)[:, :, ::-1]

        # run detector
        result = self.inferencer(img_np, return_vis=False)
        preds = result["predictions"][0]

        boxes: List[List[List[float]]] = []

        # Prefer polygons (more precise). Fallback to bboxes if polygons missing.
        if "polygons" in preds and preds["polygons"]:
            # polygons: list of [x1, y1, x2, y2, ..., xN, yN]
            for poly in preds["polygons"]:
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                box = [
                    [float(x_min), float(y_min)],
                    [float(x_max), float(y_min)],
                    [float(x_max), float(y_max)],
                    [float(x_min), float(y_max)],
                ]
                boxes.append(box)
        elif "bboxes" in preds and preds["bboxes"]:
            # bboxes: [min_x, min_y, max_x, max_y]
            for x1, y1, x2, y2 in preds["bboxes"]:
                box = [
                    [float(x1), float(y1)],
                    [float(x2), float(y1)],
                    [float(x2), float(y2)],
                    [float(x1), float(y2)],
                ]
                boxes.append(box)

        return boxes


ocr_model: OCRModel | None = None
layout_detector: MMOCRDetector | None = None

@app.on_event("startup")
def load_models():
    global ocr_model, layout_detector
    ocr_model = OCRModel(MODEL_NAME)
    layout_detector = MMOCRDetector(model_name="DBNet")  # or custom config name



@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "DBNet + TrOCR OCR API. POST an image to /ocr or /ocr_regions",
        "model": MODEL_NAME,
    }


def sort_boxes_reading_order(boxes: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    Sort boxes roughly top-to-bottom, then left-to-right.
    """
    def box_key(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(ys), min(xs))

    return sorted(boxes, key=box_key)


def crop_box(image: Image.Image, box: List[List[float]], margin: int = 2) -> Image.Image:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min = max(int(min(xs)) - margin, 0)
    y_min = max(int(min(ys)) - margin, 0)
    x_max = min(int(max(xs)) + margin, image.width)
    y_max = min(int(max(ys)) + margin, image.height)
    return image.crop((x_min, y_min, x_max, y_max))


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    High-level endpoint: detect text regions with DBNet, run TrOCR on each,
    and return concatenated text in reading order.
    """
    if ocr_model is None or layout_detector is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Expected image/*",
        )

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    try:
        boxes = layout_detector.detect_boxes(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    if not boxes:
        # Fallback: run TrOCR on whole image
        try:
            text = ocr_model.recognize(image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
        return JSONResponse({"text": text, "regions": []})

    boxes_sorted = sort_boxes_reading_order(boxes)
    lines: List[str] = []
    regions: List[Dict[str, Any]] = []

    for box in boxes_sorted:
        cropped = crop_box(image, box)
        try:
            line_text = ocr_model.recognize(cropped)
        except Exception as e:
            line_text = f"<error: {e}>"

        lines.append(line_text)
        regions.append(
            {
                "box": box,  # list of 4 [x,y] points
                "text": line_text,
            }
        )

    full_text = "\n".join(t for t in lines if t.strip())
    return JSONResponse({"text": full_text, "regions": regions})