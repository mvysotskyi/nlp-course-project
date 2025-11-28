# worker.py
import json
import os
import time
import io

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageOps
import pytesseract


SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
INPUT_BUCKET = os.environ["INPUT_BUCKET"]           # e.g. "legal-ocr-input"
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")     # e.g. "legal-ocr-output"
OCR_LANG = os.environ.get("OCR_LANG", "ukr")        # "ukr" or "ukr+eng"
DELETE_INPUT = os.environ.get("DELETE_INPUT", "true").lower() == "true"

sqs = boto3.client("sqs", region_name=os.getenv("AWS_REGION"))
s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))


def preprocess_image(image: Image.Image) -> Image.Image:
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    return image


def get_image_from_s3(bucket: str, key: str) -> Image.Image:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()
        image = Image.open(io.BytesIO(content))
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image from s3://{bucket}/{key}: {e}")


def put_ocr_result_to_s3(bucket: str, key: str, text: str):
    """
    Save OCR text to S3, e.g. for input key 'incoming/abc.png'
    produce output key 'results/incoming/abc.txt'
    """
    base_key = key.rsplit(".", 1)[0]  # drop extension
    out_key = f"results/{base_key}.txt"

    try:
        s3.put_object(
            Bucket=bucket,
            Key=out_key,
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        print(f"[OK] OCR result written to s3://{bucket}/{out_key}")
    except Exception as e:
        raise RuntimeError(f"Failed to write OCR result to s3://{bucket}/{out_key}: {e}")


def delete_input_object(bucket: str, key: str):
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"[OK] Deleted input object s3://{bucket}/{key}")
    except Exception as e:
        print(f"[WARN] Failed to delete input object s3://{bucket}/{key}: {e}")


def handle_s3_event_record(record: dict):
    bucket = record["detail"]["bucket"]["name"]
    key = record["detail"]["object"]["key"]

    if bucket != INPUT_BUCKET:
        print(f"[SKIP] Record for bucket {bucket}, expected {INPUT_BUCKET}")
        return

    print(f"[JOB] Processing s3://{bucket}/{key}")

    image = get_image_from_s3(bucket, key)
    image = preprocess_image(image)

    # Run Tesseract
    try:
        text = pytesseract.image_to_string(image, lang=OCR_LANG)
        text = text.strip()
    except pytesseract.TesseractError as e:
        raise RuntimeError(f"Tesseract error on s3://{bucket}/{key}: {e}")
    except Exception as e:
        raise RuntimeError(f"OCR failed on s3://{bucket}/{key}: {e}")

    if not text:
        print(f"[WARN] Empty OCR result for s3://{bucket}/{key}")

    # Write OCR result
    if OUTPUT_BUCKET:
        put_ocr_result_to_s3(OUTPUT_BUCKET, key, text)

    # Optionally delete input
    if DELETE_INPUT and text:
        delete_input_object(bucket, key)


def process_message(message: dict):
    body = message.get("Body", "")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print("[WARN] Non-JSON SQS message, skipping:", body[:200])
        return

    if "s3" not in payload["source"]:
        print("[WARN] SQS message without S3 event, skipping:", payload)
        return

    handle_s3_event_record(payload)


def worker_loop():
    print("[START] OCR worker starting polling loop...")
    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=5,
                WaitTimeSeconds=20,      # long polling
                VisibilityTimeout=60,    # seconds to process
            )
        except ClientError as e:
            print(f"[ERROR] Failed to receive SQS messages: {e}")
            time.sleep(5)
            continue

        messages = resp.get("Messages", [])
        if not messages:
            continue  # no messages, loop again

        for msg in messages:
            receipt_handle = msg["ReceiptHandle"]
            try:
                process_message(msg)
                # Delete message on success
                sqs.delete_message(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt_handle,
                )
                print("[OK] Message processed and deleted.")
            except Exception as e:
                # Do not delete → will be retried after visibility timeout
                print(f"[ERROR] Failed to process message, leaving in queue: {e}")


if __name__ == "__main__":
    worker_loop()
