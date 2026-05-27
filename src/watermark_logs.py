import hashlib
import hmac
import os
from datetime import datetime

LOG_PATH = "logs/access_logs.txt"
WATERMARKED_LOG_PATH = "logs/watermarked_access_logs.txt"
SECRET_KEY_PATH = "models/secret_key.txt"


def load_or_create_secret_key():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(SECRET_KEY_PATH):
        secret = os.urandom(32).hex()
        with open(SECRET_KEY_PATH, "w", encoding="utf-8") as f:
            f.write(secret)

    with open(SECRET_KEY_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate_watermark(log_content, secret_key):
    return hmac.new(
        secret_key.encode("utf-8"),
        log_content.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def write_watermarked_log(username, status, confidence):
    os.makedirs("logs", exist_ok=True)

    secret_key = load_or_create_secret_key()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_content = (
        f"{timestamp} | user={username} | "
        f"status={status} | confidence={confidence:.2f}"
    )

    watermark = generate_watermark(log_content, secret_key)

    final_line = f"{log_content} | watermark={watermark}\n"

    with open(WATERMARKED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(final_line)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_content + "\n")

    print(final_line.strip())


def verify_watermarked_logs():
    secret_key = load_or_create_secret_key()

    if not os.path.exists(WATERMARKED_LOG_PATH):
        print("No watermarked log file found.")
        return

    total = 0
    valid = 0
    invalid = 0

    with open(WATERMARKED_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Verifying watermarked logs...")
    print("-" * 70)

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        total += 1

        if " | watermark=" not in line:
            print(f"Line {line_number}: INVALID - missing watermark")
            invalid += 1
            continue

        log_content, stored_watermark = line.rsplit(" | watermark=", 1)
        recalculated_watermark = generate_watermark(log_content, secret_key)

        if hmac.compare_digest(stored_watermark, recalculated_watermark):
            print(f"Line {line_number}: VALID")
            valid += 1
        else:
            print(f"Line {line_number}: INVALID - log was modified")
            invalid += 1

    print("-" * 70)
    print(f"Total logs: {total}")
    print(f"Valid logs: {valid}")
    print(f"Invalid logs: {invalid}")
