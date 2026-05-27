import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def send_security_alert(subject, body, image_path=None):
    sender = os.getenv("ALERT_EMAIL_SENDER")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    receiver = os.getenv("ALERT_EMAIL_RECEIVER")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not sender or not password or not receiver:
        print("EMAIL ALERT NOT SENT: missing email environment variables.")
        print("Required: ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_RECEIVER")
        return False

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body)

    if image_path:
        path = Path(image_path)

        if path.exists():
            image_data = path.read_bytes()
            msg.add_attachment(
                image_data,
                maintype="image",
                subtype="jpeg",
                filename=path.name
            )

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        print("EMAIL ALERT SENT SUCCESSFULLY.")
        return True

    except Exception as error:
        print(f"EMAIL ALERT FAILED: {error}")
        return False
