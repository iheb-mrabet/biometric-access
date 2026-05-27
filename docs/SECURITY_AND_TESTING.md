# Security and Testing Notes

## Protected Assets

The application creates security-sensitive artifacts during use. They must remain local:

| Asset | Reason it is excluded from Git |
| --- | --- |
| `data/faces/` | Biometric enrollment images |
| `data/spoofgate_cdcn_dataset/` | Real and attack capture data |
| `captures/` | Images of security events |
| `data/*.db` | Worker identities, permissions, and event history |
| `logs/` | Operational audit history |
| `models/secret_key.txt` | HMAC verification secret |
| `models/*` | Models may encode local biometric training material |

## Email Credentials

SMTP alerts use environment variables:

```powershell
$env:ALERT_EMAIL_SENDER="your_sender@gmail.com"
$env:ALERT_EMAIL_PASSWORD="your_google_app_password"
$env:ALERT_EMAIL_RECEIVER="security_receiver@gmail.com"
$env:SMTP_SERVER="smtp.gmail.com"
$env:SMTP_PORT="587"
```

Use an app password, never an account password. If an app password appears in a chat, terminal recording, screenshot, or repository, revoke it and generate a new one.

## Audit Log Integrity

`src/watermark_logs.py` signs access events with HMAC-SHA256. The log verifier recalculates the watermark and marks a record invalid when any signed field has changed.

The Streamlit dashboard also reports invalid lines and, when SMTP is configured, sends one email for each distinct integrity violation set. The deduplication state is stored locally and is ignored by Git.

## Testing Guidance

Run from a configured Windows environment:

```powershell
python src\test_email.py
python src\test_cdcn_spoofgate.py --cpu
python -m streamlit run src\dashboard.py
```

For a log-integrity demonstration, use only disposable test records or back up local audit files first. A manual alteration is intentionally persistent until the original content is restored.

## Anti-Spoofing Limitations

CDCN is a passive RGB-based presentation attack detector. It uses learned surface and structure cues but does not receive physical depth or infrared measurements from the camera. Internal tests are evidence for this prototype configuration; they do not constitute standardized PAD certification.

