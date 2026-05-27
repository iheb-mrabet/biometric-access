from email_alerts import send_security_alert
from datetime import datetime

subject = "[TEST] Biometric Access Control Email Alert"
body = f"""
This is a direct email test.

Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

ok = send_security_alert(subject, body)

print("EMAIL TEST RESULT:", ok)
