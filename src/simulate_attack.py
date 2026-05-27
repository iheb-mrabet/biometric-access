from pathlib import Path

LOG_FILE = Path("logs/watermarked_access_logs.txt")

if not LOG_FILE.exists():
    print("watermarked_access_logs.txt not found.")
    exit()

lines = LOG_FILE.read_text(encoding="utf-8").splitlines()

if len(lines) == 0:
    print("No logs found.")
    exit()

print("Original first line:")
print(lines[0])
print("-" * 70)

# Simulated attack: change ACCESS_GRANTED to ACCESS_DENIED or user name
if "ACCESS_GRANTED" in lines[0]:
    lines[0] = lines[0].replace("ACCESS_GRANTED", "ACCESS_DENIED", 1)
elif "ACCESS_DENIED" in lines[0]:
    lines[0] = lines[0].replace("ACCESS_DENIED", "ACCESS_GRANTED", 1)
else:
    lines[0] = lines[0].replace("user=ihab", "user=attacker", 1)

LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Modified first line:")
print(lines[0])
print("-" * 70)
print("Attack simulation completed.")
print("Now run: python src\\verify_logs.py")
