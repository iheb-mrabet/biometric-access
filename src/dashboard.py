import streamlit as st
import pandas as pd
import sqlite3
import subprocess
import sys
import shutil
import json
import hashlib
import time
from pathlib import Path

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from database import (
    init_database,
    
    get_departments,
    get_workers_with_access,
    add_or_update_worker,
    delete_worker
)

from watermark_logs import (
    WATERMARKED_LOG_PATH,
    generate_watermark,
    load_or_create_secret_key
)

from email_alerts import send_security_alert

DB_PATH = PROJECT_ROOT / "data" / "access_control.db"
WATERMARKED_LOG_FILE = PROJECT_ROOT / WATERMARKED_LOG_PATH
REGISTRATION_STATUS_DIR = PROJECT_ROOT / "data" / "registration_status"
FACE_CHECK_RESULT_FILE = PROJECT_ROOT / "data" / "face_check_result.txt"
ACCESS_LIVE_STATUS_FILE = PROJECT_ROOT / "data" / "access_live_status.json"
LOG_INTEGRITY_ALERT_STATE_FILE = PROJECT_ROOT / "data" / "log_integrity_alert_state.txt"

st.set_page_config(
    page_title="NexusGate Biometric Access",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 35%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 30%),
        linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
}

.block-container {
    padding-top: 1.2rem;
    max-width: 1320px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 60%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    padding: 34px;
    border-radius: 30px;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 64, 175, 0.95)),
        radial-gradient(circle at top right, rgba(34, 211, 238, 0.45), transparent 35%);
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.28);
    border: 1px solid rgba(255,255,255,0.15);
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1.2px;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.88;
    line-height: 1.6;
}

.pill {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.18);
    margin-right: 8px;
    margin-top: 14px;
    font-weight: 700;
    font-size: 13px;
}

.metric-card {
    padding: 20px;
    border-radius: 24px;
    background: rgba(255,255,255,0.84);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(226, 232, 240, 0.9);
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.09);
}

.metric-title {
    font-size: 13px;
    color: #64748b;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #0f172a;
    margin-top: 6px;
}

.glass-panel {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(226,232,240,0.85);
    box-shadow: 0 16px 45px rgba(15,23,42,0.08);
    border-radius: 26px;
    padding: 24px;
    margin-bottom: 20px;
}

.worker-card {
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 28px;
    padding: 24px;
    margin-bottom: 20px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.88));
    box-shadow: 0 18px 48px rgba(15, 23, 42, 0.10);
}

.worker-name {
    font-size: 29px;
    font-weight: 900;
    margin-bottom: 10px;
    color: #0f172a;
    letter-spacing: -0.7px;
}

.worker-line {
    font-size: 15px;
    margin-bottom: 8px;
    color: #334155;
}

.tag-green {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
    padding: 10px 14px;
    border-radius: 14px;
    display: inline-block;
    font-weight: 900;
    box-shadow: inset 0 0 0 1px rgba(22,101,52,0.08);
}

.tag-blue {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    color: #1e40af;
    padding: 10px 14px;
    border-radius: 14px;
    display: inline-block;
    font-weight: 900;
}

.tag-red {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
    padding: 10px 14px;
    border-radius: 14px;
    display: inline-block;
    font-weight: 900;
}

.danger-zone {
    border: 1px solid #fecaca;
    background: #fff1f2;
    border-radius: 20px;
    padding: 18px;
}

.small-muted {
    color: #64748b;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

init_database()


if "page" not in st.session_state:
    st.session_state.page = "Overview"


def run_script_wait(script_name):
    script_path = PROJECT_ROOT / "src" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    return result.stdout + result.stderr



def run_face_check():
    script_path = PROJECT_ROOT / "src" / "check_face_registered.py"

    FACE_CHECK_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FACE_CHECK_RESULT_FILE.write_text(
        "RUNNING: Face verification window opened. Waiting for result.",
        encoding="utf-8",
    )

    subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def read_face_check_result():
    if FACE_CHECK_RESULT_FILE.exists():
        return FACE_CHECK_RESULT_FILE.read_text(encoding="utf-8").strip()

    return ""


def read_access_live_status():
    if not ACCESS_LIVE_STATUS_FILE.exists():
        return {}

    try:
        return json.loads(ACCESS_LIVE_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_access_live_status(selected_department):
    status = read_access_live_status()

    if not status:
        st.info("No live access decision yet. Start biometric access control and face the camera.")
        return

    if status.get("target_department") != selected_department:
        st.warning(
            f"Latest decision is for {status.get('target_department', '-')}, "
            f"not {selected_department}."
        )

    decision = status.get("decision", "-")
    status_code = status.get("status", "-")

    if "GRANTED" in decision:
        st.success("Current access result: ACCESS GRANTED")
    elif "SCANNING" in decision:
        st.info("Current access result: scanning...")
    else:
        st.error("Current access result: ACCESS DENIED")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Face recognized:** {status.get('face_recognized', '-')}")
        st.markdown(f"**Job title:** {status.get('job_title', '-')}")
        st.markdown(f"**Target department:** {status.get('target_department', '-')}")

    with col2:
        st.markdown(f"**Access to department:** {status.get('access_to_department', '-')}")
        st.markdown(f"**Decision:** {decision}")
        st.markdown(f"**Status:** {status_code}")

    st.caption(
        f"Liveness: {status.get('liveness_decision', '-')} "
        f"score={status.get('liveness_score', '-')} | "
        f"Face confidence={status.get('confidence', '-')} | "
        f"Updated: {status.get('updated_at', '-')}"
    )


def run_recognition(department):
    script_path = PROJECT_ROOT / "src" / "recognize_cdcn.py"
    runtime_log = PROJECT_ROOT / "logs" / "access_runtime.log"
    runtime_log.parent.mkdir(parents=True, exist_ok=True)

    if ACCESS_LIVE_STATUS_FILE.exists():
        ACCESS_LIVE_STATUS_FILE.unlink()

    log_file = open(runtime_log, "a", encoding="utf-8")

    subprocess.Popen(
        [sys.executable, str(script_path), "--department", department],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

def run_registration_new_console(username, full_name, job_title, role, departments):
    script_path = PROJECT_ROOT / "src" / "register_worker.py"

    command = [
        sys.executable,
        str(script_path),
        "--username",
        username,
        "--full-name",
        full_name,
        "--job-title",
        job_title,
        "--role",
        role,
        "--departments"
    ] + departments

    subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def delete_worker_assets(username):
    username = username.strip().lower()

    face_dir = PROJECT_ROOT / "data" / "faces" / username
    preview_file = PROJECT_ROOT / "data" / "worker_previews" / f"{username}.jpg"
    status_file = PROJECT_ROOT / "data" / "registration_status" / f"{username}.txt"

    if face_dir.exists():
        shutil.rmtree(face_dir)

    if preview_file.exists():
        preview_file.unlink()

    if status_file.exists():
        status_file.unlink()

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "train_model.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )


def load_history():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "id", "username", "full_name", "department", "status", "confidence", "timestamp"
        ])

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT id, username, full_name, department, status, confidence, timestamp
        FROM access_history
        ORDER BY id DESC
    """, conn)
    conn.close()
    return df


def load_workers():
    rows = get_workers_with_access()

    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "username",
            "full_name",
            "job_title",
            "role",
            "is_active",
            "registration_status",
            "face_image_path",
            "allowed_departments"
        ]
    )



def send_log_integrity_alert_if_needed(details):
    invalid_details = [
        item for item in details
        if item.get("integrity") == "INVALID"
    ]

    if not invalid_details:
        return

    fingerprint_source = json.dumps(
        invalid_details,
        sort_keys=True,
        ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()

    LOG_INTEGRITY_ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    previous_fingerprint = ""
    if LOG_INTEGRITY_ALERT_STATE_FILE.exists():
        previous_fingerprint = LOG_INTEGRITY_ALERT_STATE_FILE.read_text(
            encoding="utf-8"
        ).strip()

    if previous_fingerprint == fingerprint:
        return

    preview = "\n".join(
        f"Line {item['line']}: {item['reason']} | {item['content']}"
        for item in invalid_details[:10]
    )

    subject = "[ALARM] Tattooed access log integrity violation"
    body = f"""Security alarm: one or more tattooed access logs were modified.

Invalid log count: {len(invalid_details)}

Invalid lines:
{preview}

Action required: inspect logs/watermarked_access_logs.txt from the Tattooed Logs dashboard page.
"""

    if send_security_alert(subject, body):
        LOG_INTEGRITY_ALERT_STATE_FILE.write_text(fingerprint, encoding="utf-8")


def check_watermarked_logs():
    secret_key = load_or_create_secret_key()

    if not WATERMARKED_LOG_FILE.exists():
        return {"total": 0, "valid": 0, "invalid": 0, "details": []}

    lines = WATERMARKED_LOG_FILE.read_text(encoding="utf-8").splitlines()

    total = 0
    valid = 0
    invalid = 0
    details = []

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        total += 1

        if " | watermark=" not in line:
            invalid += 1
            details.append({
                "line": line_number,
                "integrity": "INVALID",
                "reason": "Missing watermark",
                "content": line
            })
            continue

        log_content, stored_watermark = line.rsplit(" | watermark=", 1)
        recalculated_watermark = generate_watermark(log_content, secret_key)

        if stored_watermark == recalculated_watermark:
            valid += 1
            details.append({
                "line": line_number,
                "integrity": "VALID",
                "reason": "Integrity verified",
                "content": log_content
            })
        else:
            invalid += 1
            details.append({
                "line": line_number,
                "integrity": "INVALID",
                "reason": "Log was modified",
                "content": log_content
            })

    if invalid > 0:
        send_log_integrity_alert_if_needed(details)

    return {"total": total, "valid": valid, "invalid": invalid, "details": details}


def read_registration_status(username):
    if not username:
        return ""

    status_file = REGISTRATION_STATUS_DIR / f"{username.strip().lower()}.txt"

    if status_file.exists():
        return status_file.read_text(encoding="utf-8").strip()

    return ""


def resolve_face_path(face_path):
    if not isinstance(face_path, str) or not face_path:
        return None

    path = Path(face_path)

    if path.exists():
        return path

    project_path = PROJECT_ROOT / path

    if project_path.exists():
        return project_path

    return None


departments = get_departments()
history_df = load_history()
workers_df = load_workers()
watermark_result = check_watermarked_logs()

registered_workers = len(workers_df[workers_df["registration_status"] == "registered"]) if not workers_df.empty else 0
total_attempts = len(history_df)
granted_attempts = len(history_df[history_df["status"].str.contains("GRANTED", na=False)]) if not history_df.empty else 0
denied_attempts = len(history_df[history_df["status"].str.contains("DENIED", na=False)]) if not history_df.empty else 0

st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ NexusGate Biometric Access</div>
    <div class="hero-subtitle">
        A company-grade access control dashboard using iPhone Camo camera,
        facial registration, department-based permissions, SQLite audit history,
        and tattooed log integrity verification.
    </div>
    <span class="pill">Face Recognition</span>
    <span class="pill">Department Authorization</span>
    <span class="pill">Tattooed Logs</span>
    <span class="pill">SQLite Audit Trail</span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Registered Workers</div><div class="metric-value">{registered_workers}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Access Attempts</div><div class="metric-value">{total_attempts}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Granted</div><div class="metric-value">{granted_attempts}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Denied</div><div class="metric-value">{denied_attempts}</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Invalid Logs</div><div class="metric-value">{watermark_result["invalid"]}</div></div>', unsafe_allow_html=True)

st.write("")

pages = [
    "Overview",
    "Register Worker",
    "Access Control",
    "Workers Management",
    "Tattooed Logs"
]

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state.page),
    key="navigation_radio"
)

st.session_state.page = page

if page == "Overview":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Security Overview")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Access Decisions")
        if history_df.empty:
            st.info("No access attempts yet.")
        else:
            status_chart = history_df["status"].value_counts().reset_index()
            status_chart.columns = ["status", "count"]
            st.bar_chart(status_chart, x="status", y="count")

    with col_right:
        st.markdown("### Department Activity")
        if history_df.empty:
            st.info("No department access data yet.")
        else:
            dept_chart = history_df["department"].value_counts().reset_index()
            dept_chart.columns = ["department", "count"]
            st.bar_chart(dept_chart, x="department", y="count")

    st.markdown("### Recent Access History")
    if history_df.empty:
        st.info("No access history yet.")
    else:
        st.dataframe(history_df.head(20), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Register Worker":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Register New Worker")

    st.info("Before registration, the system checks whether this face is already enrolled. If yes, registration is blocked.")

    username = st.text_input("Username", placeholder="example: sara", key="register_username")
    full_name = st.text_input("Full Name", placeholder="example: Sara Dahmen", key="register_full_name")
    job_title = st.text_input("Job Title", placeholder="example: SOC Analyst", key="register_job_title")
    role = st.selectbox("Role", ["worker", "admin"], key="register_role")

    if role == "admin":
        selected_departments = departments
        st.success("Admin role selected: access to all departments.")
    else:
        selected_departments = st.multiselect(
            "Allowed Departments",
            departments,
            default=[],
            key="register_departments"
        )

    st.markdown("### Step 1 — Verify if face is already registered")

    col_check_a, col_check_b = st.columns([1, 2])

    with col_check_a:
        if st.button("Check Face Already Registered", use_container_width=True, key="check_face_registered_button"):
            st.session_state["face_check_waiting"] = True
            run_face_check()
            st.info("Face verification window opened. Look at the Camo camera window.")

    with col_check_b:
        face_result = read_face_check_result()

        st.markdown("**Face Verification Result**")

        if face_result.startswith("FOUND"):
            st.session_state["face_check_waiting"] = False
            st.error(face_result)
        elif face_result.startswith("NOT_FOUND"):
            st.session_state["face_check_waiting"] = False
            st.success(face_result)
        elif face_result.startswith("CANCELLED") or face_result.startswith("FAILED"):
            st.session_state["face_check_waiting"] = False
            st.warning(face_result)
        elif face_result.startswith("RUNNING"):
            st.info(face_result)
            if st_autorefresh is not None:
                st_autorefresh(interval=1000, key="face_check_status_refresh")
            else:
                time.sleep(1)
                st.rerun()
        elif face_result:
            st.session_state["face_check_waiting"] = False
            st.info(face_result)
        elif st.session_state.get("face_check_waiting", False):
            st.info("Waiting for face verification result...")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("No verification result yet.")

    st.markdown("### Step 2 — Register the new worker")

    if st.button("Start Secure Face Registration", use_container_width=True, key="register_button"):
        if not username or not full_name or not job_title:
            st.error("Please fill username, full name, and job title.")
        elif role != "admin" and not selected_departments:
            st.error("Please select at least one allowed department.")
        else:
            REGISTRATION_STATUS_DIR.mkdir(parents=True, exist_ok=True)
            status_file = REGISTRATION_STATUS_DIR / f"{username.strip().lower()}.txt"

            if status_file.exists():
                status_file.unlink()

            run_registration_new_console(username, full_name, job_title, role, selected_departments)
            st.success("Registration window opened once. It will first check duplicate face, then scan to 100% if the face is new.")

    check_username = st.text_input("Check registration status for username", value=username, key="check_status_username")

    if st.button("Check Registration Status", use_container_width=True, key="check_status_button"):
        message = read_registration_status(check_username)

        if message.startswith("SUCCESS"):
            st.success(message)
            st.session_state.page = "Workers Management"
            st.rerun()
        elif message.startswith("FAILED"):
            st.error(message)
        else:
            st.info("Registration is still running or no status found yet.")

    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Access Control":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Department Access Control")

    selected_department = st.selectbox(
        "Department to enter",
        departments,
        key="access_department_select"
    )

    if st.button("Start Biometric Access Control", use_container_width=True, key="start_access_control"):
        run_recognition(selected_department)
        st.success(f"Access control window opened once for: {selected_department}")

    if st_autorefresh is not None:
        st_autorefresh(interval=1500, key="access_status_refresh")

    st.markdown("### Live Access Result")
    render_access_live_status(selected_department)

    st.caption("Press Q once or close the camera window to stop.")

    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Workers Management":
    st.subheader("Workers Management")

    workers_df = load_workers()

    if workers_df.empty:
        st.info("No workers registered yet.")
    else:
        for _, worker in workers_df.iterrows():
            st.markdown('<div class="worker-card">', unsafe_allow_html=True)

            col_img, col_info, col_access, col_actions = st.columns([1.4, 3.0, 2.2, 1.4], gap="large")

            with col_img:
                face_path = resolve_face_path(worker["face_image_path"])
                if face_path:
                    st.image(str(face_path), caption=str(worker["username"]), width=180)
                else:
                    st.warning("No clear face preview")

            with col_info:
                st.markdown(f'<div class="worker-name">{worker["full_name"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="worker-line"><b>Username:</b> {worker["username"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="worker-line"><b>Job:</b> {worker["job_title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="worker-line"><b>Role:</b> {worker["role"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="worker-line"><b>Registration:</b> {worker["registration_status"]}</div>', unsafe_allow_html=True)

            with col_access:
                st.markdown("**Allowed departments**")
                allowed = worker["allowed_departments"]

                if worker["role"] == "admin":
                    st.markdown('<span class="tag-green">All departments</span>', unsafe_allow_html=True)
                elif isinstance(allowed, str) and allowed:
                    st.markdown(f'<span class="tag-blue">{allowed}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-red">No department access</span>', unsafe_allow_html=True)

            with col_actions:
                st.markdown("**Actions**")
                confirm_key = f"confirm_delete_{worker['username']}"
                delete_key = f"delete_{worker['username']}"

                confirm = st.checkbox("Confirm delete", key=confirm_key)

                if st.button("Delete User", key=delete_key, use_container_width=True, disabled=not confirm):
                    username_to_delete = str(worker["username"])

                    ok = delete_worker(username_to_delete)
                    delete_worker_assets(username_to_delete)

                    if ok:
                        st.success(f"Deleted worker: {username_to_delete}")
                    else:
                        st.error("Worker not found.")

                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("Update Worker Permissions")

        worker_username = st.selectbox(
            "Select worker",
            workers_df["username"].tolist(),
            key="manage_worker_select"
        )

        selected_worker = workers_df[workers_df["username"] == worker_username].iloc[0]

        current_role = selected_worker["role"]
        role_index = 0 if current_role == "worker" else 1

        new_role = st.selectbox(
            "New role",
            ["worker", "admin"],
            index=role_index,
            key="manage_role_select"
        )

        if new_role == "admin":
            new_departments = departments
            st.success("Admin role gives access to all departments.")
        else:
            current_allowed_text = selected_worker["allowed_departments"]
            current_allowed = []

            if isinstance(current_allowed_text, str) and current_allowed_text:
                current_allowed = [item.strip() for item in current_allowed_text.split(",")]

            new_departments = st.multiselect(
                "New allowed departments",
                departments,
                default=current_allowed,
                key="manage_department_multiselect"
            )

        if st.button("Update Worker Permissions", use_container_width=True, key="update_permissions_button"):
            add_or_update_worker(
                username=worker_username,
                full_name=selected_worker["full_name"],
                job_title=selected_worker["job_title"],
                allowed_departments=new_departments,
                role=new_role
            )
            st.success("Worker permissions updated.")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


elif page == "Tattooed Logs":
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("Tattooed / Watermarked Log Integrity")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Logs", watermark_result["total"])
    col_b.metric("Valid Logs", watermark_result["valid"])
    col_c.metric("Invalid Logs", watermark_result["invalid"])

    if st.button("Verify Logs Now", use_container_width=True, key="verify_logs_button"):
        output = run_script_wait("verify_logs.py")
        st.code(output)

    if st.button("Simulate Manual Attack", use_container_width=True, key="simulate_attack_button"):
        output = run_script_wait("simulate_attack.py")
        st.code(output)
        st.warning("Attack simulated. Verify logs again or refresh.")

    st.markdown("### Verification Details")

    details_df = pd.DataFrame(watermark_result["details"])

    if details_df.empty:
        st.info("No watermarked logs found.")
    else:
        st.dataframe(details_df, use_container_width=True)

    if WATERMARKED_LOG_FILE.exists():
        with st.expander("Show raw tattooed logs"):
            st.text(WATERMARKED_LOG_FILE.read_text(encoding="utf-8"))

    st.markdown('</div>', unsafe_allow_html=True)
