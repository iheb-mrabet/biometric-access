import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/access_control.db")

DEFAULT_DEPARTMENTS = [
    "SOC",
    "Administration",
    "Server Room"
]


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            job_title TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'worker',
            is_active INTEGER NOT NULL DEFAULT 1,
            face_image_path TEXT,
            registration_status TEXT NOT NULL DEFAULT 'profile_created',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS worker_access (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER NOT NULL,
            department_id INTEGER NOT NULL,
            allowed INTEGER NOT NULL DEFAULT 0,
            UNIQUE(worker_id, department_id),
            FOREIGN KEY(worker_id) REFERENCES workers(id),
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            full_name TEXT,
            department TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    cursor.execute("PRAGMA table_info(workers)")
    worker_columns = [row[1] for row in cursor.fetchall()]

    if "role" not in worker_columns:
        cursor.execute("ALTER TABLE workers ADD COLUMN role TEXT NOT NULL DEFAULT 'worker'")
    if "face_image_path" not in worker_columns:
        cursor.execute("ALTER TABLE workers ADD COLUMN face_image_path TEXT")
    if "registration_status" not in worker_columns:
        cursor.execute("ALTER TABLE workers ADD COLUMN registration_status TEXT NOT NULL DEFAULT 'profile_created'")

    cursor.execute("PRAGMA table_info(access_history)")
    history_columns = [row[1] for row in cursor.fetchall()]

    if "full_name" not in history_columns:
        cursor.execute("ALTER TABLE access_history ADD COLUMN full_name TEXT")
    if "department" not in history_columns:
        cursor.execute("ALTER TABLE access_history ADD COLUMN department TEXT NOT NULL DEFAULT 'Unknown'")

    for department in DEFAULT_DEPARTMENTS:
        cursor.execute("""
            INSERT OR IGNORE INTO departments (name)
            VALUES (?)
        """, (department,))

    conn.commit()
    conn.close()


def get_departments():
    init_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM departments ORDER BY id ASC")

    departments = [row[0] for row in cursor.fetchall()]
    conn.close()

    return departments


def add_or_update_worker(username, full_name, job_title, allowed_departments, role="worker"):
    init_database()

    username = username.strip().lower()
    full_name = full_name.strip()
    job_title = job_title.strip()
    role = role.strip().lower()

    if role == "admin":
        allowed_departments = get_departments()

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("SELECT id FROM workers WHERE username = ?", (username,))
    existing = cursor.fetchone()

    if existing is None:
        cursor.execute("""
            INSERT INTO workers (
                username, full_name, job_title, role,
                is_active, registration_status, created_at
            )
            VALUES (?, ?, ?, ?, 1, 'profile_created', ?)
        """, (username, full_name, job_title, role, now))
    else:
        cursor.execute("""
            UPDATE workers
            SET full_name = ?,
                job_title = ?,
                role = ?,
                is_active = 1
            WHERE username = ?
        """, (full_name, job_title, role, username))

    cursor.execute("SELECT id FROM workers WHERE username = ?", (username,))
    worker_id = cursor.fetchone()[0]

    cursor.execute("SELECT id, name FROM departments")
    departments = cursor.fetchall()

    for department_id, department_name in departments:
        allowed = 1 if department_name in allowed_departments else 0

        cursor.execute("""
            INSERT INTO worker_access (worker_id, department_id, allowed)
            VALUES (?, ?, ?)
            ON CONFLICT(worker_id, department_id) DO UPDATE SET
                allowed = excluded.allowed
        """, (worker_id, department_id, allowed))

    conn.commit()
    conn.close()


def mark_worker_registration_success(username, face_image_path):
    init_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE workers
        SET registration_status = 'registered',
            face_image_path = ?
        WHERE username = ?
    """, (face_image_path, username.strip().lower()))

    conn.commit()
    conn.close()


def get_worker(username):
    init_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, full_name, job_title, role, is_active
        FROM workers
        WHERE username = ?
    """, (username.strip().lower(),))

    result = cursor.fetchone()
    conn.close()

    return result


def is_worker_allowed(username, department_name):
    init_database()

    worker = get_worker(username)

    if worker is None:
        return False

    worker_id, username, full_name, job_title, role, is_active = worker

    if is_active != 1:
        return False

    if role == "admin":
        return True

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT wa.allowed
        FROM workers w
        JOIN worker_access wa ON w.id = wa.worker_id
        JOIN departments d ON wa.department_id = d.id
        WHERE w.username = ?
        AND d.name = ?
        AND w.is_active = 1
    """, (username.strip().lower(), department_name))

    result = cursor.fetchone()
    conn.close()

    if result is None:
        return False

    return result[0] == 1


def save_access_attempt(username, full_name, department, status, confidence):
    init_database()

    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO access_history (username, full_name, department, status, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, full_name, department, status, confidence, timestamp))

    conn.commit()
    conn.close()


def get_workers_with_access():
    init_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            w.id,
            w.username,
            w.full_name,
            w.job_title,
            w.role,
            w.is_active,
            w.registration_status,
            w.face_image_path,
            GROUP_CONCAT(
                CASE WHEN wa.allowed = 1 THEN d.name END,
                ', '
            ) AS allowed_departments
        FROM workers w
        LEFT JOIN worker_access wa ON w.id = wa.worker_id
        LEFT JOIN departments d ON wa.department_id = d.id
        GROUP BY
            w.id, w.username, w.full_name, w.job_title,
            w.role, w.is_active, w.registration_status, w.face_image_path
        ORDER BY w.id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows


def delete_worker(username):
    init_database()

    username = username.strip().lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM workers WHERE username = ?", (username,))
    result = cursor.fetchone()

    if result is None:
        conn.close()
        return False

    worker_id = result[0]

    cursor.execute("DELETE FROM worker_access WHERE worker_id = ?", (worker_id,))
    cursor.execute("DELETE FROM workers WHERE id = ?", (worker_id,))

    conn.commit()
    conn.close()

    return True


def show_access_history():
    init_database()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, full_name, department, status, confidence, timestamp
        FROM access_history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    print("\nACCESS HISTORY")
    print("-" * 120)

    if not rows:
        print("No access history found.")
        return

    for row in rows:
        access_id, username, full_name, department, status, confidence, timestamp = row
        print(
            f"ID={access_id} | user={username} | name={full_name} | "
            f"department={department} | status={status} | "
            f"confidence={confidence:.2f} | time={timestamp}"
        )

    print("-" * 120)


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully.")
    print("Departments:", ", ".join(get_departments()))
