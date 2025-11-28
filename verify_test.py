import requests
import time
import json
import os
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8000"
SECRET = "floxy"
EMAIL = "24f1002364@ds.study.iitm.ac.in     "  # NOTE: Using the correct, expected email here
TARGET_URL = "https://tds-llm-analysis.s-anand.net/demo"


def run_verification():
    # 1. Test Invalid Secret (Expect 403)
    print("--- 1. Testing Invalid Secret (Expected 403) ---")
    try:
        payload_wrong_secret = {
            "email": EMAIL,
            "secret": "WRONG",
            "url": TARGET_URL
        }
        resp = requests.post(f"{BASE_URL}/submit_task", json=payload_wrong_secret, timeout=5)
        print(f"   Response Status Code: {resp.status_code}")
        if resp.status_code != 403:
            print(f"Failed: Secret check. Expected 403, got {resp.status_code}")
            return
        print("Passed: Invalid Secret (403)")
    except Exception as e:
        print(f"Connection failed during secret check: {e}")
        return

    # 2. Test Invalid Email (Expect 403)
    print("\n--- 2. Testing Invalid Email (Expected 403) ---")
    try:
        payload_wrong_email = {
            "email": "WRONG",
            "secret": SECRET,
            "url": TARGET_URL
        }
        resp = requests.post(f"{BASE_URL}/submit_task", json=payload_wrong_email, timeout=5)
        print(f"   Response Status Code: {resp.status_code}")
        if resp.status_code != 403:
            print(f"Failed: Email check. Expected 403, got {resp.status_code}")
            return
        print("Passed: Invalid Email (403)")
    except Exception as e:
        print(f"Connection failed during invalid email check: {e}")
        return

    # 3. Test Invalid JSON/Payload (Expect 400)
    print("\n--- 3. Testing Invalid JSON/Payload (Expected 400) ---")
    try:
        payload_bad_json = {
            "email": EMAIL,
            "secret": SECRET
            # Missing 'url'
        }
        resp = requests.post(f"{BASE_URL}/submit_task", json=payload_bad_json, timeout=5)
        print(f"   Response Status Code: {resp.status_code}")
        if resp.status_code != 400:
            print(f"Failed: Invalid JSON check. Expected 400, got {resp.status_code}")
            return
        print("Passed: Invalid JSON (400)")
    except Exception as e:
        print(f"Connection failed during invalid JSON check: {e}")
        return

    # 4. Test Valid Task (Expect 200 and Success)
    print("\n--- 4. Testing Valid Task Submission (Expected 200) ---")
    payload = {
        "email": EMAIL,
        "secret": SECRET,
        "url": TARGET_URL
    }

    try:
        resp = requests.post(f"{BASE_URL}/submit_task", json=payload, timeout=10)
        print(f"   Submission Status Code: {resp.status_code}")

        if resp.status_code != 200:
            print(f"Failed: Valid submission. Expected 200, got {resp.status_code}")
            return

        task_id = resp.json().get("task_id")
        print(f"Passed: Valid Submission (200). Task ID: {task_id}")

    except Exception as e:
        print(f"Connection failed during valid submission: {e}")
        return

    # Poll Status
    print("\n--- 5. Polling for Task Status (Max 170s) ---")
    start_time = time.time()

    while True:
        elapsed_time = int(time.time() - start_time)
        try:
            status_resp = requests.get(f"{BASE_URL}/status/{task_id}", timeout=5)

            if status_resp.status_code != 200:
                print(f"   [{elapsed_time}s] Status API Error: {status_resp.status_code}. Retrying...")
                time.sleep(1)
                continue

            task_data = status_resp.json()
            status = task_data.get("status")

            if status == "running":
                print(f"   [{elapsed_time}s] Status: {status}")
            elif status == "finished":
                print(f"   [{elapsed_time}s] Status: {status}")
                if check_success(task_data):
                    print("\nCorrect")
                else:
                    print("\nNot Correct (Did not achieve final success)")
                break
            elif status == "failed":
                print(f"   [{elapsed_time}s] Status: {status}")
                print(f"Not Correct (Task Failed: {task_data.get('result')})")
                break
            else:
                # Initial state might be 'not_found' or 'pending'
                print(f"   [{elapsed_time}s] Status: {status}")

            if time.time() - start_time > 170:
                print(f"   [{elapsed_time}s] Status: Timeout reached.")
                print("Not Correct (Timeout)")
                break

            time.sleep(1)

        except Exception as e:
            print(f"   [{elapsed_time}s] Connection failed during status poll: {e}. Retrying...")
            time.sleep(1)


def check_success(task_data):
    logs = task_data.get("logs", [])
    for log in reversed(logs):
        if "Tool Output:" in log:
            output = log.split("Tool Output: ")[1]
            # Normalize and check for success indicators
            output_lower = output.lower().replace(" ", "")
            if '"correct":true' in output_lower and '"url":null' in output_lower:
                return True
    return False

if __name__ == "__main__":
    run_verification()
