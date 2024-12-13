import subprocess
import sys

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        sys.exit(1)

def main():
    scripts = [
        "classification.py",
        "image_resize.py",
        "predict.py",
        "segment.py",
        "data.py"
    ]

    for script in scripts:
        run_script(script)

    print("All scripts executed successfully.")

if __name__ == "__main__":
    main()
