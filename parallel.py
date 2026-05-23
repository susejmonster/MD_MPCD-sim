import threading
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

if __name__ == "__main__":
    Cluster_Read_thread = threading.Thread(target=run_script, args=("clustering.py",))
    script2_thread = threading.Thread(target=run_script, args=("script2.py",))

    Cluster_Read_thread.start()
    script2_thread.start()

    Cluster_Read_thread.join()
    script2_thread.join()

    print("Both scripts have finished executing.")