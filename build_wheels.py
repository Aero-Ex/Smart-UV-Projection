import os
import subprocess
import sys
import shutil

def build_wheels():
    """Build local wheel and sdist for Smart-UV-Projection."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(project_root, "dist")
    
    print(f"--- Building Smart-UV-Projection in {project_root} ---")
    
    # 1. Clean old builds
    if os.path.exists(dist_dir):
        print("Cleaning old dist/ directory...")
        shutil.rmtree(dist_dir)
    
    # 2. Build using 'build' module
    try:
        import build
    except ImportError:
        print("Module 'build' not found. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "build"])
        
    print("Running python -m build...")
    subprocess.check_call([sys.executable, "-m", "build"], cwd=project_root)
    
    print("\n--- Build Complete ---")
    print("Files ready in dist/:")
    for f in os.listdir(dist_dir):
        print(f"  - {f}")
    
    print("\nYou can now install the wheel locally with:")
    print(f"pip install dist/*.whl")

if __name__ == "__main__":
    build_wheels()
