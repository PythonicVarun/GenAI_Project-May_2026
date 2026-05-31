import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run_cmd(cmd, check=True):
    """Utility function to run a shell command."""
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0 and check:
        print(f"Error executing command: {cmd}")
        print(f"Stdout: {res.stdout}")
        print(f"Stderr: {res.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res


def get_latest_version(username, model_slug, instance_slug):
    """Fetch the latest version number for a model variation."""
    model_instance = f"{username}/{model_slug}/pytorch/{instance_slug}"
    print(f"Fetching version list for {model_instance}...")
    res = run_cmd(
        f"kaggle models instances versions list -v {model_instance}", check=True
    )
    lines = res.stdout.strip().split("\n")
    if len(lines) <= 1:
        return 1

    reader = csv.reader(lines)
    next(reader)
    versions = []
    for row in reader:
        if row:
            try:
                versions.append(int(row[0]))
            except ValueError:
                pass

    if versions:
        return max(versions)

    return 1


def main():
    """
    Main function to deploy model to Kaggle.
    """
    project_root = Path(__file__).resolve().parent.parent

    api_token = os.environ.get("KAGGLE_API_TOKEN")
    if not api_token:
        raise ValueError(
            "Kaggle API Token (KAGGLE_API_TOKEN) not found in environment."
        )

    # 2. Extract Kaggle Username from kernel-metadata.json
    username = None
    orig_metadata_file = project_root / "notebooks" / "kernel-metadata.json"
    if orig_metadata_file.exists():
        try:
            with open(orig_metadata_file, "r") as f:
                meta = json.load(f)
                meta_id = meta.get("id", "")
                if "/" in meta_id:
                    username = meta_id.split("/")[0]
        except Exception as e:
            raise ValueError(f"Failed to parse notebooks/kernel-metadata.json: {e}")

    print(f"Kaggle Username: {username}")

    # Paths
    build_dir = project_root / "kaggle_build"
    shutil.rmtree(build_dir, ignore_errors=True)

    model_build = build_dir / "model"
    kernel_build = build_dir / "kernel"

    model_build.mkdir(parents=True, exist_ok=True)
    kernel_build.mkdir(parents=True, exist_ok=True)

    # Package Model Weights and Source Code
    print("Copying weights and source code to model build folder...")
    files_to_copy = [
        "outputs/local/local_model_best.pt",
        "outputs/local/vocab.pkl",
        "outputs/local/retriever.pkl",
    ]

    for f_rel in files_to_copy:
        src_file = project_root / f_rel
        if not src_file.exists():
            raise FileNotFoundError(f"Required model artifact not found: {src_file}")

        shutil.copy(src_file, model_build / src_file.name)

    src_dir = project_root / "src"
    if src_dir.exists():
        print("Zipping source code to src.zip...")
        shutil.make_archive(str(model_build / "src"), "zip", src_dir)

    # Kaggle Model Creation and Versioning
    model_slug = "smart-mcq-solver"
    instance_slug = "local"

    # Check if the model container exists
    print(f"Checking if model {username}/{model_slug} exists on Kaggle...")
    check_model = run_cmd(f"kaggle models get {username}/{model_slug}", check=False)

    if check_model.returncode != 0:
        print(f"Model container {username}/{model_slug} does not exist. Creating...")
        # Write model-metadata.json
        model_metadata = {
            "ownerSlug": username,
            "title": "Smart MCQ Solver",
            "slug": model_slug,
            "isPrivate": True,
            "description": "Smart MCQ Solver Model Registry for Local and Fine-tuned models.",  # noqa: E501
        }

        with open(model_build / "model-metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        run_cmd(f"kaggle models create -p {model_build}")
        (model_build / "model-metadata.json").unlink()
    else:
        print(f"Model container {username}/{model_slug} exists.")

    # Write model-instance-metadata.json for the variation
    model_instance_metadata = {
        "ownerSlug": username,
        "modelSlug": model_slug,
        "instanceSlug": instance_slug,
        "framework": "pytorch",
        "licenseName": "MIT",
    }
    with open(model_build / "model-instance-metadata.json", "w") as f:
        json.dump(model_instance_metadata, f, indent=2)

    try:
        old_version = get_latest_version(username, model_slug, instance_slug)
    except Exception:
        old_version = 0

    print(f"Current version before deployment: {old_version}")

    # Check if the variation exists
    print(
        f"Checking if variation {instance_slug} exists under {username}/{model_slug}..."
    )
    check_instance = run_cmd(
        f"kaggle models instances get {username}/{model_slug}/pytorch/{instance_slug}",
        check=False,
    )

    if check_instance.returncode != 0:
        print(f"Variation {instance_slug} does not exist. Creating variation...")
        run_cmd(f"kaggle models instances create -p {model_build}")
        target_version = 1
    else:
        print(f"Variation {instance_slug} exists. Creating a new version...")
        run_cmd(
            f"kaggle models instances versions create -p {model_build} "
            f"{username}/{model_slug}/pytorch/{instance_slug}"
        )
        target_version = old_version + 1

    # Wait for the new version number to register on Kaggle
    print(f"Waiting for Kaggle to register version {target_version} in the registry...")

    registered_version = 0
    start_register_time = time.time()
    while time.time() - start_register_time < 60:
        try:
            current_version = get_latest_version(username, model_slug, instance_slug)
            if current_version >= target_version:
                registered_version = current_version
                print(f"Version {target_version} has been registered in the Kaggle.")
                break
        except Exception:
            pass

        time.sleep(5)

    if not registered_version:
        print(
            f"Warning: Version {target_version} did not appear in version list."
            "Forcing target_version."
        )
        registered_version = target_version

    version_number = registered_version
    print(f"Latest model version determined: {version_number}")

    # Load and adapt kernel-metadata.json
    orig_metadata_file = project_root / "notebooks" / "kernel-metadata.json"
    if not orig_metadata_file.exists():
        raise FileNotFoundError(
            f"Missing kernel-metadata.json at: {orig_metadata_file}"
        )

    with open(orig_metadata_file, "r") as f:
        kernel_metadata = json.load(f)

    notebook_name = kernel_metadata["code_file"]
    notebook_src_file = project_root / "notebooks" / notebook_name
    if not notebook_src_file.exists():
        raise FileNotFoundError(f"Notebook file not found at: {notebook_src_file}")

    shutil.copy(notebook_src_file, kernel_build / notebook_name)

    # Adapt kernel ID
    orig_id = kernel_metadata["id"]
    if "/" in orig_id:
        slug = orig_id.split("/")[-1]
        kernel_metadata["id"] = f"{username}/{slug}"
    else:
        kernel_metadata["id"] = f"{username}/{orig_id}"

    # Add model source: pythonicvarun/smart-mcq-solver/pytorch/local/1
    model_source = f"{username}/{model_slug}/pytorch/{instance_slug}/{version_number}"
    if "model_sources" not in kernel_metadata:
        kernel_metadata["model_sources"] = []

    # Make sure model_source is listed
    if model_source not in kernel_metadata["model_sources"]:
        kernel_metadata["model_sources"].append(model_source)

    with open(kernel_build / "kernel-metadata.json", "w") as f:
        json.dump(kernel_metadata, f, indent=2)

    # Push Kernel (Runs notebook on Kaggle)
    print(f"Pushing and running kernel {kernel_metadata['id']} on Kaggle...")
    run_cmd(f"kaggle kernels push -p {kernel_build}")

    print("Deploy successfully completed! Kaggle Model updated and run triggered.")


if __name__ == "__main__":
    main()
