import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REGISTRATION_TIMEOUT = 5 * 60


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


def get_latest_version(username, model_slug, instance_slug, framework):
    """Fetch the latest version number for a model variation."""
    model_instance = f"{username}/{model_slug}/{framework}/{instance_slug}"
    print(f"Fetching version list for {model_instance}...")
    res = run_cmd(
        f"kaggle models instances versions list -v --page-size 200 {model_instance}",
        check=True,
    )

    # Drop "Next Page Token = ..." banner
    lines = [
        line
        for line in res.stdout.strip().splitlines()
        if line.strip() and not line.lower().startswith("next page token")
    ]
    if not lines or not lines[0].startswith("version"):
        return 0

    versions = [
        int(row["version"])
        for row in csv.DictReader(lines)
        if row.get("variation") == instance_slug and str(row["version"]).isdigit()
    ]

    return max(versions, default=0)


def ensure_model_container(username, model_slug, model_build):
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


def package_artifacts(project_root, model_build, files_to_copy):
    print(f"Copying weights and source code to {model_build}...")
    shutil.rmtree(model_build, ignore_errors=True)
    model_build.mkdir(parents=True, exist_ok=True)

    for f_rel in files_to_copy:
        src_file = project_root / f_rel
        if not src_file.exists():
            raise FileNotFoundError(f"Required model artifact not found: {src_file}")

        shutil.copy(src_file, model_build / src_file.name)

    src_dir = project_root / "src"
    if src_dir.exists():
        print("Zipping source code to src.zip...")
        shutil.make_archive(str(model_build / "src"), "zip", src_dir)


def deploy_instance(username, model_slug, instance_slug, framework, model_build):
    print(f"\n=== Deploying variation '{instance_slug}' ({framework}) ===")

    # Write model-instance-metadata.json for the variation
    model_instance_metadata = {
        "ownerSlug": username,
        "modelSlug": model_slug,
        "instanceSlug": instance_slug,
        "framework": framework,
        "licenseName": "MIT",
    }
    with open(model_build / "model-instance-metadata.json", "w") as f:
        json.dump(model_instance_metadata, f, indent=2)

    try:
        old_version = get_latest_version(username, model_slug, instance_slug, framework)
    except Exception as e:
        print(f"Could not read current version ({e}). Assuming none exists.")
        old_version = 0

    print(f"Current version before deployment: {old_version}")

    # Check if the variation exists
    instance_path = f"{username}/{model_slug}/{framework}/{instance_slug}"
    print(
        f"Checking if variation {instance_slug} exists under {username}/{model_slug}..."
    )
    check_instance = run_cmd(
        f"kaggle models instances get {instance_path}",
        check=False,
    )

    if check_instance.returncode != 0:
        print(f"Variation {instance_slug} does not exist. Creating variation...")
        run_cmd(f"kaggle models instances create -p {model_build}")
        target_version = 1
    else:
        print(f"Variation {instance_slug} exists. Creating a new version...")
        run_cmd(
            f"kaggle models instances versions create -p {model_build} {instance_path}"
        )
        target_version = old_version + 1

    # Wait for the new version number to register on Kaggle
    print(f"Waiting for Kaggle to register version {target_version} in the registry...")

    registered_version = 0
    latest_seen = old_version
    start_register_time = time.time()
    while time.time() - start_register_time < REGISTRATION_TIMEOUT:
        try:
            latest_seen = get_latest_version(
                username, model_slug, instance_slug, framework
            )
            if latest_seen >= target_version:
                registered_version = latest_seen
                print(f"Version {target_version} has been registered in the Kaggle.")
                break
        except Exception:
            pass

        time.sleep(5)

    if not registered_version:
        raise RuntimeError(
            f"Version {target_version} of {instance_path} did not register within "
            f"{REGISTRATION_TIMEOUT}s (latest version seen: {latest_seen})."
        )

    print(f"Latest '{instance_slug}' version determined: {registered_version}")
    return registered_version


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

    # extract username from kernel-metadata.json
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

    kernel_build = build_dir / "kernel"
    kernel_build.mkdir(parents=True, exist_ok=True)

    model_slug = "smart-mcq-solver"

    variations = [
        (
            "local",
            "PyTorch",
            [
                "outputs/local/local_model_best.pt",
                "outputs/local/vocab.pkl",
                "outputs/local/retriever.pkl",
            ],
        ),
        (
            "baseline",
            "ScikitLearn",
            ["outputs/baseline/baseline_model.pkl"],
        ),
    ]

    first_build = build_dir / "model" / variations[0][0]
    first_build.mkdir(parents=True, exist_ok=True)
    ensure_model_container(username, model_slug, first_build)

    model_sources = []
    for instance_slug, framework, files in variations:
        model_build = build_dir / "model" / instance_slug
        package_artifacts(project_root, model_build, files)
        version_number = deploy_instance(
            username, model_slug, instance_slug, framework, model_build
        )
        instance_path = f"{username}/{model_slug}/{framework}/{instance_slug}"
        model_sources.append(f"{instance_path}/{version_number}")

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
    deployed_prefixes = tuple(
        source.rsplit("/", 1)[0].lower() + "/" for source in model_sources
    )
    kernel_metadata["model_sources"] = [
        source
        for source in kernel_metadata.get("model_sources", [])
        if not source.lower().startswith(deployed_prefixes)
    ] + model_sources

    with open(kernel_build / "kernel-metadata.json", "w") as f:
        json.dump(kernel_metadata, f, indent=2)

    # Push Kernel (Runs notebook on Kaggle)
    print(f"Pushing and running kernel {kernel_metadata['id']} on Kaggle...")
    run_cmd(f"kaggle kernels push -p {kernel_build}")

    print("Deploy successfully completed! Kaggle Model updated and run triggered.")


if __name__ == "__main__":
    main()
