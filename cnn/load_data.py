from huggingface_hub import list_repo_files

files = list_repo_files("mohanty/PlantVillage", repo_type="dataset")
print(len(files), "files")
for f in files[:40]:
    print(f)