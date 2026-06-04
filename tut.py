import os
import zipfile
import shutil
import huggingface_hub

# # login
# huggingface_hub.login(token='<YOUR_HF_TOKEN>')

audio_datasets_path = "/mnt/lynx3/datasets"

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/Beijing-Opera", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "Beijing-Opera"))

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/CREMA-D", repo_type="dataset", 
#                                     local_dir=os.path.join(audio_datasets_path, "CREMA-D"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=1,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/ESC50", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "ESC50"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=2,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/ESC50-Actions", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "ESC50-Actions"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=1,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/GT-Music-Genre", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "GT-Music-Genre"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=2,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/NS-Instruments", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "NS-Instruments"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=2,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )
# zipfile_path = os.path.join(audio_datasets_path, 'NS-Instruments', 'NS-Instruments.zip')
# with zipfile.ZipFile(zipfile_path,"r") as zip_ref:
#     zip_ref.extractall(os.path.join(audio_datasets_path, 'NS-Instruments'))
# shutil.move(os.path.join(audio_datasets_path, 'NS-Instruments','NS-Instruments', 'audios'), os.path.join(audio_datasets_path, 'NS-Instruments'))
# shutil.move(os.path.join(audio_datasets_path, 'NS-Instruments','NS-Instruments', 'train.csv'), os.path.join(audio_datasets_path, 'NS-Instruments'))
# shutil.move(os.path.join(audio_datasets_path, 'NS-Instruments','NS-Instruments', 'test.csv'), os.path.join(audio_datasets_path, 'NS-Instruments'))
# shutil.rmtree(os.path.join(audio_datasets_path, 'NS-Instruments', 'NS-Instruments'))
# os.remove(zipfile_path)

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/RAVDESS", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "RAVDESS"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=1,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/SESA", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "SESA"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=2,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/TUT2017", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "TUT2017"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=2,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )

# audio_datasets_path = "/mnt/lynx1/datasets"
# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/UrbanSound8K", repo_type="dataset", local_dir=os.path.join(audio_datasets_path, "UrbanSound8K"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=3,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )


# if not os.path.exists(audio_datasets_path): print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="MahiA/VocalSound", repo_type="dataset", 
#                                     local_dir=os.path.join(audio_datasets_path, "VocalSound"),
#                                     local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                     resume_download=True,          # 이어받기
#                                     max_workers=3,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                   )
# zipfile_path = os.path.join(audio_datasets_path, 'VocalSound', 'VocalSound.zip')
# with zipfile.ZipFile(zipfile_path,"r") as zip_ref:
#     zip_ref.extractall(os.path.join(audio_datasets_path, 'VocalSound'))
# shutil.move(os.path.join(audio_datasets_path, 'VocalSound','VocalSound', 'audios'), os.path.join(audio_datasets_path, 'VocalSound'))
# shutil.move(os.path.join(audio_datasets_path, 'VocalSound','VocalSound', 'train.csv'), os.path.join(audio_datasets_path, 'VocalSound'))
# shutil.move(os.path.join(audio_datasets_path, 'VocalSound','VocalSound', 'test.csv'), os.path.join(audio_datasets_path, 'VocalSound'))
# shutil.rmtree(os.path.join(audio_datasets_path, 'VocalSound', 'VocalSound'))
# os.remove(zipfile_path)


# if not os.path.exists(audio_datasets_path): 
#   print(f"Given {audio_datasets_path=} does not exist. Specify a valid path ending with 'Audio-Datasets' folder.")
# huggingface_hub.snapshot_download(repo_id="christopher/birdclef-2025", repo_type="dataset", 
#                                   local_dir=os.path.join(audio_datasets_path, "Birdclef-2025"),
#                                   local_dir_use_symlinks=False,  # 실제 파일로 저장(심볼릭링크 회피)
#                                   resume_download=True,          # 이어받기
#                                   max_workers=3,                 # 병렬 최소화 → rate limit에 거의 안 걸림
#                                 )
