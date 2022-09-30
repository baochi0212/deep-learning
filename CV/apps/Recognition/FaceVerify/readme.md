## conflict 
- Case 1: add overlap
- Case 2: delete several 1 or 2 -> not affect because server only reload when added new person, or when restart -> add unknowns class 
- Case 3: Adding new person to server but the /predict gate doesn't update 
## Env var
- deploy: `export dir='pwd'` (` not ') 
- data_path: `export data_path="${dir}/database"`
- uvicorn: `uvicorn my_fastapi:app --reload --reload-dir $data_path --reload-include "*.jpg" --reload-include ".txt"`

## Image type:
jpg, jpeg only !!!

## Other OS
Change all the os.system() command

## Data
- bins: contains temporary removed members
- test_images: raw images
- test_images_cropped: cropped faces