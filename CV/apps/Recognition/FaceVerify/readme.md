## conflict 
- Case 1: add overlap
- Case 2: delete several 1 or 2 -> not affect because server only reload when added new person, or when restart -> add unknowns class 
- Case 3: Adding new person to server but the /predict gate doesn't update 
## Env var
- deploy: `export dir='pwd'` (` not ') 
- data_path: `export data_path="${dir}/database/test_images"`
- uvicorn: `uvicorn my_fastapi:app --reload --reload-dir $data_path --reload-include "*.jpg"`

## Image type:
jpg, jpeg only !!!
