python benchmark/tasks/MTT/preprocess.py ./data

# remove 3 corrupted files from train.tsv
FILE="data/MTT/train.tsv"
cp "$FILE" "$FILE.bak"
sed -i '/6\/norine_braun-now_and_zen-08-gently-117-146.mp3/d; /9\/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3/d; /8\/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3/d' "$FILE"
