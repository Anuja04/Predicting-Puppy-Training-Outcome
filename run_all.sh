# Set Up Environment - Only Install using PIP
pip install -r "./Part 0 - ENVIRONMENT SETUP/requirements.txt"

# Run Part 1
cd "./Part 1 - ALL"
python "./all_data_prep.py" > all_data_prep_output.txt
# nano all_data_prep_output.txt
cd ".."

# Run Part 2
cd "./Part 2 - ALL"
spark-shell -i "./naive_bayes_training.scala" > naive_bayes_training_output.txt
# nano naive_bayes_training_output.txt
cd ".."

# Run Part 3
cd "./Part 3 - ALL"
spark-shell -i "./dayinlife_training.scala" > dayinlife_training_output.txt
# nano dayinlife_training_output.txt
cd ".."

# Run Bonus
cd "./Part 4 - BONUS ALL"
python "./bonus_merge_csv.py" > bonus_merge_csv.txt
spark-shell -i "./bonus_text_training.scala" > bonus_text_training_output.txt
# nano bonus_text_training_output.txt
cd ".."

# Verify Outputs One-By-One
cd "./Part 1 - ALL"
nano all_data_prep_output.txt
cd ",,"
cd "./Part 2 - ALL"
nano naive_bayes_training_output.txt
cd ",,"
cd "./Part 3 - ALL"
nano dayinlife_training_output.txt
cd ",,"
cd "./Part 4 - BONUS ALL"
nano bonus_text_training_output.txt
cd ",,"
