import optuna
import os


study_path = ".db"  # Replace with the path to your .db file

study_name = os.path.splitext(os.path.basename(study_path))[0]
storage_name = f"sqlite:///{study_path}"  # Replace with the path to your .db file

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_name)


# Get all completed trials and sort by value in descending order
completed_trials = [
    trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
]
top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=False)

# Print the trials in descending order
for i, trial in enumerate(top_trials, 1):
    print(f"Top {i} trial number: {trial.number}")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  State: {trial.state}")
    print(f"  Duration: {trial.duration}")
    print(f"  User attrs: {trial.user_attrs}")
    print("-" * 20)
