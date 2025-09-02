import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")  # stil Seaborn  sns.set_theme(style="darkgrid")

class JSONDataPreparer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        self.users_data = {} # {username: {...}}
        

    def load_json(self):
        with open(self.file_path, 'r') as f:
            self.raw_data = json.load(f)

        for record in self.raw_data:
            username = record.get("username", "Unknown")
            data_block = record.get("data", {})

            self.users_data[username] = {
                "collection_date": datetime.fromisoformat(
                    record.get("collectionDate").replace("Z", "")
                ),
                "error_values": data_block.get("ErrorValues", []),
                "last_second_error": data_block.get("LastSecondError"),
                "play_time_seconds": data_block.get("PlayTimeSeconds"),
                "parameter_changes": data_block.get("ParameterChanges"),
            }

        return self

    def to_dataframe(self, username: str) -> pd.DataFrame:
        if username not in self.users_data:
            raise ValueError(f"Userul {username} nu există în date.")
        
        errors = self.users_data[username]["error_values"]
        df = pd.DataFrame({
            "Index": range(len(errors)),
            "ErrorValue": errors
        })
        return df




    
    def plot_user_barchart(self):
        usernames = sorted(self.users_data.keys(), key=str.lower)
        play_times = [self.users_data[user]["play_time_seconds"] for user in usernames]
        param_changes = [self.users_data[user]["parameter_changes"] for user in usernames]

    # Intercalăm valorile în ordinea corectă: pentru fiecare user, PlayTimeSeconds apoi ParameterChanges
        values = []
        for pt, pc in zip(play_times, param_changes):
            values.extend([pt/60, pc,60*pc/pt])

        df_bar = pd.DataFrame({
        "User": np.repeat(usernames, 3),
        "Metric": ["PlayTimeMinutes", "ParameterChanges","RateOfChanges"] * len(usernames),
        "Value": values
        })

        plt.figure(figsize=(12, 6))
        sns.barplot(x="User", y="Value", hue="Metric", data=df_bar, palette=["#ffbb33", "#79c7e9","#e95959"])

        plt.xlabel("Users")
        plt.ylabel("Values")
        plt.title("Play Time in Minutes, Number of Parameter Changes per Session and Rate of Parameter Change for each User - After")
        plt.xticks(rotation=45, ha="right")

        ax = plt.gca()
        for p in ax.patches:
            height = p.get_height()
            if(height<1):
                continue
            ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3),
                    textcoords='offset points')
        ax.set_ylim(0,100)

        plt.tight_layout()
        plt.show()
    

# Exemplu folosire:
preparer = JSONDataPreparer("All_A.json")
preparer.load_json()



# Bar chart comparativ pentru fiecare utilizator
preparer.plot_user_barchart()
