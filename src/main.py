### Author: Marko Mijovic, mijovicmarko11@gmail.com
### Date: March 26, 2021
"""
This program is a part of the 2020/21 mechanical engineering capstone project. The program preforms
statistical analysis and creates unique visualizations on bearing accelerometer data obtained from
https://www.kaggle.com/vinayak123tyagi/bearing-dataset?select=1st_test
Program is for academic and learning purposes only.
"""
from readData import *
import matplotlib.pyplot as plt
import seaborn as sns

class Acceleration:

    def __init__(self, path) -> None:
        self.path = path

    def get_acceleration_from_db(self) -> pd.DataFrame:
        dat = sqlite3.connect(self.path_db)
        query = dat.execute("SELECT * From Acceleration")
        cols = [column[0] for column in query.description]
        return pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        
    def create_box_plot(self, df):
        fig, axes = plt.subplots(2, 2)
        fig.suptitle('Acceleration Statistical Distribution. Failure on 1')
        sns.boxplot(ax=axes[0, 0], x=df["a1"])
        sns.boxplot(ax=axes[0, 1], x=df["a2"])
        sns.boxplot(ax=axes[1, 0], x=df["a3"])
        sns.boxplot(ax=axes[1, 1], x=df["a4"])
        plt.show()
    
    def create_heat_plot(self, df):
        sns.heatmap(df.corr(), linewidths=.5, annot=True, fmt=".2f")
        plt.show()
'''
    def create_line_plot(self, df):
        df_reshaped = reshape_data(df)
        sns.lineplot(data=df_reshaped, x="Time", y="Acceleration", hue="Label", alpha=0.5)
        print("finished plotting. opening the plot...")
        plt.show()

# Helper function for create_line_plot
def reshape_data(df):
    time = list(range(1,df["x1"].count()+1))*8
    old_length = df['x1'].count()
    combined_accel = pd.Series(df.values.ravel('F'))
    new_length = len(combined_accel)
    label = [None] * new_length
    label_values = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    for i in range(1, 9):
        label[(i - 1) * old_length:i * old_length] = [label_values[i - 1]] * old_length
    df_new = pd.DataFrame({
        'Time' : time,
        'Acceleration' : combined_accel,
        'Label' : label
    })
    return df_new
'''

if __name__ == "__main__":
    db_path = 'data/processed/acceleration2.db'
    p_absolute = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', db_path))
    accel = Acceleration(p_absolute)
    print(accel.path)