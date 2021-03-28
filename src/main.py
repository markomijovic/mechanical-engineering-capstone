### Author: Marko Mijovic, mijovicmarko11@gmail.com
### Date: March 26, 2021
"""
This program is a part of the 2020/21 mechanical engineering capstone project. The program preforms
statistical analysis and creates unique visualizations on bearing accelerometer data obtained from
https://www.kaggle.com/vinayak123tyagi/bearing-dataset?select=1st_test
Program is for academic and learning purposes only.
"""
from readdata import *

def get_acceleration():
    dat = sqlite3.connect(r"C:\Users\Marko\Desktop\School\CS - Personal\Projects\Personal - Capstone\main\data\processed\acceleration.db")
    query = dat.execute("SELECT * From Acceleration")
    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
    return df


if __name__ == "__main__":
    df = get_acceleration()
    #print(type(df['x1'][0]))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
