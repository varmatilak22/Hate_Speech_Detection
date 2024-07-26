import sqlite3
import pandas as pd

#Setup the connection with tweets database 
conn=sqlite3.connect("tweets.db")

cursor=conn.cursor()

#Read the csv file
data_frame=pd.read_csv("labeled_data.csv")
print(data_frame)

#Write a dataframe to a database
data_frame.to_sql('tweets',conn,index=False,if_exists='replace')

# commit() method is used to save the changes made in the db
# commit() after executing a series of sql statements that modify the database


#Close the connection
## Close methods is used to close the database connection.
## It is important to close the connection when it is no longer to free up database resources
#When done with the database
conn.close()