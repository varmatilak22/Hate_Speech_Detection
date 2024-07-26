import sqlite3 

#Connection setup using sqlite3 which is tweets database
conn=sqlite3.connect("tweets.db")

#Setup a cursor for applying sql statements in the python 
cursor=conn.cursor()

cursor.execute("select * from tweets;")

data=cursor.fetchall()
#print(data)

#Display information of the table
cursor.execute("pragma table_info(tweets);")
data_1=cursor.fetchall()
print("Table Information :Tweets table")
for i in data_1:
    print(i)