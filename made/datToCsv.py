import pandas as pd
import os
import numpy as np

PathDIR = "D:\df\Master\ml-1m"

ages_mapping = {1:"Under 18",18:"18-24",25:"25-34",35:"35-44",45:"45-49",50:"50-55",56:"56+"}
occupations_mapping = {0: "other",1: "academic/educator",2: "artist",3: "clerical/admin",4: "college/grad student",5: "customer service",6: "doctor/health care",7: "executive/managerial",8: "farmer",9: "homemaker",10: "K-12 student",11: "lawyer",12: "programmer",13: "retired",14: "sales/marketing",15: "scientist",16: "self-employed",17: "technician/engineer",18: "tradesman/craftsman",19: "unemployed",20: "writer"}

users = pd.read_csv(os.path.join(PathDIR,"users.dat"),sep="::",encoding="latin-1",engine="python",names=["user_id","gender","age","occupation","zipcode"])
users["user_ID"] = np.arange(stop=users.shape[0]) 
users.to_csv("Data/users.csv", sep="\t",header=True,encoding="latin-1",columns=["user_id","gender","age","occupation","zipcode"],index=False)
print(users.head())


movies = pd.read_csv(os.path.join(PathDIR,"movies.dat"),sep="::",engine="python",names=["movie_id","title","genres"],encoding="latin-1")
movies["movie_ID"] = np.arange(stop=movies.shape[0]) 
movies.to_csv("Data/movies.csv",sep="\t",header=True,encoding="latin-1",columns=["movie_id","title","genres"],index=False)
print(movies.head(10))


ratings = pd.read_csv(os.path.join(PathDIR,"ratings.dat"),sep="::",engine="python",encoding="latin-1",names=["user_id","movie_id","rating","timestamp"])
ratings = pd.merge(pd.merge(ratings,users[["user_id","user_ID"]],on="user_id"),movies[["movie_id","movie_ID"]],on="movie_id")
ratings.to_csv("Data/ratings.csv",sep="\t",header=True,encoding="latin-1",columns=["user_id","movie_id","rating","timestamp","user_ID","movie_ID"],index=False)
print(ratings.head(10))



