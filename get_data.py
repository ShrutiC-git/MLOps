import os
import wget

#defining the url of the file
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv"

file = wget.download(url)