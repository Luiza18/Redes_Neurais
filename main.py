from classes.RedeNeural import RedeNeural
from classes.Interface import Interface
from pandas import read_csv
from os import path

diretorio = path.dirname(path.abspath(__file__))

file_path = path.join(diretorio,"document","winequality-red.csv")

dados = read_csv(file_path, sep=';')

X = dados.drop("quality", axis=1) 
y = dados["quality"]

rede_neural = RedeNeural(X, y)
app = Interface(rede_neural)
app.executar()