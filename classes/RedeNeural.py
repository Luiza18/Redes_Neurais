from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RedeNeural:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def treinar_e_avaliar(self, dados_treino_teste_igual, tempo_treinamento, taxa_aprendizado, momentum, n_neuronios1, n_neuronios2, func_ativacao):
        if dados_treino_teste_igual:
            X_treino, X_teste, y_treino, y_teste = self.X, self.X, self.y, self.y
        else:
            X_treino, X_teste, y_treino, y_teste = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(n_neuronios1, n_neuronios2),
            max_iter=tempo_treinamento,
            learning_rate_init=taxa_aprendizado,
            momentum=momentum,
            activation=func_ativacao,
            random_state=42
        )
        
        mlp.fit(X_treino, y_treino)
        
        y_pred = mlp.predict(X_teste)
        acuracia = accuracy_score(y_teste, y_pred)
        return acuracia