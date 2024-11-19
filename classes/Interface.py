import PySimpleGUI as sg

class Interface:
    def __init__(self, rede_neural):
        self.rede_neural = rede_neural
        self.layout = [
            [sg.Text("Opção de Dados de Treino/Teste")],
            [sg.Combo(
                ["Dados de treino = Dados de Teste", "Dados de treino (80%) e Dados de Teste (20%)"],
                key="opcao_treino_teste",
                readonly=True,
                default_value="Dados de treino (80%) e Dados de Teste (20%)",
                size=(40, 1) 
            )],
            [sg.Text("Tempo de Treinamento"), sg.InputText("200", key="tempo_treinamento", size=(10, 1))],
            [sg.Text("Taxa de Aprendizado"), sg.InputText("0.001", key="taxa_aprendizado", size=(10, 1))],
            [sg.Text("Momentum"), sg.InputText("0.9", key="momentum", size=(10, 1))],
            [sg.Text("Neurônios na Primeira Camada Oculta"), sg.InputText("100", key="neuronios1", size=(10, 1))],
            [sg.Text("Neurônios na Segunda Camada Oculta"), sg.InputText("50", key="neuronios2", size=(10, 1))],
            [sg.Text("Função de Ativação")],
            [sg.Combo(["Identity", "Logistic", "Tanh", "Relu"], 
                      default_value="Relu", 
                      key="func_ativacao", 
                      readonly=True, 
                      size=(10, 1))], 
            [sg.Button("Executar RNA")],
            [sg.Multiline(size=(40, 5), font=('Courier New', 12), disabled=True, no_scrollbar=True, key='-RESULTADO-')]
        ]
        self.janela = sg.Window("Configuração da Rede Neural", self.layout)

    def validar_entrada(self, valores):
        validadores = {
            "opcao_treino_teste": (lambda x: x in ["Dados de treino = Dados de Teste", "Dados de treino (80%) e Dados de Teste (20%)"],
                                   "Selecione uma opção de treino/teste."),
            "tempo_treinamento": (lambda x: x.isdigit() and int(x) > 0,
                                  "Tempo de treinamento deve ser um número inteiro positivo."),
            "taxa_aprendizado": (lambda x: x.replace('.', '', 1).isdigit() and 0 < float(x) < 1,
                                 "Taxa de aprendizado deve ser um número entre 0 e 1."),
            "momentum": (lambda x: x.replace('.', '', 1).isdigit() and 0 <= float(x) <= 1,
                         "Momentum deve ser um número entre 0 e 1."),
            "neuronios1": (lambda x: x.isdigit() and int(x) > 0,
                           "Número de neurônios na primeira camada deve ser um número inteiro positivo."),
            "neuronios2": (lambda x: x.isdigit() and int(x) >= 0,
                           "Número de neurônios na segunda camada deve ser um número inteiro maior ou igual a 0."),
            "func_ativacao": (lambda x: x in ["Identity", "Logistic", "Tanh", "Relu"],
                              "Selecione uma função de ativação válida.")
        }

        for campo, (condicao, mensagem) in validadores.items():
            if not condicao(valores[campo]):
                return False, mensagem

        return True, ""

    def executar(self):
        while True:
            evento, valores = self.janela.read()
            if evento == sg.WINDOW_CLOSED:
                break
            elif evento == "Executar RNA":
                valido, mensagem_erro = self.validar_entrada(valores)
                if not valido:
                    sg.popup_error("Erro de Validação", mensagem_erro)
                    continue

                dados_treino_teste_igual = valores["opcao_treino_teste"] == "Dados de treino = Dados de Teste"
                tempo_treinamento = int(valores["tempo_treinamento"])
                taxa_aprendizado = float(valores["taxa_aprendizado"])
                momentum = float(valores["momentum"])
                n_neuronios1 = int(valores["neuronios1"])
                n_neuronios2 = int(valores["neuronios2"])
                func_ativacao = valores["func_ativacao"].lower()

                try:
                    acuracia = self.rede_neural.treinar_e_avaliar(
                        dados_treino_teste_igual, tempo_treinamento, taxa_aprendizado, momentum,
                        n_neuronios1, n_neuronios2, func_ativacao
                    )
                    self.janela["-RESULTADO-"].update(f"Acurácia: {acuracia * 100:.2f}%")
                except Exception as erro:
                    sg.popup_error("Erro durante a execução da RNA", str(erro))

        self.janela.close()
