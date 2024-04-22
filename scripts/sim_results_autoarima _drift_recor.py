import os
import warnings
import numpy as np
import pandas as pd
from river import (linear_model,
                   stream,
                   metrics,
                   preprocessing,
                   optim,
                   time_series)
import pmdarima as pm
warnings.filterwarnings("ignore")


def aplicar_drift(df, coluna='V1', tipo='abrupto',
                  pontos_drift=None, aleatorio=False, 
                  num_pontos=1, duracao_drift=None, **kwargs):

    n = len(df)
    if aleatorio and tipo != 'incremental':
        pontos_drift = np.sort(np.random.choice(df.index, num_pontos, replace=False))
    elif aleatorio and tipo == 'incremental' and duracao_drift is not None:
        inicio_drift = np.random.randint(0, n - duracao_drift)
        termino_drift = inicio_drift + duracao_drift
        pontos_drift = [inicio_drift, termino_drift]
    elif pontos_drift is None:
        pontos_drift = [n // 2]  # Ponto médio se nenhum ponto for fornecido e aleatorio=False

    if tipo == 'abrupto':
        delta = kwargs.get('delta', np.random.normal(10, 1))  # Default: distribuição normal N(10, 1)
        for ponto in pontos_drift:
            df.loc[df.index >= ponto, coluna] += delta

    elif tipo == 'incremental':
        if duracao_drift is None:
            duracao_drift = np.random.randint(50, 150) 
        if aleatorio:
            inicio_drift = np.random.randint(0, n - duracao_drift)
            termino_drift = inicio_drift + duracao_drift
            print(inicio_drift)
            print(termino_drift)
        else:
            inicio_drift = kwargs.get('inicio_drift', 0)
            termino_drift = kwargs.get('termino_drift', inicio_drift + duracao_drift)

        incremento_base = kwargs.get('incremento', np.random.normal(0.1, 0.05))
        variancia_base = kwargs.get('variancia', 0.02)
        amplitude_variacao = kwargs.get('amplitude_variacao', 0.05)  # Amplitude da variação no incremento
        frequencia_variacao = kwargs.get('frequencia_variacao', np.pi/(termino_drift - inicio_drift))  # Define a frequência da variação sinusoidal

        for i in range(inicio_drift, termino_drift + 1):
            # Calcula um incremento que varia ao longo do tempo, adicionando uma componente sinusoidal
            variacao_sinusoidal = amplitude_variacao * np.sin(frequencia_variacao * (i - inicio_drift))
            incremento_atual = incremento_base + variacao_sinusoidal + np.random.normal(0, variancia_base)

            # Aplica o incremento, ajustando pela distância do início do drift
            df.loc[i, coluna] += (i - inicio_drift) * incremento_atual
        
        # Calcula o aumento a ser aplicado aos valores após o término do drift
        aumento = df.loc[termino_drift, coluna]
        print(aumento)
        
        # Aplica o aumento aos valores restantes da série
        for i in range(termino_drift + 1, n):
            df.loc[i, coluna] += aumento

    elif tipo == 'recorrente':
        # Gera um período aleatório entre 50 e 200. Isso assegura uma variabilidade na frequência das oscilações
        # Menores valores de período gerarão mais oscilações, maiores valores gerarão menos oscilações
        periodo = kwargs.get('periodo', np.random.randint(50, 201))

        # Gera uma amplitude aleatória entre 3 e 10. Isso assegura variabilidade na intensidade das oscilações
        # Amplitudes maiores farão as oscilações serem mais pronunciadas
        amplitude = kwargs.get('amplitude', np.random.uniform(3, 10))

        for i in range(n):
            df.loc[i, coluna] += amplitude * np.sin(2 * np.pi * df.loc[i, 't'] / periodo)

    else:
        raise ValueError("Tipo de drift não suportado. Escolha entre 'abrupto','incremental', 'recorrente'.")

    return df


def get_data(sim_path):

    df = pd.read_csv(sim_path + "sim_df.csv")
    df = aplicar_drift(df, tipo='abrupto', aleatorio=True, num_pontos=2, delta=5)
    df_meta = pd.read_csv(sim_path + "/meta.csv")

    return df, df_meta


def prepare_ts(df):
    X = df[['t']]
    y = df.pop('V1')

    return X, y


def get_arima_incremental_results(X, y):

    model = time_series.SNARIMAX(
                                p=1,
                                d=1,
                                q=1,
                                regressor=(
                                    preprocessing.StandardScaler() |
                                    linear_model.LinearRegression(optimizer=optim.AdaGrad(),
                                                                  intercept_lr=0.1)
                                )
    )

    metrica = metrics.RMSE()
    Y_list = []
    Y_real_nowcast = []
    Y_pred_list = []
    Y_nowcasting_list = []
    met_list = []
    error_incremental_list = []

    for xi, yi in stream.iter_pandas(X, y):
        Y_list.append(yi)
        y_pred = model.forecast(horizon=1)
        Y_pred_list.append(y_pred)
        model.learn_one(yi)
        metrica.update(yi, y_pred[-1])
        erro = yi - y_pred[-1]
        y_pred = model.forecast(horizon=1)
        Y_nowcasting_list.append(y_pred[-1])
        met_list.append(metrica.get())
        error_incremental_list.append(erro)

    # Inicializações
    soma_erros_quadrados = 0  # Soma dos erros quadrados
    rmse_incremental = []  # Lista para armazenar o RMSE incremental
    Y_real_nowcast = []  # Lista para armazenar os valores reais usados para calcular o erro de nowcasting
    nowcasting_error_incremental_list = []  # Lista para armazenar os erros de nowcasting

    for i in range(len(Y_list)-1):  # Subtrai 1 porque não há um próximo valor para o último item
        # O valor real para calcular o erro de nowcasting é o próximo valor em Y_list
        valor_real = Y_list[i+1]
        Y_real_nowcast.append(valor_real)  # Guarda o valor real subsequente para calcular o erro de nowcasting

        # O erro de nowcasting é a diferença entre o valor real subsequente e a previsão de nowcasting
        nowcasting_error = valor_real - Y_nowcasting_list[i]
        nowcasting_error_incremental_list.append(nowcasting_error)

        # Atualiza a soma dos erros quadrados
        soma_erros_quadrados += nowcasting_error ** 2

        # Calcula o RMSE incremental
        n = i + 1  # Número de observações até o momento
        rmse_atual = np.sqrt(soma_erros_quadrados / n)
        rmse_incremental.append(rmse_atual)

    return (nowcasting_error_incremental_list,
            rmse_incremental,
            Y_real_nowcast,
            Y_nowcasting_list)


def get_arima_results(y):
    min_obs_for_first_arima = 10  # Número mínimo de observações para o primeiro ajuste do ARIMA
    adjust_every = 1  # Reajustar o modelo a cada X observações após o primeiro ajuste
    n = len(y)  # Total de pontos de dados

    arima_nowcasting_predictions = [0] * (min_obs_for_first_arima - 1)  # Previsões iniciais serão 0
    arima_nowcasting_error_list = []
    rmse_at_each_step = []

    for i in range(min_obs_for_first_arima - 1, n - 1):
        if (i == min_obs_for_first_arima - 1) or ((i - min_obs_for_first_arima + 1) % adjust_every == 0):
            # Ajusta o modelo na primeira vez após 10 pontos e, depois, a cada 1 pontos
            # Utilizando os últimos 30 pontos (ou todos os pontos disponíveis antes do primeiro ajuste)
            start_index = max(0, i - 30 + 1)
            try:
                model_arima = pm.auto_arima(y[start_index:i+1],
                                            start_p=1, start_q=1,
                                            max_p=3, max_q=3, m=0,
                                            start_P=0, seasonal=False,
                                            d=None, D=0, trace=False,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True)
                forecast = model_arima.predict(n_periods=1).to_list()[0]
            except ValueError as e:
                print(f"Erro ao ajustar o Auto ARIMA na iteração {i}: {e}")
                forecast = 0 
        else:
            forecast = model_arima.predict(n_periods=1)

        actual_value = y[i+1] 
        error = actual_value - forecast
        arima_nowcasting_predictions.append(forecast)
        arima_nowcasting_error_list.append(error)

        # Calcula o RMSE com os erros até o momento
        current_rmse = np.sqrt(np.mean(np.square(arima_nowcasting_error_list)))
        rmse_at_each_step.append(current_rmse)

    return arima_nowcasting_predictions, arima_nowcasting_error_list


def calcular_regret(arima_nowcasting_error_list,
                    nowcasting_error_incremental_list):
    if len(arima_nowcasting_error_list) != len(nowcasting_error_incremental_list):
        raise ValueError("As listas devem ter o mesmo tamanho.")
    # Calcula o regret acumulado
    regret_acumulado = 0
    regrets = [] 
    for arima_error, incr_error in zip(arima_nowcasting_error_list,
                                       nowcasting_error_incremental_list):
        arima_error = abs(arima_error)
        incr_error = abs(incr_error)
        regret_acumulado += abs(incr_error - arima_error)
        regrets.append(regret_acumulado)

    x = list(range(0, len(arima_nowcasting_error_list)))
    coef_angular, coef_linear = np.polyfit(x, regrets, 1)
    return regrets, coef_angular


def simulate(i):

    sim_path = f"./data/simulations/{i}/"
    df, df_meta = get_data(sim_path)

    X, y = prepare_ts(df)

    (nowcasting_error_incremental_list,
     rmse_incremental,
     Y_real_nowcast,
     Y_nowcasting_list) = get_arima_incremental_results(X, y)

    (arima_nowcasting_predictions,
     arima_nowcasting_error_list) = get_arima_results(y)

    regrets, coef_angular = calcular_regret(arima_nowcasting_error_list,
                                            nowcasting_error_incremental_list[9:])

    dic = {
        "y_real_nowcast": Y_real_nowcast[9:],
        "y_pred_incremental_nowcast": Y_nowcasting_list[9:-1],
        "rmse_incremental_nowcast": rmse_incremental[9:],
        "erro_incremental_nowcast": nowcasting_error_incremental_list[9:],
        "y_pred_arima_nowcast": arima_nowcasting_predictions[9:],
        "erro_arima_nowcast": arima_nowcasting_error_list,
        "regrets": regrets,

           }

    df_nowcast = pd.DataFrame(dic)

    df_nowcast.to_csv(sim_path + "sim_results_drift_recor.csv")

    return df_nowcast


def main():
    iteracoes_com_erro = []  # Lista para guardar as iterações que falharam

    for i in range(800, 1000):
        try:
            print("Começo iteração: " + str(i))
            _ = simulate(i)
            print("Fim iteração: " + str(i))
        except Exception as e:  # Captura qualquer exceção
            print(f"Erro na iteração: {i}")
            print(f"Erro: {e}")
            iteracoes_com_erro.append(i)  # Adiciona o índice da iteração à lista
            continue  # Pula para a próxima iteração

    # No final do loop, verifica se há iterações com erro e salva em um CSV
    if iteracoes_com_erro:
        df_iteracoes_com_erro = pd.DataFrame(iteracoes_com_erro, columns=['Iterações com Erro'])
        df_iteracoes_com_erro.to_csv('iteracoes_com_erro_no_drift.csv', index=False)
        print("Iterações com erro salvas em 'iteracoes_com_erro.csv'.")


if __name__ == "__main__":
    main()
