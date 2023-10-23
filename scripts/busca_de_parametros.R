library("tidyverse")
library("forecast")

# Função para extrair parâmetros ARIMA de janelas de dados
extract_arima_params <- function(data, window_size = 365, step_size = 30, N = nrow(data)) {
  
  # Certifique-se de que N não é maior que o número de linhas no dataframe
  N <- min(N, nrow(data))
  
  # Se N for especificado, considere apenas os primeiros N pontos de dados
  data <- data[1:N, ]
  
  # Inicializar uma lista para armazenar os data frames de parâmetros
  param_list <- list()
  
  # Inicializar valores para rastrear o máximo p, d, e q
  max_p <- 0
  max_q <- 0
  
  # Loop através da série temporal
  for(start in seq(1, nrow(data) - window_size, by = step_size)) {
    end <- start + window_size - 1
    
    # Certificar de que a janela não exceda o limite de N
    end <- min(end, N)
    
    # Extrair a janela de dados
    window_data <- data$price[start:end]
    
    # Ajustar o modelo ARIMA
    arima_model <- auto.arima(window_data)
    
    # Verificar se há coeficientes
    if(length(arima_model$coef) == 0) {
      next  # Pula para a próxima iteração se não houver coeficientes
    } else {
      # Extrair os coeficientes em um tibble
      ar_params <- enframe(arima_model$coef, name = "coef_name", value = "coef_value")
      # Adicionando a Variancia do termo aleatorio
      sigma2 <- var(residuals(arima_model))
      # Adicionando as datas de início e fim
      start_date = data$t[start]
      end_date = data$t[end]
      # Transformando o formato longo para largo
      ar_params <- ar_params %>% 
          pivot_wider(names_from = coef_name, values_from = coef_value) %>%
          mutate(across(where(is.numeric), ~ round(.x, digits = 2)))  # Ajustado para a nova sintaxe
      ar_params$sigma2 = sigma2
      ar_params$start_date = start_date
      ar_params$end_date = end_date
      # Adicionar os parâmetros à lista
      param_list[[length(param_list) + 1]] <- ar_params
      
      # Atualizar os valores máximos de p e q
    arima_order <- arimaorder(arima_model)
    max_p <- max(max_p, arima_order[1])  # p é o primeiro elemento de arima_order
    max_q <- max(max_q, arima_order[3])  
    }
  }
  
  # Criar o data frame final com todas as colunas necessárias
  print(max_p)
  print(max_q)
  all_cols <- c(paste0("ar", seq_len(max_p)), paste0("ma", seq_len(max_q)), "sigma2", "start_date", "end_date")
  param_df <- bind_rows(param_list) %>%
    select(all_of(all_cols))
  
  return(param_df)
}

df <- read_csv("/workspaces/estatistica.tcc/data/ebsp_clean.csv")
df$t <- as.Date(df$t, format="%b. %d, %Y", tz="UTC")
arima_params <- extract_arima_params(df, window_size = 365, step_size = 30)

# Exportar os parâmetros para um arquivo CSV
write.csv(arima_params, "/workspaces/estatistica.tcc/data/arima_parameters.csv", row.names = FALSE)