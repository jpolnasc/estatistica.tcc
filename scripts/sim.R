library("tidyverse")
library("simts")
library("forecast")

# Set seed for reproducibility
set.seed(1337)

df = read_csv("/workspaces/estatistica.tcc/data/arima_parameters.csv") %>%
    mutate(
        across(everything(), replace_na, 0)
    )

sim_num = 0

while (sim_num <= 1000) {

    sampled_row <- df %>% sample_n(1)

    ### Verificar Colunas zeradas 

    colunas_a_verificar <- c("ar1", "ar2", "ar3", "ar4")
    todas_zeros <- all(apply(sampled_row[colunas_a_verificar], 2, function(x) all(x == 0)))

    if(todas_zeros == TRUE){
        next
    } else{

    #### Paths ####
    sim_base_path = paste0("/workspaces/estatistica.tcc/data/simulations/", as.character(sim_num),"/")
    # Verifica se o diretório existe
    if (!dir.exists(sim_base_path)) {
    # Cria o diretório caso não exista
    dir.create(sim_base_path, recursive = TRUE)
    }

    sim_df_name = "sim_df.csv"
    sim_df_path = paste0(sim_base_path,sim_df_name)
    metadata_df_name = "meta.csv"
    metadata_path = paste0(sim_base_path, metadata_df_name)

    #### Extração dos coeficientes ####

    ar_coeffs <- sampled_row %>% dplyr::select(starts_with("ar")) %>% unlist() %>% as.numeric()
    ma_coeffs <- sampled_row %>% dplyr::select(starts_with("ma")) %>% unlist() %>% as.numeric()
    sigma2_val <- sampled_row$sigma2 %>% as.numeric()
    diff_order <- sampled_row$order %>% as.numeric()

    #### Geração dos dados

    N <- 1000
    sim_data <- gen_arima(N, ar_coeffs, diff_order, ma_coeffs, sigma2_val)
    df_sim = sim_data %>% as_tibble()
    df_sim = df_sim %>% mutate(t = seq(1, nrow(df_sim)))

    print(df_sim)

    write.csv(df_sim, sim_df_path, row.names = FALSE)
    write.csv(sampled_row, metadata_path, row.names = FALSE)
    sim_num = sim_num + 1
    }
        }