library("tidyverse")
library("simts")

# Set seed for reproducibility
set.seed(1337)

# Number of observations
#n = 100

#sim_arima = gen_gts(n, ARIMA(ar = 1, i = 1, ma = 1, sigma2 = 1))

#df = sim_arima %>% as_tibble()
#df = df %>% mutate(t = seq(1, nrow(df)))
#df %>% ggplot() + geom_line(aes(x = t, y = Observed))