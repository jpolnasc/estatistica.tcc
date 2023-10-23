library("tidyverse")

convert_month <- function(date) {
  date <- gsub("jan", "Jan", date)
  date <- gsub("fev", "Feb", date)
  date <- gsub("mar", "Mar", date)
  date <- gsub("abr", "Apr", date)
  date <- gsub("mai", "May", date)
  date <- gsub("jun", "Jun", date)
  date <- gsub("jul", "Jul", date)
  date <- gsub("ago", "Aug", date)
  date <- gsub("set", "Sep", date)
  date <- gsub("out", "Oct", date)
  date <- gsub("nov", "Nov", date)
  date <- gsub("dez", "Dec", date)
  return(date)
}

my_locale <- locale(decimal_mark = ",")
# Lê o arquivo CSV com a configuração de locale definida
df <- read_csv("/workspaces/estatistica.tcc/data/ebsp.csv", locale = my_locale)
# Aplica a função convert_month ao dataframe
df$t <- sapply(df$t, convert_month)
df$t <- as.Date(df$t, format="%b. %d, %Y", tz="UTC")

write_csv(df, "/workspaces/estatistica.tcc/data/ebsp_clean.csv")