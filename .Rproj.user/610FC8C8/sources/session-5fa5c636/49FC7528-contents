#install.packages(c("shiny", "bslib", "quantmod", "tidyquant", "dplyr", "tidyr", "tibble", "httr", "rvest", "purrr", "stringr", "quadprog", "tidyverse", "PerformanceAnalytics", "timetk", "lubridate", "gt", "yfR"))

library(shiny)
library(tidyquant)
library(dplyr)
library(gt)
library(yfR)
library(rvest)
library(stringr)
library(tibble)
library(tidyr)
library(purrr)
library(quadprog)
library(PerformanceAnalytics)
library(timetk)
library(lubridate)
library(bslib)

# Função para coletar dados fundamentalistas do Fundamentus
get_fundamentus_data <- function() {
  url <- "https://www.fundamentus.com.br/resultado.php"
  httr::GET(url, httr::set_cookies(`flag` = "true")) %>%
    read_html() %>%
    html_table(fill = TRUE) %>%
    .[[1]] %>%
    janitor::clean_names() %>%
    mutate(ticker = str_replace(ticker, "\\d+", ""),
           pl = as.numeric(str_replace(pl, ",", ".")),
           roe = as.numeric(str_replace(roe, ",", ".")),
           dy = as.numeric(str_replace(dy, ",", ".")),
           div_br_patrim = as.numeric(str_replace(div_br_patrim, ",", ".")),
           liquidez_media_diaria = as.numeric(gsub("\\.", "", liquidez_media_diaria)),
           papel = paste0(ticker, ".SA")) %>%
    filter(!is.na(roe), !is.na(dy)) %>%
    select(papel, setor, pl, roe, dy, div_br_patrim, liquidez_media_diaria)
}

# Selecionar automaticamente as melhores ações
seleciona_acoes <- function(df, n_setores = 8, n_por_setor = 2) {
  df %>%
    filter(liquidez_media_diaria > 1000000, pl > 0, dy > 0) %>%
    group_by(setor) %>%
    slice_max(order_by = dy + roe, n = n_por_setor, with_ties = FALSE) %>%
    ungroup() %>%
    slice_head(n = n_setores * n_por_setor)
}

# Função para obter retornos diários dos ativos selecionados
get_retorno_diario <- function(tickers, first_date, last_date) {
  yf_get(tickers = tickers, first_date = first_date, last_date = last_date) %>%
    select(ref_date, ticker, price_close) %>%
    group_by(ticker) %>%
    arrange(ref_date) %>%
    mutate(ret = price_close / lag(price_close) - 1) %>%
    filter(!is.na(ret)) %>%
    ungroup()
}

# Otimização de portfólio
otimiza_portfolio <- function(retornos) {
  returns_wide <- retornos %>%
    select(ref_date, ticker, ret) %>%
    pivot_wider(names_from = ticker, values_from = ret) %>%
    drop_na()
  
  R <- as.matrix(returns_wide %>% select(-ref_date))
  mu <- colMeans(R)
  Sigma <- cov(R)
  n <- ncol(R)
  
  Dmat <- 2 * Sigma
  dvec <- rep(0, n)
  Amat <- cbind(rep(1, n), diag(n))
  bvec <- c(1, rep(0, n))
  meq <- 1
  
  sol <- solve.QP(Dmat, dvec, Amat, bvec, meq)
  weights <- sol$solution
  data.frame(Ação = colnames(R), Peso = round(weights, 4))
}

# Interface Shiny
ui <- fluidPage(
  theme = bs_theme(bootswatch = "sandstone"),
  titlePanel("Carteira Otimizada com Seleção Automática de Ações"),
  sidebarLayout(
    sidebarPanel(
      numericInput("periodo", "Período (anos):", value = 5, min = 1, max = 10),
      actionButton("gerar", "Gerar Carteira")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Carteira", gt_output("tabelaPesos")),
        tabPanel("Curva de Retorno", plotOutput("plotRetorno"))
      )
    )
  )
)

# Servidor Shiny
server <- function(input, output) {
  dados_reativos <- eventReactive(input$gerar, {
    tryCatch({
      fundamentus <- get_fundamentus_data()
      selecionadas <- seleciona_acoes(fundamentus)
      tickers <- selecionadas$papel
      first_date <- Sys.Date() - years(input$periodo)
      last_date <- Sys.Date()
      retornos <- get_retorno_diario(tickers, first_date, last_date)
      pesos <- otimiza_portfolio(retornos)
      list(pesos = pesos, retornos = retornos)
    }, error = function(e) {
      showNotification(paste("Erro:", e$message), type = "error")
      NULL
    })
  })
  
  output$tabelaPesos <- render_gt({
    req(dados_reativos())
    dados_reativos()$pesos %>%
      gt() %>%
      fmt_percent(columns = "Peso", decimals = 2)
  })
  
  output$plotRetorno <- renderPlot({
    req(dados_reativos())
    retornos <- dados_reativos()$retornos
    pesos <- dados_reativos()$pesos
    retorno_completo <- retornos %>%
      pivot_wider(names_from = ticker, values_from = ret) %>%
      drop_na()
    
    pesos_vetor <- pesos$Peso
    R <- as.matrix(retorno_completo %>% select(-ref_date))
    retorno_portfolio <- R %*% pesos_vetor
    
    xts_portfolio <- xts::xts(retorno_portfolio, order.by = retorno_completo$ref_date)
    chart.CumReturns(xts_portfolio, main = "Retorno Acumulado da Carteira", wealth.index = TRUE)
  })
}

shinyApp(ui = ui, server = server)
