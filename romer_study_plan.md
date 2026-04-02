# Plano de Estudo: Advanced Macroeconomics - Romer (5a Ed.)
## Macro com Dados e Visualizacao

> **Objetivo duplo:** dominar os modelos formais do Romer e desenvolver a capacidade de implementa-los em Python com visualizacao e dados reais. Cada capitulo termina com um notebook funcional.

---

## Instrucoes para o Agente

Voce e um tutor de macroeconomia avancada e programacao cientifica. Ao trabalhar neste plano:

1. **Por capitulo**, seguir sempre a sequencia: (1) revisao teorica -> (2) implementacao do modelo em Python -> (3) visualizacao dos diagramas canonicos -> (4) extensao com dados reais.
2. **Stack padrao:** Python, com `numpy`, `scipy`, `matplotlib`, `pandas`. Para dados: `pandas-datareader` ou `wbdata` (World Bank API) ou `ipeadatapy` (dados brasileiros via IPEA).
3. **Convencoes de codigo:** cada modelo deve ser encapsulado em uma classe ou conjunto de funcoes documentadas. Os parametros devem ser configuraveis via dicionario `params = {...}`.
4. **Outputs esperados por modulo:** um arquivo `.py` (o modelo) + um arquivo `.ipynb` ou script de plotagem. Nomes no padrao `ch01_solow.py`, `ch01_solow_plots.py`.
5. **Ao iniciar um modulo**, perguntar: quais parametros o usuario quer explorar? Qual extensao empirica interessa (dados BR, dados Penn World Tables, dados FRED)?
6. **Diagrama de fase e obrigatorio** para qualquer modelo com dinamica (Solow, RCK, Diamond, RBC). Usar `matplotlib` com `quiver` ou `streamplot` para o campo vetorial.

---

## Estrutura Geral

```text
romer-study/
|- romer_study_plan.md
|- data/
|- ch01_solow/
|  |- ch01_solow.py
|  |- ch01_solow_plots.py
|  `- ch01_solow_empirics.py
|- ch02_rck_diamond/
|  |- ch02_rck.py
|  |- ch02_rck_plots.py
|  `- ch02_diamond.py
|- ch03_endogenous_growth/
|- ch04_cross_country/
|- ch05_rbc/
|- ch06_nominal_rigidity/
|- ch07_dsge_nk/
|- ch08_consumption/
|- ch09_investment/
|- ch10_financial/
|- ch11_unemployment/
|- ch12_monetary/
`- ch13_fiscal/
```

---

## Modulo 1 - Capitulo 1: The Solow Growth Model

**Referencia:** Romer Cap. 1 (pp. 6-49)

### Teoria a revisar

- Funcao de producao com retornos constantes: `Y = F(K, AL)`
- Forma intensiva: `y = f(k)`, onde `k = K/AL`
- Equacao de acumulacao de capital: `k_dot = s * f(k) - (n + g + delta) * k`
- Estado estacionario: condicao `s f(k*) = (n + g + delta) k*`
- Regra de Ouro: `f'(k_gold) = n + g + delta`
- Convergencia condicional e incondicional
- Contabilidade do crescimento (Secao 1.7)

### Implementacao Python - `ch01_solow.py`

```python
# Agente: implemente a classe SolowModel com os seguintes metodos:
# - steady_state() -> calcula k* e y* analiticamente (Cobb-Douglas)
# - transition_path(k0, T) -> simula a trajetoria discreta de k(t)
# - golden_rule() -> calcula k_gold e s_gold
# - growth_accounting(data) -> decompoe crescimento em K, L, A

params = {
    "alpha": 0.33,
    "s": 0.20,
    "n": 0.01,
    "g": 0.02,
    "delta": 0.05,
    "A0": 1.0,
    "L0": 1.0,
}
```

### Visualizacoes obrigatorias

1. **Diagrama de Solow classico:** curvas `s f(k)` e `(n+g+delta) k`, marcar `k*`
2. **Diagrama de fase de `k`:** `k_dot` em funcao de `k`, mostrar convergencia
3. **Trajetoria de transicao:** `k(t)`, `y(t)`, `c(t)` partindo de `k0 != k*`
4. **Efeito de choque em `s`:** shift do estado estacionario + trajetoria
5. **Curva de Ouro:** mostrar `s_gold` vs `s` atual

### Extensao empirica - `ch01_solow_empirics.py`

```python
# Agente: use Penn World Tables (PWT 10.0) ou World Bank API para:
# 1. Plotar Y/L (PIB per trabalhador) para um conjunto de paises ao longo do tempo
# 2. Scatter: taxa de investimento x PIB per capita (teste do Solow simples)
# 3. Exercicio de convergencia: OCDE vs mundo
# 4. Contabilidade do crescimento para o Brasil (1960-2020)
#    usando series do IPEA ou IBGE via ipeadatapy
```

**Questoes para discutir ao terminar o modulo:**

- O modelo Solow responde as perguntas centrais do crescimento? Por que nao?
- O que a contabilidade do crescimento revela sobre o Brasil?
- Como interpretar a convergencia condicional nos dados?

---

## Modulo 2 - Capitulo 2, Parte A: Ramsey-Cass-Koopmans

**Referencia:** Romer Cap. 2, pp. 50-75

### Teoria a revisar

- Familias com horizonte infinito maximizando utilidade intertemporal
- Funcao utilidade CES: `u(c) = c^(1-theta)/(1-theta)`
- Equacao de Euler (Keynes-Ramsey): `c_dot / c = (r - rho) / theta = (f'(k) - delta - rho) / theta`
- Restricao de acumulacao de capital: `k_dot = f(k) - c - (n+g) k`
- Sistema dinamico 2D em `(k, c)`
- Estado estacionario: `k**` onde `f'(k**) = rho + theta g + delta`
- Trajetoria de sela (saddle path)
- Analise comparativa: queda na taxa de desconto `rho`

### Implementacao Python - `ch02_rck.py`

```python
# Agente: implemente o modelo RCK com os seguintes componentes:
# - Sistema de EDOs: dk/dt e dc/dt
# - Steady state (solucao numerica via scipy.optimize.fsolve)
# - Shooting algorithm para encontrar a saddle path
# - Phase diagram com streamplot ou quiver

params_rck = {
    "alpha": 0.33,
    "rho": 0.04,
    "theta": 2.0,
    "n": 0.01,
    "g": 0.02,
    "delta": 0.05,
}
```

### Visualizacoes obrigatorias

1. **Diagrama de fase (k, c):** isoclinas `c_dot=0` e `k_dot=0`, campo vetorial, saddle path
2. **Comparacao RCK vs Solow:** trajetoria de `k(t)` nos dois modelos partindo do mesmo `k0`
3. **Choque em `rho` (fall in discount rate):** nova saddle path + dinamica de transicao
4. **Efeito de gastos do governo `G`:** deslocamento da isoclina `k_dot=0`
5. **Consumo otimo vs Solow:** comparar `c(t)` nos dois modelos

### Extensao empirica

```python
# Agente: calcule a taxa de desconto implicita rho calibrando o RCK
# para o Brasil usando dados do IPEA/IBGE:
# - Taxa real de juros de longo prazo (proxy para r)
# - Crescimento do consumo agregado per capita
# - Estimar theta via equacao de Euler com dados de consumo
```

**Questoes para discutir ao terminar o modulo:**

- Qual a diferenca fundamental entre o RCK e o Solow?
- Por que o saddle path e relevante para a estabilidade do equilibrio?
- O que a condicao de transversalidade significa economicamente?

---

## Modulo 3 - Capitulo 2, Parte B: Diamond (OLG)

**Referencia:** Romer Cap. 2, pp. 76-97

### Teoria a revisar

- Geracoes sobrepostas (OLG): jovens e velhos
- Decisao de poupanca da geracao jovem
- Equacao de dinamica do capital: `k_{t+1} = s(w_t)/(1+n)`
- Possibilidade de ineficiencia dinamica (overaccumulation)
- Papel da divida publica no Diamond

### Implementacao - `ch02_diamond.py`

```python
# Agente: implemente o Diamond OLG com:
# - Mapa de k_{t+1} em funcao de k_t (Cobb-Douglas + log utility)
# - Diagrama de 45 graus mostrando estado(s) estacionario(s)
# - Multiplos equilibrios se existirem
# - Exercicio de ineficiencia dinamica: comparar k* com k_gold

params_diamond = {
    "alpha": 0.33,
    "beta": 0.5,
    "n": 0.01,
    "delta": 1.0,
}
```

### Visualizacoes obrigatorias

1. **Mapa de `k_{t+1}(k_t)`:** linha de 45 graus, identificar estado estacionario
2. **Diagrama de fase discreto:** convergencia a partir de `k0` diferente
3. **Ineficiencia dinamica:** mostrar quando `k* > k_gold`

---

## Modulo 4 - Capitulo 3: Endogenous Growth

**Referencia:** Romer Cap. 3 (pp. 99-147)

### Modelos a implementar

1. **Modelo AK:** `Y = A K`, crescimento endogeno sem estado estacionario de `k`
2. **Modelo sem capital (secao 3.2):** economias de escala no conhecimento
3. **Modelo de Romer (1990):** variedade de bens intermediarios, P&D endogena

```python
# ch03_ak.py -> modelo AK: taxa de crescimento g = s A - delta
# ch03_romer_model.py -> modelo de variedade: simular alocacao de L entre Y e P&D
```

### Visualizacoes

- Comparacao AK vs Solow: trajetorias de `log y(t)` (paralelas vs convergentes)
- Efeito do tamanho da populacao no crescimento (escala)
- Fronteira de possibilidades P&D vs producao

### Extensao empirica

```python
# Teste de crescimento endogeno: correlacao entre P&D/PIB e crescimento
# Dados: OCDE MSTI database ou World Bank R&D data
# Para o Brasil: PINTEC (IBGE) + dados de crescimento do IPEA
```

---

## Modulo 5 - Capitulo 4: Cross-Country Income Differences

**Referencia:** Romer Cap. 4 (pp. 149-186)

### Teoria

- Solow aumentado com capital humano (Mankiw-Romer-Weil 1992)
- Decomposicao de diferencas de renda: capital fisico, capital humano, PTF
- Social infrastructure e diferencas de renda (Hall & Jones 1999)

### Implementacao

```python
# ch04_mrw.py -> replicar Mankiw-Romer-Weil com dados do PWT
# Regressao: log(y) ~ log(sk) + log(sh) + log(n+g+delta)
# Comparar R² com e sem capital humano
```

### Visualizacao

- Scatter: log(PIB/trabalhador) x log(investimento/PIB) para 100+ paises
- Decomposicao contabil de diferencas de renda (barchart)
- Mapa-mundi colorido por PTF relativa

---

## Modulo 6 - Capitulo 5: Real Business Cycle Theory

**Referencia:** Romer Cap. 5 (pp. 188-236)

### Teoria

- Modelo RBC base: preferencias, tecnologia, equilibrio
- Oferta de trabalho intertemporal (labor-leisure)
- Choque de produtividade (TFP shock)
- Calibracao e simulacao

### Implementacao - `ch05_rbc.py`

```python
# Agente: implemente o caso especial do RBC (secao 5.5)
# onde a solucao analitica existe com log-utility e delta=1
# Simule impulse-response functions para um choque de tecnologia

params_rbc = {
    "alpha": 0.33,
    "beta": 0.99,
    "delta": 0.025,
    "rho_z": 0.95,
    "sigma_z": 0.007,
}
```

### Visualizacoes

- **Impulse-Response Functions (IRFs):** `Y`, `C`, `I`, `L` apos choque de TFP
- **Simulacao estocastica:** serie temporal simulada vs dados do PIB BR
- **Diagrama de fase do modelo RBC:** espaco `(k, z)`

### Extensao empirica

```python
# Calibrar o RBC para o Brasil:
# - alpha: participacao do capital na renda (Contas Nacionais IBGE)
# - delta: estimativa de depreciacao do capital (IBGE)
# - Filtro HP sobre PIB, consumo, investimento BR para extrair ciclo
# - Comparar momentos simulados vs momentos dos dados
```

---

## Modulo 7 - Capitulos 6-7: Nominal Rigidity + New Keynesian DSGE

**Referencia:** Romer Caps. 6-7 (pp. 238-366)

### Teoria

- Rigidez nominal exogena: IS curve, Phillips curve
- Menu costs (Mankiw), Ball-Romer (real rigidities)
- Modelo Novo-Keynesiano canonico (3 equacoes):
  - IS dinamica: `x_t = E[x_{t+1}] - (1/sigma) (i_t - E[pi_{t+1}] - r_n)`
  - NKPC: `pi_t = beta E[pi_{t+1}] + kappa x_t`
  - Regra de Taylor: `i_t = phi_pi pi_t + phi_x x_t + v_t`

### Implementacao - `ch07_nk_canonical.py`

```python
# Agente: implemente o modelo NK canonico em forma matricial
# Resolva com metodo de coeficientes indeterminados ou metodo de Sims (gensys)
# Plote IRFs para choque de demanda e choque de politica monetaria

params_nk = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.1,
    "phi_pi": 1.5,
    "phi_x": 0.5,
    "rho_v": 0.5,
}
```

### Visualizacoes

- IRFs do modelo NK: `pi`, `x`, `i` apos choque de demanda e oferta
- Grafico de estabilidade: regiao de determinacao de Blanchard-Kahn
- Fronteira eficiente `variancia(pi) x variancia(x)`

### Extensao empirica

```python
# Estimar regra de Taylor para o Brasil (BCB)
# Dados: IPCA (IBGE/IPEA), Selic (BCB), Hiato do produto (FMI/BCB)
```

---

## Modulo 8 - Capitulo 8: Consumption

**Referencia:** Romer Cap. 8 (pp. 368-418)

### Modelos

- Hipotese da renda permanente (Friedman/Hall)
- Random Walk hypothesis (Hall 1978)
- Buffer-stock saving + programacao dinamica

### Implementacao - `ch08_consumption.py`

```python
# 1. Simular a hipotese da renda permanente com renda AR(1)
# 2. Implementar buffer-stock saving via value function iteration
# 3. Replicar teste de Campbell-Mankiw: regressao delta c ~ delta y
```

### Visualizacoes

- Consumo otimo vs renda ao longo do tempo (PIH)
- Policy function `c*(a, y)` do buffer-stock model (surface plot 3D)
- Distribuicao de riqueza simulada

---

## Modulo 9 - Capitulo 9: Investment

**Referencia:** Romer Cap. 9 (pp. 420-456)

### Modelos

- Custo de ajuste e `q` de Tobin
- Equacao dinamica de investimento
- Incerteza e opcoes reais

### Implementacao - `ch09_investment.py`

```python
# Sistema (k, q): diagrama de fase com isoclinas q_dot=0 e k_dot=0
# IRFs para choque de demanda e choque de taxa de juros
# Replicar scatter q x I/K com dados de empresas (Compustat ou CVM Brasil)
```

---

## Modulo 10 - Capitulo 10: Financial Markets & Crises

**Referencia:** Romer Cap. 10 (pp. 458-519)

### Modelos

- Acelerador financeiro (Bernanke-Gertler-Gilchrist)
- Diamond-Dybvig (bank runs)
- Contagio e crises

### Implementacao - `ch10_financial.py`

```python
# Diamond-Dybvig: modelo de 3 periodos, equilibrio de corrida bancaria
# Visualizar os dois equilibrios (bom e ruim)
# Financial accelerator: IRFs ampliadas vs modelo sem friccoes financeiras
```

---

## Modulo 11 - Capitulo 11: Unemployment

**Referencia:** Romer Cap. 11 (pp. 520-576)

### Modelos

- Efficiency wages (Shapiro-Stiglitz)
- Search and matching (Diamond-Mortensen-Pissarides)
- Curva de Beveridge

### Implementacao - `ch11_unemployment.py`

```python
# Shapiro-Stiglitz: diagramas de NSC e ZPC no espaco (w, u)
# DMP: Curva de Beveridge simulada
# Calibrar DMP para o Brasil: dados PME/PNAD (IBGE via sidrapy)
```

---

## Modulo 12 - Capitulo 12: Monetary Policy

**Referencia:** Romer Cap. 12 (pp. 578-658)

### Topicos

- Regra de Taylor e independencia do banco central
- Zero Lower Bound (ZLB)
- Inconsistencia dinamica (Kydland-Prescott)
- Senhoriagem e inflacao

### Implementacao - `ch12_monetary.py`

```python
# Simular ZLB: modelo NK com restricao i >= 0
# Inconsistencia dinamica: jogo entre BC e agentes
# Curva de Laffer da senhoriagem
# Estimar regra de Taylor para o Brasil (2000-2024)
```

---

## Modulo 13 - Capitulo 13: Budget Deficits & Fiscal Policy

**Referencia:** Romer Cap. 13 (pp. 660-714)

### Topicos

- Restricao orcamentaria intertemporal do governo
- Equivalencia Ricardiana
- Tax smoothing (Barro)
- Crises de divida soberana

### Implementacao - `ch13_fiscal.py`

```python
# Calcular sustentabilidade da divida publica brasileira
# Condicao de solvencia: r < g -> dinamica de Ponzi permitida?
# Dados: divida/PIB BCB + taxa real de crescimento IBGE
# Simular trajetoria de divida sob diferentes cenarios de ajuste fiscal
```

---

## Cronograma Sugerido

| Semana | Modulo | Entregavel |
|--------|--------|------------|
| 1-2 | Cap. 1: Solow | `ch01_solow.py` + plots + contabilidade BR |
| 3-4 | Cap. 2A: RCK | `ch02_rck.py` + phase diagram + calibracao |
| 5 | Cap. 2B: Diamond | `ch02_diamond.py` |
| 6-7 | Cap. 3: Endogenous Growth | `ch03_*.py` + teste empirico |
| 8 | Cap. 4: Cross-Country | regressao MRW + decomposicao |
| 9-10 | Cap. 5: RBC | `ch05_rbc.py` + calibracao BR |
| 11-12 | Caps. 6-7: NK DSGE | `ch07_nk_canonical.py` + Taylor BR |
| 13-14 | Cap. 8-9: C e I | policy functions + q de Tobin |
| 15-16 | Caps. 10-13 | Diamond-Dybvig, DMP, fiscal BR |

---

## Recursos e Dependencias Python

```bash
pip install numpy scipy matplotlib pandas pandas-datareader
pip install wbdata ipeadatapy python-bcb sidrapy
pip install jupyter notebook ipywidgets
pip install pwt
pip install linearmodels
```

### Fontes de Dados Principais

| Fonte | Acesso | Variaveis |
|-------|--------|-----------|
| Penn World Tables 10.0 | `pwt` / arquivo Excel direto | Y, K, L, TFP entre paises |
| World Bank | `wbdata` | PIB, capital humano, investimento |
| IPEA Data | `ipeadatapy` | PIB BR, consumo, inflacao, juros |
| BCB (Banco Central) | `python-bcb` | Selic, credito, cambio |
| IBGE SIDRA | `sidrapy` | PNAD, PIBR, IPCA, contas nacionais |
| OCDE | `pandas-datareader` | dados para paises avancados |

---

## Convencao de Output dos Notebooks

Cada modulo deve terminar com um **sumario de 3 paragrafos** respondendo:

1. O que o modelo diz formalmente?
2. O que os dados mostram para o Brasil ou o mundo?
3. Qual a conexao com os debates contemporaneos (crescimento, ciclos, politica)?

---

*Plano criado em Abril de 2026 - PPGEco UFES*  
*Referencia: Romer, David. Advanced Macroeconomics, 5a ed. McGraw-Hill, 2019.*
