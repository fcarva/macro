# Romer Study

Projeto de estudo de `Advanced Macroeconomics` (Romer, 5a ed.) com dois objetivos:

- dominar a derivacao formal dos modelos;
- implementar os modelos em Python com visualizacao e extensoes empiricas.

O repositorio esta organizado por capitulo. Hoje, os capitulos efetivamente implementados e documentados sao:

- Capitulo 1: Solow
- Capitulo 2: Ramsey-Cass-Koopmans (RCK)
- Capitulo 2: notas teoricas do modelo de Diamond

## Inicio rapido

Da raiz do projeto:

```bash
python ch01_solow/ch01_solow_plots.py
python ch01_solow/ch01_solow_empirics.py
python ch02_rck_diamond/ch02_rck_plots.py
python ch02_rck_diamond/ch02_rck_empirics.py
python -m unittest discover -s tests
```

Para recompilar as notas em LaTeX:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_derivations.ps1
```

## Padrao visual dos graficos

Os graficos dos capitulos 1 e 2 seguem agora um padrao editorial claro, inspirado
na linguagem de explicacao visual do Nexo Jornal e apoiado na paleta
`Flexoki light`.

Convencoes principais:

- titulos e subtitulos em portugues;
- rodape com `Fonte:` e, quando necessario, `Obs.:`;
- unidades jornalisticas em pt-BR, como `R$ bi`, `R$ tri`, `%` e `p.p.`;
- exportacao dupla para cada figura: `PNG` e `SVG`;
- graficos teoricos mantem notacao formal (`k`, `c`, `y`, `\dot{k}`, `\dot{c}`), mas com textos de apoio legiveis.

## Navegacao rapida

Arquivos centrais do projeto:

- [Plano geral](./romer_study_plan.md)
- [Parametros compartilhados](./params.py)
- [Estilo grafico compartilhado](./plotting_style.py)
- [Utilitarios de dados](./data_utils.py)
- [Dependencias](./requirements.txt)
- [Build das notas LaTeX](./build_derivations.ps1)
- [Guia de derivacoes ch. 1 e 2](./notes/guia_derivacoes_ch01_ch02.pdf)

Capitulo 1, Solow:

- [Modelo](./ch01_solow/ch01_solow.py)
- [Plots](./ch01_solow/ch01_solow_plots.py)
- [Empiria](./ch01_solow/ch01_solow_empirics.py)
- [Notas em LaTeX](./ch01_solow/notes/ch01_solow_derivations.tex)
- [Notas em PDF](./ch01_solow/notes/ch01_solow_derivations.pdf)
- [Figuras](./ch01_solow/figures/)
- [Outputs empiricos](./ch01_solow/empirical_outputs/)

Capitulo 2, RCK e Diamond:

- [Modelo RCK](./ch02_rck_diamond/ch02_rck.py)
- [Plots RCK](./ch02_rck_diamond/ch02_rck_plots.py)
- [Empiria RCK](./ch02_rck_diamond/ch02_rck_empirics.py)
- [Notas em LaTeX](./ch02_rck_diamond/notes/ch02_rck_diamond_derivations.tex)
- [Notas em PDF](./ch02_rck_diamond/notes/ch02_rck_diamond_derivations.pdf)
- [Figuras](./ch02_rck_diamond/figures/)
- [Outputs empiricos](./ch02_rck_diamond/empirical_outputs/)

Guia transversal de estudo:

- [Fonte LaTeX do guia](./notes/guia_derivacoes_ch01_ch02.tex)
- [PDF do guia](./notes/guia_derivacoes_ch01_ch02.pdf)

Testes:

- [Testes do Solow](./tests/test_solow.py)
- [Testes do RCK](./tests/test_rck.py)
- [Helpers empiricos](./tests/test_empirical_helpers.py)

## Estrutura do repositorio

```text
romer-study/
├── ch01_solow/
├── ch02_rck_diamond/
├── ch03_endogenous_growth/
├── ch04_cross_country/
├── ch05_rbc/
├── ch06_nominal_rigidity/
├── ch07_dsge_nk/
├── ch08_consumption/
├── ch09_investment/
├── ch10_financial/
├── ch11_unemployment/
├── ch12_monetary/
├── ch13_fiscal/
├── data/
├── tests/
├── build_derivations.ps1
├── data_utils.py
├── params.py
├── requirements.txt
└── romer_study_plan.md
```

## O que ja esta pronto

### Capitulo 1, Solow

Implementado em [ch01_solow.py](./ch01_solow/ch01_solow.py):

- funcao de producao Cobb-Douglas em forma intensiva;
- `steady_state()`;
- `transition_path()`;
- `golden_rule()`;
- `growth_accounting()`.

Visualizacoes em [ch01_solow_plots.py](./ch01_solow/ch01_solow_plots.py):

- diagrama classico do Solow;
- diagrama de fase unidimensional;
- trajetoria de transicao de `k(t)`, `y(t)` e `c(t)`;
- choque na taxa de poupanca;
- comparacao com a Regra de Ouro.

Camada empirica em [ch01_solow_empirics.py](./ch01_solow/ch01_solow_empirics.py):

- bloco Brasil com `IBGE SIDRA`, `SCN` e `BCB` quando aplicavel;
- comparacao internacional com `World Bank` no bloco cross-country;
- growth accounting para o Brasil;
- painel tidy `brazil_official_series.csv`;
- validacao `CNT vs SCN` em `brazil_validation_residuals.csv`.

Arquivos mais uteis do modulo:

- [Growth accounting do Brasil](./ch01_solow/empirical_outputs/brazil_growth_accounting.csv)
- [Painel de convergencia](./ch01_solow/empirical_outputs/convergence_panel.csv)
- [Metadados da empiria](./ch01_solow/empirical_outputs/solow_empirics_metadata.json)

### Capitulo 2, RCK e Diamond

Implementado em [ch02_rck.py](./ch02_rck_diamond/ch02_rck.py):

- sistema dinamico continuo em `(k, c)`;
- `steady_state()` analitico e validacao numerica;
- `simulate()`;
- shooting algorithm em `find_saddle_path()`;
- geracao de dados para diagrama de fase.

Visualizacoes em [ch02_rck_plots.py](./ch02_rck_diamond/ch02_rck_plots.py):

- diagrama de fase com isoclinas;
- comparacao Solow vs RCK;
- choque em `rho`;
- deslocamento por `G`;
- trajetorias de `c(t)` e `k(t)`.

Camada empirica em [ch02_rck_empirics.py](./ch02_rck_diamond/ch02_rck_empirics.py):

- consumo real per capita com base oficial brasileira;
- proxy operacional de juros reais via `BCB SGS`;
- calibracao brasileira de `rho`;
- painel tidy `brazil_official_series.csv`;
- residual de validacao anual.

Arquivos mais uteis do modulo:

- [Painel de calibracao do RCK](./ch02_rck_diamond/empirical_outputs/rck_brazil_calibration_panel.csv)
- [Metadados da empiria do RCK](./ch02_rck_diamond/empirical_outputs/rck_empirics_metadata.json)

Observacao:

- o Diamond ainda nao esta implementado em Python nesta fase;
- ele aparece apenas nas notas teoricas do Capitulo 2.

## Notas de derivacao

As notas em LaTeX agora tem tres camadas complementares:

- uma nota completa do Capitulo 1;
- uma nota completa do Capitulo 2;
- um guia transversal curto, pensado para revisar o esqueleto das derivacoes sem reler tudo.

Capitulo 1:

- [Fonte LaTeX do Solow](./ch01_solow/notes/ch01_solow_derivations.tex)
- [PDF do Solow](./ch01_solow/notes/ch01_solow_derivations.pdf)

Capitulo 2:

- [Fonte LaTeX do RCK e Diamond](./ch02_rck_diamond/notes/ch02_rck_diamond_derivations.tex)
- [PDF do RCK e Diamond](./ch02_rck_diamond/notes/ch02_rck_diamond_derivations.pdf)

Guia complementar:

- [Fonte LaTeX do guia de derivacoes](./notes/guia_derivacoes_ch01_ch02.tex)
- [PDF do guia de derivacoes](./notes/guia_derivacoes_ch01_ch02.pdf)

Cobertura atual das notas:

- Capitulo 1: forma intensiva, dinamica de `k`, steady state, crescimento balanceado, Cobb-Douglas, Regra de Ouro, convergencia local e growth accounting.
- Capitulo 2A: normalizacao do problema, Hamiltoniano, FOCs, equacao de Euler, isoclinas, steady state, linearizacao local, saddle path, TVC e gasto do governo.
- Capitulo 2B: problema de duas idades, regra de poupanca, mapa `k_{t+1}(k_t)`, steady state, Regra de Ouro e ineficiencia dinamica.
- Guia complementar: mapa de notacao, templates de derivacao e erros comuns dos capitulos 1 e 2.

## Fontes de dados

Hierarquia planejada:

1. Brasil real: `IBGE -> IPEA -> BCB`
2. Brasil monetario e financeiro: `BCB`
3. Internacional: `PWT -> World Bank`

Implementacao atual:

- Solow Brasil: `IBGE SIDRA` com apoio de `SCN` anual e validacao `CNT`;
- RCK Brasil: `IBGE SIDRA/SCN` para consumo e populacao, `BCB SGS` para Selic e IPCA;
- Internacional: `World Bank API` apenas no bloco comparativo do Solow;
- `rbcb` fica documentado como referencia principal para uma futura ponte em R, mas nao e dependencia obrigatoria agora.

## Outputs principais

Capitulo 1:

- [Figuras do Solow](./ch01_solow/figures/)
- [Outputs empiricos do Solow](./ch01_solow/empirical_outputs/)
- [Painel brasileiro oficial](./ch01_solow/empirical_outputs/brazil_official_series.csv)
- [Validacao CNT vs SCN](./ch01_solow/empirical_outputs/brazil_validation_residuals.csv)

Capitulo 2:

- [Figuras do RCK](./ch02_rck_diamond/figures/)
- [Outputs empiricos do RCK](./ch02_rck_diamond/empirical_outputs/)
- [Painel brasileiro oficial](./ch02_rck_diamond/empirical_outputs/brazil_official_series.csv)
- [Validacao CNT vs SCN](./ch02_rck_diamond/empirical_outputs/brazil_validation_residuals.csv)

Observacao:

- em `figures/` e `empirical_outputs/`, os graficos passam a existir em `PNG` e `SVG`, com o mesmo nome-base.

## Testes e validacao

Rodar todos os testes:

```bash
python -m unittest discover -s tests
```

Escopo atual dos testes:

- consistencia do steady state do Solow;
- Regra de Ouro do Solow;
- convergencia das trajetorias do Solow;
- steady state e shooting do RCK;
- helpers da camada empirica e validacoes de agregacao.

## Estado atual do projeto

Ja implementado:

- Capitulo 1 em codigo, plots, empiria e notas;
- Capitulo 2A em codigo, plots, empiria e notas;
- Capitulo 2B em notas teoricas.

Ainda nao implementado:

- `ch02_diamond.py`
- Capitulo 3 em diante

## Proximos passos naturais

- implementar o modelo de Diamond em Python;
- adicionar notebooks leves por modulo;
- expandir a camada brasileira com trabalho via PNAD/PNAD Continua;
- usar NTN-B/ETTJ como proxy mais estrutural de juros reais no RCK;
- avancar para o Capitulo 3 de crescimento endogeno.
