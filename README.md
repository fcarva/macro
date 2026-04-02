# Romer Study

Projeto de estudo para `Advanced Macroeconomics` (Romer, 5a ed.) com foco duplo:

- dominar os modelos formais;
- implementar cada modelo em Python com visualizacao e extensoes empiricas.

## Modulos implementados

- `ch01_solow/ch01_solow.py`: nucleo numerico do modelo de Solow.
- `ch01_solow/ch01_solow_plots.py`: 5 visualizacoes obrigatorias do modulo.
- `ch01_solow/ch01_solow_empirics.py`: extensao empirica Brasil-first com comparacao internacional.
- `ch02_rck_diamond/ch02_rck.py`: nucleo numerico do modelo RCK.
- `ch02_rck_diamond/ch02_rck_plots.py`: diagrama de fase, comparacoes e choques.
- `ch02_rck_diamond/ch02_rck_empirics.py`: calibracao brasileira de `rho`.
- `params.py`: calibracoes compartilhadas.

## Fontes de dados

Hierarquia planejada:

1. Brasil real: `IBGE -> IPEA -> BCB`.
2. Brasil monetario e financeiro: `BCB`.
3. Internacional: `PWT -> World Bank`.

Implementacao atual da v1:

- bloco Brasil em `Solow` com `IBGE SIDRA` (`CNA 6784`, `CNT 1846`, `CNT 1620`) e referencia anual da `SCN tab05`;
- bloco Brasil em `RCK` com `IBGE SIDRA/SCN` para consumo e populacao, e `BCB SGS` para Selic e IPCA;
- `World Bank API` apenas no bloco internacional do `Solow`;
- `rbcb` documentado como referencia principal para uma ponte futura com os web services do BCB, sem ser dependencia obrigatoria deste ambiente.

## Como rodar

Executar os scripts diretamente a partir da raiz do projeto:

```bash
python ch01_solow/ch01_solow_plots.py
python ch01_solow/ch01_solow_empirics.py
python ch02_rck_diamond/ch02_rck_plots.py
python ch02_rck_diamond/ch02_rck_empirics.py
python -m unittest discover -s tests
```

## Saidas

- figuras em `ch01_solow/figures/` e `ch02_rck_diamond/figures/`;
- arquivos empiricos e `metadata` em `ch01_solow/empirical_outputs/` e `ch02_rck_diamond/empirical_outputs/`;
- `brazil_official_series.csv` com o painel tidy usado em cada modulo;
- `brazil_validation_residuals.csv` com a comparacao entre a soma anualizada da CNT e a referencia anual oficial da SCN.
