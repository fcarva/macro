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

- `BCB SGS` direto para juros e inflacao no modulo RCK;
- `World Bank API` para a camada real usada nos scripts empiricos atuais;
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
- arquivos empiricos e `metadata` em `ch01_solow/empirical_outputs/` e `ch02_rck_diamond/empirical_outputs/`.
