# Artificial Intelligence and Sectoral Wages in European Countries: The Moderating Role of ICT Capacity

Master's thesis repository — data processing, panel regression models, and a complementary
text-mining analysis of the EU AI Act.

## Thesis information

- **University:** University of Warsaw, Faculty of Economic Sciences
- **Author:** Yuliya Martyniuk (474075)
- **Degree:** Master's thesis
- **Field of study:** Data Science and Business Analytics
- **Supervisors:** Ph.D. Grzegorz Wesołowski and Ph.D. Mehmet Burak Turgut — Department of
  Macroeconomics and International Trade Theory, WNE UW
- **Date:** Warsaw, September 2026

## Summary

This thesis examines the association between enterprise Artificial Intelligence (AI) adoption and
average sectoral real wages across the European countries, and assesses whether this relationship
varies with pre-existing Information and Communication Technology (ICT) capacity. Using an
unbalanced panel of 25 European countries and eleven sectors from 2021 to 2025, the study applies a
Correlated Random Effects (CRE) Mundlak specification to distinguish between cross-sectional
country-sector differences and year-to-year changes within entities. Country-sector entities with
higher average AI adoption tend to report higher average real wages. However, this positive
cross-sectional association becomes smaller as the share of enterprises employing ICT specialists
increases, contrary to the expected moderating role of ICT capacity. No statistically detectable
relationship is found between year-to-year changes in AI adoption and wage changes within
country-sector entities during the observed period. A complementary textual analysis of the EU AI
Act indicates that the Regulation gives greater attention to technical AI requirements and deployer
competencies than to sector-specific workforce digital-upskilling requirements. Overall, the study
provides evidence that the link between AI adoption and wages varies with ICT capacity across
European country-sector entities, while longer and more granular data are needed to examine the
underlying mechanisms and causal relationships.

## Repository structure

```
Master_thesis/
├── ai_regulation/                  # EU AI Act text corpus and its scraper
│   ├── output/                     # 340 scraped .txt files (recitals, articles, annexes)
│   └── regulation_scraper.py       # scrapes Regulation (EU) 2024/1689 from EUR-Lex
├── data_panel/                     # cleaned panel CSVs (per-variable + master panels)
├── model_functions/                # wage-model functions used by model_estimation.ipynb
│   ├── cre_additive_full.py        #   CRE/Mundlak, additive, with FSI & productivity
│   ├── cre_interaction_full.py     #   CRE/Mundlak, interaction, with FSI & productivity
│   ├── cre_additive_3y.py          #   CRE/Mundlak, additive, without FSI & productivity
│   ├── cre_interaction_3y.py       #   CRE/Mundlak, interaction, without FSI & productivity
│   ├── cre_additive_geo.py         #   CRE/Mundlak, additive, with country fixed effects
│   ├── cre_interaction_geo.py      #   CRE/Mundlak, interaction, with country fixed effects
│   ├── fe_additive.py              #   Two-way FE, additive
│   └── fe_interaction.py           #   Two-way FE, interaction
├── data_preprocessing_panel.ipynb  # downloads & cleans Eurostat data into the panel CSVs
├── EDA.ipynb                       # exploratory data analysis
├── model_estimation.ipynb          # estimates the wage models & robustness checks
├── regulation_txt_mining.ipynb     # text mining of the EU AI Act (TF-IDF + clustering)
├── .gitignore
└── README.md
```

## Data sources

- **Eurostat** — enterprise ICT usage, AI adoption, wages, productivity, firm size, education,
  employment, unemployment, inflation and GDP per capita (compiled in `data_panel/`).
- **EUR-Lex** — Regulation (EU) 2024/1689 (the EU AI Act), scraped into `ai_regulation/output/`.

The main model inputs are `data_panel/full_panel_master.csv` (with macroeconomic controls) and
`data_panel/panel_master.csv` (core variables).

## Analysis pipeline

Run the notebooks in this order:

1. `data_preprocessing_panel.ipynb` — downloads and cleans the Eurostat indicators and builds the
   country × sector × year panel (writes `data_panel/`).
2. `EDA.ipynb` — exploratory analysis of the panel.
3. `model_estimation.ipynb` — estimates the CRE/Mundlak and two-way fixed-effects wage models and
   the robustness checks (sector heterogeneity splits, marginal effects, country fixed-effects
   variants). It imports the specification functions from `model_functions/`.
4. `regulation_txt_mining.ipynb` — text-mines the EU AI Act (labour-market term coverage,
   obligation-sentence extraction, and TF-IDF clustering of "training" sentences).

## Dependencies

The analysis uses Python with, in particular:

- `pandas`, `numpy`
- `statsmodels`, `linearmodels` (panel regressions)
- `scikit-learn` (TF-IDF, K-means, PCA)
- `matplotlib` (figures)
- `eurostat` (data download)
- `selenium` (regulation scraper)

