# Projeto aplicado — classificação temática de abstracts médicos

## Resumo executivo

Uma instituição hospitalar de referência — cliente fictício deste estudo de
caso — contratou a equipe para desenvolver um protótipo que classifique
abstracts médicos em categorias temáticas. A necessidade de negócio é organizar
publicações científicas para facilitar sua catalogação e recuperação por áreas
de conhecimento. A entrega é uma API de inferência conteinerizada, acompanhada
de retreinamento reproduzível, automação de qualidade e observabilidade local.

O projeto usa o **Medical Abstracts TC Corpus**, recomendado no enunciado, cujos
rótulos correspondem a cinco categorias de doenças. Essa escolha permite
demonstrar classificação multiclasse com dados existentes, mas delimita o
problema: **a solução classifica abstracts científicos por assunto; não
classifica laudos, não determina prioridade assistencial e não prevê urgência**.
O resultado é um protótipo acadêmico, não um produto clínico nem uma implantação
em hospital real.

| Item | Definição do projeto |
| --- | --- |
| Cliente / cenário | Hospital de referência fictício, no contexto do desafio |
| Usuário do protótipo | Equipe que cataloga ou pesquisa literatura médica |
| Entrada | Texto de um abstract científico |
| Saída | Uma das cinco categorias temáticas aprendidas do corpus e confiança do modelo |
| Fora de escopo | Diagnóstico, classificação de paciente/laudo, triagem de urgência e decisão clínica |

## Problema, objetivo e critérios de sucesso

O cliente precisa de uma prova de conceito que receba texto sob demanda e
devolva uma categoria temática de forma consistente, com uma API simples de
integrar. Como a finalidade é consulta interativa, o objetivo técnico é
disponibilizar inferência HTTP de baixa latência e manter separado o ciclo de
treinamento do serviço online.

O aceite do protótipo é verificável por estas demonstrações:

1. Treinar com o CSV do corpus e gerar artefatos scikit-learn e ONNX.
2. Submeter um abstract ao endpoint REST e receber uma categoria do modelo.
3. Executar lint e testes automaticamente em push e pull request.
4. Acionar a DAG para validar a fonte e orquestrar o retreinamento.
5. Observar contagem de requisições, latência p95 e respostas HTTP 5xx no
   dashboard provisionado.

Não foi definido pelo enunciado um limiar de acurácia ou SLA numérico. Portanto,
o trabalho demonstra o ciclo de engenharia e reporta as medições efetivamente
obtidas; não declara desempenho mínimo, adequação clínica ou validação
institucional. Os testes automatizados verificam contratos e integração local,
mas não substituem avaliação do modelo, validação científica ou revisão por
profissionais.

## Entregáveis

| Entregável acordado no estudo de caso | Evidência no projeto |
| --- | --- |
| API de classificação documentada | `triage/api.py`, OpenAPI em `/docs` e teste da API |
| Pipeline de treino e otimização | `triage/training.py`, artefatos `.joblib`/`.onnx` e comparação de latência |
| Ingestão do corpus | `triage/data.py`; CSV baixado localmente pelo usuário |
| Automação de qualidade | `.github/workflows/ci.yml` com Ruff e pytest |
| Orquestração de retreino | `airflow/dags/train_model.py` |
| Stack de observabilidade | `docker-compose.yml`, Prometheus e dashboard Grafana provisionado |
| Decisão técnica e operação | Este README, incluindo arquitetura, limites, comandos e proposta de cloud |
| Demonstração de resultado | Vídeo STAR a ser gravado após executar o projeto com o corpus |

**Estado da entrega:** código, testes e configurações estão no repositório. O
dataset é um insumo externo obtido conforme os termos da fonte. Uma cópia local
foi usada para este experimento, mas não foi adicionada ao Git. A proposta de
cloud é arquitetural, não uma implantação real.

## Dataset recomendado

O **Medical Abstracts TC Corpus** indicado no PDF está disponível no Kaggle e
no repositório original:

- [Página do corpus no Kaggle](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus)
- [Repositório do corpus e referência acadêmica](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)

Para esta execução, os CSVs foram obtidos do repositório original do corpus em
uma revisão fixada, correspondente ao conjunto espelhado no Kaggle. A cópia
local e o arquivo da licença são mantidos fora do Git. Também é possível obter
os arquivos pela página Kaggle. Organize-os localmente assim:

```text
data/medical_abstracts/
├── medical_tc_train.csv
├── medical_tc_test.csv
└── medical_tc_labels.csv
```

O CSV de treino contém `medical_abstract` e `condition_label`; o arquivo de
teste usa o mesmo esquema; o CSV de nomes contém `condition_label` e
`condition_name`. O carregador também aceita CSV genérico com `text` e `label`.
Se o arquivo de nomes estiver junto do CSV de treino com o nome
`medical_tc_labels.csv`, o treino resolve automaticamente os IDs numéricos para
as categorias:

1. Neoplasms
2. Digestive system diseases
3. Nervous system diseases
4. Cardiovascular diseases
5. General pathological conditions

O corpus contém abstracts médicos reais. Respeite os termos, atribuição e
condições de uso da fonte; não publique cópias dos dados nem os use para decisões
assistenciais. O dataset não tem rótulos de urgência, portanto não pode treinar
nem validar o requisito original de triagem `normal/atenção/urgente`. A adaptação
deste estudo de caso é, portanto, uma classificação **temática de abstracts**,
não uma alegação de que o requisito de urgência foi atendido.

### Atribuição e licença da fonte

O repositório original informa licença **Creative Commons Attribution-ShareAlike
3.0 Unported (CC BY-SA 3.0)**. A cópia local utilizada inclui
`data/medical_abstracts/SOURCE_LICENSE.txt`; consulte a licença e confirme os
termos aplicáveis à cópia obtida antes de redistribuir dados ou artefatos
derivados. No momento desta implementação, a API do Kaggle retorna a licença do
espelho como **Unknown**; a indicação de CC BY-SA é a licença declarada pelo
repositório original, não uma confirmação jurídica independente do espelho.

Crédito acadêmico: Tim Schopf, Daniel Braun e Florian Matthes, “Evaluating
Unsupervised Text Classification: Zero-Shot and Similarity-Based Approaches”,
NLPIR '22, 2023. DOI: [10.1145/3582768.3582795](https://doi.org/10.1145/3582768.3582795).

## Arquitetura

```text
Medical Abstracts TC Corpus (download manual; não versionado)
  └── CSV: medical_abstract + condition_label
        └── scikit-learn: TF-IDF + LogisticRegression
              ├── artifacts/model.joblib (referência)
              ├── artifacts/model.onnx (inferência)
              └── artifacts/training_metrics.json (classes e latência)
                    └── FastAPI (/predict, /health, /metrics)
                          └── Prometheus ── Grafana (dashboard provisionado)

GitHub Actions: Ruff + pytest em push e pull request
Airflow: validar CSV → treinar → exportar modelo ONNX
```

## Decisões de arquitetura

As escolhas abaixo partem dos requisitos e do perfil de uso do cenário
acadêmico. São decisões justificadas para um protótipo; não são resultados de
uma avaliação operacional em produção.

| Decisão | Justificativa para este cenário | Alternativa considerada / limite |
| --- | --- | --- |
| Inferência síncrona por REST | O usuário envia um abstract e espera uma categoria; FastAPI facilita validação, documentação OpenAPI e integração com clientes. | Batch reduziria custo em grandes lotes, mas não atende a consulta interativa demonstrada. |
| TF-IDF + Logistic Regression | Baseline supervisionado leve, rápido de treinar, funciona bem como referência para texto e permite inspecionar pesos e classes. | Transformers podem capturar semântica mais rica, mas aumentam custo, tamanho e latência; são uma evolução possível, não necessária para demonstrar o ciclo de deploy. |
| ONNX Runtime para inferência | Separa o formato de serving do treinamento scikit-learn e possibilita comparar latência na mesma máquina e entradas. | A conversão adiciona complexidade e não garante ganho; o projeto mede os dois caminhos e deve reportar o resultado observado, sem presumir aceleração. |
| Airflow para retreinamento | A DAG explicita a validação da fonte e o treinamento como tarefas reexecutáveis e observáveis, atendendo ao requisito de orquestração. | Para uma única execução manual, um script bastaria; Airflow é usado pelo valor didático do ciclo orquestrado, não por necessidade de escala local. |
| Prometheus + Grafana | Métricas de requisições, latência e respostas 5xx tornam o comportamento do serviço visível durante a demonstração. | Logs ajudam a investigar eventos, mas sozinhos não resumem tendências temporais nem percentis. |
| Docker Compose no desenvolvimento | Sobe a API e a stack de monitoração com configuração reproduzível em uma máquina. | Não é uma arquitetura de produção altamente disponível; em produção seriam necessários gestão de segredos, escalabilidade, redes e armazenamento operacionais. |

### Caminho de deploy em nuvem (proposta)

A API atende inferência online, então uma proposta coerente em AWS seria publicar
uma imagem versionada no Amazon ECR e executá-la no Amazon ECS Fargate atrás de
um Application Load Balancer. O modelo aprovado poderia ser armazenado no S3 e
consumido por uma versão imutável do serviço; credenciais e segredos ficariam no
AWS Secrets Manager. O treinamento periódico poderia ser orquestrado em Amazon
MWAA, separado da API de inferência.

Essa proposta permite escalar o serviço online independentemente do retreino,
evita administrar servidores diretamente e mantém artefatos versionáveis. Ela
não foi implantada nem testada na AWS neste trabalho: o ambiente entregue e
reproduzível é local, com Docker Compose. Antes de adoção real seriam necessários
testes de carga, autenticação/autorização, proteção de dados, custos, estratégia
de rollback e avaliação institucional.

### Fluxo operacional proposto

1. Obter uma cópia autorizada do corpus e validar colunas e rótulos.
2. Executar o treinamento e gerar modelo scikit-learn, exportação ONNX e
   relatório de latência.
3. Revisar os artefatos; um retreino não promove automaticamente uma versão.
4. Iniciar a API com o artefato escolhido e observar métricas no Prometheus e
   Grafana.
5. Em uma evolução para cloud, versionar e promover explicitamente a imagem e o
   modelo aprovados, mantendo um caminho de rollback.

O artefato do modelo e o CSV são entradas externas locais; ambos ficam fora da
imagem e do controle de versão. Essa separação evita distribuir o corpus junto
do serviço e deixa explícito qual modelo foi treinado e promovido.

## Pré-requisitos

- Python 3.11 (suporte declarado: `>=3.11,<3.13`).
- CSV do corpus baixado e descompactado localmente.
- Docker Desktop com Docker Compose v2, para executar a stack containerizada.
- PowerShell no Windows para os comandos abaixo (ou shell equivalente).

## Treinar e executar localmente

Na raiz do repositório, crie o ambiente e instale o projeto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Treine usando os arquivos do corpus baixados para `data/medical_abstracts/`:

```powershell
python -m triage.training `
  --data-path data/medical_abstracts/medical_tc_train.csv `
  --labels-path data/medical_abstracts/medical_tc_labels.csv
uvicorn triage.api:app --host 127.0.0.1 --port 8000
```

Se `medical_tc_test.csv` estiver ao lado do CSV de treino, o comando o detecta
automaticamente. Também é possível apontar explicitamente um holdout com
`--test-data caminho\para\teste.csv`.

Se o CSV de nomes não estiver disponível, omita `--labels-path`; a API devolverá
os IDs de classe do dataset. O carregador valida cabeçalhos, células vazias e
IDs sem correspondência em vez de inventar dados ou rótulos.

Em outro terminal, consulte a saúde e teste a classificação:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"text":"Paste an abstract from your permitted local dataset"}'
Invoke-WebRequest http://127.0.0.1:8000/metrics
```

O treino salva `artifacts/model.joblib`, `artifacts/model.onnx` e
`artifacts/training_metrics.json`. Quando existe conjunto de teste, o JSON
registra acurácia, F1 macro e ponderado, métricas por classe, matriz de confusão
e concordância entre previsões scikit-learn e ONNX. A API lê o caminho ONNX de
`MODEL_PATH` (por padrão `artifacts/model.onnx`) e as classes do arquivo de
métricas vizinho. Execute o treino antes da API; erros de arquivo/modelo ausente
são explícitos.

## Docker Compose e observabilidade

Treine o modelo localmente antes de subir a stack. Os artefatos locais são
montados como somente leitura no container; o corpus não é copiado para a imagem:

```powershell
python -m triage.training `
  --data-path data/medical_abstracts/medical_tc_train.csv `
  --labels-path data/medical_abstracts/medical_tc_labels.csv
$env:GRAFANA_ADMIN_USER = "admin"
$env:GRAFANA_ADMIN_PASSWORD = Read-Host "Senha local do Grafana"
docker compose up --build
```

Defina a senha por variável de ambiente; ela não é armazenada no repositório nem
incluída na imagem. Acesse:

- API: <http://localhost:8000/docs>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (credenciais definidas acima)

O dashboard **Medical Abstract Classifier API** é provisionado automaticamente
com painéis de requisições por segundo, latência p95 e proporção de respostas
HTTP 5xx. Envie requisições para popular os gráficos. Encerre com `Ctrl+C`; remova
os containers com `docker compose down`. O volume `grafana-storage` preserva os
dados do Grafana.

## Retreinamento com Airflow

A DAG `airflow/dags/train_model.py` valida o CSV e treina/exporta o modelo. O
arquivo padrão é `data/medical_abstracts/medical_tc_train.csv`; para apontar a
outro CSV, defina `TRIAGE_DATA_PATH`. Se
`medical_tc_labels.csv` estiver junto do CSV, os nomes das categorias são
resolvidos automaticamente. Instale Airflow em um ambiente Python 3.11 separado,
usando as constraints oficiais da mesma versão:

```powershell
$env:AIRFLOW_HOME = (Join-Path (Get-Location) "airflow")
python -m pip install "apache-airflow==2.10.5" `
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
python -m pip install -e .
airflow db migrate
airflow users create --username admin --firstname Local --lastname User `
  --role Admin --email local@example.invalid
airflow standalone
```

O Airflow descobre a DAG em `airflow/dags`; se executar o scheduler a partir de
outro diretório, configure também `AIRFLOW__CORE__DAGS_FOLDER` com o caminho
absoluto dessa pasta. Defina `TRIAGE_PROJECT_ROOT` se a raiz do projeto não for
inferida do arquivo DAG. Monte o dataset no mesmo caminho visível para os
workers. O retreino grava em `artifacts/`, mas não publica ou substitui
automaticamente um modelo já em serviço; revise e promova artefatos
explicitamente.

## CI, testes e medição

O GitHub Actions executa lint com Ruff e testes com pytest em push e pull
request. Para repetir localmente:

```powershell
ruff check .
pytest -q
```

`training_metrics.json` registra a mediana de latência do classificador
scikit-learn e do ONNX Runtime, medida em 100 inferências sequenciais por versão
em textos do conjunto de treino; quando o corpus traz `medical_tc_test.csv`,
também calcula métricas de classificação em separado, sem treinar com o holdout.
É uma comparação local informativa, não um SLA nem uma garantia de que ONNX será
sempre mais rápido. Para comparações válidas, registre hardware, versões,
tamanho de entrada, aquecimento e carga concorrente; meça também a latência fim
a fim da API no Prometheus.

### Resultado desta execução

O resultado detalhado, incluindo matriz de confusão, F1 por classe, hash dos
arquivos, revisão de origem e ambiente, está em
[reports/model-evaluation.json](./reports/model-evaluation.json). A avaliação
usa o conjunto de teste separado do corpus; o modelo foi ajustado apenas nos
11.550 exemplos de treino.

| Métrica no holdout (2.888 abstracts) | Resultado |
| --- | ---: |
| Acurácia ONNX | 55,75% |
| F1 macro ONNX | 54,22% |
| F1 ponderado ONNX | 55,57% |
| Concordância de rótulos ONNX vs scikit-learn | 99,13% (25 divergências) |
| Mediana scikit-learn por texto | 0,5037 ms |
| Mediana ONNX Runtime por texto | 0,0931 ms |
| Razão observada neste microbenchmark | 5,41x |

O ONNX foi mais rápido neste microbenchmark local, mas as duas versões
divergiram em 25 predições do teste; a acurácia reportada é a do artefato ONNX
que serve a API. Essa diferença de conversão merece investigação antes de
qualquer uso mais amplo. O classificador apresenta desempenho modesto, com
variação entre classes — por exemplo, F1 de 42,21% para doenças digestivas e
68,75% para neoplasias — e não deve ser apresentado como sistema pronto para
produção clínica. As latências são medianas de chamadas isoladas em um processo,
não p95 da API sob carga, e não demonstram SLA.

Este é um experimento de referência numa máquina Windows 11, Python 3.12.13,
Intel Core Ultra 5 135U, scikit-learn 1.9.1 e ONNX Runtime 1.31.0. Para uma
comparação de performance mais rigorosa, repetir com aquecimento controlado,
múltiplas execuções, carga concorrente e medição fim a fim pelo dashboard.

## Roteiro sugerido para vídeo STAR (5 minutos)

- **Situation:** motivação da classificação de abstracts e escopo limitado do
  corpus.
- **Task:** requisitos de API, CI, retreino, observabilidade e latência.
- **Action:** arquitetura, dataset recomendado, treino TF-IDF, exportação ONNX,
  DAG e dashboard.
- **Result:** execução do Compose, requisições, painéis e comparação local de
  latência; destaque que categorias de doença não significam urgência clínica.

## Estrutura

```text
airflow/dags/train_model.py                   DAG de validação e retreino
monitoring/prometheus.yml                      Scrape da API
monitoring/grafana/dashboards/                 Dashboard com três painéis
monitoring/grafana/provisioning/               Datasource e dashboard provisionados
tests/                                         Testes de CSV, API e ONNX
triage/api.py                                  API e métricas HTTP
triage/data.py                                 Leitura do corpus e nomes das classes
triage/predictor.py                            Inferência ONNX
triage/training.py                             Treino, exportação e benchmark
Dockerfile, docker-compose.yml                 Execução containerizada
.github/workflows/ci.yml                       Lint e testes automatizados
```
