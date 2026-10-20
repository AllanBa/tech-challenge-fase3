# Roteiro de gravação — demonstração STAR (5 minutos)

**Objetivo:** gravar uma demonstração técnica do protótipo acadêmico. Substitua
os trechos entre colchetes pelos resultados da execução final dos workflows.
Não apresente o cenário fictício como contratação real e não use a palavra
"triagem" para descrever o classificador de categorias de abstracts.

## Preparação antes de gravar

- Baixe o corpus autorizado, respeitando a licença e a atribuição indicadas no
  README.
- Execute o treino e confira `reports/model-evaluation.json`.
- Suba a stack e confirme que API, Prometheus e Grafana estão saudáveis.
- Abra `/docs`, o dashboard e o histórico dos workflows do GitHub Actions.
- Não mostre credenciais, tokens, dados privados, nem envie abstracts a serviços
  de terceiros. Use uma amostra do corpus apenas localmente.

## 00:00–00:45 — Situation

> “Neste Tech Challenge, vou apresentar um estudo de caso acadêmico. No cenário
> fictício, uma instituição hospitalar precisa organizar abstracts de
> publicações médicas por tema para apoiar catalogação e recuperação. O corpus
> recomendado contém cinco categorias de doenças. Ele não contém rótulos de
> urgência e não representa laudos de pacientes; portanto, o protótipo não faz
> triagem nem toma decisões clínicas.”

**Na tela:** README, tabela de escopo e origem do corpus.

## 00:45–01:20 — Task

> “A entrega precisava demonstrar o ciclo de vida de um classificador leve:
> treino reproduzível, API REST em container, automação de qualidade,
> retreinamento orquestrado, métricas operacionais e uma comparação de
> latência. Como o enunciado não define um SLA nem um mínimo de acurácia, vou
> reportar medições observadas e as limitações, sem declarar um critério que não
> foi especificado.”

**Na tela:** estrutura do repositório e workflow CI.

## 01:20–02:30 — Action: arquitetura e modelo

> “Escolhi TF-IDF com regressão logística balanceada como baseline porque é
> leve, rápido de treinar e adequado para uma comparação supervisionada
> transparente. Comparei cinco configurações numa divisão estratificada
> separada do teste e selecionei pela F1 macro. Depois, retreinei no conjunto de
> treino e exportei para ONNX Runtime. A API FastAPI recebe um abstract e
> responde com uma categoria e uma confiança. O Docker Compose reúne a API,
> Prometheus e Grafana; a DAG do Airflow valida a fonte e orquestra o retreino.”

**Na tela:** endpoint `/docs`, resposta de `/predict`, arquivos ONNX/JSON e
workflow/DAG. Evite afirmar ganho sem mostrar as medições.

## 02:30–03:35 — Action: observabilidade e operação

> “Instrumentei contagem de requisições, duração HTTP e respostas de erro. O
> dashboard mostra taxa de requisições, p95 e proporção de respostas 5xx. O
> benchmark de carga aquece o serviço e mede o tempo ponta a ponta visto pelo
> cliente, em vez de apenas cronometrar a função do modelo.”

**Na tela:** três painéis Grafana; use a execução real do workflow de integração
para provar o estado dos serviços.

## 03:35–04:35 — Result

> “No holdout oficial de 2.888 abstracts, a configuração escolhida alcançou
> acurácia de 59,49%, F1 macro de 59,70% e F1 ponderado de 58,10%. O F1 da classe
> general pathological conditions ficou em 42,20%, então ainda há espaço claro
> para melhoria. Num ensaio local de 500 requisições com concorrência 10, medi
> p50 de 26,11 milissegundos, p95 de 58,25, p99 de 77,54 e cerca de 317
> requisições por segundo. Esses números são de uma única máquina e não
> representam SLA.”

**Na tela:** `reports/model-evaluation.json` e `reports/api-load-benchmark.json`.

## 04:35–05:00 — Fechamento e limitações

> “O protótipo demonstra API, exportação ONNX, CI, retreinamento e
> observabilidade. As previsões scikit-learn e ONNX divergiram em 21 exemplos,
> principalmente perto da fronteira entre classes. Antes de qualquer uso além
> do exercício, eu investigaria essa paridade, avaliaria outras divisões e
> configurações, faria testes de carga repetidos e uma revisão adequada ao
> contexto. Este trabalho classifica temas de abstracts científicos; não é
> produto clínico e não classifica urgência.”

**Entregável após gravar:** substitua esta nota pelo link do vídeo autorizado e
confira que a audiência consegue acessá-lo.
