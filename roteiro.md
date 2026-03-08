# Roteiro do Vídeo — Datathon Passos Mágicos (5 minutos)

> Formato gerencial. Pelo menos uma pessoa do grupo aparece.
> Foco: problema → solução → pipeline → demonstração → resultados.

---

## PARTE 1 — Contexto e Problema (0:00 – 0:50)


"Olá! Eu sou o Emanuel Borges e junto com o Rafael Cunha nós vamos apresentar a nossa solução para o TechChallenge 5 da FIAP, desenvolvida em parceria com a Associação Passos Mágicos.

A Passos Mágicos atende centenas de jovens em situação de vulnerabilidade social em Embu-Guaçu. O grande desafio da organização é conseguir antecipar quais alunos estão em risco de defasagem escolar — antes que esse risco se concretize.

Hoje, essa identificação é feita de forma manual e reativa. Ou seja, quando o problema é detectado, geralmente já se instalou e exige muito mais esforço para ser revertido. Isso gera um custo alto — tanto para a equipe quanto para o aluno.

A proposta do TechChallenge é clara: desenvolver uma solução inteligente, baseada em Machine Learning, que transforme essa identificação de risco em um processo preditivo, automatizado e contínuo."

---

## PARTE 2 — Solução Proposta (0:50 – 1:40)


"Então, foi construido uma pipeline completa de Machine Learning — desde o tratamento dos dados brutos até o modelo rodando em produção com API REST.

A arquitetura tem duas partes principais. A primeira é a Pipeline Offline, que cuida de todo o processamento dos dados, criação de variáveis, treinamento e avaliação do modelo. A segunda é a Pipeline Online, que é a API em si — feita com FastAPI — onde a gente recebe os dados de um aluno, aplica as mesmas transformações do treino, faz a predição em tempo real e ainda monitora se está tendo drift nos dados.

No final, o modelo classifica o aluno em duas categorias: fora do grupo de risco ou em grupo de risco. E se os dados de entrada estiverem muito diferentes do que o modelo treinou, ele já dispara um alerta de drift.

Como métrica principal a gente escolheu o F1-score. Por quê? Porque ele equilibra precisão e recall. Isso é fundamental aqui: a gente não quer deixar passar aluno em risco sem identificar, mas também não pode gerar alerta demais e sobrecarregar a equipe."

O F1-Score é uma métrica de desempenho em aprendizado de máquina que combina precisão e revogação (recall) em uma única média harmônica, variando de 0 a 1.

**[Tela]** Mostrar o diagrama de arquitetura do README (fluxo Offline → Online).

---

## PARTE 3 — Etapas do Pipeline de ML (1:40 – 3:00)

"Vamos agora às etapas da nossa pipeline.

Na primeira etapa, o pré-processamento, os dados brutos são carregados, as colunas são padronizadas, duplicatas são removidas e valores ausentes são tratados. É aqui também que criamos a variável-alvo — o indicador de risco — com base numa combinação ponderada dos indicadores educacionais do aluno.

A segunda etapa é a engenharia de variáveis. Criamos indicadores derivados, como a média de performance e uma flag de baixo engajamento, e removemos informações sensíveis que não devem influenciar o modelo.

Na terceira etapa, o treinamento. Os dados são divididos em 80% para treino e 20% para teste. O pipeline combina tratamento de variáveis numéricas e categóricas com um classificador RandomForest — um modelo robusto e de fácil interpretação.

A quarta etapa é a avaliação, onde medimos accuracy, precision, recall, F1-score e ROC-AUC. O F1 é o nosso critério principal de decisão.

E por último, o pós-processamento. O modelo treinado e as estatísticas são salvos como artefatos que a API consome em produção. As mesmas transformações aplicadas no treino são reaplicadas a cada requisição, garantindo total consistência."

**[Tela]** Mostrar o terminal com o `train.py` executando as 8 etapas, ou slides resumindo o fluxo.

---

## PARTE 4 — Demonstração da API (3:00 – 4:15)

**[Fala]**

"Agora vamos à demonstração prática. A aplicação está rodando em Docker."

**[Terminal — executar e mostrar]**

```bash
docker build -t passos-magicos-api .
docker run --rm -p 8000:8000 passos-magicos-api
```

**[Navegador — abrir Swagger]**

"Acessando a documentação interativa da API, temos 9 endpoints organizados na ordem lógica de execução do projeto:"

```
1. POST /train      → Treinamento do modelo via API
2. GET  /health     → Status da aplicação e do modelo
3. GET  /metrics    → Métricas de desempenho do treino
4. POST /predict    → Predição de risco escolar
5. GET  /monitoring → Indicadores operacionais
6. GET  /monitoring/predictions → Histórico de predições
7. GET  /drift      → Status do monitoramento de drift
8. GET  /drift/history   → Histórico de alertas
9. GET  /drift/dashboard → Painel visual de monitoramento
```

**[Demonstrar no Swagger ou Postman]**

"Vamos fazer duas predições ao vivo. Primeiro, um aluno com indicadores educacionais elevados:"

```json
{
  "INDE": 7.5, "IDA": 8.0, "IEG": 7.0, "IAA": 8.5,
  "IPS": 7.0, "IPP": 7.5, "IPV": 0.8, "IAN": 7.5,
  "FASE": 2, "PEDRA": "Ametista"
}
```

"Resultado: classificado como fora do grupo de risco. Exatamente o esperado. ✅"

"Agora, um aluno com indicadores significativamente baixos:"

```json
{
  "INDE": 3.0, "IDA": 2.5, "IEG": 3.5, "IAA": 2.0,
  "IPS": 3.0, "IPP": 2.8, "IPV": 0.2, "IAN": 2.5,
  "FASE": 0, "PEDRA": "Quartzo"
}
```

"Resultado: classificado como em grupo de risco, acompanhado de alertas de drift — indicando que esses valores estão fora do padrão observado no treinamento. ⚠️"

**[Navegador]** Abrir `http://localhost:8000/drift/dashboard` e mostrar o painel HTML.

"Aqui temos o dashboard de monitoramento, com visão em tempo real de predições, latência, taxa de erro e histórico de drift."

---

## PARTE 5 — Resultados, Testes e Encerramento (4:15 – 5:00)

**[Fala]**

"Para finalizar, os resultados obtidos. O modelo atingiu 85% de acurácia, 78% de precisão, 56% de recall, F1-score de 0.65 e ROC-AUC de 0.89.

O ROC-AUC de 0.89 demonstra uma boa capacidade do modelo em distinguir alunos em risco dos demais. E o F1 de 0.65 traduz o equilíbrio entre identificar corretamente os alunos que precisam de atenção sem sobrecarregar a equipe com falsos alertas.

Em relação à qualidade do código, o projeto conta com 157 testes automatizados e 90% de cobertura."

**[Terminal — mostrar rapidamente]**

```bash
.venv/bin/python -m pytest --cov=src --cov=app tests/ -v --tb=short
# 157 passed — 90% coverage
```

**[Fala — encerramento]**

"Resumindo a entrega: código organizado e documentado no GitHub, documentação completa no README, API funcional com Swagger e monitoramento integrado, e tudo empacotado com Docker pronto para deploy.

Com essa solução, a Passos Mágicos passa a contar com uma ferramenta proativa para identificar alunos em risco, permitindo direcionar os recursos pedagógicos de forma mais estratégica e eficiente.

Obrigado pela atenção."

---

## Checklist de Gravação

- [ ] Pelo menos uma pessoa do grupo aparece no vídeo
- [ ] Duração máxima: 5 minutos
- [ ] Formato gerencial (foco no problema e na solução, não em código)
- [ ] Mostrar: Swagger (endpoints em ordem), predição ao vivo, dashboard de drift
- [ ] Mostrar terminal: testes passando (157 passed, 90% coverage)
- [ ] Mencionar: F1-score como métrica principal + justificativa
- [ ] Mencionar: Docker como forma de deploy

## Comandos úteis para a gravação

```bash
# Subir a API via Docker
docker build -t passos-magicos-api .
docker run --rm -p 8000:8000 passos-magicos-api

# Rodar testes com cobertura
.venv/bin/python -m pytest --cov=src --cov=app tests/ -v --tb=short

# Endpoints para demonstrar
# Swagger:     http://localhost:8000/docs
# Health:      http://localhost:8000/health
# Métricas:    http://localhost:8000/metrics
# Dashboard:   http://localhost:8000/drift/dashboard
```