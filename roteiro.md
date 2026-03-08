# Roteiro do Vídeo — Datathon Passos Mágicos (5 minutos)

> Formato gerencial. ~600 palavras de fala + demos na tela.
> Foco: problema → solução → pipeline → demo → resultados.

---

## PARTE 1 — Problema (0:00 – 0:40)

"Olá! Eu sou o Emanuel Borges e junto com o Rafael Cunha vamos apresentar nossa solução para o TechChallenge 5 da FIAP, em parceria com a Associação Passos Mágicos.

A Passos Mágicos atende centenas de jovens em vulnerabilidade social em Embu-Guaçu. O desafio: identificar quais alunos estão em risco de defasagem escolar antes que o problema se instale.

Hoje isso é feito de forma manual e reativa — quando se detecta, já é tarde. Nossa proposta é transformar essa identificação em algo preditivo e automatizado, usando Machine Learning."

---

## PARTE 2 — Solução e Arquitetura (0:40 – 1:20)

"Construímos uma pipeline completa, do dado bruto ao modelo em produção.

A arquitetura tem duas camadas: a offline, que trata dados, treina e avalia o modelo; e a online, uma API FastAPI que recebe os dados do aluno, faz a predição em tempo real e monitora drift.

O modelo classifica o aluno como fora de risco ou em grupo de risco. Se os dados estiverem fora do padrão do treino, o sistema emite alerta de drift automaticamente."

**[Tela]** Mostrar diagrama de arquitetura do README.

---

## PARTE 3 — Pipeline de ML (1:20 – 2:20)

"Nossa pipeline tem cinco etapas.

Primeiro, o pré-processamento: carregamos os dados, padronizamos colunas, tratamos ausentes e criamos a variável de risco a partir dos indicadores educacionais.

Segundo, engenharia de variáveis: criamos indicadores derivados como média de performance e flag de baixo engajamento.

Terceiro, treinamento: split 80/20, pipeline scikit-learn com ColumnTransformer e RandomForest.

Quarto, avaliação: medimos accuracy, precision, recall, F1-score e ROC-AUC. O F1 é nosso critério principal — equilibra precisão e recall, ideal pra não deixar aluno em risco passar e também não gerar falsos alertas.

Quinto, salvamento: modelo, métricas e estatísticas de treino são persistidos como artefatos para a API consumir."

**[Tela]** Mostrar terminal com `train.py` rodando, ou slide das etapas.

---

## PARTE 4 — Demonstração (2:20 – 3:50)

"Vamos à demo. A aplicação roda em Docker."

**[Terminal]**

```bash
docker build -t passos-magicos-api .
docker run --rm -p 8000:8000 passos-magicos-api
```

**[Navegador — Swagger]**

"No Swagger temos 9 endpoints em ordem de execução: treino, health, métricas, predição, monitoramento e drift."

**[Predição ao vivo]**

"Aluno com indicadores altos:"

```json
{
  "INDE": 7.5, "IDA": 8.0, "IEG": 7.0, "IAA": 8.5,
  "IPS": 7.0, "IPP": 7.5, "IPV": 0.8, "IAN": 7.5,
  "FASE": 2, "PEDRA": "Ametista"
}
```

"Resultado: fora do grupo de risco. ✅"

"Agora, indicadores baixos:"

```json
{
  "INDE": 3.0, "IDA": 2.5, "IEG": 3.5, "IAA": 2.0,
  "IPS": 3.0, "IPP": 2.8, "IPV": 0.2, "IAN": 2.5,
  "FASE": 0, "PEDRA": "Quartzo"
}
```

"Em grupo de risco, com alertas de drift. ⚠️"

**[Navegador]** Abrir `http://localhost:8000/drift/dashboard`.

"E aqui o dashboard de monitoramento com predições, latência e histórico de drift em tempo real."

---

## PARTE 5 — Resultados e Encerramento (3:50 – 5:00)

"Resultados do modelo: 85% de acurácia, F1 de 0.65 e ROC-AUC de 0.89. O modelo distingue bem as classes e equilibra a detecção sem gerar falsos alertas em excesso.

Em qualidade de código: 157 testes automatizados, 90% de cobertura."

**[Terminal]**

```bash
.venv/bin/python -m pytest --cov=src --cov=app tests/ -v --tb=short
# 157 passed — 90% coverage
```

"Resumindo a entrega: código documentado no GitHub, API funcional com Swagger e monitoramento, tudo empacotado com Docker.

Com essa solução, a Passos Mágicos ganha uma ferramenta proativa para identificar alunos em risco e direcionar recursos pedagógicos de forma estratégica.

Obrigado pela atenção."

---

## Checklist de Gravação

- [ ] Pelo menos uma pessoa do grupo aparece no vídeo
- [ ] Duração máxima: 5 minutos
- [ ] Formato gerencial (foco no problema e na solução, não em código)
- [ ] Mostrar: Swagger, predição ao vivo, dashboard de drift
- [ ] Mostrar: testes passando (157 passed, 90% coverage)
- [ ] Mencionar: F1-score + justificativa
- [ ] Mencionar: Docker

## Comandos úteis

```bash
docker build -t passos-magicos-api .
docker run --rm -p 8000:8000 passos-magicos-api
.venv/bin/python -m pytest --cov=src --cov=app tests/ -v --tb=short
# Swagger:   http://localhost:8000/docs
# Dashboard: http://localhost:8000/drift/dashboard
```