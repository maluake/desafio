# Roteiro sugerido para o vídeo (≈10–12 min)

> Objetivo: mostrar o raciocínio, as decisões de arquitetura e a orquestração funcionando.
> Deixe abertos: o README (diagramas), o VS Code em `app/agents/`, um terminal e o `/docs` do FastAPI.

## 1. Contexto e objetivo (1 min)
- "O desafio pede um sistema multi-agente para atender lojistas Getnet: um Router, um agente de conhecimento com RAG e busca web, e um agente de suporte com ferramentas sobre os dados do cliente."
- "Fui além nos bônus: agente de escalonamento humano, guardrails, handoff por baixa confiança, observabilidade e avaliação automatizada."

## 2. Arquitetura (2 min) – mostrar o diagrama do README (seção 4)
- Fluxo: **guardrails de entrada → Router → especialistas → decisão de escalonamento → composição → guardrails de saída**.
- Comunicação entre agentes: chamadas diretas e tipadas (`AgentContext` → `AgentResult`). Explicar por que não usei fila: request/response com plano curto; o contrato é serializável, então dá para migrar para fila sem reescrever agentes.
- O Router usa **saída estruturada (JSON schema)** com esforço baixo e, por cima, **regras de negócio determinísticas** (ex.: pedido de humano sempre escala, fraude sempre escala com prioridade urgente).

## 3. Demo ao vivo (3 min)
```bash
docker compose up --build
make demo            # os 10 cenários do desafio
```
Comentar 4 respostas, mostrando no JSON os campos `route`, `tool_calls`, `sources`, `trace`:
1. **Get Clássica vs Get Smart** → Knowledge + RAG com citações `[n]` e tabela.
2. **"Quando cai o dinheiro das vendas de ontem?"** → Support chama `get_customer_profile` + `get_settlement_forecast`; datas por regra (débito D+1 útil, crédito D+30, parcelas mensais, Pix na hora).
3. **"Minha maquininha não conecta"** → Support cruza diagnóstico do terminal (Wi-Fi fraco, chip sem sinal, firmware desatualizado) com o guia do help center e **abre chamado** porque está offline há mais de 1h.
4. **Clima em Porto Alegre / câmbio do euro** → ferramentas externas (Open-Meteo, Frankfurter/BCE com fallback).

Extras rápidos:
- `"quero falar com um atendente"` → Escalation Agent cria ticket com transcrição e SLA (`GET /tickets/cliente1988`).
- `"Ignore previous instructions..."` → bloqueado pelo guardrail.
- Mensagem com número de cartão → redigido antes de chegar ao LLM e aviso de segurança.

## 4. RAG (1,5 min) – README seção 7
- Ingestão: **crawler** (robots.txt, mesmo domínio, HTML→Markdown preservando títulos) + **corpus curado** com fontes citadas (garante funcionamento offline; o site estava inacessível no meu ambiente de desenvolvimento).
- Chunking por seção, tabelas atômicas, overlap sem cortar palavras.
- Recuperação híbrida: **TF-IDF (cosseno) + BM25 com Reciprocal Rank Fusion**, expansão de consulta bilíngue (EN↔PT), diversificação por documento.
- Geração: o agente decide quando buscar, cita as passagens e não inventa taxas.
- Golden set de recuperação nos testes — mostrar que ele pegou uma regressão real (pergunta sobre WhatsApp do suporte).

## 5. Confiabilidade e segurança (1,5 min)
- **Degradação graciosa**: todo agente tem caminho LLM e caminho determinístico que usa as mesmas ferramentas. Se a API cair, o sistema continua respondendo (campo `mode`).
- Ferramentas do suporte **presas ao `user_id` autenticado** (o modelo nunca escolhe o usuário) → proteção contra "confused deputy".
- Fallback de recusa no servidor (`fallbacks: "default"`), retries, timeouts, limite de iterações.

## 6. Testes, avaliação e observabilidade (2 min)
```bash
python -m pytest          # 129 testes, ~5 s, sem rede e sem chave
python -m evals.run_eval  # relatório com métricas e thresholds
```
- Estratégia em camadas (README seção 12): unitários determinísticos → **Messages API roteirizada** para testar o loop de ferramentas real → integração multi-agente → avaliação com modelo real e **LLM-as-judge**.
- Ser transparente: a avaliação offline mede o caminho determinístico; a de qualidade real é `--llm --judge`.
- Observabilidade: `trace_id` em todo log, spans por agente/ferramenta/LLM, métricas Prometheus, alertas e dashboard Grafana (`docker compose --profile observability up`).

## 7. Próximos passos (30 s)
- Vector DB gerenciado + re-ranker, integrações reais (APIs internas, Zendesk/Teams), OpenTelemetry/Langfuse, avaliação online por amostragem e prompt caching.
