# Quick Start - NeuroPredict

Guia rápido para começar a usar o NeuroPredict em 5 minutos!

## Pré-requisitos

- Docker & Docker Compose instalados
- Python 3.10+ (para desenvolvimento local)
- 16GB RAM mínimo recomendado
- Git

## Início Rápido com Docker (Recomendado)

### 1. Clone o repositório

```bash
git clone https://github.com/nathadriele/neuroPredict-sistema-medicina-precisao-epilepsia-refrataria
cd neuropredict
```

### 2. Configure variáveis de ambiente

```bash
cp .env.example .env
# Edite .env com suas credenciais (especialmente as API keys de LLMs)
nano .env
```

**Mínimo necessário no .env:**
```bash
# LLM (obrigatório para RAG)
LLM_API_KEY=sua-api-key-aqui
LLM_PROVIDER=openai  # ou anthropic

# Senhas dos serviços
DB_PASSWORD=senha-segura-postgres
NEO4J_PASSWORD=senha-segura-neo4j
REDIS_PASSWORD=senha-segura-redis
```

### 3. Inicie os serviços

```bash
# Construir imagens
docker-compose build

# Iniciar todos os serviços
docker-compose up -d

# Verificar status
docker-compose ps
```

### 4. Gere dados sintéticos

```bash
# Criar diretórios
mkdir -p data/raw data/processed

# Gerar dados de exemplo
docker-compose exec api python scripts/generate_synthetic_data.py
```

### 5. Treine o modelo

```bash
# Treinamento rápido (sem HPO)
docker-compose exec api python scripts/train_model.py \
    --clinical-data data/raw/clinical_data.csv \
    --genetic-data data/raw/genetic_data.csv

# Treinamento completo (com HPO e CV)
docker-compose exec api python scripts/train_model.py \
    --clinical-data data/raw/clinical_data.csv \
    --genetic-data data/raw/genetic_data.csv \
    --hpo \
    --cv \
    --mlflow
```

### 6. Acesse as interfaces

Após iniciar os serviços, acesse:

- **API**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **MLFlow**: http://localhost:5000
- **Neo4j Browser**: http://localhost:7474
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Desenvolvimento Local

### 1. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 2. Instalar dependências

```bash
# Produção
pip install -e .

# Desenvolvimento
pip install -e ".[dev]"
```

### 3. Configurar ambiente

```bash
cp .env.example .env
# Edite .env com suas credenciais
```

### 4. Iniciar serviços de infraestrutura

```bash
# Apenas bancos de dados e serviços
docker-compose up -d postgres neo4j redis mlflow
```

### 5. Executar aplicação

```bash
# API
uvicorn neuropredict.api.main:app --reload

# Dashboard
streamlit run src/neuropredict/dashboard/app.py

# CLI
python -m neuropredict.cli --help
```

## Uso Básico

### Via CLI

```bash
# Treinar modelo
neuropredict train \
    --clinical-data data/raw/clinical_data.csv \
    --genetic-data data/raw/genetic_data.csv \
    --output models/

# Fazer predições
neuropredict predict \
    --model models/ensemble_model_v1.pkl \
    --input data/new_patients.csv \
    --output predictions.csv

# Iniciar API
neuropredict serve --port 8000

# Iniciar Dashboard
neuropredict dashboard --port 8501

# Construir base de conhecimento
neuropredict build-kb \
    --pdf-dir data/papers/ \
    --pubmed-queries "epilepsy treatment" \
    --pubmed-queries "antiepileptic drugs"
```

### Via Python

```python
from neuropredict.models.predictor import TreatmentPredictor

# Carrega modelo
predictor = TreatmentPredictor.load("models/ensemble_model_v1.pkl")

# Dados do paciente
patient_data = {
    "patient_id": "PAT001",
    "age": 35,
    "sex": "M",
    "seizure_type": "focal_impaired_awareness",
    "seizure_frequency_per_month": 4.5,
    "age_at_onset": 12,
    "epilepsy_duration_years": 23.0,
    "previous_treatments": ["levetiracetam", "lamotrigine"],
    "genetic_variants": [
        {"gene": "SCN1A", "variant": "p.R1648H", "variant_type": "missense"},
        {"gene": "KCNQ2", "variant": "p.A306T", "variant_type": "missense"}
    ],
}

# Predição
result = predictor.predict(patient_data)
print(f"Tratamento: {result['recommended_treatment']}")
print(f"Probabilidade: {result['response_probability']:.2%}")

# Predição com explicações
result = predictor.predict_with_explanation(patient_data)
print("Top features:", result['explanation']['top_features'])
```

### Via API REST

```python
import requests

# Endpoint
url = "http://localhost:8000/predict"

# Dados
data = {
    "patient": {
        "patient_id": "PAT001",
        "age": 35,
        "sex": "M",
        "seizure_type": "focal_impaired_awareness",
        "seizure_frequency_per_month": 4.5,
        "age_at_onset": 12,
        "epilepsy_duration_years": 23.0,
        "previous_treatments": ["levetiracetam"],
        "genetic_variants": []
    },
    "explain": True
}

# Request
response = requests.post(url, json=data)
result = response.json()

print(result)
```

## Testes

```bash
# Todos os testes
make test

# Testes rápidos
make test-fast

# Com cobertura
pytest tests/ --cov=neuropredict --cov-report=html
```

## Próximos Passos

1. **Leia a documentação completa**: [docs/](docs/)
2. **Explore os notebooks**: [notebooks/](notebooks/)
3. **Customize o modelo**: Ajuste hiperparâmetros em `config.py`
4. **Adicione seus dados**: Substitua dados sintéticos por dados reais
5. **Deploy em produção**: Veja [deployment/](deployment/)

## Troubleshooting

### Problema: Serviços não iniciam

```bash
# Verifique logs
docker-compose logs -f

# Reinicie serviços
docker-compose down
docker-compose up -d
```

### Problema: Erro de memória

- Aumente memória do Docker (mínimo 8GB)
- Reduza `MODEL_BATCH_SIZE` no `.env`
- Use menos workers: `API_WORKERS=2`

### Problema: Erro de API Key

```bash
# Verifique se a key está no .env
cat .env | grep LLM_API_KEY

# Teste a key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $LLM_API_KEY"
```

### Problema: Porta já em uso

```bash
# Mude a porta no docker-compose.yml ou mate o processo
lsof -ti:8000 | xargs kill -9  # Mata processo na porta 8000
```

## Recursos Adicionais

- [Documentação Completa](https://neuropredict.readthedocs.io)
- [Tutoriais](docs/tutorials/)
- [Arquitetura](docs/architecture.md)
- [Contribuindo](CONTRIBUTING.md)
- [Discussions](https://github.com/seu-usuario/neuropredict/discussions)

## Dicas

- Use `make help` para ver todos os comandos disponíveis
- Configure pre-commit hooks: `pre-commit install`
- Monitore com Grafana para visualizações avançadas
- Use MLFlow para tracking de experimentos
- Explore o Neo4j Browser para visualizar o grafo de conhecimento

## Observação

Este sistema é apenas para pesquisa, aprendizado e testes. 
