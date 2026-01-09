# Testes Unitários do Algoritmo Genético

Este diretório contém testes unitários abrangentes para todos os componentes da implementação do Algoritmo Genético.

## Cobertura de Testes

A suíte de testes cobre todos os componentes do AG:

### 1. **Mutação** (`test_mutation.py`)
- `mutate()` - Mutação individual com diferentes taxas de mutação
- `apply_mutation()` - Mutação em nível de população
- Testes para probabilidade de mutação, validação de valores e preservação de estrutura

### 2. **Seleção** (`test_selection.py`)
- `elitism_selection()` - Mecanismo de seleção elitista
- `tournament_selection()` - Seleção por torneio com vários tamanhos
- `apply_selection()` - Seleção combinada com elites e pais
- Testes para casos extremos (população vazia, tamanhos grandes de torneio)

### 3. **Crossover** (`test_crossover.py`)
- `crossover()` - Operação de crossover única entre dois pais
- `apply_crossover()` - Crossover em nível de população com controle de taxa
- Testes para probabilidade de crossover, geração de descendentes e casos extremos

### 4. **Fitness** (`test_fitness.py`)
- `calculate_fitness()` - Cálculo da métrica F1 usando sklearn
- Testes para predições perfeitas, correspondências parciais e casos extremos

### 5. **Geração de Indivíduo** (`test_generate_individual.py`)
- `generate_individual()` - Criação aleatória de indivíduo
- Testes para validação de valores, reprodutibilidade e consistência de estrutura

### 6. **Geração de População** (`test_generate_population.py`)
- `generate_population()` - Inicialização da população
- Testes para tamanho da população, validade dos indivíduos e reprodutibilidade

### 7. **Utilitários** (`test_utils.py`)
- `set_random_seed()` - Configuração de semente aleatória para reprodutibilidade
- Testes para os módulos `random` e `numpy.random`

### 8. **GA Runner** (`test_ga_runner.py`)
- `run_ga()` - Fluxo completo de execução do AG
- Testes para rastreamento de fitness, manutenção de histórico e passagem de parâmetros
- Usa mocking para evitar treinamento real de modelos

## Executando os Testes

### Usando unittest (biblioteca padrão):
```bash
python3 -m unittest discover -s fase_2/ga/tests -p "test_*.py" -v
```

### Usando pytest (se instalado):
```bash
python3 -m pytest fase_2/ga/tests/ -v
```

### Executar arquivo de teste específico:
```bash
python3 -m unittest fase_2.ga.tests.test_mutation -v
```

### Executar teste específico:
```bash
python3 -m unittest fase_2.ga.tests.test_mutation.TestMutation.test_mutate_no_mutation_when_rate_zero -v
```

## Estatísticas dos Testes

- **Total de Testes**: 84
- **Todos Passando**: ✅
- **Cobertura**: Todos os componentes do AG testados

## Estrutura dos Testes

Cada arquivo de teste segue a estrutura padrão do unittest:
- `setUp()` - Inicializa fixtures de teste
- Métodos de teste prefixados com `test_`
- Docstrings abrangentes explicando cada teste
- Cobertura de casos extremos (entradas vazias, valores de limite, etc.)

## Observações

- Os testes usam mocking extensivamente para evitar dependências de modelos de ML reais
- A semente aleatória é definida nos testes para reprodutibilidade
- Casos extremos são testados minuciosamente (listas vazias, valores zero, etc.)
- Todos os testes verificam tanto a funcionalidade quanto a integridade da estrutura de dados
