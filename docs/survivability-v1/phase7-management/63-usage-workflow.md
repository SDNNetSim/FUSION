# Phase 7: Project Management

## 63 - Example Usage Workflow

**Section Reference**: Section 11 - Example Usage Workflow

**Purpose**: Provide end-to-end workflow examples for common survivability experiments.

---

## Workflow 1: Generate Offline Dataset

```bash
# 1. Generate offline dataset with KSP-FF + epsilon-mix
python -m fusion.cli.run_sim \
  --config_path fusion/configs/templates/survivability_experiment.ini \
  --failure_type link \
  --failed_link_src 3 \
  --failed_link_dst 9 \
  --log_offline_dataset true \
  --dataset_output_path datasets/ksp_ff_link_failures.jsonl \
  --epsilon_mix 0.1 \
  --num_requests 50000 \
  --seed_list 42 43 44 45 46 \
  --run_id dataset_generation

# Output: datasets/ksp_ff_link_failures.jsonl (~50k transitions per seed)
```

---

## Workflow 2: Train BC Model (External)

```bash
# 2. Train BC model (external script, not in simulator)
python scripts/train_bc_model.py \
  --dataset datasets/ksp_ff_link_failures.jsonl \
  --output models/bc_model.pt \
  --epochs 50 \
  --batch_size 256

# Output: models/bc_model.pt
```

---

## Workflow 3: Evaluate BC Policy Under Failures

```bash
# 3. Evaluate BC policy under geographic failures
python -m fusion.cli.run_sim \
  --config_path fusion/configs/templates/survivability_experiment.ini \
  --failure_type geo \
  --geo_center_node 5 \
  --geo_hop_radius 2 \
  --policy_type bc \
  --bc_model_path models/bc_model.pt \
  --fallback_policy ksp_ff \
  --seed_list 42 43 44 45 46 \
  --export_csv true \
  --csv_output_path results/bc_geo_failure_results.csv \
  --run_id bc_evaluation

# Output: results/bc_geo_failure_results.csv
```

---

## Workflow 4: Compare Against Baseline

```bash
# 4. Run baseline (KSP-FF) with same configuration
python -m fusion.cli.run_sim \
  --config_path fusion/configs/templates/survivability_experiment.ini \
  --failure_type geo \
  --geo_center_node 5 \
  --geo_hop_radius 2 \
  --policy_type ksp_ff \
  --seed_list 42 43 44 45 46 \
  --export_csv true \
  --csv_output_path results/ksp_ff_geo_failure_results.csv \
  --run_id baseline_evaluation

# Output: results/ksp_ff_geo_failure_results.csv
```

---

## Workflow 5: Aggregate and Compare Results

```bash
# 5. Aggregate results and generate comparison plots
python scripts/aggregate_results.py \
  --baseline results/ksp_ff_geo_failure_results.csv \
  --rl results/bc_geo_failure_results.csv \
  --metrics bp_overall recovery_time_mean_ms \
  --output figures/bp_comparison.png

# Output: figures/bp_comparison.png, summary statistics
```

---

## Workflow 6: 1+1 Protection Evaluation

```bash
# 6. Evaluate 1+1 protection under SRLG failure
python -m fusion.cli.run_sim \
  --config_path fusion/configs/templates/survivability_experiment.ini \
  --failure_type srlg \
  --srlg_links "[(0,1), (2,3), (5,6)]" \
  --protection_mode 1plus1 \
  --policy_type one_plus_one \
  --seed_list 42 43 44 45 46 \
  --export_csv true \
  --csv_output_path results/protection_srlg_results.csv \
  --run_id protection_evaluation

# Output: results/protection_srlg_results.csv
```

---

## Common Scenarios

### Scenario A: Quick Test (Single Seed)

```bash
python -m fusion.cli.run_sim \
  --config_path fusion/configs/templates/survivability_experiment.ini \
  --failure_type link \
  --failed_link_src 0 \
  --failed_link_dst 1 \
  --policy_type ksp_ff \
  --num_requests 1000 \
  --seed 42
```

### Scenario B: Multi-Seed Batch Run

```bash
for seed in 42 43 44 45 46; do
  python -m fusion.cli.run_sim \
    --config_path config.ini \
    --seed $seed \
    --csv_output_path results/run_${seed}.csv
done

# Aggregate
python scripts/aggregate_csv.py results/run_*.csv > results/aggregated.csv
```

### Scenario C: Load Sweep

```bash
for load in 100 150 200 250 300; do
  python -m fusion.cli.run_sim \
    --config_path config.ini \
    --erlang $load \
    --seed_list 42 43 44 45 46 \
    --csv_output_path results/load_${load}.csv
done
```

---

## Configuration Files for Common Experiments

### Link Failure + KSP-FF
```ini
[failure_settings]
failure_type = link
failed_link_src = 3
failed_link_dst = 9

[offline_rl_settings]
policy_type = ksp_ff
```

### Geographic Failure + BC Policy
```ini
[failure_settings]
failure_type = geo
geo_center_node = 5
geo_hop_radius = 2

[offline_rl_settings]
policy_type = bc
bc_model_path = models/bc_model.pt
fallback_policy = ksp_ff
```

### SRLG Failure + 1+1 Protection
```ini
[failure_settings]
failure_type = srlg
srlg_links = [(0,1), (2,3), (5,6)]

[protection_settings]
protection_mode = 1plus1
protection_switchover_ms = 50.0
```

---

## Troubleshooting

### Dataset Generation Issues
```bash
# Check dataset format
head -n 1 data/datasets/offline_data.jsonl | jq .

# Validate transitions
python scripts/validate_dataset.py data/datasets/offline_data.jsonl
```

### Model Loading Issues
```bash
# Test model loading
python -c "import torch; model = torch.load('models/bc_model.pt'); print('OK')"

# Check model architecture
python scripts/inspect_model.py models/bc_model.pt
```

### Performance Issues
```bash
# Profile simulation
python -m cProfile -o profile.stats run_sim.py --config config.ini

# Analyze bottlenecks
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

---

## Acceptance Criteria

- [x] All common workflows documented
- [x] Example commands run without errors
- [x] Troubleshooting guide covers common issues
- [x] Configuration examples provided

---

**Related Documents**:
- [12-configuration.md](../phase2-infrastructure/12-configuration.md) (Configuration options)
- [62-traceability.md](62-traceability.md) (Paper experiments)
- [64-checklist.md](64-checklist.md) (Pre-flight checklist)
