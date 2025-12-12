from module_diabetes import load_dataset, prepare_dataset, build_mlp_baseline, evaluate_model

df = load_dataset()
df_processed = prepare_dataset(df)
model, X_test, y_test = build_mlp_baseline(df_processed)
report = evaluate_model(model, X_test, y_test)

print("\n=== Baseline Report ===")
print(report)
