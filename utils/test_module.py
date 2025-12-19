from module_diabetes import load_dataset, prepare_dataset, train_mlp, evaluate_model

df = load_dataset()
df_processed = prepare_dataset(df)
baseline_params = {
    "hidden_layer_sizes": (16, 8)
}

model, X_test, y_test = train_mlp(df_processed, baseline_params)
report = evaluate_model(model, X_test, y_test)

print("\n=== Baseline Report ===")
print(report)
