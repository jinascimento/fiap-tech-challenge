from sklearn.metrics import f1_score

def calculate_fitness(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

