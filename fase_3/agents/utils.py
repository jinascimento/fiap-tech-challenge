import os
import sqlite3

from config.settings import settings


def setup_database():
    conn = sqlite3.connect(settings.DATABASE_FILE)
    cursor = conn.cursor()

    # Tabela de pacientes (dados anonimizados)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY,
            idade INTEGER,
            genero TEXT,
            historico TEXT,
            risco_previo TEXT
        )
    """)

    # Tabela de exames
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            tipo_exame TEXT,
            resultado TEXT,
            data TEXT,
            FOREIGN KEY (patient_id) REFERENCES pacientes (id)
        )
    """)

    # Adiciona dados sintéticos para teste
    pacientes_teste = [
        (123, 45, "M", "Hipertensão leve, ex-fumante", "MÉDIO"),
        (456, 30, "F", "Sem comorbidades, atleta", "BAIXO"),
        (789, 68, "M", "Diabetes Tipo 2, Insuficiência Renal Crônica", "ALTO"),
    ]

    exames_teste = [
        (123, "Creatinina", "1.4 mg/dL", "2023-10-01"),
        (123, "Glicose", "105 mg/dL", "2023-10-01"),
        (789, "Creatinina", "2.8 mg/dL", "2023-10-05"),
        (789, "Ureia", "90 mg/dL", "2023-10-05"),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO pacientes VALUES (?,?,?,?,?)", pacientes_teste
    )
    cursor.executemany(
        "INSERT OR REPLACE INTO exames (patient_id, tipo_exame, resultado, data) VALUES (?,?,?,?)",
        exames_teste,
    )

    conn.commit()
    conn.close()

    print("✅ Banco de dados 'hospital.db' criado e populado com sucesso!")


if __name__ == "__main__":
    setup_database()
