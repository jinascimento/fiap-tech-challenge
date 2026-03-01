SELECT_PATIENT: str = """
SELECT idade, historico, risco_previo FROM pacientes WHERE id = ?
"""

SELECT_EXAMS: str = """
SELECT tipo_exame, resultado, data
FROM exames WHERE patient_id = ?
ORDER BY data DESC LIMIT 3
"""
