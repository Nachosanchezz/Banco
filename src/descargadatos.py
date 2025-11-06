import yfinance as yf
import pandas as pd

# Definir los símbolos de las acciones
bbva_ticker = "BBVA.MC"   # BBVA (Bolsa de Madrid)
santander_ticker = "SAN.MC"  # Banco Santander (Bolsa de Madrid)

# Definir el rango de fechas
start_date = "2000-01-01"
end_date   = "2025-11-01"

# Descargar los datos históricos desde Yahoo Finance
bbva_data = yf.download(bbva_ticker, start=start_date, end=end_date, auto_adjust=False, actions=True)
santander_data = yf.download(santander_ticker, start=start_date, end=end_date, auto_adjust=False, actions=True)

# Mostrar resumen por consola
print("BBVA data:")
print(bbva_data.head())
print("\nSantander data:")
print(santander_data.head())

# Guardar los datos en CSV
bbva_data.to_csv("data/bbva_data.csv")
santander_data.to_csv("data/santander_data.csv")