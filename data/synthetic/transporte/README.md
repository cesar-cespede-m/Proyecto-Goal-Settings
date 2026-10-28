# Negocio de prueba con varios archivos y fechas

Uno de los negocios de la suite de genericidad (`tests/support/domains.py`). Es 100 %
sintético y se regenera igual con `python scripts/generate_transport_datasets.py`. Nada del
producto puede depender de estos nombres.

| Archivo | Filas | Qué tiene |
|---|---|---|
| `transporte_estaciones.csv` | 30 | Una fila por estación: tipo, línea, comuna, torniquetes |
| `transporte_transacciones.csv` | ≈ 78 000 | Viajes de tres semanas con fecha y hora, tipo de usuario, medio de pago y tarifa |
| `transporte_buses.csv` | 155 | Buses por hora por estación y franja |
| `transporte_encuestas.csv` | 1 500 | Satisfacción 1–5, espera declarada, edad, frecuencia |

Patrones deliberados que la plataforma debe encontrar sin programar: puntas a las 7–8 y 18–19 h
en días hábiles; fin de semana con aproximadamente la mitad de viajes; demanda muy distinta entre
estaciones; cinco estaciones con poca oferta de buses para su demanda; satisfacción que baja con
la espera; `Estación_31` sólo en buses y `Estación_30` sin encuestas (desajustes de clave).
