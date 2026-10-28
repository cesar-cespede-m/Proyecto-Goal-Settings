# Datos sintéticos de recorridos

Estos cuatro archivos son **100 % sintéticos**: no contienen datos personales ni registros de
personas, empresas o sistemas reales. Se generan con la semilla fija `20261021` mediante:

```powershell
python scripts/generate_journey_datasets.py
```

También se puede elegir otra carpeta con `--output` o repetir una semilla concreta con `--seed`.
Con la misma versión del entorno y la misma semilla, el contenido de cada CSV es idéntico.

| Archivo | Entidad o filas | Contenido |
|---|---:|---|
| `altas_entidades.csv` | 3.000 incorporaciones | Atributos fijos de modalidad, zona e interés inicial |
| `altas_eventos.csv` | Varias filas por incorporación | Fecha y hora, paso, canal y valor de actividad desde registro hasta segunda compra |
| `solicitudes_entidades.csv` | 2.000 gestiones | Prioridad, área resolutora, vía de recepción y complejidad |
| `solicitudes_eventos.csv` | Varias filas por gestión | Fecha y hora, estado, grupo en turno y esfuerzo hasta resolución o cierre sin resolver |

En las altas, el avance cambia de forma plausible según el canal y el valor de la actividad. En
las solicitudes, la prioridad y el área alteran los tiempos entre estados; la complejidad influye
en el avance y en el tipo de cierre. Hay casos que terminan en cada etapa para poder probar
embudos, tiempos entre etapas y futuras recetas de resultado sin depender de nombres concretos.
