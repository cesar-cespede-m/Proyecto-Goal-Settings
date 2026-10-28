# Variantes sintéticas para monitoreo y pruebas de fallo

Archivos **100 % falsos** derivados de los cuatro datasets de `data/synthetic/`. Salen de
`scripts/generate_synthetic_variants.py` con semilla fija; regenerarlos produce exactamente los
mismos bytes (lo comprueba `tests/unit/test_synthetic_variants.py`).

```bash
python scripts/generate_synthetic_variants.py
```

Cada dominio tiene su carpeta (`mantenimiento_maquinas/`, `abandono_suscripcion/`,
`retraso_entregas/`, `reservas_restaurante/`) con los mismos 27 archivos. Los identificadores de
los lotes nuevos llevan un código propio (por ejemplo `MAQ-U03-00012`) para no confundirse con
los del entrenamiento.

## Cómo se simulan los cambios

- **Cambio de población:** se genera un conjunto grande con el mismo generador y se re-muestrea
  dando más peso a un extremo de una columna clave. La relación entre los datos y el resultado
  no cambia; sólo llegan otros casos.
- **Cambio de comportamiento:** en el lote fuerte y en el de reentrenamiento, parte de los casos
  de bajo riesgo pasa a tener el evento y parte de los de alto riesgo deja de tenerlo. Las
  columnas que se conocen después del evento se recalculan para seguir siendo coherentes.

| Dominio | Columna clave del cambio | Categoría nueva | Unidades cambiadas |
|---|---|---|---|
| Mantenimiento | `desgaste_herramienta_min` ↑ | `tipo_maquina = D` | `temperatura_aire_c` en °F |
| Suscripción | `sesiones_ultimo_mes` ↓ | `plan = familiar` | `minutos_promedio_sesion` en horas |
| Entregas | `distancia_km` ↑ | `clima_previsto = nieve` | `distancia_km` en millas |
| Reservas | `dias_anticipacion` ↑ | `canal_reserva = whatsapp` | `dias_anticipacion` en horas |

Prevalencia del evento (mismo orden: base → estable → leve → fuerte → reentrenamiento):
mantenimiento 12 % → 12 % → 17 % → 30 % → 28 %; suscripción 17 % → 18 % → 19 % → 26 % → 25 %;
entregas 22 % → 22 % → 29 % → 32 % → 32 %; reservas 14 % → 16 % → 15 % → 26 % → 25 %.

## Archivos por dominio

### Tamaños para entrenar

| Archivo | Filas | Para qué |
|---|---|---|
| `entrenamiento_grande.csv` | 15 000 | Entrenar con más datos; probar tiempos |
| `entrenamiento_pequeno.csv` | 300 | Pocos casos del evento (~35) |
| `entrenamiento_minimo.csv` | 60 | Al límite: pocos casos del evento para separar entrenamiento, validación y prueba |

### Lotes de uso del modelo

| Archivo | Qué tiene | Qué debería pasar |
|---|---|---|
| `uso_01_estable.csv` | 1000 casos como los del entrenamiento | Sin cambios; aciertos parecidos a la prueba final |
| `uso_02_cambio_leve.csv` | 1000, población algo desplazada | Cambio moderado en la columna clave |
| `uso_03_cambio_fuerte.csv` | 1500, población y comportamiento nuevos | Cambio fuerte **y** caída de aciertos |
| `uso_04_sin_resultado.csv` | 800, sin la columna del resultado | Prioriza; no puede medir aciertos |
| `uso_05_pocos_casos.csv` | 20 | Muy pocos para concluir nada |
| `uso_06_categoria_nueva.csv` | 30 % con una categoría nunca vista | Funciona; debe avisarse la categoría nueva |
| `uso_07_muchos_vacios.csv` | 45 % vacíos en una columna | Funciona (se completan); debe avisarse |
| `uso_08_unidades_cambiadas.csv` | Una columna en otra unidad | Funciona **en silencio** con resultados absurdos; debe detectarse |
| `uso_09_columnas_reordenadas_y_extra.csv` | Columnas en orden inverso y una extra | Funciona; la extra se ignora |
| `uso_10_filas_ya_vistas.csv` | 200 filas del entrenamiento + 200 nuevas | Las vistas no cuentan para medir aciertos |

### Lotes que deben fallar o bloquearse

| Archivo | Qué tiene | Dónde se detiene |
|---|---|---|
| `falla_01_columna_faltante.csv` | Falta una columna del modelo | Al usar: `MISSING_FEATURE_COLUMNS` |
| `falla_02_columna_renombrada.csv` | Una columna con otro nombre | Al usar: `MISSING_FEATURE_COLUMNS` |
| `falla_03_resultado_desconocido.csv` | Resultado «pendiente» y «Sí» con mayúscula | Al usar: `UNKNOWN_OUTCOME_VALUES` |
| `falla_04_punto_y_coma_decimal_coma.csv` | `;` como separador y `,` decimal | Al cargar: `AMBIGUOUS_DELIMITER` |
| `falla_05_codificacion_latin1.csv` | Latin-1 en vez de UTF-8 | Al cargar: `INVALID_ENCODING` |
| `falla_06_texto_en_columna_numerica.csv` | «n/d» en una columna numérica | Al usar: `NON_NUMERIC_FEATURE_VALUES` |
| `falla_07_solo_encabezado.csv` | Sólo encabezado | Al cargar: `NO_DATA_ROWS` |

### Reentrenamiento

`reentrenamiento_reciente.csv`: 6000 casos con la población y el comportamiento del lote fuerte,
con resultado. Sirve para entrenar un modelo nuevo y compararlo con el actual sobre
`uso_03_cambio_fuerte.csv`.

### Auxiliares para cruzar varios CSV (con la tabla base del dominio)

Todos se cruzan con el archivo base (`data/synthetic/<dominio>.csv`), que comparte sus claves,
salvo `auxiliar_*_para_entrenamiento_grande.csv`, que corresponde a `entrenamiento_grande.csv`.
Cruzar un auxiliar con otro archivo da `JOIN_HAS_NO_MATCHES`: son casos distintos, y cruzar
agrega columnas, no filas.

| Archivo | Qué tiene | Comportamiento (20/10) |
|---|---|---|
| `auxiliar_*_ok.csv` | Una fila por caso, clave idéntica | Cruza todas las filas |
| `auxiliar_*_claves_duplicadas.csv` | 150 claves repetidas | Rechazo: `AUXILIARY_KEY_NOT_UNIQUE` |
| `auxiliar_*_claves_faltantes.csv` | 30 % de los casos sin fila | Cruza 70 %; el resto queda vacío y se completa |
| `auxiliar_*_claves_otro_formato.csv` | Claves en minúscula y con `_` | Rechazo: `JOIN_KEYS_FORMAT_MISMATCH` (antes se aceptaba con 0 coincidencias) |
| `auxiliar_*_para_entrenamiento_grande.csv` | Una fila por caso de `entrenamiento_grande.csv` | Cruza las 15 000 filas de ese archivo |
| `auxiliar_*_columnas_repetidas.csv` | Trae columnas con el nombre del resultado y de una variable | Acepta y advierte `AUXILIARY_REPEATS_PRIMARY_COLUMNS` como posible fuga |

## Resultado real con un modelo de mantenimiento (20/10, prueba automatizada)

Modelo entrenado y aprobado sobre `mantenimiento_maquinas.csv` (Random Forest de prueba, 30
árboles). Cifras de una corrida de prueba, **no de desempeño de negocio**:

| Lote | Cambio en los datos | Alertas | Precisión | Cobertura |
|---|---|---|---|---|
| Estable | Estable (PSI máx. 0,02) | 227 / 1000 | 0,33 | 0,63 |
| Cambio leve | Vigilar (0,17) | 345 / 1000 | 0,37 | 0,74 |
| Cambio fuerte | Revisar (1,34) | 819 / 1500 | 0,24 | 0,43 |
| Unidades cambiadas | Revisar (8,23; 100 % fuera de rango) | **1000 / 1000** | 0,13 | 1,00 |

El lote de unidades cambiadas muestra por qué hace falta monitorear los datos: el modelo no
falla, pero pone todo en alerta. Desde MON-01 Goal Settings Analytics Studio lo marca «Revisar». Además: categoría
nueva → Revisar; muchos vacíos → Revisar; pocos casos → «muy pocos casos para concluir»;
columnas reordenadas, filas ya vistas y sin resultado → Estable.
