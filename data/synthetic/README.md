# Datasets sintéticos de Goal Settings Analytics Studio

Cuatro archivos **100 % falsos**, creados a propósito para demos y pruebas. Ningún valor proviene
de personas, empresas ni sistemas reales: salen de `scripts/generate_synthetic_datasets.py` con
semilla fija, así que regenerarlos produce exactamente los mismos bytes (lo comprueba
`tests/unit/test_synthetic_datasets.py`). Son parte del código del proyecto y comparten su
licencia propietaria (`LICENSE`).

```bash
python scripts/generate_synthetic_datasets.py
```

Clasificación al cargarlos en Goal Settings Analytics Studio: **Sintético**. Procedencia sugerida:
«Generado por scripts/generate_synthetic_datasets.py (datos falsos)».

**Variantes para monitoreo y pruebas de fallo:** tamaños distintos, lotes con cambios, archivos
mal formados y auxiliares con claves problemáticas en [`variantes/`](variantes/README.md).

## Trampas deliberadas

Cada archivo incluye trampas para comprobar que el asistente y la persona las detectan:

| Trampa | Por qué importa | Decisión correcta |
|---|---|---|
| Identificador único por fila | No ayuda a predecir; el modelo podría memorizar | Excluir |
| Columna constante | No aporta información | Excluir |
| Valores vacíos en una variable | Prueba el completado automático | Usar (se completa dentro del proceso) |
| Varias columnas con dos valores | El resultado a anticipar no es obvio | Elegir el evento correcto |
| Dato conocido **después** del evento | Fuga de información: resultados engañosamente perfectos | Marcar «no existe al momento de decidir» |

## 1. `mantenimiento_maquinas.csv` — fallas de máquinas

4000 filas · 11 columnas · evento `falla_maquina = sí` en 482 filas (12,0 %).

| Columna | Descripción | Trampa |
|---|---|---|
| `id_maquina` | Código MAQ-00000… | Identificador |
| `tipo_maquina` | A, B o C | |
| `temperatura_aire_c`, `temperatura_proceso_c` | Grados Celsius | |
| `velocidad_rpm` | Revoluciones por minuto | 6 % vacíos |
| `torque_nm`, `desgaste_herramienta_min` | Torque y desgaste | |
| `turno_noche` | sí/no | Segunda columna de dos valores |
| `planta` | Siempre «Planta Demo» | Constante |
| `horas_detenida_post_falla` | Horas detenida tras fallar (0 si no falló) | **Posterior al evento** |
| `falla_maquina` | sí/no | Resultado a anticipar |

Pregunta de negocio sugerida: «¿Qué máquinas revisar al inicio del turno?».

## 2. `abandono_suscripcion.csv` — bajas de un servicio de suscripción

3000 filas · 11 columnas · evento `abandona = sí` en 519 filas (17,3 %).

| Columna | Descripción | Trampa |
|---|---|---|
| `id_suscriptor` | Código SUS-00000… | Identificador |
| `meses_antiguedad`, `plan`, `sesiones_ultimo_mes` | Uso y contrato | |
| `minutos_promedio_sesion` | Minutos por sesión | 8 % vacíos |
| `tickets_soporte_90d`, `region` | Soporte y zona | |
| `pago_automatico` | sí/no | Segunda columna de dos valores |
| `moneda` | Siempre «CLP-demo» | Constante |
| `encuesta_salida_completada` | Encuesta que se contesta al darse de baja | **Posterior al evento** (sutil) |
| `abandona` | sí/no | Resultado a anticipar |

Pregunta de negocio sugerida: «¿A quiénes contactar este mes para evitar que se vayan?».

## 3. `retraso_entregas.csv` — envíos que llegan tarde

3500 filas · 12 columnas · evento `entregado_tarde = sí` (más de 40 minutos) en 779 filas (22,3 %).

| Columna | Descripción | Trampa |
|---|---|---|
| `id_envio` | Código ENV-000000… | Identificador |
| `distancia_km`, `peso_kg`, `tipo_servicio`, `dia_semana` | Datos del envío | `tipo_servicio` también tiene dos valores |
| `clima_previsto` | despejado / lluvia / tormenta | 5 % vacíos |
| `pedidos_dia_centro` | Carga del centro de distribución | |
| `conductor_nuevo` | sí/no | Otra columna de dos valores |
| `pais_operacion` | Siempre «País Demo» | Constante |
| `minutos_retraso_real` | Retraso medido al entregar | **Posterior al evento** (lo define) |
| `reclamo_cliente` | sí/no, reclamo tras la entrega | **Posterior al evento** y de dos valores |
| `entregado_tarde` | sí/no | Resultado a anticipar |

Pregunta de negocio sugerida: «¿Qué envíos reforzar antes de despacharlos?».

## 4. `reservas_restaurante.csv` — reservas a las que el cliente no llega (dataset «sorpresa»)

2500 filas · 14 columnas · evento `no_asistio = sí` en 355 filas (14,2 %). Pensado para
mostrar en vivo que el mismo producto sirve en un dominio nuevo sin cambiar código.

| Columna | Descripción | Trampa |
|---|---|---|
| `id_reserva` | Código RES-00000… | Identificador |
| `dias_anticipacion`, `personas_mesa`, `dia_semana`, `turno`, `canal_reserva` | Datos de la reserva | `turno` también tiene dos valores |
| `cliente_frecuente`, `confirmo_recordatorio` | sí/no | Otras columnas de dos valores |
| `cancelaciones_previas` | Historial del cliente | |
| `pronostico_lluvia` | sí/no | 4 % vacíos |
| `cadena` | Siempre «Restaurantes Demo» | Constante |
| `propina_registrada`, `minutos_espera_mesa` | Sólo existen si la persona llegó | **Posteriores al evento** |
| `no_asistio` | sí/no | Resultado a anticipar |

Pregunta de negocio sugerida: «¿A qué reservas llamar esta mañana para liberar mesas?».

## Comprobación exploratoria (20/10/2026, no oficial)

Entrenamiento real con la estrategia fija, medida principal «calidad global de las alertas»
(average precision) en la prueba final. Sólo muestra el efecto de las trampas; **no es evidencia
de desempeño de negocio** y debe repetirse en la corrida oficial con el responsable.

| Dataset | Sin las columnas posteriores al evento | Usándolas (trampa) |
|---|---|---|
| Mantenimiento | 0,430 | 1,000 |
| Suscripción | 0,340 | 0,860 |
| Entregas | 0,729 | 1,000 |

Un resultado casi perfecto es una señal de alarma, no un éxito.
