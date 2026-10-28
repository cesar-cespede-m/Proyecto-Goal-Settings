# Instrucciones de repositorio: Proyecto-Goal-Settings

**Fuente de verdad de producto:** `PLAN_MAESTRO_GOAL_SETTINGS.md`. El contexto es el objetivo anual de diseño de producto con IA y analítica de datos en Babel. No inventar el texto oficial, indicadores, resultados o aprobaciones corporativas.

## Reglas de trabajo

1. Mantener este repositorio autónomo. No configurar remotos ni publicar cambios sin autorización expresa.
2. Trabajar una historia prioritaria verificable a la vez. Registrar pregunta de diseño, evidencia necesaria y criterio de aceptación.
3. Usar sólo datos sintéticos, sandbox o anonimizados y autorizados. Nunca incluir datos de producción, información personal, claves o documentos internos de Babel.
4. Separar capacidades comprobadas, hipótesis y propuestas. Una demo sintética no acredita un resultado de negocio ni el cumplimiento del objetivo anual.
5. Para ML, separar entrenamiento, validación y prueba, prevenir fuga de información y comparar con baseline. Un LLM no sustituye al ML.
6. Mantener `core/` independiente de Streamlit y del proveedor LLM. La IA propone e interpreta; no ejecuta código arbitrario ni inventa métricas.
7. Exigir aprobación humana vigente antes de entrenar o publicar y auditar decisiones.
8. Para cambios relevantes: declarar alcance, archivos y pruebas; implementar; ejecutar comprobaciones; informar resultados, deuda y riesgos.
9. No cambiar silenciosamente contratos persistidos, variables de entorno, rutas de datos o evidencia firmada. Planificar compatibilidad antes de importar material existente.

## Entrega por historia

Indicar ID, resumen, archivos cambiados, comandos y resultados reales, supuestos, riesgos y siguiente paso. Marcar como pendiente toda integración con Babel que no esté verificada.
