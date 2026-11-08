# Benchmark de programación en Rust (LFM3-1.7B, 2026-11-07)

Primera medición de la capacidad del modelo para escribir código Rust, y de si el modo código
del harness mejora el resultado. Herramienta: `core/examples/codebench.rs`. Tareas y tests
ocultos: `core/bench/rust_tasks.txt` (16 tareas). Soluciones de referencia que validan los
tests: `core/bench/refs.txt` (`codebench --refs`, 16/16 pasan).

## Condiciones

- Modelo: LFM3-1.7B Q4_K_M (el GGUF de uso diario, `LFM3-1.7B-Uncensored-AD`; ver ROADMAP, item 6).
- Ejecutado en PC (CPU), no en el móvil. Los resultados de calidad no dependen del dispositivo;
  el tiempo sí.
- Verificación: `rustc --edition 2021 --test` con tests ocultos añadidos como `mod hidden_tests`.
  Límite de 20 s por ejecución. Resultados: `pass`, `test_fail`, `compile_error`, `no_code`,
  `timeout`, `forbidden`.
- `pass@1`: la primera respuesta pasa todos los tests. `tras reparar`: alguna de las
  respuestas (primera o reparaciones) pasa. Reparaciones: 2 por tarea.
- Muestreo no determinista (temperatura > 0), una muestra por tarea. Con 16 tareas, **una
  tarea equivale a 6 puntos**; diferencias de 1–2 tareas son ruido.

## Pasadas

| Pasada | Configuración | pass@1 | tras reparar | Tokens generados (media por tarea) |
|---|---|---|---|---|
| A | sin modo código, sin razonamiento, repair 2 | **2/16** | **5/16** | 2201 |
| B | sin modo código, con razonamiento | incompleta (6 de 16) | 0/6 → 2/6 | 6991 |
| C | modo código, sin razonamiento, repair 2 (repair a temp 0.15) | **1/16** | **3/16** | 1269 |
| D | modo código + razonamiento limitado a 384 tokens, repair 2 (repair a temp 0.15) | **5/16** | **7/16** | 2503 |
| C′ | igual que C, reparaciones a temp ≥ 0.6 y semilla nueva | **1/16** | **3/16** | 1424 |

Datos crudos en `docs/benchmarks/data/` (JSONL por intento, y resumen en TXT).
Recuento con `docs/benchmarks/data/score.py`.

Notas sobre cada pasada:

- **A** es la referencia: pasa `fizzbuzz` y `palindrome` a la primera; tras reparar suma
  `binary_search` y `matmul`.
- **B** se detuvo a mitad (tarea `matmul`) porque el razonamiento sin límite consume todo el
  presupuesto: 12 de 17 intentos terminaron sin código (`no_code`, 3072 tokens).
- **C** es peor que A: el modo código sin razonamiento empeora el primer intento. El muestreo
  fijo (temp 0.15) produce la misma respuesta en 9 de las 28 reparaciones.
- **D** es la mejor configuración medida. Razonar un poco (384 tokens) ayuda; sin límite, el
  modelo no llega a escribir código.
- **C′** prueba si subir la temperatura de la reparación (≥ 0.6) y cambiar la semilla rompe el
  bucle de respuestas idénticas. Resultado: no mejora (1/16 → 3/16, igual que C; aprueban `palindrome` y `binary_search`
  tras reparar). Las reparaciones idénticas bajan de 9 a 5 de 28, pero el cambio no se
  traduce en más aciertos: el modelo sigue sin corregir errores como E0599 (`stack`,
  `matmul`, `group_by`) aunque cambie de respuesta. Datos: `bench_harness_repairtemp.*`.

## Modelo oficial (LFM3-1.7B QAD-Q4_0, 2026-11-08)

Mismas 16 tareas, mismas condiciones (PC, CPU, repair 2). Archivo oficial
`LFM3-1.7B-QAD-Q4_0.gguf` (1,05 GB, SHA-256 `43a7d09a…7452`, coincide con Hugging Face).

| Modelo | Configuración | pass@1 | tras reparar | Tokens (media/tarea) |
|---|---|---|---|---|
| Fine-tune Q4_K_M | A: sin modo código | 2/16 | 5/16 | 2201 |
| Oficial QAD-Q4_0 | sin modo código | 1/16 | 2/16 | 1687 |
| Fine-tune Q4_K_M | D: modo código + razonamiento 384 | 5/16 | 7/16 | 2503 |
| Oficial QAD-Q4_0 | modo código + razonamiento 384 | 4/16 | 5/16 | 2184 |

- Datos: `data/official_nothink.*` y `data/official_harness_think.*`.
- El modelo oficial no mejora al fine-tune en código; queda 1–3 tareas por debajo, dentro del
  ruido de una muestra por tarea pero sin ventaja visible. El modo código con razonamiento
  sigue siendo mucho mejor que sin él (4/16 frente a 1/16), igual que con el fine-tune.
- Aprueba a la primera `fizzbuzz`, `palindrome`, `stack` y `matmul` (modo código); `group_by`
  tras reparar.
- En el CUBOT P90: 9,3–9,4 tok/s de generación (el fine-tune daba ~10), RAM ~2,4 GB (PSS).
  Q4_0 no es más rápido que Q4_K_M en este SoC con el build actual.
- Incidente: la segunda pasada se cortó dos veces porque un programa generado para `fib_iter`
  consumía hasta 11 GB al ejecutar los tests y `systemd-oomd` mataba el proceso. `codebench`
  ahora limita cada ejecución a 2 GiB (`prlimit --as`). Las 11 primeras tareas de esa pasada
  se ejecutaron antes del límite y las 5 restantes (`fib_iter` … `group_by`) después, en una
  segunda ejecución con `--only`; `official_nothink.jsonl` es la unión de ambas.

## Qué falla

Clasificación de los fallos en la pasada C (sin razonamiento), donde hay más casos:

1. **Prosa mezclada con código y bloques partidos.** El modelo escribe varios bloques o texto
   antes del código. El extractor toma el bloque Rust más largo; aun así algunos quedan
   truncados.
2. **Corte por límite de tokens.** 15 de 42 intentos de la pasada A se cortaron en el límite
   (1536 tokens). En B, 12 de 17 intentos agotaron los 3072 tokens sin escribir código.
3. **Métodos inventados (E0599).** Llama a métodos que no existen en la biblioteca estándar o en el tipo
   usado. Aparece en `stack`, `rpn`, `shapes` y `palindrome`, entre otras.
4. **Desajuste de tipos (E0308, E0277).** Mezcla `&str`/`String`, `usize`/`i32`, o devuelve
   el tipo equivocado. Es el fallo más frecuente en `matmul`, `binary_search` y `rpn`.
5. **Regresiones tras reparar.** `group_by` duplica el nombre de la función; `palindrome`
   inventa una función `eq_impl` no pedida; `linked_list` deja delimitadores sin cerrar.
6. **Reparaciones idénticas.** En C, 9 de 28 reparaciones devolvieron la misma respuesta que
   la anterior aunque recibieron el error de rustc. Causa probable: temperatura 0.15 con
   semilla fija. Motiva C′.
7. **Errores de sintaxis que `syn` detecta** (`rust_syntax_error`): en D, `shapes` no logra
   una sintaxis válida en ninguna de sus 3 rondas.

## Análisis

- **El modo código por sí solo no basta.** C (forzar bloque, prefill, parada y muestreo
  preciso) empeora frente a A. Forzar formato no compensa la falta de capacidad del modelo.
- **El razonamiento limitado sí aporta.** D frente a C cambia solo el presupuesto de
  razonamiento (384 tokens): 1 → 5 aciertos a la primera. Consistente con que el modelo
  necesita "pensar" un poco antes de escribir, pero no mucho.
- **`syn` y la reparación automática funcionan.** En D reparó automáticamente la sintaxis en
  varios casos (`palindrome`, `word_freq`, `rpn`); en C′ la reparación con
  temperatura alta no resultó mejor que la de 0.15.
- **Los fallos lógicos dominan al final.** Los casos que ya compilan (`rpn`, `longest_word`,
  `fib_iter`) fallan los tests por lógica, no por tipos. Ese error no lo detecta `syn` ni
  rustc; solo los tests.
- **Muestra pequeña.** 16 tareas, una muestra por tarea y temperatura > 0. Las conclusiones
  son orientativas. Para decidir entre configuraciones con más seguridad hacen falta más
  tareas y varias muestras por tarea.

## Propuestas (roadmap, item 7)

1. C′ confirmado sin mejora: la temperatura alta en la reparación reduce las respuestas
   repetidas (9 → 5 de 28) pero no los fallos; el cuello de botella es la capacidad del modelo.
2. Usar D como configuración de referencia para el modo código: razonamiento limitado a
   384 tokens y repair 2.
3. Añadir a las reparaciones una frase fija sobre el error más frecuente de la tarea (E0308,
   E0599) en lugar de solo el mensaje de rustc.
4. Medir el prompt de sistema con una hoja de errores frecuentes de Rust, con y sin ella.
5. Ampliar el benchmark a más tareas y 3 muestras por tarea antes de cambiar el valor por
   defecto de forma definitiva.
6. Repetir la misma batería con el modelo oficial (item 6 del roadmap) antes de decidir.

## Reproducir

```bash
cd core
# validar tests con soluciones de referencia
cargo run --release --example codebench -- --refs bench/refs.txt
# una pasada con modo código y reparación
cargo run --release --example codebench -- MODEL.gguf --harness --repair 2 --out res.jsonl
# con razonamiento limitado
cargo run --release --example codebench -- MODEL.gguf --harness --think --repair 2
```
