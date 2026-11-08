# LFM Harness

App Android personal para chatear con un LLM **100 % local** (LFM3-1.7B en GGUF), con el
núcleo en **Rust** y la interfaz en **Kotlin/Jetpack Compose**. Nada sale del teléfono: no
hay red, ni cuentas, ni telemetría.

> Estado: MVP funcional verificado en un CUBOT P90 (Helio G99, 12 GB, Android 15).
> Plan y próximos pasos en [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Qué hace

- **Chat en streaming** con el modelo cargado desde un `.gguf` del almacenamiento (se copia
  a la carpeta privada de la app).
- **Historial persistente** (SQLite): lista de chats, renombrar, borrar, se restaura el
  último chat al abrir.
- **Herramientas en sandbox** que el modelo puede llamar: calculadora, fecha/hora, buscar en
  chats y notas (listar, leer, guardar, borrar). Sin red, archivos ni shell.
- **Modo código**: si pides código, la respuesta se fuerza a un único bloque, con muestreo
  preciso, y el Rust se valida sintácticamente en el propio móvil con `syn`.
- **Markdown** renderizado de forma nativa (títulos, listas, tablas, código…).
- **Razonamiento** opcional (`<think>`), mostrado en una sección plegable.

## Arquitectura

```
lfm-harness/
├── core/                     crate Rust `harness_core` (cdylib + rlib)
│   ├── src/engine.rs         llama.cpp: carga, streaming, caché con checkpoints,
│   │                         prefill/stop/límite de razonamiento
│   ├── src/chat.rs           formato de chat LFM3 (reimplementa la plantilla Jinja)
│   ├── src/harness.rs        bucle de herramientas + modo código
│   ├── src/tools.rs          herramientas, lista blanca, límites, saneado
│   ├── src/codecheck.rs      detección de código, `syn`, resumen de errores de rustc
│   ├── src/store.rs          SQLite: chats, mensajes, notas, búsqueda
│   ├── src/markdown.rs       pulldown-cmark → árbol de bloques para Compose
│   ├── src/jni.rs            puente JNI (JSON en la frontera)
│   ├── src/bin/spike.rs      chat de terminal (PC y `adb shell`)
│   ├── examples/codebench.rs benchmark de programación en Rust
│   └── bench/rust_tasks.txt  16 tareas con tests ocultos
├── android/                  app Gradle (Kotlin + Compose); compila el núcleo vía cargo-ndk
├── scripts/build-rust.sh     cargo-ndk → android/app/src/main/jniLibs
└── docs/                     roadmap y resultados de benchmarks
```

Decisiones clave:

- **llama.cpp vía `llama-cpp-2`** fijado a un commit de git: la versión de crates.io no
  incluye la arquitectura `lfm3`.
- **Plantilla de chat propia**: llama.cpp no ejecuta la plantilla Jinja del GGUF; `chat.rs`
  la reproduce (herramientas, turnos de herramienta y reglas de `<think>`), con tests.
- **Caché en modelo híbrido**: LFM3 tiene estado recurrente y no admite recortar la caché;
  se guardan *checkpoints* parciales (fin del prompt y fin del turno de sistema) para
  reutilizar el prefijo entre turnos y entre chats.
- **Toda la lógica en Rust**; Kotlin solo dibuja y persiste lo que devuelve el núcleo.

## Seguridad de las herramientas

El modelo (1.7B) se equivoca a menudo, así que el harness no confía en él:

| Regla | Detalle |
|---|---|
| Lista blanca | 7 herramientas; desconocidas/desactivadas o con parámetros extra → error, no se ejecuta nada |
| Límites | 1 llamada por ronda, máx. 3 por mensaje; argumentos ≤ 2 KB; resultados ≤ 2000 caracteres; ≤ 100 notas de ≤ 4000 caracteres |
| Confirmación | `save_note` y `delete_note` piden permiso al usuario |
| Rechazo | termina el turno con un mensaje fijo (el modelo llegó a afirmar que había borrado una nota rechazada) |
| Saneado | los resultados neutralizan `<\|im_end\|>`, `<tool_call>`, etc. para evitar inyecciones |
| Aviso | la app muestra «⚠ N acción(es) no realizada(s)» diga lo que diga el modelo |

## Compilar y ejecutar

Requisitos: Rust (target `aarch64-linux-android`), `cargo-ndk`, Android NDK (r30), SDK
Android 36, JDK 17.

```bash
# Núcleo en el PC (tests y herramientas de desarrollo)
cd core
cargo test --release --lib
cargo run --release --bin spike -- ~/models/LFM3-1.7B.gguf --tools

# App (Gradle compila primero el núcleo con scripts/build-rust.sh)
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

`GGML_CPU_ARM_ARCH` (por defecto `armv8.2-a+dotprod+fp16`) selecciona las extensiones ARM de
ggml; tras cambiarlo hay que limpiar `llama-cpp-sys-2` (ver `scripts/build-rust.sh`).

## Rendimiento medido (LFM3-1.7B Q4_K_M)

| Dispositivo | Generación | Prompt | Carga |
|---|---|---|---|
| CUBOT P90 (Helio G99, 2 hilos grandes) | ~10 tok/s | ~40 tok/s | ~4 s |
| PC (CPU) | ~25 tok/s | ~200 tok/s | <1 s |

## Benchmark de programación en Rust

`codebench` pide código al modelo, lo compila con `rustc --test` junto a tests ocultos y,
opcionalmente, le devuelve los errores para que repare. Resultados y análisis en
[`docs/benchmarks/2026-11-07-rust-codebench.md`](docs/benchmarks/2026-11-07-rust-codebench.md).

```bash
cd core
cargo run --release --example codebench -- MODEL.gguf --harness --repair 2
cargo run --release --example codebench -- --refs REFS.txt   # valida los tests
```
