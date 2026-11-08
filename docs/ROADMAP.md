# Roadmap

App personal, local y contenida: el modelo nunca tiene acceso amplio al dispositivo y todo
lo que modifica datos pasa por el usuario. Las decisiones se toman con datos medidos
(`spike`, `codebench`, pruebas en el CUBOT P90).

## Hecho

| Fase | Contenido | Commit |
|---|---|---|
| 1. Spike | `llama-cpp-2` (git, con `lfm3`) compilado con cargo-ndk; ~10 tok/s en Helio G99 | `9014677` |
| 2. Núcleo | streaming, cancelación, plantilla LFM3 propia, caché con checkpoints (modelo híbrido), recorte de historial | `9014677` |
| 3. App | Kotlin/Compose: importar GGUF, chat, detener, ajustes | `9014677` |
| 4. Persistencia | SQLite en Rust: chats, mensajes, menú lateral, renombrar/borrar | `a004673` |
| 5. Herramientas | sandbox con lista blanca, límites, confirmación, saneado, avisos | `d889c05` |
| — | Markdown nativo (pulldown-cmark → Compose) | `6446896` |
| — | `codebench` + modo código (prefill, muestreo, límite de razonamiento, `syn`, errores resumidos) | este cambio |

## Próximo

### 6. Evaluar el modelo oficial LFM3-1.7B (en curso: QAD-Q4_0 elegido el 2026-11-08)
El modelo de uso diario es el fine-tune `LFM3-1.7B-Uncensored-AD`. Antes de cambiar:
- [x] Verificar los pesos (hash SHA-256 de QAD-Q4_0 coincide; falta comparar metadatos GGUF): hash SHA-256 contra Hugging Face, metadatos GGUF (arquitectura,
      plantilla, tokens especiales) y cuantización equivalente (Q4_K_M / Q6_K).
- [x] Velocidad en el móvil (9,3 tok/s, ~2,4 GB RAM; sin mejora sobre el fine-tune) (tok/s de generación y prompt, RAM, tiempo de carga) con `spike`.
- [x] Calidad (4/16→5/16 con modo código; fine-tune 5/16→7/16): `codebench` (sin/con razonamiento, con/sin modo código) y un conjunto pequeño
      de chats y llamadas a herramientas.
- [ ] Comprobar si el oficial respeta `enable_thinking=false` (el fine-tune razona igualmente).
- [ ] Decidir: modelo por defecto o ambos seleccionables.

### 7. Precisión del modo código (según `docs/benchmarks/`)
- [x] Razonamiento de 384 tokens siempre activo en modo código (configuración D: 5/16 → 7/16).
- [x] Variar la temperatura en las reparaciones (probado en C′: menos repeticiones, sin más aciertos). Antes, con 0.15 el modelo repetía respuestas
      idénticas aunque reciba el error (visto en `rpn`).
- [ ] Comprobar en el móvil la firma pedida (nombre, nº de parámetros) con `syn` y avisar si
      no coincide.
- [ ] Hoja de errores frecuentes de Rust en el prompt de sistema del modo código (E0308,
      E0599, E0277) y probar si mejora el pass@1.
- [ ] Explicación opcional tras el bloque de código (segunda generación corta).

### 8. Gestión de modelos
- [ ] Varios modelos importados, cambiar entre ellos, borrar y ver el espacio usado.
- [ ] Perfiles de muestreo por modelo.

### 9. Robustez y experiencia
- [ ] Servicio en primer plano para no perder la generación en segundo plano.
- [ ] Indicador de contexto usado y resumen de turnos antiguos al acercarse al límite.
- [ ] Exportar/importar chats y notas (JSON) para copias de seguridad.
- [ ] Reintentar/editar el último mensaje.

### 10. Rendimiento
- [ ] Evaluar el backend Vulkan (Mali-G57) frente a la CPU.
- [ ] Cuantizaciones Q4_0 (re-empaquetado ARM) frente a Q4_K_M en velocidad y calidad.

### 11. Calidad del proyecto
- [ ] CI en GitHub Actions: `cargo test` del núcleo (sin modelo) y `assembleDebug`.
- [ ] Ejecutar `codebench` automáticamente al cambiar el núcleo o el modelo y guardar
      el histórico en `docs/benchmarks/`.

## Fuera de alcance (por decisión)
- Acceso a red, archivos del sistema o shell desde las herramientas del modelo.
- Acciones irreversibles sin confirmación del usuario.
