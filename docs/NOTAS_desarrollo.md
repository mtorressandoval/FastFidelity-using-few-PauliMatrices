# Notas de desarrollo (resumen de la inspección, decisiones y estado)

## Inspección del repositorio original
- `tomography/FastFidelity.py`: `Mean_Direct_Fidelity` (DFE de Flammia–Liu con muestreo
  por χ², valores esperados con el `Estimator` V1 de Aer y una caché `Measures` de valores
  esperados ya medidos), `Random_Measurements`.
- `tomography/Tomography.py`: SGQT con CSPSA y post-procesado `NearSparseTomography`.
- Notebooks y `SGQT_for_mixed.py` de simulaciones de tomografía.
Todo esto se ha movido sin cambios a `tomography_sgqt/` y funciona con Qiskit 2.6.2.

## Decisiones
1. **DFE sin almacén y con almacén.** `FreshDFE` es la DFE de la librería para objetivos
   estabilizadores (χ² uniforme sobre el grupo) con un disparo por Pauli. `StoreDFE` comparte
   el mismo muestreo: con el almacén vacío consume los mismos números aleatorios y da el mismo
   resultado (prueba `test_empty_store_reproduces_fresh_dfe`).
2. **Estimador estratificado** (petición del 12/11/2026): estrato determinista = Paulis del
   grupo que ya están en el almacén (todos, con peso 1/|U| y la media de sus resultados,
   completados hasta ⌈N/|U|⌉ resultados); estrato aleatorio = resto del grupo, muestreado con
   ⌈N|R|/|U|⌉ disparos. Insesgado y nunca más ruidoso que una DFE nueva (pruebas 5–6 con 10⁴
   repeticiones). Sustituye al estimador por «átomos» del prototipo: mi implementación de ese
   estimador mostró sesgo (|z| hasta 11) en secuencias de consultas solapadas; no se usa.
3. **Filtro de direcciones** p̂ₓ > (3−2√2) p̂₀: activado por defecto. Solo descarta direcciones
   que no pueden mejorar; reduce los disparos ~4× sin perder éxito (E5).
4. **Vaciar el almacén antes de cada reintento**: opción `refresh_on_retry`, desactivada por
   defecto (mejora marginal con estados densos, el doble de disparos).
5. **n grande**: la transformada de Walsh sobre 2ⁿ direcciones se sustituye por una búsqueda
   por prefijos sobre subgrupos (`walsh_block`), comprobada contra la exacta para n ≤ 8.
6. **Simulaciones**: todas con n ≤ 8; los scripts de `experiments/large/` quedan preparados
   (mezclas de estabilizadores hasta n = 100; Aer MPS y GPU) y solo se probaron como humo.

## Limitaciones conocidas
- El almacén es insesgado para secuencias de consultas fijadas de antemano. En la búsqueda,
  las decisiones dependen de los datos reutilizados; por eso la fidelidad final se estima
  siempre con copias nuevas.
- El error estándar del almacén con subcadenas ignora las correlaciones entre subcadenas de un
  mismo disparo (la varianza real medida es menor que la de una DFE nueva, prueba 6).

## Resultados principales (n ≤ 8; detalles en results/summary.txt)
- E1: óptimo en 1080/1080 ejecuciones (arranque a 3 pasos). Ahorro 3,79 ± 0,05 (almacén) y
  5,21 ± 0,08 (con subcadenas); pendiente con n: +0,10 ± 0,04 y −0,07 ± 0,07 por qubit.
- E2: óptimo siempre para τ ≥ 0,8 desde arranque aleatorio (n ≤ 5); sin certificados falsos;
  cobertura del IC al 99 %: 99,1 %.
- E4: desde arranque aleatorio, Grewal et al. (Bell, dos copias) usa ~5×10³ copias (teoría) o
  ~1,4×10³ (ajustado) frente a 1,1–2,5×10⁴ del protocolo. Para n = 7–8 la búsqueda sin almacén
  no sale de la región plana (fidelidades ~2⁻ⁿ < ruido 1/√N) y agota las 200 iteraciones; el
  almacén llega al 90 % (N = 1000) y al 100 % (N = 4000, E4b).
- E5: K no influye; el filtro reduce ~4× los disparos; la distancia del arranque es el parámetro
  más importante.

## Errores corregidos durante el desarrollo
- Indexación booleana con arrays `object` (n > 31) en `StoreDFE._known`, que bloqueaba la ruta de n grande.
- `cliff.stab` devuelve arrays en Qiskit 2.x (no PauliList): se usan `stab_x`, `stab_z`, `stab_phase`.

## Revisión 2 (simulaciones del artículo)
- Simulación 1 (`sim1_methods.py`): comparación de los tres estimadores con estados
  estabilizadores perturbados (τ = 0,9), arranque aleatorio, n = 3–8; trayectorias por iteración.
- Simulación 2 (`sim2_families.py`): solo almacén + subcadenas; familias: estabilizador +
  perturbación (τ = 0,86; 0,9; 0,95), estados T-dopados C(T|+⟩)^⊗t|0⟩ (t = 1, 2, 3) y
  estabilizadores despolarizados (p = 0,1; 0,3; 0,5). F_STAB exacto para todo n.
- F_STAB((T|+⟩)^⊗t) = cos^{2t}(π/8) verificado por enumeración para t ≤ 4.
- La primera ronda de simulaciones (E1–E5) queda en `experiments/archive/`.

## Revisión 3 (más qubits)
- Backends exactos `TDopedBackend` (C(T|+⟩)^⊗t|0⟩, coste polinómico) y
  `DepolarizedStabilizerBackend`; verificados contra la simulación densa (n ≤ 6).
- Aer `extended_stabilizer` funciona con `QiskitSamplerBackend`, pero cuesta ~0,1 s por disparo.
- Cambios de signo confirmados por defecto (`confirm_flips=True`): el máximo de 2ⁿ−1 estimaciones
  ruidosas produce cambios espurios; margen √(4n ln2)·σ(p̂₀). Simulaciones 1–2 repetidas con él
  (versión anterior en `results/round2/`).
- n grande (sim3): con N fijo = 1000 el éxito cae (n = 16 T-dopado: 2/6 con almacén, 3/6 sin).
  Con N = 125 n y `refresh_on_retry=True`: 100 % hasta n = 16; disparos ∝ n^{2.1±0.3} (T-dopado)
  y n^{2.0±0.1} (despolarizado).
- Limitación: el estrato determinista solo cubre una fracción ~N/2ⁿ del grupo, así que el ahorro
  del almacén desaparece para n = 16. Reutilizar las muestras antiguas de T_x como muestras del
  estrato aleatorio podría mantenerlo (trabajo futuro).

## Revisión 4
- Eliminada la Fig. 5 (n grande separada). La Fig. 3 se extiende a n = 10 para T-dopados (t = 2) y
  despolarizados (p = 0,3), con la misma configuración (almacén + subcadenas, arranque aleatorio,
  N = 1000): 0/5 óptimos en ambas familias; la búsqueda se detiene en la región plana tras ~10⁵ disparos.
- n = 12 no se completó (más de 10 min por ejecución); los demás parámetros quedan hasta n = 8.
- El resultado con arranque cercano y N = 125 n (hasta n = 16, solo con almacén) se cita en el texto
  como solución, sin figura (`results/archive/sim3_large`).

## Revisión 5 (extensión de la Fig. 3)
- n = 10 completo (15 estados por punto, seis parámetros): 2/90 óptimos; mediana 9·10⁴ disparos.
- n = 12: 3 estados por parámetro en esta versión (0/18 óptimos; mediana 4·10⁵ disparos). La
  ampliación a 15 estados está en marcha (`sim2_extension.py 12 15 ... --from 3 --dense`).
- Aceleración: `StoreDFE._known` enumera el grupo (2^m) en lugar de recorrer todas las cadenas
  guardadas (resultados idénticos); cachés limitadas a 4000 entradas (memoria de 3,7 GB a 0,8 GB).
- n = 14 no es viable: el espectro denso necesita ~6 GB y los simuladores exactos >20 min por ejecución.
- `make_figures.py` admite `N12_STATES=3` para usar solo los primeros estados de n = 12.

## Revisión 6
- Fig. 3 hasta n = 10 (n = 9 añadido, n = 12 retirado y archivado en `results/archive/n12`);
  el estabilizador perturbado se mantiene hasta n = 8.
- Éxito conjunto T-dopados + despolarizados: 71 % (n = 8), 28 % (n = 9), 2 % (n = 10).
- Figs. 2 y 3 con mediana y rango intercuartílico.

## Revisión 7 (arranque por bloques)
- `fastfidelity/blocks.py`: 2^k bloques contiguos de ≤ 8 qubits (k mínimo), búsqueda en cada bloque
  sobre su estado reducido, fusión por pares partiendo del producto, búsqueda global.
  `FrameBackend`: bloques en el marco de una Clifford conocida (mide C P C†; sin subcadenas).
- Resultados exploratorios, no incluidos en el artículo (T-dopados, t = 2, 3 estados por caso):
  n = 10: vecinos y pares aleatorios de profundidad 2 → 3/3 con bloques (0/3 con arranque aleatorio);
  Clifford aleatoria → 0/3 con bloques, 3/3 en el marco de C.
  n = 16 (N = 125 n): vecinos 3/3; pares aleatorios 1/3; Clifford aleatoria en el marco de C 3/3.
- Las subcadenas solo se usan en bloques de ≤ 12 qubits (memoria).
- Pendiente: elegir los bloques por correlaciones medidas (circuitos poco profundos con pares arbitrarios).

## Revisión 8 (reinicio local, exploratorio, fuera del artículo)
- Idea: el almacén con subcadenas ya contiene las estimaciones de todos los Paulis de peso ≤ k.
  `fastfidelity/local.py` construye con ellas un candidato (generadores k-locales que conmutan,
  ordenados por |ō| − 2/√m) y `local_restart=k` salta a él si su DFE supera a p̂₀ (cada 5
  iteraciones y cuando no hay mejora). Para n > 12, `LocalStats` guarda solo subcadenas de peso ≤ k.
- Simulación 5 (T-dopados t = 2, arranque aleatorio, N = 1000, k = 4): circuitos de profundidad 2
  (vecinos y pares aleatorios) → 100 % óptimo hasta n = 20; disparos ∝ n^{3.8±0.3} y n^{4.0±0.3}
  (n = 10–20). Sin reinicio: 10 % en n = 10. Clifford profunda: 100 % (n = 8), 80 % (10), 20 % (12).
- Por qué la meseta: sin ruido, la búsqueda voraz global encuentra siempre el óptimo en n = 9
  (`quick_plateau_n9.py`); el fallo es estadístico (valores ~2⁻ⁿ frente a ruido 1/√N).

## Revisión 9 (nueva propuesta en el artículo)
- Propuesta: almacén + subcadenas + reinicio local (k = 4). Eliminadas la comparación con/sin almacén
  (antigua Fig. 2), la Fig. 4 (copias frente a Grewal et al.) y la Tabla I; comparación cualitativa en
  «Discussion and conclusions». Nueva Sec. III D «Local restart».
- Fig. 2 nueva (`sim6_families.py`, `make_figures.py` con NMAX=10): 4 familias, n ≤ 10 (perturbado ≤ 8):
  perturbado (profundo), T-dopado profundo O(n), T-dopado O(log n) (⌈log₂ n⌉ capas entre vecinos,
  t = 0–3) y despolarizado O(log n). 1590 ejecuciones: 98,1 % óptimo, 0 certificados falsos.
- Umbral del candidato local: |ō| − 3/√m > 0,2.
- Datos de n = 12 ya calculados (T-dopado O(log n) completo, despolarizado parcial) se conservan
  pero no entran en la figura; extensión a n = 14–20 pendiente de decisión.

## Revisión 10 (k = max(4, 2⌈log₂ n⌉), resultados hasta n = 10)
- Con k = 4 fijo el candidato local quedaba incompleto en una fracción creciente de estados de
  profundidad log₂ n (generadores de hasta 2⌈log₂ n⌉ qubits). Regla nueva: k = max(4, 2⌈log₂ n⌉).
  Resultados con k = 4 archivados en `results/archive/sim6_k4`.
- Fig. 2 (n ≤ 10, perturbado ≤ 8): 1590/1590 ejecuciones en el óptimo, 0 certificados falsos.
  Disparos de búsqueda ∝ n^{0.58±0.04} (T-dopado O(log n)), n^{0.66±0.08} (despolarizado),
  n^{0.86±0.07} (T-dopado profundo), n^{0.90±0.07} (perturbado); exponencial igual de compatible.
- Con k = 2⌈log₂ n⌉ los circuitos profundos también funcionan hasta n = 10 (≈770 elementos de peso
  ≤ 8); el límite se espera hacia n ≈ 48–64. Peor caso del coste: 4^k ≈ n⁴.
- Datos parciales de n = 12 y 14 con la regla nueva quedan en `results/` (no entran en la figura).

## Revisión 11 (hasta n = 14)
- 1920/1920 ejecuciones en el óptimo (n ≤ 14; perturbado ≤ 8). Disparos de búsqueda en n = 14:
  4,4·10⁴ (T-dopado O(log n)), 3,2·10⁴ (despolarizado), 7,5·10⁴ (Clifford profunda).
- El salto en n = 14 coincide con el cambio de régimen del almacén (solo subcadenas de peso ≤ k).
- Ajustes n = 3–14: exponencial mejor (β = 0,145, 0,139, 0,206) que potencia (α ≈ 0,97, 0,95, 1,39).
- `LocalStats` descarta cadenas con < 3 resultados al superar 1,5·10⁷ entradas (memoria ≤ 2,3 GB).

## Notación (revisión del manuscrito)
- Cadenas de Pauli: $P$ (base $P_k$); subcadenas $P'\preceq P$; resultado de una medición $P_j=\pm1$;
  estimación de $\langle P\rangle$: $\bar P$ con $m_P$ resultados. Producto de desestabilizadores: $D_x$
  (antes $Q_x$). Estado de la búsqueda: $|\phi\rangle$ (antes $|z\rangle$), base $\{|\phi_x\rangle\}$ con
  $|\phi_0\rangle=|\phi\rangle$, candidato local $|\tilde\phi\rangle$, vecinos $|\nu_{x,m}\rangle$; estados
  estabilizadores genéricos $|\eta\rangle$, $|\eta_1\rangle$, $|\eta_2\rangle$.
