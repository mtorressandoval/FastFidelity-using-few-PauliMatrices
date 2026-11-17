# Prompt: fidelidad estabilizadora autoguiada con DFE y almacén de Paulis

Lee este documento completo antes de empezar. Las secciones 1 a 6 son instrucciones; las 7 y 8 son material de apoyo. El perfil de escritura está en `docs/perfil_escritura.md` y los prototipos de referencia son `vecinos_con_almacen.py` y `estabilizador_cercano.py`.

## 1. Objetivo, entorno y reglas

Vas a extender mi librería de estimación directa de fidelidad (DFE), en `<RUTA_DEL_REPO>`, con un algoritmo autoguiado que estima la fidelidad estabilizadora de un estado desconocido ρ, puro o mixto, usando solo mediciones de cadenas de Pauli sobre una copia:

$$F_{\rm STAB}(\rho)=\max_{|\phi\rangle\in{\rm Stab}_n}\langle\phi|\rho|\phi\rangle .$$

Cada fidelidad se estima por DFE, y un almacén de resultados de Pauli reduce el número de mediciones. Después producirás las simulaciones y el borrador del artículo.

**Entorno.** Python ≥ 3.10 y la última versión estable de Qiskit (serie 2.x): compruébala con `pip index versions qiskit` y fíjala en `requirements.txt`. Usa `qiskit.quantum_info` (`Pauli`, `PauliList`, `Clifford`, `StabilizerState`, `Statevector`, `DensityMatrix`, `random_clifford`, `random_density_matrix`) y un *sampler* de Qiskit para simular mediciones en bases de Pauli. Consulta la documentación de la versión instalada antes de usar cualquier API.

**Reglas.**

1. Antes de programar, inspecciona el repositorio, resume su API de DFE y propón un plan. Espera mi confirmación antes de cualquier cambio que rompa la API.
2. El almacén es opcional y la ruta sin almacén debe dar los mismos resultados que hoy con la misma semilla.
3. Toda aleatoriedad lleva semilla explícita. Cada número o figura del artículo sale de un script versionado que escribe CSV y figura en `results/`.
4. No inventes resultados ni referencias. Si algo es ambiguo, pregunta.

**Fases:** inspección y plan → algoritmo y almacén con sus pruebas (secciones 2 a 4) → simulaciones (sección 5) → artículo (sección 6).

## 2. El algoritmo

El iterado z es siempre un estado estabilizador; en cada paso se mueve al mejor de sus vecinos y para cuando ninguno mejora. Todas las fórmulas valen para ρ mixto.

**Objetos.** z se guarda por n generadores con signo (`StabilizerState` o `Clifford`); su grupo S tiene 2^n elementos s_P·P. Con desestabilizadores d_i, la base propia es φ_x = Q_x z, con Q_x = ∏ d_i^{x_i}. Los vecinos son ν_{x,m} = (z + i^m φ_x)/√2, con x ≠ 0 y m ∈ {0,1,2,3}: 4(2^n − 1) estados estabilizadores, cada uno una rotación de Pauli de π/2 aplicada a z. Para construir sus generadores, toma un generador g_0 que anticonmute con Q_x, multiplica por g_0 los demás que anticonmutan, y sustituye g_0 por ±Q_x (m par) o ±iQ_x g_0 (m impar).

**Fórmulas.** Con p_x = ⟨φ_x|ρ|φ_x⟩ y r_x = ⟨z|ρ|φ_x⟩:

$$f(z)=\frac{1}{2^n}\sum_{P\in S}s_P\,{\rm tr}(\rho P),\qquad p_x=\mathbb E_{u}\big[(-1)^{x\cdot u}s_u\,{\rm tr}(\rho\,g_u)\big],$$

$$f(\nu_{x,m})=\frac{p_0+p_x}{2}+{\rm Re}\big(i^m r_x\big),\qquad |r_x|\le\sqrt{p_0\,p_x}.$$

Por tanto, solo pueden mejorar las direcciones con p_x > (3 − 2√2) p_0 ≈ 0,17 p_0. Además, z y sus cuatro vecinos en la dirección x usan solo cuatro clases laterales del subgrupo T_x ⊂ S que conmuta con Q_x. Dos de ellas son de z, así que por dirección solo hacen falta dos clases nuevas.

**Pasos.**

1. Arranque: el estado que el experimento quería preparar, o uno aleatorio si no hay ninguno.
2. DFE sobre S: da f̂(z) y, por la transformada de Walsh de los mismos datos, todos los p̂_x.
3. Si algún p̂_x > p̂_0, cambia los signos de los generadores con x_i = 1 (z ← φ_x) y vuelve al paso 2.
4. Candidatos: las K direcciones con mayor p̂_x que cumplan p̂_x > 0,17 p̂_0.
5. Para cada candidato, estima los cuatro vecinos y acepta el mejor si supera a z; vuelve al paso 2.
6. Si ningún candidato mejora, repite la comparación con 2N y luego 4N disparos; si sigue sin mejora, para.
7. Salida: f(z) estimada con copias nuevas y su intervalo de confianza. Si la cota inferior supera cos²(π/8) ≈ 0,854, z es el óptimo global. Esto vale también para ρ mixto, porque dos estados estabilizadores distintos cumplen ⟨φ₁|ρ|φ₁⟩ + ⟨φ₂|ρ|φ₂⟩ ≤ 1 + 1/√2.

**Parámetros por defecto:** N = 1000 disparos por estimación, K = 3, 2 reintentos, 10⁴ disparos en la estimación final, confianza del 99%.

**Devuelve:** el estado z, F̂ con su intervalo, la marca de certificado global, y un registro con el camino, los disparos por paso y la tasa de reutilización del almacén.

## 3. Almacén con subcadenas

Cada disparo mide una cadena de Pauli aleatoria y alimenta el almacén con todas sus subcadenas. El estimador reutiliza esos datos sin sesgo y nunca es menos preciso que una DFE nueva con el mismo N.

**Qué se guarda.** Medir la cadena P en su base producto da una cadena de bits. El resultado de cualquier subcadena Q ⊑ P (P con algunas letras cambiadas por I) es la paridad de esos bits sobre el soporte de Q. Para cada Q sin signo se guardan el número de resultados y su suma. El signo s_Q lo aporta el estado objetivo en cada consulta, así que un mismo dato sirve a objetivos con signos opuestos. Como una cadena de peso w tiene 2^w subcadenas, guarda la base y la cadena de bits y calcula las subcadenas al consultarlas, o materialízalas solo hasta un peso máximo configurable.

**Estimador.** Para el grupo U de un estado objetivo:

1. Ubicaciones: hacen falta N Paulis uniformes en U. Se reutilizan las de estimaciones anteriores donde siguen siendo uniformes. Para eso U se divide en átomos, que son los Paulis de U con el mismo historial de pertenencia a los conjuntos muestreados antes, y cada átomo a recibe ⌈N |a|/|U|⌉ ubicaciones. Las que falten se sortean uniformemente dentro del átomo.
2. Valor en cada ubicación Q: la media ō_Q de todos los resultados guardados de Q, directos o como subcadena. Una ubicación nueva que ya tiene datos no cuesta disparos; si no los tiene, se mide Q una vez y se guardan sus subcadenas.
3. Estimación:

$$\hat A_U=\sum_{a}\frac{|a|}{|U|}\,\frac{1}{n_a}\sum_{Q\in a}s_Q\,\bar o_Q .$$

Es insesgado porque las ubicaciones son uniformes dentro de cada átomo y cada ō_Q es insesgada para tr(ρQ), con independencia de dónde cayó la ubicación. Con la asignación proporcional, su varianza no supera la de una DFE nueva con N disparos. Como un mismo disparo alimenta varias subcadenas, los intervalos de confianza se calculan remuestreando por disparo.

**Implementación.** API mínima: `store.estimate(group, N, weights=None) -> (valor, error_estándar, disparos_nuevos)`. Haz una versión exacta con conjuntos explícitos para n ≤ 10, que sirve de oráculo en las pruebas, y una versión escalable. En el algoritmo todos los conjuntos son clases laterales de T_x, así que la versión escalable puede guardar los átomos como subespacios afines de F_2^{2n}, con pertenencia y tamaños por álgebra lineal sobre F_2. La fidelidad que se reporta al final se estima siempre con copias nuevas, nunca desde el almacén.

## 4. Pruebas

El código está listo cuando pasan estas pruebas.

**Exactas** (contra `DensityMatrix`, con ρ aleatorios puros y mixtos):

1. Productos y fases de Paulis coinciden con `qiskit.quantum_info.Pauli` para n ≤ 3.
2. Vecinos: 12, 28 y 60 distintos para n = 2, 3 y 4, todos con solapamiento 1/2 con z.
3. La fórmula de vecinos, la descomposición en cuatro clases y la de Walsh para p_x son exactas (tolerancia 10⁻¹⁰, n ≤ 6).
4. La paridad de bits de una subcadena Q, medida en la base de P, tiene valor esperado tr(ρQ).

**Estadísticas** (semillas fijas, 10⁴ repeticiones, tolerancia de 3 errores estándar):

5. DFE y almacén son insesgados, incluso en secuencias de consultas solapadas con subcadenas reutilizadas.
6. La varianza con almacén no supera 1,1 veces la de una DFE nueva con el mismo N.
7. Las versiones exacta y escalable del almacén son estadísticamente indistinguibles para n ≤ 8.

**Control** (deben reproducir el prototipo sin subcadenas, N = 1000): n = 4 con τ = 0,9 y arranque aleatorio, y n = 8 con τ = 0,9 y arranque a 3 pasos, alcanzan el óptimo en el 100% de los casos con un ahorro de disparos de al menos 4×. Con subcadenas, el ahorro no puede ser menor.

## 5. Simulaciones del artículo

Cinco experimentos y cuatro figuras bastan.

**Estados de prueba.**

- Puros: ψ = √τ φ + √(1−τ) χ con φ estabilizador, estados de Clifford con pocas compuertas T, y estados de Haar (n ≤ 5).
- Mixtos: ρ = (1−p)|ψ⟩⟨ψ| + p I/2^n, que cumple F_STAB(ρ) = (1−p) F_STAB(ψ) + p/2^n; mezclas de estados estabilizadores Σ w_i |φ_i⟩⟨φ_i|, cuyos valores esperados de Pauli son analíticos y permiten llegar a n ≥ 50; y estados aleatorios de rango k con `random_density_matrix` (n ≤ 5).
- Referencia exacta de F_STAB: enumeración de todos los estados estabilizadores para n ≤ 5 (comprueba que es viable para n = 5). Para n mayor, un óptimo plantado con ⟨φ₁|ρ|φ₁⟩ > cos²(π/8), que es único.

**Experimentos.**

| Id | Pregunta | Configuración |
| --- | --- | --- |
| E1 | ¿Cuánto ahorra el almacén y cómo escala con n? | Sin almacén, con almacén y con almacén más subcadenas; n = 4…12 con `DensityMatrix` y n = 20…100 con mezclas de estabilizadores |
| E2 | ¿Cuándo se alcanza el óptimo? | P(éxito) y F̂/F_STAB frente a τ (puros) y frente a p (mixtos), n ≤ 5 |
| E3 | ¿Qué pasa con estados densos? | Haar y mixtos aleatorios de rango k: distribución de F̂/F_STAB |
| E4 | ¿Cuántas copias frente a los métodos publicados? | Copias para estimar F_STAB con error ε frente a n (sección 7) |
| E5 | ¿Qué parámetros importan? | Ablación de K, N y distancia del arranque |

**E4.** Simula el algoritmo de Grewal et al. con muestreo de diferencias de Bell (n ≤ 10, régimen τ > cos²(π/8)). Para el *stabilizer bootstrapping*, implementa una versión simplificada si es viable; si no, compara con su complejidad publicada y dilo explícitamente. Indica en cada caso el tipo de medición: Bell sobre dos copias frente a Pauli sobre una copia. Comprueba si cada método está formulado para estados puros o también mixtos, y limita la comparación con estados mixtos a donde aplican sus garantías.

**Figuras.** Fig. 1: esquema de los vecinos (el octaedro de un qubit) y del algoritmo. Fig. 2: disparos frente a n (E1). Fig. 3: P(éxito) frente a τ y p (E2). Fig. 4: copias frente a n para todos los métodos (E4). E3 y E5 van al apéndice.

## 6. Redacción del artículo

El artículo es para Physical Review A, en inglés, con 4 a 8 páginas de texto principal sin contar apéndices. El texto principal solo lleva las ideas centrales; demostraciones, detalles de simulación y figuras secundarias van a los apéndices.

**Formato.** REVTeX 4.2 (`\documentclass[pra,reprint,superscriptaddress,amsmath,amssymb]{revtex4-2}`), bibliografía en BibTeX con DOI. Verifica cada referencia de la sección 7 antes de citarla.

**Estructura.**

1. Introduction.
2. Preliminaries: A. Stabilizer states and stabilizer fidelity (definición, grupo estabilizador, F_STAB para ρ puro o mixto, y por qué calcularla es difícil: hay 2^{O(n²)} estados estabilizadores); B. Direct fidelity estimation (estimador para objetivos estabilizadores, muestreo uniforme del grupo y coste independiente de n).
3. Method: A. Stabilizer neighbors; B. Self-guided search; C. Pauli storage.
4. Numerical simulations: E1, E2 y E4, con la Tabla I comparando métodos (tipo de medición, copias, garantía, resultado).
5. Conclusions.
6. Apéndices: A. demostraciones (sección 8); B. insesgadez y varianza del estimador; C. detalles de simulación, E3 y E5; D. pseudocódigo.

Nivel de rigor: resultados como proposiciones con demostración breve en el apéndice, para lectores de física y no de teoría de la computación.

**Perfil de escritura.**

- Sigue `docs/perfil_escritura.md`: tiene prioridad sobre cualquier otra indicación de estilo de este documento, y su checklist se pasa antes de entregar cada borrador. Lo esencial está en los puntos siguientes.
- Un nombre neutro y fijo para la propuesta, por ejemplo *self-guided estimation protocol*, sin sinónimos. El mensaje central se repite en el resumen, la introducción, los resultados y las conclusiones: la fidelidad estabilizadora de un estado desconocido se estima con mediciones de Pauli sobre una copia, y el almacén reduce el número de mediciones por un factor XX.
- Texto principal sin derivaciones: resultados finales explicados con palabras y la referencia al apéndice, por ejemplo «(Appendix A)».
- Inglés sencillo en primera persona del plural; presente para describir el trabajo, pasado para las simulaciones y la discusión. Oraciones de unas 25 palabras encadenadas con *which*, *so that* y *that is,*.
- Párrafos de 6 a 10 oraciones: contexto → limitación (*However,*) → solución (*we…*) → consecuencia (*Thereby,* / *Therefore,*) → cierre. Uno o dos conectores del perfil por párrafo.
- Introducción: contexto general → métodos existentes y sus limitaciones (*It has been shown [] that… However,*) → propuesta → resultados → comparación con otros métodos. Sin párrafo sobre la organización del artículo.
- Figuras: subfiguras como *inset (a)*, referencias en frases completas (*as shown in Fig. 1(a)*), y pies que empiezan con *Schematic representation of…* o terminan en *… as a function of … in logarithmic scale*. Nunca *panel* ni *slope*.
- Números: coeficientes ajustados con barra de error (α = 0.76 ± 0.19), escalamientos en su forma más simple, y un único caso concreto para ilustrar un extremo; no se describen rangos en el texto. Nunca inventes números ni citas: deja `[?]` o `XX` y lístalos al final.
- LaTeX: `Eq.\thinspace\eqref{…}`, `Fig.\thinspace\ref{…}(a)`, ecuaciones integradas en la oración, iniciales de autores sin espacio.
- Longitud orientativa: 150 palabras de resumen, 600 de introducción, 500 de preliminares, 900 de método, 900 de resultados y 300 de conclusiones. Redacta en este orden: preliminares, método, resultados, introducción, conclusiones, resumen y título (propón tres títulos).

## 7. Trabajos previos

Los métodos publicados con garantías usan mediciones de Bell sobre dos copias o solo deciden si F_STAB es alta o baja; ninguno estima F_STAB con mediciones de Pauli sobre una copia. Las referencias sin enlace son de memoria y hay que confirmarlas.

| Trabajo | Qué hace con F_STAB | Mediciones | Garantía |
| --- | --- | --- | --- |
| [Chen, Gong, Ye y Zhang, arXiv:2408.06967](https://arxiv.org/abs/2408.06967) | La estima con error ε y devuelve el testigo (*stabilizer bootstrapping*) | Bell, dos copias | Global; tiempo n³·quasipoly(1/ε) |
| [Grewal, Iyer, Kretschmer y Liang, arXiv:2304.13915](https://arxiv.org/html/2304.13915) | Encuentra el estabilizador más cercano; test tolerante | Bell, dos copias | Exponencial en n/τ⁴ en general; polinómico si τ > cos²(π/8) |
| [Aprendizaje agnóstico de qudits, arXiv:2607.15559](https://arxiv.org/abs/2607.15559) | La estima para qudits de dimensión prima impar | Varias copias | Demostrada |
| [Arunachalam y Dutt, arXiv:2408.06289](https://arxiv.org/abs/2408.06289); [Arunachalam, Bravyi y Dutt, arXiv:2410.22220](https://arxiv.org/html/2410.22220v1); [entradas mixtas, arXiv:2411.08765](https://arxiv.org/pdf/2411.08765) | Deciden si es ≥ ε o ≤ ε^C | Varias copias | poly(1/ε) copias; solo decide |
| [Hinsche y Helsen, arXiv:2410.07986](https://arxiv.org/pdf/2410.07986) | Decide si el estado es estabilizador | Una copia | O(n) copias; no tolerante |
| Gross, Nezami y Walter, CMP 385 (2021) | Test con 6 copias; base del muestreo de diferencias de Bell | Bell, dos copias | No tolerante |
| [Hamaguchi et al., arXiv:2406.16673](https://arxiv.org/html/2406.16673) | La calcula exactamente a partir del vector de estado | Clásico | Exacto; hasta n = 9 en estados de Haar |
| Leone, Oliviero y Hamma, PRL 128, 050402 (2022) | Entropías estabilizadoras; dan F_STAB ≤ 2^{−M2/4} | Bell o aleatorizadas | Solo una cota |
| Aaronson et al., arXiv:2211.00747 | Estados pseudoaleatorios con F_STAB inverso-superpolinómica | — | Ningún método eficiente puede estimarla en ese régimen |

Herramientas que usa la propuesta y deben citarse, sin compararse: DFE (Flammia y Liu, PRL 106, 230501 (2011); da Silva, Landon-Cardinal y Poulin, PRL 107, 210404 (2011)), la definición de F_STAB (Bravyi et al., Quantum 3, 181 (2019)), la geometría de vecinos ([García, Markov y Cross, arXiv:1711.07848](https://arxiv.org/pdf/1711.07848)) y la tomografía autoguiada como antecedente del término ([Ferrie, arXiv:1406.4101](https://arxiv.org/abs/1406.4101)).

## 8. Afirmaciones que hay que demostrar

Siete afirmaciones sostienen el artículo; todas tienen demostraciones cortas salvo el ahorro, que es numérico.

| # | Afirmación | Dónde |
| --- | --- | --- |
| 1 | Los vecinos de z son las rotaciones e^{±iπP/4} z con P ∉ ±S, y son 4(2^n − 1) | Apéndice A |
| 2 | f(ν_{x,m}) = (p_0 + p_x)/2 + Re(i^m r_x), con \|r_x\| ≤ √(p_0 p_x), para ρ puro o mixto | Apéndice A |
| 3 | Solo mejoran direcciones con p_x > (3 − 2√2) p_0, y hay a lo sumo (1 − p_0)/((3 − 2√2) p_0) | Apéndice A |
| 4 | z y sus 4 vecinos en la dirección x usan 4 clases laterales de T_x | Apéndice A |
| 5 | Si ⟨φ\|ρ\|φ⟩ > cos²(π/8), φ es el único óptimo, también para ρ mixto, porque λ_max(\|φ₁⟩⟨φ₁\| + \|φ₂⟩⟨φ₂\|) = 1 + \|⟨φ₁\|φ₂⟩\| ≤ 1 + 1/√2 | Apéndice A |
| 6 | El estimador con almacén y subcadenas es insesgado y su varianza no supera la de una DFE nueva | Apéndice B |
| 7 | El ahorro de disparos no depende de n | E1 (hasta ahora, unas 5× para n ≤ 8 sin subcadenas) |
