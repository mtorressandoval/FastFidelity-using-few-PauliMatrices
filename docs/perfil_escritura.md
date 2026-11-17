# Perfil de escritura (v2)

Perfil para redactar o revisar artículos científicos en inglés con mi estilo. La versión 1
(`perfil_escritura_v1.md`) describía sobre todo las oraciones; esta versión añade la estructura
del artículo, de las secciones y de los párrafos, la forma de analizar los resultados numéricos y
la estructura de la discusión. Se basa en la versión 1 y en dos artículos míos:

- A. D. Muñoz-Moller, L. Pereira, L. Zambrano, J. Cortés-Vega, A. Delgado, *Variational
  determination of multiqubit geometrical entanglement in noisy intermediate-scale quantum
  computers*, arXiv:2110.03709 (VDGE).
- L. Zambrano, A. D. Muñoz-Moller, M. Muñoz, L. Pereira, A. Delgado, *Avoiding barren plateaus in
  the variational determination of geometric entanglement*, arXiv:2304.13388 (iVDGE).

**Estilo en una línea:** inglés científico sencillo, en primera persona del plural; cada párrafo
desarrolla una sola idea, que se anuncia en la primera oración y se cierra con una consecuencia;
los resultados se leen en el texto a partir de las figuras, con cifras concretas y una explicación
física de cada tendencia.

---

## 1. Preferencias explícitas

Tienen prioridad sobre el resto del perfil.

- **Vocabulario simple.** "we can see", "we observe", "we obtain", "then", "thereby", "therefore",
  "in order to", "according to", "approach", sin abusar.
- **Nombre de la propuesta.** Descriptivo y neutro ("method", "protocol"), definido una vez con su
  sigla si la tiene, y usado siempre igual. Los términos más cargados se reservan para los métodos
  de otros autores.
- **Mensaje central** repetido en el resumen, la introducción, los resultados y la discusión.
- **Sin derivaciones en el texto principal.** Se dan los resultados, se explican en palabras y se
  remite al apéndice ("(see Appendix B for details)").
- **Figuras.** Las subfiguras son "inset (a)". Las referencias van en frases completas: "as shown
  in Fig. 2(a)", "Fig. 3 shows…".
- **Evitar "slope".** Describir cómo crece una cantidad ("increases faster with … than …").
- **Coeficientes ajustados con barra de error** ($\alpha=0.76\pm0.19$); escalamientos en su forma
  más simple ($n^2$, $e^{\beta n}$); un único caso concreto para ilustrar un extremo.
- **Extensión.** Al revisar, mantener la estructura y la longitud; el texto puede acortarse, no
  alargarse, salvo que se pida.

---

## 2. Estructura del artículo

### 2.1 Resumen (6 a 7 oraciones, sin cifras salvo tamaños)

1. **Contexto o problema**, en una oración afirmativa: "The barren plateau phenomenon is one of the
   main obstacles to …", "Current noise levels … limit …".
2. **Limitación de lo existente** (opcional, con "However").
3. **Propuesta**, siempre con "Here, we introduce / we propose / we follow …".
4. **Cómo funciona o qué requiere**, en una oración: "The method is based on …", "The algorithm
   requires only single-qubit gates and measurements, so it is well suited for NISQ devices."
5. **Resultado analítico**: "We analytically demonstrate that …".
6. **Simulaciones**, con tamaño y familia: "Numerical simulations up to 18-qubit GHZ and W states
   demonstrate that the method converges to the exact value."
7. **Experimento o robustez** (si lo hay): "… are in agreement with experiments carried out on …".

### 2.2 Introducción (4 a 5 párrafos, cada uno con un papel fijo)

1. **Contexto amplio.** El campo y su importancia, con muchas citas agrupadas por aplicación
   ("finding applications in quantum chemistry \cite{…}, finances \cite{…}, and quantum tomography
   \cite{…}, among others").
2. **Herramienta concreta.** La cantidad o el método en el que se apoya el trabajo: definición en
   palabras, quién la introdujo y para qué se ha usado.
3. **Lo que falta.** "However, …" seguido del problema concreto, de cómo se ha intentado resolver
   (lista con citas) y de una oración que deja abierto el problema: "Despite the advances on this
   topic, there is no general indication …, and moreover, it is not clear which technique …".
4. **Propuesta.** "In this article, we propose …". El nombre del método, su idea principal en una
   o dos oraciones y sus propiedades clave (cotas, estimadores insesgados, coste).
5. **Resultados.** "We carried out exhaustive numerical simulations to test …". Cada resultado con
   su cifra más representativa ("being 10% and 71%, respectively") y, al final, el experimento o
   la comparación con otros métodos. Sin párrafo sobre la organización del artículo.

### 2.3 Secciones

- **Fundamentos** ("Theoretical background" o la primera subsección de "Results"): definición
  formal de la cantidad, el problema de optimización y el método base del que parte el trabajo.
- **Método** en subsecciones cortas, una por componente. La subsección del método nuevo empieza
  por el motivo ("To prevent … we resort to …"), sigue con la definición, enuncia las propiedades
  (cotas, estimador insesgado, error) remitiendo al apéndice, y termina con el algoritmo completo:
  "The algorithm that we propose, which we call …, arises from concatenating the following two
  stages: (i) …, and (ii) …". Los pasos van en lista nominal y, si hace falta, en pseudocódigo.
- **Resultados numéricos** en subsecciones por pregunta: primero los casos pequeños y conocidos,
  después el régimen difícil (muchos qubits, ruido) y al final el experimento.
- **Discusión** (sección 5 de este perfil).
- **Apéndices** con las demostraciones, los detalles del estimador, el cálculo de los valores
  exactos de referencia y el optimizador.

---

## 3. Párrafos

- **Una idea central por párrafo.** Si aparece una segunda idea, empieza un párrafo nuevo.
- **La primera oración anuncia la idea o el objetivo:** "We first test the performance of our
  method with …", "To comparatively study the performance of … in the high qubit number regime,
  …, we perform …", "A drawback present in … is its landscape, which …".
- **El desarrollo** encadena 4 a 8 oraciones con conectores lógicos ("In particular", "This is
  because", "Due to this", "On the other hand", "Notice that").
- **La última oración da la consecuencia**, con "Thereby,", "Therefore," o "This suggests that …".
  Al final de una sección: "All of this indicates that …", "Overall, …".
- **Tipos de párrafo y su esquema:**
  - *Definición:* "Let us consider …" → ecuación → significado en palabras → propiedad clave.
  - *Limitación:* ventaja → "However," → ejemplo concreto → consecuencia para el método.
  - *Solución:* problema → "To overcome this problem, we employ …" → cómo funciona → resultado.
  - *Resultado numérico:* sección 4 de este perfil.
  - *Transición:* una o dos oraciones ("The previous simulations are restricted to small numbers of
    qubits. In order to extend …").

---

## 4. Análisis de los resultados numéricos

### 4.1 Plantilla del párrafo de resultados

1. **Objetivo** del experimento numérico: "We first test …", "To study the ability of … to …, we
   performed …".
2. **Configuración completa**, en este orden: estados (familia, número, distribución), número de
   qubits, repeticiones y condiciones iniciales, iteraciones, disparos por evaluación, valor exacto
   de referencia y cómo se obtuvo, y medida de error ("the median of $|\hat E-E|$").
3. **Lectura de la figura:** "Fig. 2 shows … as a function of …, for … from bottom to top. Points
   represent the median and bars correspond to the interquartile range. Solid lines correspond to
   the exact value obtained via …".
4. **Observación con cifras:** "As is apparent from this figure, …", "We see that …", con valores
   concretos ("with mean errors of 0.00011, 0.00091 …", "being 10% and 71%, respectively").
5. **Explicación física de cada tendencia:** "This behavior is to be expected because the dimension
   increases exponentially with the number of qubits and the fidelity is measured with a constant
   number of shots."
6. **Salvedades** cuando corresponde: "This behaviour might be controlled by fine-tuning the
   hyper-parameters …".
7. **Cierre:** "Thereby, our simulations indicate that …".

### 4.2 Reglas

- **Valor exacto de referencia** siempre, obtenido con un método independiente y descrito en un
  apéndice. Las familias de prueba se eligen para que este valor se conozca para todo $n$.
- **Mediana y rango intercuartílico** como estadísticos por defecto; la media solo si se dice.
- **Criterio de éxito o de fallo explícito:** "We consider that the algorithm is trapped in a BP if,
  after the optimization, the GME is still greater than 0.9."
- **Comparaciones con igualdad de recursos:** "Thereby, our implementations of … and … use the same
  total amount of shots."
- **Tablas** para porcentajes y medianas frente a un recurso (disparos); **figuras** para
  tendencias frente a $n$ o frente a las iteraciones.
- **Conectar regímenes:** casos pequeños con referencia exacta → régimen grande con estados
  especiales → ruido o experimento, y explicar las diferencias entre ellos.
- **Dar la escala del problema** cuando ayuda a entender un número ("a thousand times smaller than
  the dimension of the system $d=2^{25}\approx3\times10^7$").

---

## 5. Discusión

La discusión es una sección completa ("Discussion" o "Conclusions") con este orden:

1. **Contexto y problema** (1 párrafo corto): por qué la cantidad importa y qué limita su
   estimación, sin repetir la introducción.
2. **Resumen corto de la propuesta** (1 párrafo): qué se hace, qué cotas o propiedades tiene y
   por qué se puede implementar.
3. **Resumen de los resultados numéricos** (1 a 2 párrafos): en el orden de las figuras, cada uno
   con su cifra clave y su explicación ("As Figs. 2 and 3 indicate, … Meanwhile, …").
4. **Interpretación** (1 párrafo): el mecanismo que explica los resultados, con un argumento de
   escala sencillo ("the fidelity … is $O(1/128)$, and using 256 shots to estimate this quantity is
   not enough to construct a good approximation of the gradient").
5. **Valoración global** (1 párrafo): "Considering the previous discussion, … is effectively an
   improvement of …", con lo que no añade complejidad.
6. **Comparación con otros resultados** (1 párrafo): en qué se parece y en qué se diferencia de
   las propuestas existentes, con citas, y en qué sentido las complementa ("The standard approach
   is …. In contrast, our approach …").
7. **Perspectiva** (1 párrafo): aplicaciones directas con dispositivos o experimentos concretos
   ("recently 27-qubit GHZ states have been implemented …"), extensiones y trabajo futuro.

## 5b. Material suplementario y apéndices

No uso enunciados formales (teoremas, proposiciones, lemas) ni bloques "Proof"/∎. Cada resultado se
presenta como una narrativa inductiva dentro de su propia sección o subsección:

- **Título descriptivo** del contenido, no del tipo de resultado: "Bounds for the infidelity",
  "Evaluation of the expectation value of $H_L$", "Fidelity of the neighbors".
- **Primera oración: qué se va a obtener y para qué.** "The aim of this section is to obtain bounds
  for …", "In this section, we show how to evaluate … using …", "Here, we express … in terms of
  quantities that can be estimated …".
- **Construcción paso a paso** a partir de una observación sencilla o de un caso concreto (un qubit,
  los primeros proyectores $P(0)$ y $P(1)$, un ejemplo numérico), y después el caso general. Las
  ecuaciones se encadenan con frases de acción: "Removing … from the sum, …", "Considering that …,
  we have that …", "Due to the normalization of $\rho$, we obtain …", "Isolating …, we achieve …",
  "Furthermore, by using a similar reasoning …, we obtain …", "Replacing … in Eq. (…), we obtain …".
- **Interpretación tras cada resultado**: "This means that …", "Thereby, …".
- **Cierre con el alcance**: si el resultado es suficiente para el método, si la cota es ajustada o
  no, o qué caso no cubre ("Equation (…) is enough to motivate our proposal. However, these bounds
  may not be tight …").
- **Resultados ya publicados:** se enuncian y se citan ("It has been shown \cite{…} that …"), sin
  desarrollar la demostración; solo se añade lo necesario para traducirlos a la notación del
  artículo o para extenderlos (por ejemplo, de estados puros a mixtos). Al principio del apéndice
  se indica qué resultados se citan y cuáles se derivan.
- **Referencias desde el texto principal** a la subsección ("Appendix A 2"), nunca a "Proposition 2".
- Las verificaciones numéricas que sustituyen a una demostración se describen con su configuración
  (estado, número de repeticiones, resultado) y la ruta del test en el repositorio.

---

## 6. Conectores y frases hechas

Se mantienen los de la versión 1 ("that is,", "Thereby,", "Therefore,", "Then,", "However,",
"so that", "In fact,", "Due to this,", "In this way,", "since"), y se añaden:

- **Escenario adverso:** "In this adverse scenario, …", "Within this adverse scenario, …".
- **Literatura:** "It has been shown that …", "Despite the advances on this topic, …".
- **Motivo de una elección:** "To prevent … we resort to …", "To overcome this problem, we employ …",
  "We do this because …".
- **Figuras:** "As is apparent from this figure, …", "As Fig. 2 indicates, …", "Fig. 3 allows us to
  analyze … from a different perspective".
- **Interpretación:** "This behavior is to be expected because …", "This suggests that …", "This
  shows that …", "This can be explained by …".
- **Cierres:** "Thereby, our simulations indicate that …", "Overall, …", "All of this indicates
  that …".
- **Discusión:** "Considering the previous discussion, …", "Our protocol finds direct application
  in …", "Moreover, our proposal represents a step forward and complements previous results …".

---

## 7. Oraciones, verbos y formato

- Oraciones de 20 a 30 palabras, con cláusulas encadenadas por "which" y "so that"; las definiciones
  entre comas ("the GME, an entanglement monotone first introduced by …,").
- Presente para el método ("we propose", "the method requires"), pasado para las simulaciones y la
  discusión ("we carried out", "we found", "VDGE exhibited"), pasiva con "can" para capacidades.
- Ecuaciones integradas en la oración ("…, where …", "…, with …"); referencias con
  `Eq.\thinspace\eqref{…}`, `Fig.\thinspace\ref{…}(a)`, `Appendix\,\ref{…}`.
- Captions: frase nominal con lo que se representa "as a function of …", el significado de puntos,
  barras, líneas y áreas sombreadas, y los parámetros de la simulación.
- Siglas definidas una vez; nombres de dispositivos en `\texttt{}`.

---

## 8. Lista de comprobación

- [ ] El resumen sigue el orden contexto → propuesta → funcionamiento → análisis → simulaciones.
- [ ] La introducción tiene un párrafo por papel y termina con resultados concretos.
- [ ] Cada párrafo tiene una idea, anunciada al principio y cerrada con una consecuencia.
- [ ] Cada párrafo de resultados da la configuración, lee la figura, da cifras y explica la tendencia.
- [ ] Hay valor exacto de referencia, mediana y rango intercuartílico, y criterio de fallo explícito.
- [ ] La discusión sigue el orden problema → propuesta → resultados → interpretación → comparación
      → perspectiva.
- [ ] Ninguna afirmación de escalamiento va más allá de lo que muestran los datos.
- [ ] Los apéndices no tienen enunciados formales: cada subsección dice qué obtiene y lo construye
      paso a paso, desde un caso concreto cuando es posible.
