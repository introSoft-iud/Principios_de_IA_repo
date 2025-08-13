<!-- Presentación general del curso

-------------------------------------
Preliminares
Presentación general del curso 
-------------------------------------

Nombre del curso: Diplomado en construcción de Aplicaciones Asistidas por IA



-->
# Diplomado en construcción de Aplicaciones Asistidas por IA

![Banner del curso](assets/images/Banner.jpg)

<!--
*********           PRELIMINARES  *****************************


Describe de manera precisa y comprensible el propósito general del curso o asignatura, así como sus particularidades, enfatizando su relevancia práctica para el estudiante.



Describe la relevancia del contenido del curso para la formación, esto es: los saberes a explorar, las habilidades a desarrollar, qué metodología se empleará, cuál es el proceso de evaluación y cómo están estructuradas los módulos del curso. 

Para redactar la presentación, ten en cuenta las siguientes recomendaciones:

Establece el objetivo general y los específicos, o los resultados de aprendizaje (según sea el caso).
Inicia con un contexto histórico o geográfico sobre el tema central del curso.
Describe brevemente el tema central, es decir, lo que el estudiante aprenderá.
Añade aspectos que resulten significativos para el estudiante, por qué es necesario conocer este tema, cómo se aplicará en su quehacer profesional o académico, para qué le servirá en el presente y a futuro. Señala también la importancia del curso en el marco del programa.
Relaciona un ejemplo, algunas cifras notables o la aplicación principal de los conceptos para demostrar su relevancia.
Menciona qué habilidades se pueden desarrollar.
Indica las unidades de las que consta el curso y los temas a abordar en cada una.
Finaliza con un párrafo de cierre en el cual motives al estudiante a realizar el curso.

 
** Esta presentación no debe superar las 300 palabras.  



-->
Bienvenidos al diplomado en construcción de aplicaciones asistidas por modelos de lenguaje de la Institución Universitaria Digital de Antioquia.

Aunque la inteligencia artificial ha existido como un campo exitoso y prometedor entre los expertos durante varias décadas, la llegada de capacidades computacionales más avanzadas —ofrecidas por las GPU modernas— y las habilidades demostradas con el lanzamiento de ChatGPT fueron una gran sorpresa para muchos.

No está del todo claro cómo este "pequeño" avance en la escala de los modelos pudo desencadenar la gran cantidad de aplicaciones asistidas por IA que estamos viendo explotar cada semana. Lo que sí es claro es que el campo de la ingeniería de software está siendo revolucionado, y que el nuevo paradigma de construcción de software ya no consiste en los tradicionales flujos de ejecución, sino que la nueva ingeniería de sistemas debe integrar a los LLM en dichos flujos.

Hemos diseñado este diplomado para introducirte a este nuevo paradigma. Este curso está basado en LangChain, el framework más popular de la actualidad para interactuar con las APIs de los principales modelos de lenguaje.

En el módulo 1 comenzarás por aprender a crear instrucciones reutilizables para los LLM, los llamados `prompt templates`, luego verás cómo encadenar estas instrucciones a través de las cadenas usando  el Lenguaje de Expresión de LangChain (LCEL). Aprenderás a especificar el formato de salida de tus cadenas de ejecución mediante los `output parsers`. A continuación, en el módulo 2, crearás tus primeros chatbots y aprenderás a gestionar sus memorias. Como ejercicio práctico, implementarás un chatbot asistente de un médico que carga en su memoria la historia clínica de los pacientes. Finalmente, en el módulo, estarás preparado para incorporar los componentes de los dos módulos anteriores junto con loaders de documentos, retrievers y bases de datos vectoriales, para crear y desplegar un sistema RAG sobre documentos en PDF, de manera que podrás conversar con tus archivos PDF.

<!--

*******************************Resultados de aprendizaje******************
Establecen las dinámicas de ENSEÑANZA-APRENDIZAJE dentro del curso y encaminan el proceso hacia lo que queremos que los estudiantes sepan, comprendan y sean capaces de hacer al finalizar el curso.


Esta información se extrae de la carta descriptiva, por esa razón es importante consultarla antes de redactar esta parte. Ten presente que la versión en Word contiene el objetivo general y los específicos; mientras que la versión en Excel contiene los resultados de aprendizaje.

-->
## Resultados de Aprendizaje

1. **Diseñar y aplicar prompt templates y cadenas de ejecución en LangChain:** los estudiantes serán capaces de crear instrucciones reutilizables (prompt templates) y encadenarlas utilizando el Lenguaje de Expresión de LangChain (LCEL), especificando formatos de salida con output parsers para interactuar eficazmente con modelos de lenguaje.

2. **Desarrollar chatbots con gestión de memoria contextual:** los estudiantes podrán construir chatbots funcionales utilizando LangChain, integrando memorias para almacenar y gestionar información contextual.

3. **Construir y desplegar sistemas RAG para interacción con documentos:** los estudiantes estarán capacitados para integrar componentes como loaders de documentos, retrievers y bases de datos vectoriales en LangChain, creando sistemas de Retrieval-Augmented Generation (RAG) que permitan conversar con archivos PDF.

<!--
Pregunta orientadora
Es un interrogante que sirve como punto de partida para la exploración del tema central del curso, y está diseñado para dirigir la atención del estudiante, reconociendo de qué manera se apropia de ese saber. A través de esta pregunta, el conocimiento se logra concretar en una respuesta que recoge la esencia del curso, guiando al estudiante hacia el descubrimiento de conceptos importantes o la resolución de problemas dentro de un contexto determinado.



Formula la pregunta hablando al estudiante, de manera directa, concisa y sin ambigüedades. 
Evita utilizar términos confusos o complejos que dificulten su comprensión.
Recuerda que el estudiante dará respuesta a la pregunta orientadora al terminar el curso, por ello, es importante contextualizarla con un dato de interés o mediante un caso específico. 
La respuesta a esta pregunta se afianza o ejercita durante todo el proceso por medio de las evidencias de aprendizaje. Se espera que, al finalizar el curso, la respuesta tenga amplia relación con la actividad final.

**Procura no emplear más de 200 palabras. 
-->
### Pregunta Orientadora

La siguiente imagen es la respuesta de ChatGPT al prompt:

*“Based on what you know about me, draw a picture of what you think my current life looks like.”*

![alt text](image.png)

*Figura 1: Representación visual generada por ChatGPT sobre la vida del autor del diplomado.*

¿Qué tan parecida es esta imagen a tu vida en tu caso? ¿Te sorprende que ChatGPT tenga tanta información sobre tu vida y tus gustos o todo lo contrario? ¿Qué opinas que va a pasar con los datos en un futuro donde todo es asistido por IA? ¿Le daremos a las IA la información sensible de las personas y las empresas?

***¿Cómo podemos asegurarnos de que nuestros datos permanezcan privados, y al mismo tiempo, aprovechar todo el poder de los modelos de lenguaje en los datos privados?***

<!--

Mapa del curso
Es una herramienta visual que proporciona una visión general de la estructura y el contenido del curso.Se presenta en forma de diagrama y muestra los temas del curso divididos en unidades temáticas.


Es importante relacionar el nombre del curso, de sus respectivas unidades y de los saberes o temáticas correspondientes a cada una de ellas. Esta información se obtiene del formato de planeación o de la carta descriptiva. 

Ejemplos:
       

** Tanto el mapa del curso como cualquier gráfico de autoría propia y/o adaptado de otros autores deben entregarse en formato editable. 

-->

## Mapa del curso
Estos son los módulos que conforman nuestro diploma:

![Mapa del curso](assets/images/Mapa_DIPLOMA_LLM.png)

<!-- Cronograma de actividades 
Organiza las actividades y evidencias de aprendizaje que deben llevarse a cabo dentro del curso. Este cronograma incluye información sobre la secuencia de las actividades en cada módulo, la ubicación temporal (en qué semana se presentan) y los porcentajes correspondientes a las evidencias de aprendizaje.



Diligencia el cuadro siguiendo las indicaciones que encontrarás en cada celda. 
Recuerda la taxonomía y las definiciones establecidas en la carta descriptiva.
Menciona las actividades y evidencias de aprendizaje que deberá realizar el estudiante durante el estudio del curso (bien sea de 2 o 3 es, según el caso).



 
-->



## Módulo 1: introducción a la Construcción de Aplicaciones con LLM
- Fundamentos de modelos de lenguaje grandes (LLM) y su integración en flujos de software
- Creación y uso de *prompt templates* para instrucciones reutilizables
- Encadenamiento de instrucciones con el Lenguaje de Expresión de LangChain (LCEL)
- Especificación de formatos de salida mediante *output parsers*

## Módulo 2: Cadenas y Memoria
- Desarrollo de chatbots funcionales con LangChain
- Gestión de memoria contextual para mantener el historial de interacciones
- Implementación práctica de un chatbot funcional

## Módulo 3: Proyecto integrador: construcción y Despliegue de un Sistema RAG
- Integración de *document loaders* para procesar archivos PDF
- Uso de *retrievers* y bases de datos vectoriales para recuperación de información
- Construcción de sistemas Retrieval-Augmented Generation (RAG)
- Despliegue de una aplicación RAG para interacción conversacional con documentos PDF

<!-- 

Cronograma de actividades 
Organiza las actividades y evidencias de aprendizaje que deben llevarse a cabo dentro del curso. Este cronograma incluye información sobre la secuencia de las actividades en cada , la ubicación temporal (en qué semana se presentan) y los porcentajes correspondientes a las evidencias de aprendizaje.



Diligencia el cuadro siguiendo las indicaciones que encontrarás en cada celda. 
Recuerda la taxonomía y las definiciones establecidas en la carta descriptiva.
Menciona las actividades y evidencias de aprendizaje que deberá realizar el estudiante durante el estudio del curso (bien sea de 2 o 3 es, según el caso).

-->
## Cronograma de actividades

| Actividad de aprendizaje                                      | Evidencia de aprendizaje                               | Semana         | Ponderación |
|---------------------------------------------------------------|--------------------------------------------------------|----------------|-------------|
| Actividad de conocimientos previos                            | Actividad de conocimientos previos                      | Semana 1       | 0%          |
| Reto formativo 1 y 2                                          | EA1: Templates y Output Parsers                         | Semanas 2 y 3  | 35%         |
| Cadenas                                                      | EA2: Generación de Informes de Salud Utilizando Archivos CSV | Semanas 4 y 5  | 35%         |
| Reto formativo: "Carga y división de archivos PDF"            | EA3: Chat con tus datos. Proyecto integrador            | Semanas 6, 7 y 8 | 30%         |
| **Total**                                                     |                                                        |                | **100%**    |


<!--
### Actividad de refuerzo
*Agrega el nombre de la actividad de refuerzo (cuando así se requiera).*
  -->

<!--
Actividad de conocimientos previos
Permite reconocer e identificar, mediante un foro o un cuestionario, cuánto conocen los estudiantes sobre los temas del curso.


Esta actividad suele ser un foro o un cuestionario. Como experto temático, puedes definir cuál escoges. Por ejemplo:
Cuestionario de preguntas con 4 opciones de respuesta (a, b, c, d)
Debes proponer 15 preguntas básicas sobre los contenidos que se van a abordar, sin ningún porcentaje.
Ofrece una realimentación a cada opción de respuesta.
Las preguntas y respuestas del cuestionario deben presentarse en el formato Plantilla_Prueba Conocimientos previos_V1.docx.
Foro
A partir de una pregunta problematizadora o el análisis de un documento, se presentan planteamientos e instrucciones que conlleven a construcciones colectivas, bien sea, entre los mismos estudiantes o con el docente.
La pregunta problematizadora debe ser diferente a lo planteado en la pregunta orientadora, pues debe percibir los conocimientos previos que tienen los estudiantes en relación con los contenidos que estudiarán.
** Ten presente que la actividad de conocimientos previos no tiene ponderación.
Nombre del foro / o nombre del cuestionario
Escribe aquí el nombre de la actividad de conocimientos previos
Objetivo del foro/ cuestionario
(No modificar)
Determinar los saberes previos sobre las temáticas que se abordarán durante la asignatura.
Pregunta problematizadora del foro

(en caso de escoger un cuestionario, entrega las preguntas en la plantilla indicada a manera de anexo)
Escribe aquí la pregunta problematizadora que utilizarás para el foro.


Anexo_No_X_Prueba_de_conocimientos_previos (si es el caso)



Guion de video de presentación del curso
Tiene como objetivo principal despertar la motivación del estudiante, ofreciéndole una visión clara del propósito y la intención del curso.
Incluye elementos interesantes sobre la temática del curso, como una breve descripción del tema principal, su relevancia para el quehacer profesional, su aplicación en la actualidad y, especialmente, los beneficios para el estudiante. 
Agrega elementos visuales que ayuden a captar la atención del estudiante y generar un impacto emocional positivo.
Recuerda que este video debe motivar al estudiante a realizar el curso y explorar sus temáticas, de manera que participe activamente de su proceso de aprendizaje.
Para la elaboración del guion, se cuenta con una plantilla con instrucciones detalladas sobre cómo se elabora. Diligencia la plantilla y adjúntala a manera de anexo. 


Anexo_N°_X_Guión_de_Video_Presentación_Curso




-->


## Actividad de conocimientos previos. Foro: Transformers y mecanismo de atención
Como actividad de conocimientos previos, te invitamos a participar en el [foro de discusión](assets/resources/conociemientosPrevios_foro.md) sobre el funcionamiento de los LLM.

## Configuración del sistema antes de comenzar

Antes de empezar a trabajar con los módulos del curso, debes configurar tu sistema para poder ejecutar los ejemplos correctamente.

La forma más sencilla de hacerlo es descargando el archivo de configuración [📄environment.yml](assets/resources/environment.yml)
, el cual creará automáticamente un entorno de Conda llamado `DiplomadoIA_env` con todas las dependencias necesarias para el curso.

### Requisitos previos

- Tener **Anaconda** instalado en tu computador.
- Usar una terminal Bash (en Windows puedes usar Anaconda Prompt, git bash, WSL o similares).

### Instalación

Una vez descargado el archivo de configuración, ejecuta el siguiente comando en tu terminal:

```bash
conda env create -f environment.yml
```
### Activación del entorno

Para activar el entorno en tu terminal, ejecuta:

```bash
conda activate diplomado_IA
```
A partir de aquí, cualquier comando que ejecutes usará las dependencias definidas para el curso.

### Uso del entorno en Visual Studio Code

!!! warning "Para tener en cuenta"
   
    Para ejecutar notebooks `.ipynb` en Visual Studio Code usando este entorno:

    1. Abre **VS Code**.
    2. Abre la carpeta del proyecto o el notebook deseado.
    3. En la parte superior derecha del notebook, haz clic en la selección de kernel.
    4. Elige el kernel correspondiente al entorno `diplomado_IA`.  
    Si no aparece, reinicia VS Code o asegúrate de haber activado el entorno desde la terminal integrada.
    5. Comienza a ejecutar celdas normalmente.

!!! tip
    Puedes asegurarte de que el entorno se registre correctamente como kernel ejecutando en la terminal:
    ```bash
    python -m ipykernel install --user --name diplomado_IA --display-name "Python (diplomado_IA)"
    ```

