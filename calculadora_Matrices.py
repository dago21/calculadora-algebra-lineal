import streamlit as st
import numpy as np
import sympy as sp
import csv
import io
from fractions import Fraction

from algebra import (
    sumar_matrices,
    multiplicar_matrices,
    transponer_matriz,
    determinante_matriz,
    matriz_inversa,
    resolver_sistema,
    analizar_autovalores,
    producto_punto,
    norma_vector,
    normalizar_vector,
    gram_schmidt,
    descomposicion_svd,
)


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Calculadora de Álgebra Lineal",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       CONTENIDO GENERAL
    ------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
    }

    h2 {
        margin-top: 1.5rem !important;
    }


    /* -------------------------------------------------------
       SUBTÍTULO
    ------------------------------------------------------- */

    .subtitulo {
        font-size: 1.10rem;
        color: #B8BCC8;
        margin-bottom: 0.25rem;
    }

    .autor {
        font-size: 0.95rem;
        color: #6EA8FE;
        margin-bottom: 2rem;
    }


    /* -------------------------------------------------------
       BOTONES
    ------------------------------------------------------- */

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        min-height: 3rem;
        font-size: 1rem;
        font-weight: 500;
        text-align: left;
        border: 1px solid #3A3F4B;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        border-color: #4C8DFF;
        color: #4C8DFF;
    }


    /* -------------------------------------------------------
       CAJA DE INFORMACIÓN
    ------------------------------------------------------- */

    .info-box {
        padding: 1rem 1.2rem;
        border: 1px solid #303641;
        border-radius: 10px;
        background-color: rgba(30, 34, 42, 0.55);
        margin-bottom: 1.5rem;
    }


    /* -------------------------------------------------------
       RESULTADO
    ------------------------------------------------------- */

    .resultado-titulo {
        color: #40C978;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .resultado-box {
        border: 1px solid #303641;
        border-radius: 12px;
        padding: 1.4rem;
        margin-top: 1rem;
        background-color: rgba(21, 25, 32, 0.6);
    }

    .matrix-label {
        text-align: center;
        color: #6EA8FE;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }


    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #252A33;
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    .sidebar-section {
        font-size: 0.8rem;
        font-weight: 600;
        color: #AEB4C0;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .sidebar-footer {
        border: 1px solid #303641;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 3rem;
        font-size: 0.85rem;
        color: #B8BCC8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def convertir_a_numero(texto):
    """
    Convierte una entrada de texto en una expresión matemática exacta.

    Ejemplos válidos:
        2
        -3
        1/2
        -7/4
        0.5
        sqrt(2)
        sqrt(3)/2
        pi
        2*pi
        (1 + sqrt(5))/2
    """

    texto = str(texto).strip()

    if texto == "":
        return sp.Integer(0)

    # Algunas variantes cómodas para el usuario
    texto = texto.replace("^", "**")

    try:

        valor = sp.sympify(
            texto,
            locals={
                "sqrt": sp.sqrt,
                "pi": sp.pi,
                "e": sp.E,
                "E": sp.E
            }
        )

        # No queremos variables simbólicas como x, y, z
        if valor.free_symbols:
            raise ValueError(
                "No se permiten variables simbólicas."
            )

        # Evitar valores infinitos o indefinidos
        if valor.has(
            sp.zoo,
            sp.oo,
            -sp.oo,
            sp.nan
        ):
            raise ValueError(
                "La expresión genera un valor indefinido."
            )

        return valor

    except Exception:
        raise ValueError(
            f"El valor '{texto}' no es válido. "
            "Puede ingresar números como 2, -3, 1/2, "
            "sqrt(2), sqrt(3)/2 o pi."
        )


def mostrar_ayuda_entrada():
    """
    Muestra una guía breve sobre cómo ingresar
    expresiones matemáticas en las matrices.
    """

    st.info(
        """
        **💡 Guía para ingresar valores**

        | Tipo | Escriba | Representa |
        |---|---|---|
        | Fracción | `1/2` | ½ |
        | Raíz cuadrada | `sqrt(2)` | √2 |
        | Raíz con fracción | `sqrt(3)/2` | √3/2 |
        | Potencia | `2^3` | 2³ |
        | Número π | `pi` | π |
        | Múltiplo de π | `2*pi` | 2π |
        | Expresión | `(1+sqrt(5))/2` | (1+√5)/2 |

        También puede ingresar **enteros** (`-3`) y **decimales** (`0.25`).
        """
    )

def numpy_a_sympy(matriz, decimales=5):
    """
    Convierte una matriz NumPy en una matriz SymPy
    redondeada para mostrarla con LaTeX.
    """

    matriz_redondeada = np.round(
        matriz,
        decimales
    )

    return sp.Matrix(
        matriz_redondeada
    )

    
def controles_generacion_aleatoria_matriz(nombre, filas, columnas):
    """
    Permite completar una matriz con enteros aleatorios dentro de un
    intervalo definido por el usuario.

    El intervalo es inclusivo: [mínimo, máximo].
    """

    with st.expander("🎲 Generar matriz aleatoriamente"):

        st.caption(
            "Defina el intervalo de valores enteros y pulse Generar. "
            "Luego puede modificar manualmente cualquier elemento."
        )

        c1, c2, c3 = st.columns([1, 1, 1.2])

        with c1:
            minimo = st.number_input(
                "Valor mínimo",
                value=-5,
                step=1,
                key=f"random_matriz_{nombre}_min"
            )

        with c2:
            maximo = st.number_input(
                "Valor máximo",
                value=5,
                step=1,
                key=f"random_matriz_{nombre}_max"
            )

        with c3:
            st.write("")
            st.write("")
            generar = st.button(
                "🎲 Generar matriz",
                use_container_width=True,
                key=f"random_matriz_{nombre}_btn"
            )

        if generar:

            minimo = int(minimo)
            maximo = int(maximo)

            if minimo > maximo:
                st.error(
                    "El valor mínimo no puede ser mayor que el valor máximo."
                )
            else:
                rng = np.random.default_rng()
                valores = rng.integers(
                    minimo,
                    maximo + 1,
                    size=(filas, columnas)
                )

                for i in range(filas):
                    for j in range(columnas):
                        st.session_state[f"{nombre}_{i}_{j}"] = str(
                            int(valores[i, j])
                        )

                st.success(
                    f"Matriz {nombre} generada con valores entre "
                    f"{minimo} y {maximo}."
                )


def cargar_matriz_csv(archivo, nombre="A"):
    """
    Lee una matriz desde un archivo CSV separado por comas.

    - No requiere encabezados.
    - Todas las filas deben tener la misma cantidad de columnas.
    - Los valores se interpretan con convertir_a_numero(), por lo que
      también se admiten fracciones y expresiones como sqrt(2).
    """

    if archivo is None:
        return None

    try:
        contenido = archivo.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(
            f"El archivo CSV de {nombre} debe estar codificado en UTF-8."
        )

    lector = csv.reader(
        io.StringIO(contenido),
        delimiter=","
    )

    filas_csv = []

    for numero_fila, fila in enumerate(lector, start=1):

        # Ignorar filas completamente vacías
        if not fila or all(str(celda).strip() == "" for celda in fila):
            continue

        filas_csv.append(
            [str(celda).strip() for celda in fila]
        )

    if not filas_csv:
        raise ValueError(
            f"El archivo CSV de {nombre} está vacío."
        )

    cantidad_columnas = len(filas_csv[0])

    if cantidad_columnas == 0:
        raise ValueError(
            f"El archivo CSV de {nombre} no contiene columnas."
        )

    for i, fila in enumerate(filas_csv, start=1):
        if len(fila) != cantidad_columnas:
            raise ValueError(
                f"El CSV de {nombre} no es rectangular: "
                f"la fila {i} tiene {len(fila)} columnas y se esperaban "
                f"{cantidad_columnas}."
            )

    matriz = []

    for i, fila in enumerate(filas_csv, start=1):
        fila_convertida = []

        for j, texto in enumerate(fila, start=1):
            if texto == "":
                raise ValueError(
                    f"La celda ({i}, {j}) del CSV de {nombre} está vacía."
                )

            try:
                valor = convertir_a_numero(texto)
            except ValueError as error:
                raise ValueError(
                    f"Error en {nombre}[{i},{j}]: {error}"
                ) from error

            fila_convertida.append(valor)

        matriz.append(fila_convertida)

    return sp.Matrix(matriz)


def cargar_vector_csv(archivo, nombre="v"):
    """
    Lee un vector desde un CSV separado por comas.

    Se admite:
    - una columna: 1\n2\n3
    - una fila: 1,2,3

    En ambos casos se devuelve un vector columna de SymPy.
    """

    matriz = cargar_matriz_csv(archivo, nombre)

    if matriz is None:
        return None

    if matriz.cols == 1:
        return matriz

    if matriz.rows == 1:
        return matriz.T

    raise ValueError(
        f"El archivo CSV de {nombre} debe contener un solo vector: "
        "una fila o una columna."
    )


def ingresar_matriz(nombre, filas, columnas):
    """
    Permite ingresar una matriz de dos formas:

    1. Manualmente o mediante generación aleatoria.
       Esta modalidad está pensada para matrices de hasta 10 × 10.
    2. Mediante un archivo CSV separado por comas.
       Las dimensiones se detectan automáticamente y pueden superar 10 × 10.
    """

    st.subheader(f"Matriz {nombre}")

    modo = st.radio(
        f"Forma de ingreso de la matriz {nombre}",
        ["Manual / aleatoria", "Archivo CSV"],
        horizontal=True,
        key=f"modo_matriz_{nombre}",
    )

    if modo == "Archivo CSV":

        st.caption(
            "El CSV debe estar separado por comas y no debe incluir encabezados. "
            "Las dimensiones se detectan automáticamente, por lo que esta opción "
            "permite trabajar con matrices mayores a 10 × 10."
        )

        archivo = st.file_uploader(
            f"Cargar CSV de la matriz {nombre}",
            type=["csv"],
            key=f"csv_matriz_{nombre}",
        )

        if archivo is None:
            st.info(
                f"Seleccione el archivo CSV correspondiente a la matriz {nombre}."
            )
            return None

        try:
            matriz_csv = cargar_matriz_csv(
                archivo,
                nombre,
            )

            st.success(
                f"Matriz {nombre} cargada correctamente: "
                f"{matriz_csv.rows} × {matriz_csv.cols}."
            )

            # Para matrices grandes mostramos una vista tabular compacta.
            if matriz_csv.rows <= 10 and matriz_csv.cols <= 10:
                mostrar_matriz(
                    nombre,
                    matriz_csv,
                    f"Vista previa de {nombre}",
                )
            else:
                st.caption(
                    "Vista previa parcial (primeras 10 filas y 10 columnas)."
                )
                vista = matriz_csv[:min(10, matriz_csv.rows), :min(10, matriz_csv.cols)]
                st.dataframe(
                    [[str(valor) for valor in fila] for fila in vista.tolist()],
                    use_container_width=True,
                )

            return matriz_csv

        except ValueError as error:
            st.error(str(error))
            return None

    # --------------------------------------------------------
    # INGRESO MANUAL / ALEATORIO
    # --------------------------------------------------------

    matriz = []
    hay_error = False

    controles_generacion_aleatoria_matriz(
        nombre,
        filas,
        columnas
    )

    st.caption("Valores admitidos: 2 · 1/2 · sqrt(2) · 2^3 · pi")

    for i in range(filas):

        columnas_ui = st.columns(columnas)

        fila = []

        for j in range(columnas):

            texto = columnas_ui[j].text_input(
                f"{nombre}[{i + 1},{j + 1}]",
                value="0",
                key=f"{nombre}_{i}_{j}"
            )

            try:
                valor = convertir_a_numero(texto)

            except ValueError as error:
                columnas_ui[j].error(str(error))
                valor = sp.Integer(0)
                hay_error = True

            fila.append(valor)

        matriz.append(fila)

    if hay_error:
        return None

    return sp.Matrix(matriz)




def numero_latex(numero):
    """
    Convierte enteros y fracciones a formato LaTeX.
    """

    if isinstance(numero, Fraction):

        if numero.denominator == 1:
            return str(numero.numerator)

        return rf"\frac{{{numero.numerator}}}{{{numero.denominator}}}"

    # Por si en alguna operación recibimos float
    if isinstance(numero, (float, np.floating)):

        if np.isclose(numero, round(numero)):
            return str(int(round(numero)))

        fraccion = Fraction(float(numero)).limit_denominator(1000)

        return rf"\frac{{{fraccion.numerator}}}{{{fraccion.denominator}}}"

    return str(numero)


def matriz_a_latex(matriz):
    """
    Convierte una matriz SymPy directamente a LaTeX.
    """

    return sp.latex(matriz)

def simplificar_expresion_visual(expr):
    """
    Busca una forma algebraicamente equivalente
    pero más compacta para mostrar al usuario.

    Se prueban distintas estrategias de SymPy
    y se conserva la representación LaTeX más corta.
    """

    candidatos = []

    # Expresión original
    candidatos.append(expr)

    funciones = [
        sp.simplify,
        sp.cancel,
        sp.factor,
        sp.together,
        sp.radsimp
    ]

    for funcion in funciones:

        try:
            candidato = funcion(expr)

            # Verificamos que sea realmente equivalente
            if sp.simplify(expr - candidato) == 0:
                candidatos.append(candidato)

        except Exception:
            pass

    # Elegimos la representación LaTeX más corta
    mejor = min(
        candidatos,
        key=lambda x: len(sp.latex(x))
    )

    return mejor


def simplificar_matriz_visual(matriz):
    """
    Simplifica individualmente cada elemento
    de una matriz buscando una presentación compacta.
    """

    return matriz.applyfunc(
        simplificar_expresion_visual
    )


##GENERA UNA MATRIZ EN DECIMALES PARA MEJOR VISTA CUANTO USAMOS NUMNEROS RACIONALES.
def matriz_decimal(matriz, digitos=6):
    """
    Devuelve una aproximación numérica de la matriz.
    """

    return matriz.applyfunc(
        lambda x: sp.N(x, digitos)
    )


def mostrar_matriz(nombre, matriz, descripcion=None):

    if descripcion:

        st.markdown(
            f'<div class="matrix-label">{descripcion}</div>',
            unsafe_allow_html=True
        )

    latex = rf"{nombre} = {matriz_a_latex(matriz)}"

    st.latex(latex)


# ============================================================
# GENERACIÓN DEL INFORME PDF DEL ANÁLISIS DE UNA MATRIZ
# ============================================================

def generar_pdf_analisis(A, resultados, errores):
    """
    Genera en memoria un informe PDF con todos los resultados
    disponibles de la sección "Análisis de una matriz".

    El PDF se construye únicamente cuando el usuario analiza
    una matriz, por lo que no se crea ningún archivo temporal.
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
        )
    except ImportError:
        raise ValueError(
            "Para descargar el informe PDF debe instalar ReportLab con: "
            "pip install reportlab"
        )

    from io import BytesIO

    buffer = BytesIO()

    documento = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Análisis de una matriz",
        author="Atilio David Gómez Riveros",
    )

    estilos = getSampleStyleSheet()

    estilos.add(
        ParagraphStyle(
            name="TituloCentrado",
            parent=estilos["Title"],
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="Seccion",
            parent=estilos["Heading2"],
            spaceBefore=10,
            spaceAfter=8,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="Subseccion",
            parent=estilos["Heading3"],
            spaceBefore=7,
            spaceAfter=5,
        )
    )

    elementos = []

    def texto_expr(expr):
        """Convierte una expresión SymPy a texto legible para el PDF."""
        try:
            return sp.sstr(sp.simplify(expr))
        except Exception:
            return str(expr)

    def tabla_matriz(matriz, titulo=None, decimales=None):
        """Agrega una matriz como tabla al documento."""

        if titulo:
            elementos.append(
                Paragraph(titulo, estilos["Subseccion"])
            )

        if isinstance(matriz, np.ndarray):
            datos_np = matriz
            datos = []
            for fila in datos_np:
                fila_pdf = []
                for valor in fila:
                    if decimales is None:
                        fila_pdf.append(str(valor))
                    else:
                        fila_pdf.append(f"{float(valor):.{decimales}f}")
                datos.append(fila_pdf)
        else:
            datos = []
            for i in range(matriz.rows):
                fila_pdf = []
                for j in range(matriz.cols):
                    fila_pdf.append(
                        texto_expr(matriz[i, j])
                    )
                datos.append(fila_pdf)

        # Una matriz puede tener columnas anchas. Reducimos el tamaño
        # de fuente de forma gradual según el número de columnas.
        columnas = len(datos[0]) if datos else 1
        tam_fuente = 9
        if columnas >= 6:
            tam_fuente = 7
        if columnas >= 9:
            tam_fuente = 6

        tabla = Table(
            datos,
            hAlign="CENTER",
            repeatRows=0,
        )

        tabla.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), tam_fuente),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        elementos.append(tabla)
        elementos.append(Spacer(1, 8))

    def agregar_mensaje_no_disponible(clave):
        mensaje = errores.get(
            clave,
            "El cálculo no está disponible para la matriz ingresada."
        )
        elementos.append(
            Paragraph(
                f"<b>No disponible:</b> {mensaje}",
                estilos["BodyText"]
            )
        )
        elementos.append(Spacer(1, 8))

    # --------------------------------------------------------
    # PORTADA / DATOS GENERALES
    # --------------------------------------------------------

    elementos.append(
        Paragraph(
            "Calculadora de Álgebra Lineal",
            estilos["TituloCentrado"]
        )
    )

    elementos.append(
        Paragraph(
            "Informe - Análisis de una matriz",
            estilos["Heading1"]
        )
    )

    elementos.append(
        Paragraph(
            f"Dimensión de A: {A.rows} x {A.cols}",
            estilos["BodyText"]
        )
    )

    elementos.append(Spacer(1, 8))
    tabla_matriz(A, "Matriz A")

    # --------------------------------------------------------
    # TRANSPUESTA
    # --------------------------------------------------------

    elementos.append(
        Paragraph("1. Matriz transpuesta", estilos["Seccion"])
    )

    if "transpuesta" in resultados:
        tabla_matriz(resultados["transpuesta"], "A^T")
    else:
        agregar_mensaje_no_disponible("transpuesta")

    # --------------------------------------------------------
    # DETERMINANTE
    # --------------------------------------------------------

    elementos.append(
        Paragraph("2. Determinante", estilos["Seccion"])
    )

    if "determinante" in resultados:
        determinante = resultados["determinante"]
        elementos.append(
            Paragraph(
                f"det(A) = {texto_expr(determinante)}",
                estilos["BodyText"]
            )
        )
        if sp.simplify(determinante) == 0:
            elementos.append(
                Paragraph(
                    "El determinante es 0; por lo tanto, la matriz no es invertible.",
                    estilos["BodyText"]
                )
            )
    else:
        agregar_mensaje_no_disponible("determinante")

    elementos.append(Spacer(1, 8))

    # --------------------------------------------------------
    # INVERSA
    # --------------------------------------------------------

    elementos.append(
        Paragraph("3. Matriz inversa", estilos["Seccion"])
    )

    if "inversa" in resultados:
        tabla_matriz(
            resultados.get("inversa_simplificada", resultados["inversa"]),
            "A^-1 - resultado exacto"
        )

        tabla_matriz(
            resultados["inversa_decimal"],
            "A^-1 - aproximación decimal"
        )

        tabla_matriz(
            resultados["verificacion_inversa"],
            "Verificación A * A^-1"
        )
    else:
        agregar_mensaje_no_disponible("inversa")

    # --------------------------------------------------------
    # AUTOVALORES / AUTOVECTORES
    # --------------------------------------------------------

    elementos.append(PageBreak())
    elementos.append(
        Paragraph(
            "4. Autovalores, autovectores y diagonalización",
            estilos["Seccion"]
        )
    )

    if "autovalores" in resultados:
        resultado_auto = resultados["autovalores"]

        for i, dato in enumerate(
            resultado_auto["autovalores"],
            start=1
        ):
            elementos.append(
                Paragraph(
                    f"Autovalor {i}: {texto_expr(dato['valor'])}",
                    estilos["Subseccion"]
                )
            )

            elementos.append(
                Paragraph(
                    "Multiplicidad algebraica: "
                    f"{dato['multiplicidad_algebraica']}<br/>"
                    "Multiplicidad geométrica: "
                    f"{dato['multiplicidad_geometrica']}",
                    estilos["BodyText"]
                )
            )

            if dato["autovectores"]:
                for j, vector in enumerate(
                    dato["autovectores"],
                    start=1
                ):
                    tabla_matriz(
                        vector,
                        f"Autovector v({i},{j})"
                    )

        diagonalizable = (
            "Sí" if resultado_auto["diagonalizable"] else "No"
        )

        elementos.append(
            Paragraph(
                f"¿La matriz es diagonalizable? {diagonalizable}",
                estilos["BodyText"]
            )
        )

        elementos.append(
            Paragraph(
                "Número de autovectores linealmente independientes: "
                f"{resultado_auto['numero_autovectores_independientes']}",
                estilos["BodyText"]
            )
        )
    else:
        agregar_mensaje_no_disponible("autovalores")

    # --------------------------------------------------------
    # CUATRO SUBESPACIOS FUNDAMENTALES
    # --------------------------------------------------------

    elementos.append(
        Paragraph(
            "5. Cuatro subespacios fundamentales",
            estilos["Seccion"]
        )
    )

    if "subespacios" in resultados:
        sub = resultados["subespacios"]

        elementos.append(
            Paragraph(
                f"Rango(A) = {sub['rango']}<br/>"
                f"dim C(A) = {sub['dimension_columna']}<br/>"
                f"dim N(A) = {sub['nulidad']}<br/>"
                f"dim C(A^T) = {sub['dimension_fila']}<br/>"
                f"dim N(A^T) = {sub['nulidad_izquierda']}",
                estilos["BodyText"]
            )
        )

        columnas_pivote = [
            indice + 1 for indice in sub["columnas_pivote"]
        ]

        elementos.append(
            Paragraph(
                "Columnas pivote de A: "
                + (
                    ", ".join(map(str, columnas_pivote))
                    if columnas_pivote
                    else "ninguna"
                ),
                estilos["BodyText"]
            )
        )

        espacios = [
            ("Base del espacio columna C(A)", sub["espacio_columna"]),
            ("Base del espacio nulo N(A)", sub["espacio_nulo"]),
            ("Base del espacio fila C(A^T)", sub["espacio_fila"]),
            (
                "Base del espacio nulo izquierdo N(A^T)",
                sub["espacio_nulo_izquierdo"]
            ),
        ]

        for nombre, base in espacios:
            elementos.append(
                Paragraph(nombre, estilos["Subseccion"])
            )

            if base:
                for i, vector in enumerate(base, start=1):
                    tabla_matriz(vector, f"Vector {i}")
            else:
                elementos.append(
                    Paragraph(
                        "El subespacio contiene únicamente al vector cero; "
                        "su base es el conjunto vacío.",
                        estilos["BodyText"]
                    )
                )
                elementos.append(Spacer(1, 6))

        elementos.append(
            Paragraph(
                "Teorema rango-nulidad: "
                f"{sub['rango']} + {sub['nulidad']} = {A.cols}",
                estilos["BodyText"]
            )
        )

        elementos.append(
            Paragraph(
                "Para A^T: "
                f"{sub['rango']} + {sub['nulidad_izquierda']} = {A.rows}",
                estilos["BodyText"]
            )
        )
    else:
        agregar_mensaje_no_disponible("subespacios")

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    elementos.append(PageBreak())
    elementos.append(
        Paragraph(
            "6. Descomposición en valores singulares (SVD)",
            estilos["Seccion"]
        )
    )

    if "svd" in resultados:
        resultado_svd = resultados["svd"]

        valores = [
            f"{float(v):.6f}"
            for v in resultado_svd["valores_singulares"]
        ]

        elementos.append(
            Paragraph(
                "Valores singulares: " + ", ".join(valores),
                estilos["BodyText"]
            )
        )

        elementos.append(Spacer(1, 6))

        tabla_matriz(resultados["U"], "Matriz U")
        tabla_matriz(resultados["Sigma"], "Matriz Sigma")
        tabla_matriz(resultados["Vt"], "Matriz V^T")
        tabla_matriz(
            resultados["reconstruida"],
            "Reconstrucción U * Sigma * V^T"
        )

        elementos.append(
            Paragraph(
                "Error de reconstrucción: "
                f"{resultado_svd['error']:.3e}",
                estilos["BodyText"]
            )
        )
    else:
        agregar_mensaje_no_disponible("svd")

    documento.build(elementos)

    buffer.seek(0)
    return buffer.getvalue()


def mostrar_resultado_suma(A, B, resultado):

    st.markdown(
        '<div class="resultado-titulo">✓ Resultado</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Matrices ingresadas")

    col1, col2 = st.columns(2)

    with col1:
        mostrar_matriz(
            "A",
            A,
            "Matriz A"
        )

    with col2:
        mostrar_matriz(
            "B",
            B,
            "Matriz B"
        )

    st.divider()

    st.markdown("### Resultado de la operación")

    expresion = (
        rf"A + B = "
        rf"{matriz_a_latex(A)}"
        rf" + "
        rf"{matriz_a_latex(B)}"
        rf" = "
        rf"{matriz_a_latex(resultado)}"
    )

    st.latex(expresion)


def mostrar_resultado_multiplicacion(A, B, resultado):

    st.markdown(
        '<div class="resultado-titulo">✓ Resultado</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Matrices ingresadas")

    col1, col2 = st.columns(2)

    with col1:
        mostrar_matriz(
            "A",
            A,
            "Matriz A"
        )

    with col2:
        mostrar_matriz(
            "B",
            B,
            "Matriz B"
        )

    st.divider()

    st.markdown("### Resultado de la operación")

    expresion = (
        rf"AB = "
        rf"{matriz_a_latex(A)}"
        rf"{matriz_a_latex(B)}"
        rf" = "
        rf"{matriz_a_latex(resultado)}"
    )

    st.latex(expresion)


def mostrar_resultado_transpuesta(A, resultado):

    st.markdown(
        '<div class="resultado-titulo">✓ Resultado</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        mostrar_matriz(
            "A",
            A,
            "Matriz A (original)"
        )

    with col2:

        mostrar_matriz(
            r"A^T",
            resultado,
            "Transpuesta de A"
        )

    st.divider()

    st.markdown("### Operación")

    st.latex(
        rf"A^T = "
        rf"{matriz_a_latex(resultado)}"
    )

def controles_generacion_aleatoria_vector(nombre, filas):
    """
    Permite completar un vector columna con enteros aleatorios dentro
    de un intervalo definido por el usuario.
    """

    with st.expander("🎲 Generar vector aleatoriamente"):

        st.caption(
            "Defina el intervalo de valores enteros y pulse Generar. "
            "Luego puede modificar manualmente cualquier componente."
        )

        c1, c2 = st.columns(2)

        with c1:
            minimo = st.number_input(
                "Valor mínimo",
                value=-5,
                step=1,
                key=f"random_vector_{nombre}_min"
            )

        with c2:
            maximo = st.number_input(
                "Valor máximo",
                value=5,
                step=1,
                key=f"random_vector_{nombre}_max"
            )

        generar = st.button(
            "🎲 Generar vector",
            use_container_width=True,
            key=f"random_vector_{nombre}_btn"
        )

        if generar:

            minimo = int(minimo)
            maximo = int(maximo)

            if minimo > maximo:
                st.error(
                    "El valor mínimo no puede ser mayor que el valor máximo."
                )
            else:
                rng = np.random.default_rng()
                valores = rng.integers(
                    minimo,
                    maximo + 1,
                    size=filas
                )

                for i in range(filas):
                    st.session_state[f"{nombre}_{i}"] = str(
                        int(valores[i])
                    )

                st.success(
                    f"Vector {nombre} generado con valores entre "
                    f"{minimo} y {maximo}."
                )


def ingresar_vector(nombre, filas):
    """
    Permite ingresar un vector columna manualmente, generarlo de forma
    aleatoria o cargarlo desde un CSV separado por comas.

    Cuando una operación utiliza dos vectores (por ejemplo, producto punto),
    cada vector dispone de su propio archivo CSV independiente.
    """

    st.subheader(f"Vector {nombre}")

    modo = st.radio(
        f"Forma de ingreso del vector {nombre}",
        ["Manual / aleatorio", "Archivo CSV"],
        horizontal=True,
        key=f"modo_vector_{nombre}",
    )

    if modo == "Archivo CSV":

        st.caption(
            "El CSV debe estar separado por comas y contener una sola fila "
            "o una sola columna, sin encabezados."
        )

        archivo = st.file_uploader(
            f"Cargar CSV del vector {nombre}",
            type=["csv"],
            key=f"csv_vector_{nombre}",
        )

        if archivo is None:
            st.info(
                f"Seleccione el archivo CSV correspondiente al vector {nombre}."
            )
            return None

        try:
            vector = cargar_vector_csv(
                archivo,
                nombre,
            )

            st.success(
                f"Vector {nombre} cargado correctamente: "
                f"dimensión {vector.rows}."
            )

            if vector.rows <= 20:
                mostrar_matriz(
                    nombre,
                    vector,
                    f"Vista previa de {nombre}",
                )
            else:
                st.caption(
                    "Vista previa parcial (primeros 20 elementos)."
                )
                st.dataframe(
                    [[str(vector[i, 0])] for i in range(min(20, vector.rows))],
                    use_container_width=True,
                )

            return vector

        except ValueError as error:
            st.error(str(error))
            return None

    # --------------------------------------------------------
    # INGRESO MANUAL / ALEATORIO
    # --------------------------------------------------------

    valores = []
    hay_error = False

    controles_generacion_aleatoria_vector(
        nombre,
        filas
    )

    for i in range(filas):

        texto = st.text_input(
            f"{nombre}[{i + 1}]",
            value="0",
            key=f"{nombre}_{i}"
        )

        try:
            valor = convertir_a_numero(texto)

        except ValueError as error:
            st.error(str(error))
            valor = sp.Integer(0)
            hay_error = True

        valores.append(valor)

    if hay_error:
        return None

    return sp.Matrix(valores)


def ingresar_vectores(cantidad, dimension):
    """
    Permite ingresar varios vectores columna.
    """

    vectores = []

    columnas = st.columns(cantidad)

    for j in range(cantidad):

        with columnas[j]:

            vector = ingresar_vector(
                f"v{j + 1}",
                dimension
            )

            vectores.append(vector)

    return vectores



# ============================================================
# ESTADO DE LA APLICACIÓN
# ============================================================

if "operacion" not in st.session_state:
    st.session_state.operacion = "Suma de matrices"

operacion = st.session_state.operacion

# ============================================================
# SIDEBAR
# ============================================================

def cambiar_operacion(nombre_operacion):
    """
    Cambia la operación seleccionada en el menú lateral.
    """
    st.session_state.operacion = nombre_operacion


with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            🧮 Calculadora de<br>
            Álgebra Lineal
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">OPERACIONES CON MATRICES</div>',
        unsafe_allow_html=True
    )

    st.button(
        "➕  Suma de matrices",
        use_container_width=True,
        key="btn_suma",
        on_click=cambiar_operacion,
        args=("Suma de matrices",)
    )

    st.button(
        "✕  Multiplicación de matrices",
        use_container_width=True,
        key="btn_multiplicacion",
        on_click=cambiar_operacion,
        args=("Multiplicación de matrices",)
    )

    st.button(
        "▦  Análisis de una matriz",
        use_container_width=True,
        key="btn_analisis",
        on_click=cambiar_operacion,
        args=("Análisis de una matriz",)
    )

    st.markdown(
        '<div class="sidebar-section">SISTEMAS Y VECTORES</div>',
        unsafe_allow_html=True
    )

    st.button(
        "Ax = b  Sistemas lineales",
        use_container_width=True,
        key="btn_sistemas",
        on_click=cambiar_operacion,
        args=("Sistemas lineales",)
    )

    st.button(
        "⊥  Ortogonalidad",
        use_container_width=True,
        key="btn_ortogonalidad",
        on_click=cambiar_operacion,
        args=("Ortogonalidad",)
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            <b>Trabajo Práctico</b><br>
            Calculadora de Matrices y Álgebra Lineal
            <br><br>
            Autor:<br>
            <span style="color:#6EA8FE;">
                Atilio David Gómez Riveros
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ENCABEZADO
# ============================================================

st.title("🧮 Calculadora de Álgebra Lineal")

st.markdown(
    """
    <div class="subtitulo">
        Trabajo Práctico - Calculadora de Matrices y Álgebra Lineal
    </div>

    <div class="autor">
       Atilio David Gómez Riveros
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NOTA SOBRE ARCHIVOS CSV
# ============================================================

# Las matrices de ingreso manual/aleatorio se mantienen limitadas a 10 × 10
# por comodidad de la interfaz. La carga CSV no tiene ese límite y detecta
# automáticamente las dimensiones del archivo.

# ============================================================
# OPERACIÓN ACTUAL
# ============================================================

operacion = st.session_state.operacion


# ============================================================
# SUMA
# ============================================================

if operacion == "Suma de matrices":

    st.header("➕ Suma de matrices")

    st.markdown(
        """
        <div class="info-box">
        Para sumar dos matrices, ambas deben tener
        exactamente las mismas dimensiones.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Dimensiones de las matrices")

    st.caption("Las dimensiones siguientes se aplican al ingreso manual/aleatorio. Si utiliza CSV, se detectan automáticamente.")

    c1, c2 = st.columns(2)

    with c1:

        filas = st.number_input(
            "Filas",
            min_value=1,
            max_value=10,
            value=2,
            key="suma_filas"
        )

    with c2:

        columnas = st.number_input(
            "Columnas",
            min_value=1,
            max_value=10,
            value=2,
            key="suma_columnas"
        )

    filas = int(filas)
    columnas = int(columnas)

    st.divider()

    mostrar_ayuda_entrada()

    col_A, col_B = st.columns(2)

    with col_A:

        A = ingresar_matriz(
            "A",
            filas,
            columnas
        )

    with col_B:

        B = ingresar_matriz(
            "B",
            filas,
            columnas
        )

    st.write("")

    if st.button(
        "🧮 Calcular A + B",
        type="primary"
    ):
        if A is None or B is None:
            st.error("Corrija los valores inválidos antes de realizar la operación")

        else:
            try:

                resultado = sumar_matrices(A, B)

                st.divider()

                mostrar_resultado_suma(
                    A,
                    B,
                    resultado
                )

            except ValueError as error:

                st.error(str(error))


# ============================================================
# MULTIPLICACIÓN
# ============================================================

elif operacion == "Multiplicación de matrices":

    st.header("✕ Multiplicación de matrices")

    st.markdown(
        """
        <div class="info-box">
        Para multiplicar <b>A × B</b>, el número de columnas
        de A debe ser igual al número de filas de B.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Dimensiones")

    st.caption("Las dimensiones siguientes se aplican al ingreso manual/aleatorio. Si utiliza CSV, se detectan automáticamente para cada archivo.")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("#### Matriz A")

        filas_A = st.number_input(
            "Filas de A",
            min_value=1,
            max_value=10,
            value=2
        )

        columnas_A = st.number_input(
            "Columnas de A",
            min_value=1,
            max_value=10,
            value=3
        )

    with c2:

        st.markdown("#### Matriz B")

        filas_B = st.number_input(
            "Filas de B",
            min_value=1,
            max_value=10,
            value=3
        )

        columnas_B = st.number_input(
            "Columnas de B",
            min_value=1,
            max_value=10,
            value=2
        )

    filas_A = int(filas_A)
    columnas_A = int(columnas_A)

    filas_B = int(filas_B)
    columnas_B = int(columnas_B)

    st.divider()

    mostrar_ayuda_entrada()

    col_A, col_B = st.columns(2)

    with col_A:

        A = ingresar_matriz(
            "A",
            filas_A,
            columnas_A
        )

    with col_B:

        B = ingresar_matriz(
            "B",
            filas_B,
            columnas_B
        )

    st.write("")

    if st.button(
        "🧮 Calcular A × B",
        type="primary"
    ):
        if A is None or B is None:
            st.error("Corrija los valores invalidos antes de realizar la operación")

        else:

            try:

                resultado = multiplicar_matrices(
                    A,
                    B
                )

                st.divider()

                mostrar_resultado_multiplicacion(
                    A,
                    B,
                    resultado
                )

            except ValueError as error:

                st.error(str(error))


# ============================================================
# ANÁLISIS DE UNA MATRIZ
# ============================================================

elif operacion == "Análisis de una matriz":

    st.header("▦ Análisis de una matriz")

    st.markdown(
        """
        <div class="info-box">
        Ingrese una única matriz <b>A</b>. La calculadora intentará obtener
        automáticamente su <b>transpuesta</b>, <b>determinante</b>,
        <b>matriz inversa</b>, <b>autovalores y autovectores</b>, los
        <b>cuatro subespacios fundamentales</b> y la
        <b>descomposición en valores singulares (SVD)</b>.
        <br><br>
        Si alguna operación no está definida para la matriz ingresada,
        se mostrará el motivo sin impedir el cálculo de las demás.
        </div>
        """,
        unsafe_allow_html=True
    )

    mostrar_ayuda_entrada()

    st.markdown("### Dimensiones de la matriz A")

    c1, c2 = st.columns(2)

    with c1:
        filas = st.number_input(
            "Número de filas",
            min_value=1,
            max_value=10,
            value=3,
            key="analisis_filas"
        )

    with c2:
        columnas = st.number_input(
            "Número de columnas",
            min_value=1,
            max_value=10,
            value=3,
            key="analisis_columnas"
        )

    filas = int(filas)
    columnas = int(columnas)

    st.divider()

    A = ingresar_matriz(
        "A",
        filas,
        columnas
    )

    st.write("")

    if st.button(
        "🧮 Analizar matriz",
        type="primary",
        use_container_width=True
    ):

        if A is None:

            st.error(
                "Corrija los valores inválidos antes de analizar la matriz."
            )

        else:

            st.divider()

            st.markdown(
                '<div class="resultado-titulo">✓ Análisis de la matriz</div>',
                unsafe_allow_html=True
            )

            mostrar_matriz(
                "A",
                A,
                "Matriz ingresada"
            )

            st.caption(
                f"Dimensión de A: {A.rows} × {A.cols}"
            )

            # ----------------------------------------------------
            # CALCULAR TODAS LAS OPERACIONES DE FORMA INDEPENDIENTE
            # ----------------------------------------------------

            resultados = {}
            errores = {}

            # Transpuesta
            try:
                resultados["transpuesta"] = transponer_matriz(A)
            except ValueError as error:
                errores["transpuesta"] = str(error)

            # Determinante
            try:
                resultados["determinante"] = determinante_matriz(A)
            except ValueError as error:
                errores["determinante"] = str(error)

            # Inversa
            try:
                inversa = matriz_inversa(A)
                resultados["inversa"] = inversa
                resultados["inversa_simplificada"] = simplificar_matriz_visual(
                    inversa
                )
                resultados["inversa_decimal"] = matriz_decimal(
                    inversa,
                    digitos=6
                )
                resultados["verificacion_inversa"] = sp.simplify(
                    A * inversa
                )
            except ValueError as error:
                errores["inversa"] = str(error)

            # Autovalores y autovectores
            try:
                resultados["autovalores"] = analizar_autovalores(A)
            except ValueError as error:
                errores["autovalores"] = str(error)

            # Cuatro subespacios fundamentales
            try:
                rango = A.rank()
                _, columnas_pivote = A.rref()

                resultados["subespacios"] = {
                    "rango": rango,
                    "columnas_pivote": columnas_pivote,
                    "espacio_columna": A.columnspace(),
                    "espacio_fila": A.rowspace(),
                    "espacio_nulo": A.nullspace(),
                    "espacio_nulo_izquierdo": A.T.nullspace(),
                    "dimension_columna": rango,
                    "dimension_fila": rango,
                    "nulidad": A.cols - rango,
                    "nulidad_izquierda": A.rows - rango,
                }
            except Exception as error:
                errores["subespacios"] = (
                    "No fue posible calcular los cuatro subespacios "
                    f"fundamentales: {error}"
                )

            # SVD
            try:
                resultado_svd = descomposicion_svd(A)

                resultados["svd"] = resultado_svd
                resultados["U"] = numpy_a_sympy(
                    resultado_svd["U"]
                )
                resultados["Sigma"] = numpy_a_sympy(
                    resultado_svd["Sigma"]
                )
                resultados["Vt"] = numpy_a_sympy(
                    resultado_svd["Vt"]
                )
                resultados["reconstruida"] = numpy_a_sympy(
                    resultado_svd["reconstruida"]
                )

            except ValueError as error:
                errores["svd"] = str(error)

            except np.linalg.LinAlgError:
                errores["svd"] = (
                    "No fue posible calcular la SVD de la matriz ingresada."
                )

            st.divider()

            # ----------------------------------------------------
            # DESCARGA DEL INFORME COMPLETO EN PDF
            # ----------------------------------------------------

            st.markdown("### 📄 Descargar resultados")

            st.caption(
                "El informe PDF incluye la matriz ingresada y todos "
                "los cálculos disponibles del análisis."
            )

            try:
                pdf_analisis = generar_pdf_analisis(
                    A,
                    resultados,
                    errores
                )

                st.download_button(
                    label="⬇️ Descargar análisis en PDF",
                    data=pdf_analisis,
                    file_name="analisis_matriz.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    on_click="ignore"
                )

            except ValueError as error:
                st.warning(str(error))

            st.divider()

            # Los resultados se organizan en pestañas para mantener
            # una única pantalla sin hacerla excesivamente larga.
            (
                tab_transpuesta,
                tab_determinante,
                tab_inversa,
                tab_auto,
                tab_subespacios,
                tab_svd
            ) = st.tabs(
                [
                    "Aᵀ Transpuesta",
                    "det Determinante",
                    "A⁻¹ Inversa",
                    "λ Autovalores",
                    "▦ Subespacios",
                    "Σ SVD"
                ]
            )

            # ====================================================
            # TRANSPUESTA
            # ====================================================

            with tab_transpuesta:

                st.markdown("### Matriz transpuesta")

                if "transpuesta" in resultados:

                    transpuesta = resultados["transpuesta"]

                    mostrar_matriz(
                        r"A^T",
                        transpuesta,
                        "Transpuesta de A"
                    )

                    st.caption(
                        "La transpuesta se obtiene intercambiando "
                        "las filas por las columnas."
                    )

                else:
                    st.info(errores["transpuesta"])

            # ====================================================
            # DETERMINANTE
            # ====================================================

            with tab_determinante:

                st.markdown("### Determinante")

                if "determinante" in resultados:

                    determinante = resultados["determinante"]

                    st.latex(
                        rf"\det(A) = {sp.latex(determinante)}"
                    )

                    if sp.simplify(determinante) == 0:
                        st.warning(
                            "El determinante es 0. Por lo tanto, "
                            "la matriz no es invertible."
                        )
                    else:
                        st.success(
                            "El determinante es distinto de 0."
                        )

                else:
                    st.info(errores["determinante"])

            # ====================================================
            # INVERSA
            # ====================================================

            with tab_inversa:

                st.markdown("### Matriz inversa")

                if "inversa" in resultados:

                    inversa_simplificada = resultados[
                        "inversa_simplificada"
                    ]

                    inversa_decimal = resultados[
                        "inversa_decimal"
                    ]

                    verificacion = resultados[
                        "verificacion_inversa"
                    ]

                    st.markdown("#### Resultado exacto")

                    st.caption(
                        "La calculadora conserva fracciones, raíces "
                        "y constantes en forma exacta."
                    )

                    mostrar_matriz(
                        r"A^{-1}",
                        inversa_simplificada,
                        "Inversa de A"
                    )

                    st.divider()

                    st.markdown("#### Aproximación decimal")

                    st.latex(
                        rf"A^{{-1}} \approx "
                        rf"{sp.latex(inversa_decimal)}"
                    )

                    st.divider()

                    st.markdown("#### Verificación")

                    st.latex(
                        rf"A A^{{-1}} = "
                        rf"{sp.latex(verificacion)}"
                    )

                    if verificacion == sp.eye(A.rows):
                        st.success(
                            "✓ Verificación correcta: A · A⁻¹ = I"
                        )

                else:
                    st.info(errores["inversa"])

            # ====================================================
            # AUTOVALORES Y AUTOVECTORES
            # ====================================================

            with tab_auto:

                st.markdown(
                    "### Autovalores, autovectores y diagonalización"
                )

                if "autovalores" in resultados:

                    resultado_auto = resultados["autovalores"]

                    for i, dato in enumerate(
                        resultado_auto["autovalores"],
                        start=1
                    ):

                        valor = dato["valor"]
                        ma = dato["multiplicidad_algebraica"]
                        mg = dato["multiplicidad_geometrica"]

                        st.markdown(
                            f"#### Autovalor {i}"
                        )

                        st.latex(
                            rf"\lambda_{i} = {sp.latex(valor)}"
                        )

                        c1, c2 = st.columns(2)

                        with c1:
                            st.write(
                                "Multiplicidad algebraica:",
                                ma
                            )

                        with c2:
                            st.write(
                                "Multiplicidad geométrica:",
                                mg
                            )

                        st.markdown("**Base del autoespacio:**")

                        for j, vector in enumerate(
                            dato["autovectores"],
                            start=1
                        ):

                            st.latex(
                                rf"v_{{{i},{j}}} = "
                                rf"{sp.latex(vector)}"
                            )

                        st.divider()

                    st.markdown("### ¿La matriz es diagonalizable?")

                    if resultado_auto["diagonalizable"]:

                        st.success(
                            "✓ Sí. La matriz es diagonalizable."
                        )

                    else:

                        st.error(
                            "✗ No. La matriz no es diagonalizable."
                        )

                    st.write(
                        "Número de autovectores linealmente independientes:",
                        resultado_auto[
                            "numero_autovectores_independientes"
                        ]
                    )

                    st.write(
                        "Orden de la matriz:",
                        A.rows
                    )

                    st.divider()

                    st.markdown("### Verificación")

                    st.caption(
                        "Para cada autovector se verifica que A·v = λ·v."
                    )

                    todos_correctos = True

                    for dato in resultado_auto["autovalores"]:

                        valor = dato["valor"]

                        for vector in dato["autovectores"]:

                            izquierda = sp.simplify(
                                A * vector
                            )

                            derecha = sp.simplify(
                                valor * vector
                            )

                            st.latex(
                                rf"A{sp.latex(vector)} = "
                                rf"{sp.latex(izquierda)}"
                            )

                            st.latex(
                                rf"{sp.latex(valor)}"
                                rf"{sp.latex(vector)} = "
                                rf"{sp.latex(derecha)}"
                            )

                            if izquierda != derecha:
                                todos_correctos = False

                    if todos_correctos:

                        st.success(
                            "✓ Verificación correcta: "
                            "A·v = λ·v para todos los autovectores."
                        )

                else:
                    st.info(errores["autovalores"])

            # ====================================================
            # CUATRO SUBESPACIOS FUNDAMENTALES
            # ====================================================

            with tab_subespacios:

                st.markdown("### Cuatro subespacios fundamentales")

                st.caption(
                    "Para una matriz A de dimensión m × n se analizan "
                    "C(A), N(A), C(Aᵀ) y N(Aᵀ)."
                )

                if "subespacios" in resultados:

                    sub = resultados["subespacios"]
                    rango = sub["rango"]
                    espacio_columna = sub["espacio_columna"]
                    espacio_fila = sub["espacio_fila"]
                    espacio_nulo = sub["espacio_nulo"]
                    espacio_nulo_izquierdo = sub[
                        "espacio_nulo_izquierdo"
                    ]
                    columnas_pivote = sub["columnas_pivote"]

                    st.markdown("#### Resumen de dimensiones")

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric(
                            "dim C(A)",
                            sub["dimension_columna"]
                        )

                    with c2:
                        st.metric(
                            "dim N(A)",
                            sub["nulidad"]
                        )

                    with c3:
                        st.metric(
                            "dim C(Aᵀ)",
                            sub["dimension_fila"]
                        )

                    with c4:
                        st.metric(
                            "dim N(Aᵀ)",
                            sub["nulidad_izquierda"]
                        )

                    st.latex(
                        rf"\operatorname{{rango}}(A) = {rango}"
                    )

                    st.latex(
                        rf"\dim C(A) + \dim N(A) = "
                        rf"{rango} + {A.cols - rango} = {A.cols}"
                    )

                    st.latex(
                        rf"\dim C(A^T) + \dim N(A^T) = "
                        rf"{rango} + {A.rows - rango} = {A.rows}"
                    )

                    st.divider()

                    # --------------------------------------------
                    # ESPACIO COLUMNA
                    # --------------------------------------------

                    st.markdown("#### 1. Espacio columna  C(A)")

                    st.write(
                        f"Es un subespacio de ℝ^{A.rows} y su "
                        f"dimensión es {rango}."
                    )

                    if columnas_pivote:
                        columnas_humanas = [
                            indice + 1 for indice in columnas_pivote
                        ]
                        st.write(
                            "Columnas pivote de A:",
                            columnas_humanas
                        )

                    if espacio_columna:
                        st.markdown("**Una base de C(A) es:**")

                        for i, vector in enumerate(
                            espacio_columna,
                            start=1
                        ):
                            st.latex(
                                rf"c_{i} = {sp.latex(vector)}"
                            )
                    else:
                        st.latex(r"\mathcal{B}_{C(A)} = \varnothing")
                        st.info(
                            "C(A) es el subespacio cero; su base es vacía."
                        )

                    st.divider()

                    # --------------------------------------------
                    # ESPACIO NULO
                    # --------------------------------------------

                    st.markdown("#### 2. Espacio nulo  N(A)")

                    st.write(
                        f"Es un subespacio de ℝ^{A.cols} y su "
                        f"nulidad es {A.cols - rango}."
                    )

                    st.latex(r"N(A)=\{x:Ax=0\}")

                    if espacio_nulo:
                        st.markdown("**Una base de N(A) es:**")

                        for i, vector in enumerate(
                            espacio_nulo,
                            start=1
                        ):
                            st.latex(
                                rf"n_{i} = {sp.latex(vector)}"
                            )

                        st.markdown("**Verificación:**")

                        verificacion_nulo = True

                        for i, vector in enumerate(
                            espacio_nulo,
                            start=1
                        ):
                            producto = (A * vector).applyfunc(
                                sp.simplify
                            )
                            st.latex(
                                rf"A n_{i} = {sp.latex(producto)}"
                            )
                            if producto != sp.zeros(A.rows, 1):
                                verificacion_nulo = False

                        if verificacion_nulo:
                            st.success(
                                "✓ Los vectores de la base satisfacen Ax = 0."
                            )
                    else:
                        st.latex(r"\mathcal{B}_{N(A)} = \varnothing")
                        st.success(
                            "La nulidad es 0; la única solución de Ax = 0 "
                            "es el vector nulo."
                        )

                    st.divider()

                    # --------------------------------------------
                    # ESPACIO FILA
                    # --------------------------------------------

                    st.markdown("#### 3. Espacio fila  C(Aᵀ)")

                    st.write(
                        f"Es un subespacio de ℝ^{A.cols} y su "
                        f"dimensión es {rango}."
                    )

                    if espacio_fila:
                        st.markdown("**Una base de C(Aᵀ) es:**")

                        for i, vector in enumerate(
                            espacio_fila,
                            start=1
                        ):
                            st.latex(
                                rf"f_{i} = {sp.latex(vector)}"
                            )
                    else:
                        st.latex(r"\mathcal{B}_{C(A^T)} = \varnothing")
                        st.info(
                            "C(Aᵀ) es el subespacio cero; su base es vacía."
                        )

                    st.divider()

                    # --------------------------------------------
                    # ESPACIO NULO IZQUIERDO
                    # --------------------------------------------

                    st.markdown("#### 4. Espacio nulo izquierdo  N(Aᵀ)")

                    st.write(
                        f"Es un subespacio de ℝ^{A.rows} y su "
                        f"dimensión es {A.rows - rango}."
                    )

                    st.latex(r"N(A^T)=\{y:A^Ty=0\}")

                    if espacio_nulo_izquierdo:
                        st.markdown("**Una base de N(Aᵀ) es:**")

                        for i, vector in enumerate(
                            espacio_nulo_izquierdo,
                            start=1
                        ):
                            st.latex(
                                rf"\ell_{i} = {sp.latex(vector)}"
                            )

                        st.markdown("**Verificación:**")

                        verificacion_izquierda = True

                        for i, vector in enumerate(
                            espacio_nulo_izquierdo,
                            start=1
                        ):
                            producto = (A.T * vector).applyfunc(
                                sp.simplify
                            )
                            st.latex(
                                rf"A^T \ell_{i} = "
                                rf"{sp.latex(producto)}"
                            )
                            if producto != sp.zeros(A.cols, 1):
                                verificacion_izquierda = False

                        if verificacion_izquierda:
                            st.success(
                                "✓ Los vectores de la base satisfacen "
                                "Aᵀy = 0."
                            )
                    else:
                        st.latex(r"\mathcal{B}_{N(A^T)} = \varnothing")
                        st.success(
                            "La nulidad izquierda es 0; la única solución "
                            "de Aᵀy = 0 es el vector nulo."
                        )

                else:
                    st.info(errores["subespacios"])

            # ====================================================
            # SVD
            # ====================================================

            with tab_svd:

                st.markdown(
                    "### Descomposición en Valores Singulares"
                )

                if "svd" in resultados:

                    resultado_svd = resultados["svd"]
                    U = resultados["U"]
                    Sigma = resultados["Sigma"]
                    Vt = resultados["Vt"]
                    reconstruida = resultados["reconstruida"]

                    st.latex(
                        r"A = U\Sigma V^T"
                    )

                    st.markdown("#### Valores singulares")

                    for i, valor in enumerate(
                        resultado_svd["valores_singulares"],
                        start=1
                    ):

                        st.latex(
                            rf"\sigma_{i} \approx {valor:.6f}"
                        )

                    st.divider()

                    st.markdown("#### Matrices de la descomposición")

                    mostrar_matriz(
                        "U",
                        U,
                        "Matriz U"
                    )
                    st.caption(
                        f"Dimensión: {U.rows} × {U.cols}"
                    )

                    mostrar_matriz(
                        r"\Sigma",
                        Sigma,
                        "Matriz Σ"
                    )
                    st.caption(
                        f"Dimensión: {Sigma.rows} × {Sigma.cols}"
                    )

                    mostrar_matriz(
                        r"V^T",
                        Vt,
                        "Matriz Vᵀ"
                    )
                    st.caption(
                        f"Dimensión: {Vt.rows} × {Vt.cols}"
                    )

                    st.divider()

                    st.markdown("#### Verificación")

                    st.caption(
                        "Se reconstruye la matriz mediante "
                        "U · Σ · Vᵀ y se compara con A."
                    )

                    st.latex(
                        rf"U\Sigma V^T \approx "
                        rf"{sp.latex(reconstruida)}"
                    )

                    error = resultado_svd["error"]

                    st.latex(
                        rf"\|A-U\Sigma V^T\| "
                        rf"\approx {error:.3e}"
                    )

                    if error < 1e-8:

                        st.success(
                            "✓ Verificación correcta: A ≈ U · Σ · Vᵀ"
                        )

                    else:

                        st.warning(
                            "La reconstrucción presenta un error "
                            "numérico mayor al esperado."
                        )

                else:
                    st.info(errores["svd"])


elif operacion == "Sistemas lineales":

    st.header("Ax = b  Sistemas de ecuaciones lineales")

    st.markdown(
        """
        <div class="info-box">
        Ingrese la matriz de coeficientes <b>A</b>
        y el vector de términos independientes <b>b</b>.
        La calculadora resolverá el sistema
        <b>Ax = b</b> y determinará si posee
        solución única, infinitas soluciones o ninguna solución.
        </div>
        """,
        unsafe_allow_html=True
    )

    mostrar_ayuda_entrada()

    # ----------------------------------------------------
    # DIMENSIONES
    # ----------------------------------------------------

    st.markdown("### Dimensiones del sistema")

    c1, c2 = st.columns(2)

    with c1:

        numero_ecuaciones = st.number_input(
            "Número de ecuaciones",
            min_value=1,
            max_value=10,
            value=2,
            key="sistema_ecuaciones"
        )

    with c2:

        numero_incognitas = st.number_input(
            "Número de incógnitas",
            min_value=1,
            max_value=10,
            value=2,
            key="sistema_incognitas"
        )

    numero_ecuaciones = int(numero_ecuaciones)
    numero_incognitas = int(numero_incognitas)

    st.divider()

    # ----------------------------------------------------
    # INGRESO A Y b
    # ----------------------------------------------------

    col_A, col_b = st.columns([3, 1])

    with col_A:

        A = ingresar_matriz(
            "A",
            numero_ecuaciones,
            numero_incognitas
        )

    with col_b:

        b = ingresar_vector(
            "b",
            numero_ecuaciones
        )

    st.write("")

    # ----------------------------------------------------
    # BOTÓN
    # ----------------------------------------------------

    if st.button(
        "🧮 Resolver sistema",
        type="primary"
    ):

        if A is None or b is None:

            st.error(
                "Corrija los valores inválidos antes "
                "de resolver el sistema."
            )

        else:

            try:

                resultado = resolver_sistema(
                    A,
                    b
                )

                st.divider()

                st.markdown(
                    '<div class="resultado-titulo">'
                    '✓ Resultado'
                    '</div>',
                    unsafe_allow_html=True
                )

                # ----------------------------------------
                # DATOS INGRESADOS
                # ----------------------------------------

                st.markdown("### Sistema ingresado")

                c1, c2 = st.columns(2)

                with c1:

                    mostrar_matriz(
                        "A",
                        A,
                        "Matriz de coeficientes"
                    )

                with c2:

                    mostrar_matriz(
                        "b",
                        b,
                        "Vector independiente"
                    )

                st.divider()

                # ----------------------------------------
                # CLASIFICACIÓN
                # ----------------------------------------

                st.markdown("### Clasificación del sistema")

                clasificacion = resultado["clasificacion"]

                if clasificacion == "Solución única":

                    st.success(
                        "✓ El sistema tiene una solución única."
                    )

                elif clasificacion == "Infinitas soluciones":

                    st.warning(
                        "El sistema tiene infinitas soluciones."
                    )

                else:

                    st.error(
                        "El sistema no tiene solución."
                    )

                # ----------------------------------------
                # RANGOS
                # ----------------------------------------

                st.markdown("### Análisis de rangos")

                st.latex(
                    rf"\operatorname{{rg}}(A)"
                    rf" = {resultado['rango_A']}"
                )

                st.latex(
                    rf"\operatorname{{rg}}([A|b])"
                    rf" = {resultado['rango_aumentada']}"
                )

                st.write(
                    "Número de incógnitas:",
                    resultado["numero_incognitas"]
                )

                # ----------------------------------------
                # MATRIZ ESCALONADA
                # ----------------------------------------

                st.divider()

                st.markdown(
                    "### Forma escalonada reducida"
                )

                st.latex(
                    sp.latex(
                        resultado["rref"]
                    )
                )

                # ----------------------------------------
                # SOLUCIÓN
                # ----------------------------------------

                if resultado["solucion"] is not None:

                    st.divider()

                    st.markdown("### Solución")

                    st.latex(
                        sp.latex(
                            resultado["solucion"]
                        )
                    )

            except ValueError as error:

                st.error(str(error))

elif operacion == "Ortogonalidad":

    st.header("⊥ Ortogonalidad y Gram–Schmidt")

    st.markdown(
        """
        <div class="info-box">
        Esta sección permite trabajar con producto punto,
        norma, normalización de vectores y el proceso de
        Gram–Schmidt para obtener bases ortogonales y
        ortonormales.
        </div>
        """,
        unsafe_allow_html=True
    )

    mostrar_ayuda_entrada()

    # ----------------------------------------------------
    # TIPO DE OPERACIÓN
    # ----------------------------------------------------

    st.markdown("### Seleccione la operación")

    tipo = st.radio(
        "Operación",
        [
            "Producto punto",
            "Norma y normalización",
            "Gram–Schmidt"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    # ====================================================
    # PRODUCTO PUNTO
    # ====================================================

    if tipo == "Producto punto":

        st.markdown("### Producto punto")

        dimension = st.number_input(
            "Dimensión de los vectores",
            min_value=2,
            max_value=10,
            value=3,
            key="producto_dimension"
        )

        dimension = int(dimension)

        c1, c2 = st.columns(2)

        with c1:
            u = ingresar_vector(
                "u",
                dimension
            )

        with c2:
            v = ingresar_vector(
                "v",
                dimension
            )

        if st.button(
            "🧮 Calcular producto punto",
            type="primary"
        ):

            if u is None or v is None:

                st.error(
                    "Corrija los valores inválidos."
                )

            else:

                try:

                    resultado = producto_punto(
                        u,
                        v
                    )

                    st.divider()

                    st.markdown(
                        '<div class="resultado-titulo">'
                        '✓ Resultado'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    c1, c2 = st.columns(2)

                    with c1:
                        mostrar_matriz(
                            "u",
                            u,
                            "Vector u"
                        )

                    with c2:
                        mostrar_matriz(
                            "v",
                            v,
                            "Vector v"
                        )

                    st.markdown(
                        "### Producto punto"
                    )

                    st.latex(
                        rf"u \cdot v = "
                        rf"{sp.latex(resultado)}"
                    )

                    # Comprobar ortogonalidad

                    if sp.simplify(resultado) == 0:

                        st.success(
                            "✓ Los vectores son ortogonales "
                            "porque u · v = 0."
                        )

                    else:

                        st.info(
                            "Los vectores no son ortogonales "
                            "porque u · v ≠ 0."
                        )

                except ValueError as error:

                    st.error(str(error))

    # ====================================================
    # NORMA Y NORMALIZACIÓN
    # ====================================================

    elif tipo == "Norma y normalización":

        st.markdown(
            "### Norma y normalización"
        )

        dimension = st.number_input(
            "Dimensión del vector",
            min_value=2,
            max_value=10,
            value=3,
            key="norma_dimension"
        )

        dimension = int(dimension)

        v = ingresar_vector(
            "v",
            dimension
        )

        if st.button(
            "🧮 Calcular norma y normalizar",
            type="primary"
        ):

            if v is None:

                st.error(
                    "Corrija los valores inválidos."
                )

            else:

                try:

                    norma = norma_vector(v)

                    normalizado = normalizar_vector(v)

                    st.divider()

                    st.markdown(
                        '<div class="resultado-titulo">'
                        '✓ Resultado'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    mostrar_matriz(
                        "v",
                        v,
                        "Vector original"
                    )

                    st.markdown("### Norma")

                    st.latex(
                        rf"\|v\| = "
                        rf"{sp.latex(norma)}"
                    )

                    st.markdown(
                        "### Vector normalizado"
                    )

                    st.latex(
                        rf"\hat{{v}} = "
                        rf"\frac{{v}}{{\|v\|}}"
                        rf" = "
                        rf"{sp.latex(normalizado)}"
                    )

                    st.markdown("### Verificación")

                    norma_resultado = sp.simplify(
                        normalizado.norm()
                    )

                    st.latex(
                        rf"\|\hat{{v}}\| = "
                        rf"{sp.latex(norma_resultado)}"
                    )

                    if norma_resultado == 1:

                        st.success(
                            "✓ El vector normalizado "
                            "tiene norma 1."
                        )

                except ValueError as error:

                    st.error(str(error))

    # ====================================================
    # GRAM-SCHMIDT
    # ====================================================

    elif tipo == "Gram–Schmidt":

        st.markdown(
            "### Proceso de Gram–Schmidt"
        )

        c1, c2 = st.columns(2)

        with c1:

            dimension = st.number_input(
                "Dimensión del espacio",
                min_value=2,
                max_value=6,
                value=3,
                key="gram_dimension"
            )

        with c2:

            cantidad = st.number_input(
                "Cantidad de vectores",
                min_value=1,
                max_value=int(dimension),
                value=min(2, int(dimension)),
                key="gram_cantidad"
            )

        dimension = int(dimension)
        cantidad = int(cantidad)

        st.divider()

        vectores = ingresar_vectores(
            cantidad,
            dimension
        )

        if st.button(
            "🧮 Aplicar Gram–Schmidt",
            type="primary"
        ):

            if any(
                vector is None
                for vector in vectores
            ):

                st.error(
                    "Corrija los valores inválidos."
                )

            else:

                try:

                    base_ortogonal, base_ortonormal = (
                        gram_schmidt(vectores)
                    )

                    st.divider()

                    st.markdown(
                        '<div class="resultado-titulo">'
                        '✓ Resultado'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    # ------------------------------------
                    # VECTORES ORIGINALES
                    # ------------------------------------

                    st.markdown(
                        "### Vectores ingresados"
                    )

                    for i, vector in enumerate(
                        vectores,
                        start=1
                    ):

                        st.latex(
                            rf"v_{i} = "
                            rf"{sp.latex(vector)}"
                        )

                    st.divider()

                    # ------------------------------------
                    # BASE ORTOGONAL
                    # ------------------------------------

                    st.markdown(
                        "### Base ortogonal"
                    )

                    for i, vector in enumerate(
                        base_ortogonal,
                        start=1
                    ):

                        st.latex(
                            rf"u_{i} = "
                            rf"{sp.latex(vector)}"
                        )

                    st.divider()

                    # ------------------------------------
                    # BASE ORTONORMAL
                    # ------------------------------------

                    st.markdown(
                        "### Base ortonormal"
                    )

                    for i, vector in enumerate(
                        base_ortonormal,
                        start=1
                    ):

                        st.latex(
                            rf"e_{i} = "
                            rf"{sp.latex(vector)}"
                        )

                    # ------------------------------------
                    # VERIFICACIÓN
                    # ------------------------------------

                    st.divider()

                    st.markdown(
                        "### Verificación"
                    )

                    correcto = True

                    # Ortogonalidad

                    for i in range(
                        len(base_ortonormal)
                    ):

                        for j in range(
                            i + 1,
                            len(base_ortonormal)
                        ):

                            producto = sp.simplify(
                                base_ortonormal[i].dot(
                                    base_ortonormal[j]
                                )
                            )

                            st.latex(
                                rf"e_{i+1}"
                                rf"\cdot "
                                rf"e_{j+1}"
                                rf" = "
                                rf"{sp.latex(producto)}"
                            )

                            if producto != 0:
                                correcto = False

                    # Normas

                    for i, vector in enumerate(
                        base_ortonormal,
                        start=1
                    ):

                        norma = sp.simplify(
                            vector.norm()
                        )

                        st.latex(
                            rf"\|e_{i}\| = "
                            rf"{sp.latex(norma)}"
                        )

                        if norma != 1:
                            correcto = False

                    if correcto:

                        st.success(
                            "✓ La base obtenida es "
                            "ortonormal."
                        )

                except ValueError as error:

                    st.error(str(error))

