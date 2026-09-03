import sympy as sp
import numpy as np


def sumar_matrices(A, B):
    """
    Suma dos matrices.

    Ambas matrices deben tener las mismas dimensiones.
    """

    if A.shape != B.shape:
        raise ValueError(
            "Las matrices deben tener las mismas dimensiones "
            "para poder sumarse."
        )

    return A + B


def multiplicar_matrices(A, B):
    """
    Multiplica las matrices A y B.

    Para que A × B exista:
    columnas de A = filas de B.
    """

    if A.cols != B.rows:
        raise ValueError(
            "El número de columnas de A debe ser igual "
            "al número de filas de B."
        )

    return A * B


def transponer_matriz(A):
    """
    Devuelve la matriz transpuesta de A.
    """

    return A.T


def calcular_operacion_combinada(A, B, operacion):
    """Construye una matriz a partir de A, B y sus transpuestas.

    Las operaciones se identifican con la misma notación que se muestra en
    la interfaz. Se validan las dimensiones mediante las funciones básicas
    para devolver mensajes comprensibles al estudiante.
    """

    operaciones = {
        "A + B": (sumar_matrices, A, B),
        "A + Bᵀ": (sumar_matrices, A, B.T),
        "Aᵀ + B": (sumar_matrices, A.T, B),
        "Aᵀ + Bᵀ": (sumar_matrices, A.T, B.T),
        "A - B": (sumar_matrices, A, -B),
        "A - Bᵀ": (sumar_matrices, A, -B.T),
        "Aᵀ - B": (sumar_matrices, A.T, -B),
        "A × B": (multiplicar_matrices, A, B),
        "A × Bᵀ": (multiplicar_matrices, A, B.T),
        "Aᵀ × B": (multiplicar_matrices, A.T, B),
        "Aᵀ × Bᵀ": (multiplicar_matrices, A.T, B.T),
    }

    if operacion not in operaciones:
        raise ValueError("La operación combinada seleccionada no es válida.")

    funcion, izquierda, derecha = operaciones[operacion]
    return funcion(izquierda, derecha).applyfunc(sp.simplify)


def analizar_subespacios(A):
    """Devuelve las bases y dimensiones de los cuatro subespacios de A."""

    rango = A.rank()
    rref, columnas_pivote = A.rref()
    return {
        "rango": rango,
        "rref": rref,
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


def determinante_matriz(A):
    """
    Calcula el determinante de una matriz cuadrada.
    """

    if A.rows != A.cols:
        raise ValueError(
            "El determinante solo puede calcularse "
            "para matrices cuadradas."
        )

    return A.det()


def matriz_inversa(A):
    """
    Calcula la matriz inversa de A.

    La matriz debe ser cuadrada y tener
    determinante distinto de cero.
    """

    if A.rows != A.cols:
        raise ValueError(
            "La matriz inversa solo existe para matrices cuadradas."
        )

    #determinante = A.det()
    determinante = sp.simplify(A.det())

    if determinante == 0:
        raise ValueError(
            "La matriz no posee inversa porque su determinante es 0."
        )

    return A.inv()

def resolver_sistema(A, b):
    """
    Resuelve y clasifica el sistema lineal Ax = b.

    Clasificaciones posibles:
    - Solución única
    - Infinitas soluciones
    - Sin solución
    """

    if b.cols != 1:
        raise ValueError(
            "b debe ser un vector columna."
        )

    # Validar dimensiones
    if A.rows != b.rows:
        raise ValueError(
            "La matriz A y el vector b deben tener "
            "el mismo número de filas."
        )

    # Matriz aumentada [A | b]
    aumentada = A.row_join(b)

    # Rangos
    rango_A = A.rank()
    rango_aumentada = aumentada.rank()

    numero_incognitas = A.cols

    # ------------------------------------------------
    # SIN SOLUCIÓN
    # ------------------------------------------------

    if rango_A != rango_aumentada:

        return {
            "clasificacion": "Sin solución",
            "rango_A": rango_A,
            "rango_aumentada": rango_aumentada,
            "numero_incognitas": numero_incognitas,
            "solucion": None,
            "rref": aumentada.rref()[0]
        }

    # ------------------------------------------------
    # SOLUCIÓN ÚNICA
    # ------------------------------------------------

    if rango_A == rango_aumentada == numero_incognitas:

        solucion = sp.linsolve(
            (A, b)
        )

        return {
            "clasificacion": "Solución única",
            "rango_A": rango_A,
            "rango_aumentada": rango_aumentada,
            "numero_incognitas": numero_incognitas,
            "solucion": solucion,
            "rref": aumentada.rref()[0]
        }

    # ------------------------------------------------
    # INFINITAS SOLUCIONES
    # ------------------------------------------------

    solucion = sp.linsolve(
        (A, b)
    )

    return {
        "clasificacion": "Infinitas soluciones",
        "rango_A": rango_A,
        "rango_aumentada": rango_aumentada,
        "numero_incognitas": numero_incognitas,
        "solucion": solucion,
        "rref": aumentada.rref()[0]
    }

def analizar_autovalores(A):
    """
    Calcula autovalores, autovectores
    y determina si una matriz es diagonalizable.
    """

    if A.rows != A.cols:
        raise ValueError(
            "Los autovalores y autovectores se calculan "
            "para matrices cuadradas."
        )

    # SymPy devuelve:
    # [(autovalor, multiplicidad_algebraica, [autovectores]), ...]
    datos = A.eigenvects()

    autovalores = []

    total_autovectores_independientes = 0

    for valor, multiplicidad_algebraica, vectores in datos:

        multiplicidad_geometrica = len(vectores)

        total_autovectores_independientes += multiplicidad_geometrica

        autovalores.append(
            {
                "valor": valor,
                "multiplicidad_algebraica": multiplicidad_algebraica,
                "multiplicidad_geometrica": multiplicidad_geometrica,
                "autovectores": vectores,
            }
        )

    diagonalizable = (
        total_autovectores_independientes == A.rows
    )

    return {
        "autovalores": autovalores,
        "diagonalizable": diagonalizable,
        "numero_autovectores_independientes":
            total_autovectores_independientes,
    }

def producto_punto(u, v):
    """
    Calcula el producto punto entre dos vectores.
    """

    if u.cols != 1 or v.cols != 1:
        raise ValueError(
            "El producto punto requiere dos vectores columna."
    )
    

    if u.rows != v.rows:
        raise ValueError(
            "Los vectores deben tener la misma dimensión."
        )

    return sp.simplify(u.dot(v))


def norma_vector(v):
    """
    Calcula la norma euclídea de un vector.
    """

    return sp.simplify(v.norm())


def normalizar_vector(v):
    """
    Devuelve el vector unitario asociado a v.
    """

    norma = norma_vector(v)

    if norma == 0:
        raise ValueError(
            "El vector nulo no puede normalizarse."
        )

    return v.applyfunc(
        lambda x: sp.simplify(x / norma)
    )


def gram_schmidt(vectores):
    """
    Aplica el proceso de Gram-Schmidt.

    Devuelve:
    - una base ortogonal;
    - una base ortonormal.
    """

    if len(vectores) == 0:
        raise ValueError(
            "Debe ingresar al menos un vector."
        )

    dimension = vectores[0].rows

    if any(v.rows != dimension for v in vectores):
        raise ValueError(
            "Todos los vectores deben tener la misma dimensión."
        )

    # Comprobar independencia lineal
    matriz = sp.Matrix.hstack(*vectores)

    if matriz.rank() < len(vectores):
        raise ValueError(
            "Los vectores ingresados son linealmente dependientes. "
            "Gram-Schmidt requiere un conjunto linealmente independiente."
        )

    base_ortogonal = []

    for v in vectores:

        u = v

        for q in base_ortogonal:

            proyeccion = (
                v.dot(q) / q.dot(q)
            ) * q

            u = u - proyeccion

        u = u.applyfunc(sp.simplify)

        base_ortogonal.append(u)

    base_ortonormal = [
        normalizar_vector(v)
        for v in base_ortogonal
    ]

    return base_ortogonal, base_ortonormal


def descomposicion_svd(A, tolerancia=1e-10):
    """
    Calcula la descomposición en valores singulares:

        A = U * Sigma * V^T

    Funciona para matrices cuadradas y rectangulares.

    Devuelve:
    - U
    - Sigma
    - Vt
    - valores singulares
    - matriz reconstruida
    - error de reconstrucción
    """

    # --------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------

    if A.rows == 0 or A.cols == 0:
        raise ValueError(
            "La matriz debe contener al menos un elemento."
        )

    # La implementación de esta calculadora trabaja
    # con SVD real, de acuerdo con A = U Σ V^T.
    for elemento in A:

        valor = complex(sp.N(elemento))

        if abs(valor.imag) > tolerancia:
            raise ValueError(
                "La SVD implementada trabaja con matrices reales."
            )

    # --------------------------------------------------
    # CONVERSIÓN DE SYMPY A NUMPY
    # --------------------------------------------------

    A_np = np.array(
        A.evalf(),
        dtype=float
    )

    # --------------------------------------------------
    # SVD
    # --------------------------------------------------

    try:

        U, valores_singulares, Vt = np.linalg.svd(
            A_np,
            full_matrices=True
        )
    except np.linalg.LinAlgError:
        raise ValueError(
            "No fue posible calcular la descomposicion SVD."
        )

    filas, columnas = A_np.shape

    # --------------------------------------------------
    # CONSTRUIR SIGMA
    # --------------------------------------------------

    Sigma = np.zeros(
        (filas, columnas)
    )

    cantidad = min(
        filas,
        columnas
    )

    Sigma[
        :cantidad,
        :cantidad
    ] = np.diag(
        valores_singulares
    )

    # --------------------------------------------------
    # RECONSTRUCCIÓN
    # --------------------------------------------------

    reconstruida = (
        U
        @ Sigma
        @ Vt
    )

    # --------------------------------------------------
    # ERROR NUMÉRICO
    # --------------------------------------------------

    error = np.linalg.norm(
        A_np - reconstruida
    )

    # Eliminar números extremadamente pequeños
    U[np.abs(U) < tolerancia] = 0
    Sigma[np.abs(Sigma) < tolerancia] = 0
    Vt[np.abs(Vt) < tolerancia] = 0
    reconstruida[
        np.abs(reconstruida) < tolerancia
    ] = 0

    return {
        "U": U,
        "Sigma": Sigma,
        "Vt": Vt,
        "valores_singulares": valores_singulares,
        "reconstruida": reconstruida,
        "error": error
    }
