import time
import numpy as np


def main():
    print("=== SYNTECXHUB - PROJECT 1: NUMPY DATA EXPLORER ===")

    # -------------------------------------------------------------
    # 1. Manipulation fondamentale : Création, Indexation, Slicing
    # -------------------------------------------------------------
    print("\n--- 1. Création, Indexation et Slicing ---")

    # Création d'un tableau 1D et 2D
    arr1d = np.array([10, 20, 30, 40, 50, 60])
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(f"Tableau 1D : {arr1d}")
    print(f"Indexation (élément à l'index 2) : {arr1d[2]}")
    print(f"Slicing 1D (éléments 1 à 4) : {arr1d[1:4]}")

    print("\nTableau 2D :\n", arr2d)
    print(f"Élément ligne 1, colonne 2 : {arr2d[1, 2]}")
    print("Sous-matrice (2 premières lignes, 2 dernières colonnes) :\n", arr2d[:2, 1:])

    # -------------------------------------------------------------
    # 2. Opérations mathématiques, statistiques et par axe
    # -------------------------------------------------------------
    print("\n--- 2. Opérations Mathématiques et Statistiques ---")

    dataset = np.array([[15, 25, 35], [45, 55, 65], [75, 85, 95]])

    # Statistiques globales
    print(f"Moyenne globale : {np.mean(dataset):.2f}")
    print(f"Écart-type : {np.std(dataset):.2f}")
    print(f"Minimum : {np.min(dataset)} | Maximum : {np.max(dataset)}")

    # Opérations par axe (axis 0 = colonnes, axis 1 = lignes)
    print(f"Somme par colonne (axis=0) : {np.sum(dataset, axis=0)}")
    print(f"Moyenne par ligne (axis=1) : {np.mean(dataset, axis=1)}")

    # -------------------------------------------------------------
    # 3. Reshaping et Broadcasting
    # -------------------------------------------------------------
    print("\n--- 3. Reshaping et Broadcasting ---")

    # Reshaping : transformer un tableau 1D de 12 éléments en matrice 3x4
    raw_data = np.arange(12)
    reshaped_data = raw_data.reshape(3, 4)
    print("Tableau original (1D) :", raw_data)
    print("Après Reshape (3x4) :\n", reshaped_data)

    # Broadcasting : ajouter un vecteur 1D à chaque ligne d'une matrice 2D
    matrix = np.ones((3, 3)) * 10
    vector = np.array([1, 2, 3])
    broadcasted_result = matrix + vector

    print("\nMatrice initiale :\n", matrix)
    print("Vecteur ajouté :", vector)
    print("Résultat du Broadcasting :\n", broadcasted_result)

    # -------------------------------------------------------------
    # 4. Opérations de Sauvegarde et Chargement
    # -------------------------------------------------------------
    print("\n--- 4. Sauvegarde et Chargement ---")

    filename = "data_array.npy"
    np.save(filename, reshaped_data)
    print(f"Tableau sauvegardé sous '{filename}'.")

    loaded_array = np.load(filename)
    print("Tableau rechargé depuis le fichier :\n", loaded_array)

    # -------------------------------------------------------------
    # 5. Comparaison des Performances (NumPy vs Listes Python)
    # -------------------------------------------------------------
    print("\n--- 5. Comparaison des Performances ---")

    size = 10_000_000

    # Test avec liste Python standard
    py_list_a = list(range(size))
    py_list_b = list(range(size))

    start_time = time.time()
    py_result = [a + b for a, b in zip(py_list_a, py_list_b)]
    py_duration = time.time() - start_time

    # Test avec NumPy
    np_arr_a = np.arange(size)
    np_arr_b = np.arange(size)

    start_time = time.time()
    np_result = np_arr_a + np_arr_b
    np_duration = time.time() - start_time

    print(f"Temps d'exécution (Liste Python) : {py_duration:.5f} secondes")
    print(f"Temps d'exécution (NumPy Array)  : {np_duration:.5f} secondes")
    print(f"--> NumPy est environ {py_duration / np_duration:.1f}x plus rapide !")


if __name__ == "__main__":
    main()