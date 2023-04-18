with open("../DDI/resources/DrugBank.txt", "r") as f:
    for line in f:
        etiqueta = line.strip().split("|")[-1]  # Obtener la etiqueta de la línea
        archivo_salida = etiqueta + ".txt"  # Nombre del archivo de salida correspondiente
        with open(archivo_salida, "a") as f_salida:
            f_salida.write(line.split("|")[0] + "\n")