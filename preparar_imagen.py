"""
preparar_imagen.py — Convierte fotos reales al formato Fashion-MNIST
=====================================================================
Uso:
    python preparar_imagen.py camisa.jpg
    python preparar_imagen.py camisa.jpg --ver          # muestra la imagen final
    python preparar_imagen.py carpeta/                  # procesa toda una carpeta

Genera archivos _mnist.png listos para subir al clasificador.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter


def procesar_imagen(ruta: Path, ver: bool = False) -> Path:
    img = Image.open(ruta).convert("RGBA")

    # ── 1. Fondo blanco debajo de transparencia ──
    fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
    fondo.paste(img, mask=img.split()[3])
    img = fondo.convert("RGB")

    # ── 2. Escala de grises ──
    img = img.convert("L")

    # ── 3. Detectar si el fondo es oscuro o claro muestreando esquinas ──
    arr = np.array(img)
    h, w = arr.shape
    esquinas = [arr[0,0], arr[0,w-1], arr[h-1,0], arr[h-1,w-1],
                arr[0,w//2], arr[h//2,0], arr[h//2,w-1], arr[h-1,w//2]]
    brillo_fondo = np.mean(esquinas)

    # Fashion-MNIST: fondo negro (0), prenda blanca (255)
    # Si el fondo es claro → invertir
    if brillo_fondo > 127:
        img = ImageOps.invert(img)
        arr = np.array(img)

    # ── 4. Recortar al bounding box de la prenda (quitar fondo negro) ──
    umbral = 30  # píxeles más oscuros que esto = fondo
    mascara = arr > umbral
    filas = np.any(mascara, axis=1)
    cols  = np.any(mascara, axis=0)
    if filas.any() and cols.any():
        r0, r1 = np.where(filas)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        # Agregar un pequeño margen
        margen = max(2, int((r1 - r0) * 0.05))
        r0 = max(0, r0 - margen)
        r1 = min(h - 1, r1 + margen)
        c0 = max(0, c0 - margen)
        c1 = min(w - 1, c1 + margen)
        img = Image.fromarray(arr[r0:r1+1, c0:c1+1])

    # ── 5. Redimensionar a 28x28 con padding cuadrado ──
    img.thumbnail((28, 28), Image.LANCZOS)
    resultado = Image.new("L", (28, 28), 0)  # fondo negro
    offset = ((28 - img.width) // 2, (28 - img.height) // 2)
    resultado.paste(img, offset)

    # ── 6. Suavizado leve para reducir ruido ──
    resultado = resultado.filter(ImageFilter.SMOOTH_MORE)

    # ── 7. Guardar ──
    salida = ruta.parent / (ruta.stem + "_mnist.png")
    resultado.save(salida)
    print(f"  ✓  {ruta.name}  →  {salida.name}  (fondo detectado: {'claro' if brillo_fondo > 127 else 'oscuro'})")

    if ver:
        resultado_grande = resultado.resize((280, 280), Image.NEAREST)
        resultado_grande.show()

    return salida


def main():
    parser = argparse.ArgumentParser(description="Convierte fotos al formato Fashion-MNIST (28x28, fondo negro)")
    parser.add_argument("entrada", help="Imagen o carpeta a procesar")
    parser.add_argument("--ver", action="store_true", help="Mostrar imagen procesada")
    args = parser.parse_args()

    ruta = Path(args.entrada)

    if ruta.is_dir():
        extensiones = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imagenes = [f for f in ruta.iterdir() if f.suffix.lower() in extensiones and "_mnist" not in f.stem]
        if not imagenes:
            print("No se encontraron imágenes en la carpeta.")
            sys.exit(1)
        print(f"\nProcesando {len(imagenes)} imagen(es) en '{ruta}'...\n")
        for img in sorted(imagenes):
            try:
                procesar_imagen(img, args.ver)
            except Exception as e:
                print(f"  ✗  {img.name}  →  Error: {e}")
    elif ruta.is_file():
        print(f"\nProcesando '{ruta}'...\n")
        procesar_imagen(ruta, args.ver)
    else:
        print(f"No se encontró: {ruta}")
        sys.exit(1)

    print("\nListo. Sube los archivos _mnist.png al clasificador.")


if __name__ == "__main__":
    main()
