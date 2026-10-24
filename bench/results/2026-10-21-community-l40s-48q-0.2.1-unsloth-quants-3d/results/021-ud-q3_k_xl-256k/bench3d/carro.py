import bpy
import math


def main():
    scene = bpy.context.scene

    # Empty 'Carro' na origem
    carro = bpy.data.objects.new("Carro", None)
    carro.location = (0.0, 0.0, 0.0)
    scene.collection.objects.link(carro)

    # Posições das quatro rodas
    wheel_positions = [
        ( 1.2,  0.8, 0.3),
        ( 1.2, -0.8, 0.3),
        (-1.2,  0.8, 0.3),
        (-1.2, -0.8, 0.3),
    ]

    # Cria as rodas como cilindros com eixo paralelo ao eixo Y
    for i, loc in enumerate(wheel_positions, start=1):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.3,
            depth=0.2,
            location=loc,
            rotation=(math.pi / 2.0, 0.0, 0.0),
        )

        roda = bpy.context.active_object
        if roda is None and bpy.context.selected_objects:
            roda = bpy.context.selected_objects[0]

        roda.name = f"Roda_{i}"

        # Torna a roda filha do Empty, preservando a transformação global
        roda.parent = carro
        roda.matrix_parent_inverse = carro.matrix_world.inverted()


if __name__ == "__main__":
    main()
